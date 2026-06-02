"""
Parametric kinematic optimization for the Huston2025_C20Bulge model.

Performs multi-dataset parametric optimization (VIRAC2, BRAVA, GIBS) under a
standardized solar and bar coordinate frame, fitting 19 kinematic parameters of
the cylindrical and streaming bulge models.
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, value_and_grad, vmap
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

# Set up project root imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../shared")))
sys.path.append(os.path.abspath(os.path.join(project_root, "synthpop")))

import synthpop
from inversion_config import get_inverter_grid_axes
from jax_kinematic_inverter import BulgeKinematicInverter


# ---------------------------------------------------------------------------
# Masked Parametric Loss Helper
# ---------------------------------------------------------------------------

def get_masked_parametric_nll(
    inverter: BulgeKinematicInverter,
    params: tuple,
    meta_batch: dict,
    obs_mu_pm: jnp.ndarray | None = None,
    obs_cov_pm: jnp.ndarray | None = None,
    obs_mu_err_pm: jnp.ndarray | None = None,
    obs_mu_rv: jnp.ndarray | None = None,
    obs_var_rv: jnp.ndarray | None = None,
    obs_mu_err_rv: jnp.ndarray | None = None,
    mask: jnp.ndarray | None = None,
    rv_weight: float = 1.0,
) -> jnp.ndarray:
    """
    Calculate masked negative log-likelihood for proper motions and/or radial velocities.

    Parameters
    ----------
    inverter : BulgeKinematicInverter
    params : tuple
        Tuple of ``(p_kosh, omega, alpha_deg, log_f_bulge)``.
    meta_batch : dict
        Dict of precomputed line-of-sight coordinates and selection fractions.
    obs_mu_pm : jnp.ndarray, optional
    obs_cov_pm : jnp.ndarray, optional
    obs_mu_err_pm : jnp.ndarray, optional
    obs_mu_rv : jnp.ndarray, optional
    obs_var_rv : jnp.ndarray, optional
    obs_mu_err_rv : jnp.ndarray, optional
    mask : jnp.ndarray, optional
    rv_weight : float, optional

    Returns
    -------
    nll : jnp.ndarray
        Negative log-likelihood value.
    """
    p_kosh, omega, alpha_deg, log_f_bulge = params
    alpha_r = jnp.radians(alpha_deg)
    n_pixels = jnp.size(meta_batch["l"])

    def flatten_spatial(x):
        if x is None:
            return None
        return x.reshape((n_pixels,) + x.shape[len(meta_batch["l"].shape) :])

    flat_meta = {
        k: (v if k == "d" else flatten_spatial(v)) for k, v in meta_batch.items()
    }
    flat_obs_mu_pm = flatten_spatial(obs_mu_pm)
    flat_obs_cov_pm = flatten_spatial(obs_cov_pm)
    flat_obs_mu_err_pm = flatten_spatial(obs_mu_err_pm)
    flat_obs_mu_rv = flatten_spatial(obs_mu_rv)
    flat_obs_var_rv = flatten_spatial(obs_var_rv)
    flat_obs_mu_err_rv = flatten_spatial(obs_mu_err_rv)
    flat_mask = mask.flatten()

    meta_axes = {
        k: (0 if (jnp.ndim(v) > 0 and v.shape[0] == n_pixels) else None)
        for k, v in flat_meta.items()
    }
    preds = vmap(
        lambda m: inverter.predict_parametric(
            p_kosh, omega, alpha_r, m, log_f_bulge=log_f_bulge
        ),
        in_axes=(meta_axes,),
    )(flat_meta)
    pred_mu_pm, pred_cov_pm, pred_mu_rv, pred_var_rv = preds

    nll = 0.0
    if flat_obs_mu_pm is not None:
        diff_pm = flat_obs_mu_pm - pred_mu_pm

        def bin_loss_pm(i, k):
            C_pred = pred_cov_pm[i, k] + jnp.eye(2) * 1e-6
            inv_C = jnp.linalg.inv(C_pred)
            _, logdet = jnp.linalg.slogdet(C_pred)
            n_eff = (
                jnp.mean(
                    jnp.diag(flat_obs_cov_pm[i, k])
                    / (flat_obs_mu_err_pm[i, k] ** 2 + 1e-10)
                )
                if flat_obs_mu_err_pm is not None
                else 100.0
            )
            n_eff = jnp.clip(n_eff, 1.0, 1e6)
            return n_eff * (
                logdet
                + jnp.trace(inv_C @ flat_obs_cov_pm[i, k])
                + diff_pm[i, k].T @ inv_C @ diff_pm[i, k]
            )

        pixel_losses = vmap(
            lambda i: jnp.sum(
                vmap(lambda k: bin_loss_pm(i, k))(jnp.arange(inverter.n_bins))
            )
        )(jnp.arange(n_pixels))
        nll += jnp.sum(pixel_losses * flat_mask) / (jnp.sum(flat_mask) + 1e-10)

    if flat_obs_mu_rv is not None:
        diff_rv = flat_obs_mu_rv - pred_mu_rv

        def bin_loss_rv(i, k):
            var_pred = pred_var_rv[i, k] + 1e-6
            n_eff = (
                (flat_obs_var_rv[i, k] / (flat_obs_mu_err_rv[i, k] ** 2 + 1e-10))
                if flat_obs_mu_err_rv is not None
                else 100.0
            )
            n_eff = jnp.clip(n_eff, 1.0, 1e6)
            return n_eff * (
                jnp.log(var_pred)
                + (flat_obs_var_rv[i, k] + (diff_rv[i, k] ** 2)) / var_pred
            )

        pixel_losses = vmap(
            lambda i: jnp.sum(
                vmap(lambda k: bin_loss_rv(i, k))(jnp.arange(inverter.n_bins))
            )
        )(jnp.arange(n_pixels))
        nll += rv_weight * jnp.sum(pixel_losses * flat_mask) / (
            jnp.sum(flat_mask) + 1e-10
        )
    return nll


# ---------------------------------------------------------------------------
# Main Optimization Block
# ---------------------------------------------------------------------------

def main() -> None:
    """Run combined parametric kinematic optimization."""
    # 0. Dynamically Extract Parameters from SynthPop Model
    print("Extracting initial parameters from SynthPop...")
    model_name = "Huston2025_C20Bulge"
    config = {
        "MANDATORY": {"name_for_output": "fit_init", "model_name": model_name},
        "SIGHTLINES": {
            "l_set": [0.0],
            "l_set_type": "list",
            "b_set": [0.0],
            "b_set_type": "list",
            "solid_angle": 0.01,
            "solid_angle_unit": "deg^2",
        },
    }
    tmp_conf = "tmp_fit_init.json"
    with open(tmp_conf, "w") as f:
        json.dump(config, f)
    sp = synthpop.SynthPop(tmp_conf)
    sp.init_populations()
    os.remove(tmp_conf)

    bulge_pop = sp.populations[0]
    kin = bulge_pop.kinematics
    dens = bulge_pop.population_density

    # Extract dynamic values
    v0_init = getattr(kin, "v0_stream", 50.0)
    y0_init = getattr(kin, "y0_stream", 341.6)
    if y0_init > 10.0:
        y0_init /= 1000.0  # kpc conversion

    s_i0 = getattr(kin, "sigma_i0", [70.0, 70.0, 70.0])
    s_i1 = getattr(kin, "sigma_i1", [140.0, 90.0, 70.0])
    s_x0, s_y0, s_z0 = s_i0[0], s_i0[1], s_i0[2]
    s_x1, s_y1, s_z1 = s_i1[0], s_i1[1], s_i1[2]

    h0_r = getattr(kin, "h0_r", [8.44, 0.86, 0.92])
    h0_z = getattr(kin, "h0_z", [3.49, 0.60, 2.49])

    C_par_r = getattr(kin, "C_par_r", 1.02)
    C_perp_r = getattr(kin, "C_perp_r", 4.78)
    C_par_z = getattr(kin, "C_par_z", 1.04)
    C_perp_z = getattr(kin, "C_perp_z", 4.39)

    omega_init = getattr(kin, "omega_p", 50.0)
    alpha_rad = getattr(dens, "bar_ang", np.radians(18.43))
    fixed_alpha = np.degrees(alpha_rad)
    x_scale = getattr(dens, "x_scale", 1.0)
    y_scale = getattr(dens, "y_scale", 1.0)
    z_scale = getattr(dens, "z_scale", 1.0)

    # f_bulge is fixed from the SynthPop model to preserve normalisation.
    raw_f_bulge = getattr(kin, "f_bulge", 0.5)
    fixed_log_f_bulge = float(np.log(raw_f_bulge / (1.0 - raw_f_bulge)))  # logit

    solar = {
        "R0": np.abs(sp.parms.sun.gal_dist),
        "U_SUN": sp.parms.sun.u,
        "V_LSR": sp.parms.lsr["v_lsr"],
        "V_SUN_REL": sp.parms.sun.v - sp.parms.lsr["v_lsr"],
        "W_SUN": sp.parms.sun.w,
        "bar_angle_rad": alpha_rad,
        "x_scale": x_scale,
        "y_scale": y_scale,
        "z_scale": z_scale,
    }

    # 1. Load Aligned Observed Data
    print("\nLoading observed data...")
    # --- VIRAC2 ---
    data_v = np.load(
        os.path.join(project_root, "results/reduction/virac2_obs_aligned_h25c20.npz")
    )
    obs_mu_pm_v = jnp.array(data_v["mu_pm"])
    obs_cov_pm_v = (
        jnp.array(data_v["cov_pm"]) + jnp.eye(2)[None, None, :, :] * (0.01**2)
    )
    obs_mu_err_v = jnp.array(data_v["mu_err"])
    mask_v = jnp.array(data_v["mask"])
    b_v_arr = data_v["b"]
    mask_v = mask_v & (jnp.abs(b_v_arr) >= 2.0)

    with open(
        os.path.join(
            project_root, "results/inversion/huston25c20_metadata_science_binned.pkl"
        ),
        "rb",
    ) as f:
        meta_v = pickle.load(f)
    n_bins_v = obs_mu_pm_v.shape[1]

    # --- BRAVA ---
    data_b = np.load(
        os.path.join(project_root, "results/reduction/brava_obs_aligned_h25c20.npz")
    )
    obs_mu_rv_b = jnp.array(data_b["mu_rv"])
    obs_var_rv_b = jnp.array(data_b["var_rv"]) + (1.0**2)
    obs_mu_err_b = jnp.array(data_b["mu_err"])
    mask_b = jnp.array(data_b["mask"])
    with open(
        os.path.join(
            project_root, "results/inversion/huston25c20_brava_obs_metadata.pkl"
        ),
        "rb",
    ) as f:
        meta_b = pickle.load(f)

    # --- GIBS ---
    data_g = np.load(
        os.path.join(project_root, "results/reduction/gibs_obs_aligned_h25c20.npz")
    )
    obs_mu_rv_g = jnp.array(data_g["mu_rv"])
    obs_var_rv_g = jnp.array(data_g["var_rv"]) + (1.0**2)
    obs_mu_err_g = jnp.array(data_g["mu_err"])
    mask_g = jnp.array(data_g["mask"])
    with open(
        os.path.join(
            project_root, "results/inversion/huston25c20_gibs_metadata_science.pkl"
        ),
        "rb",
    ) as f:
        meta_g = pickle.load(f)

    # 2. Setup Inverters
    grid_axes = get_inverter_grid_axes()
    inverter_v = BulgeKinematicInverter(grid_axes, solar, n_bins=n_bins_v)
    inverter_b = BulgeKinematicInverter(grid_axes, solar, n_bins=1)
    inverter_g = BulgeKinematicInverter(grid_axes, solar, n_bins=1)

    # 3. Define Masked Parametric Loss
    @jit
    def combined_loss(params):
        p_kosh = params[:18]
        omega = params[18]
        alpha_d = fixed_alpha
        loss_v = get_masked_parametric_nll(
            inverter_v,
            (p_kosh, omega, alpha_d, fixed_log_f_bulge),
            meta_v,
            obs_mu_pm=obs_mu_pm_v,
            obs_cov_pm=obs_cov_pm_v,
            obs_mu_err_pm=obs_mu_err_v,
            mask=mask_v,
        )
        loss_b = get_masked_parametric_nll(
            inverter_b,
            (p_kosh, omega, alpha_d, fixed_log_f_bulge),
            meta_b,
            obs_mu_rv=obs_mu_rv_b,
            obs_var_rv=obs_var_rv_b,
            obs_mu_err_rv=obs_mu_err_b,
            mask=mask_b,
        )
        loss_g = get_masked_parametric_nll(
            inverter_g,
            (p_kosh, omega, alpha_d, fixed_log_f_bulge),
            meta_g,
            obs_mu_rv=obs_mu_rv_g,
            obs_var_rv=obs_var_rv_g,
            obs_mu_err_rv=obs_mu_err_g,
            mask=mask_g,
        )
        return loss_v + loss_b + loss_g

    val_and_grad_fn = jit(value_and_grad(combined_loss))

    # 4. Optimization
    p_init = np.array([
        v0_init,
        y0_init,  # v0_str, y0_str
        s_x0,
        s_x1,
        s_y0,
        s_y1,
        s_z0,
        s_z1,  # sigmas
        h0_r[0],
        h0_r[1],
        h0_r[2],  # h0_r
        h0_z[0],
        h0_z[1],
        h0_z[2],  # h0_z
        C_par_r,
        C_perp_r,
        C_par_z,
        C_perp_z,  # C params
        omega_init,  # omega
    ])

    bounds = [
        (0, 400),
        (0.01, 10.0),  # v0_str, y0_str
        (10, 400),
        (0, 600),
        (10, 400),
        (0, 600),
        (10, 400),
        (0, 600),  # sigmas
        (0.1, 15.0),
        (0.1, 15.0),
        (0.1, 15.0),  # h0_r
        (0.1, 15.0),
        (0.1, 15.0),
        (0.1, 15.0),  # h0_z
        (0.5, 10.0),
        (0.5, 20.0),
        (0.5, 10.0),
        (0.5, 20.0),  # C params
        (0, 150),  # omega
    ]

    p_names = [
        "v0_str",
        "y0_str",
        "sig_i0_x",
        "sig_i1_x",
        "sig_i0_y",
        "sig_i1_y",
        "sig_i0_z",
        "sig_i1_z",
        "h0_rx",
        "h0_ry",
        "h0_rz",
        "h0_zx",
        "h0_zy",
        "h0_zz",
        "C_par_r",
        "C_perp_r",
        "C_par_z",
        "C_perp_z",
        "omega",
    ]

    print("\n--- Combined OBSERVED DATA Parametric Optimization (H25C20) ---")
    print(f"Alpha dynamically fixed at: {fixed_alpha:.2f} deg")
    print(f"R0 dynamically set at: {solar['R0']:.3f} kpc")
    print(
        f"f_bulge fixed at: {raw_f_bulge:.4f} (logit={fixed_log_f_bulge:.4f}) — not a free parameter"
    )

    def scipy_obj(x):
        l, g = val_and_grad_fn(jnp.array(x))
        if not hasattr(scipy_obj, "n"):
            scipy_obj.n = 0
        if scipy_obj.n % 10 == 0:
            print(
                f"  Iter {scipy_obj.n:3d}: Loss={l:12.4f}, v0={x[0]:.2f}, omega={x[18]:.2f}, f_bulge={raw_f_bulge:.4f} (fixed)"
            )
        scipy_obj.n += 1
        return float(l), np.array(g).astype(np.float64)

    max_iter_val = 5 if os.environ.get("FAST_TEST") == "1" else 10000
    res = minimize(
        scipy_obj,
        p_init,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": max_iter_val, "ftol": 1e-12, "gtol": 1e-12},
    )
    print("\nOptimization Complete.")
    for i, name in enumerate(p_names):
        print(f"{name:<15} | {res.x[i]:10.2f}")

    output_path = os.path.join(
        project_root, "results/inversion/parametric_fit_results_obs_h25c20.npz"
    )
    full_params = np.zeros(21)
    full_params[:19] = res.x  # 19 free params
    full_params[19] = fixed_alpha  # alpha (fixed)
    full_params[20] = raw_f_bulge  # f_bulge (fixed from SynthPop)
    np.savez(output_path, params=full_params, names=p_names + ["alpha", "f_bulge"])
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
