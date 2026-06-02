"""
Joint non-parametric kinematic inversion for the bulge model.

Performs joint non-parametric coordinate inversion using a multi-dataset proper motion
and radial velocity grid (VIRAC2, BRAVA, GIBS), applying an anisotropic KL divergence
penalty relative to the optimized parametric prior.
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
sys.path.append(os.path.join(project_root, "src/shared"))

from inversion_config import get_inverter_grid_axes
from jax_kinematic_inverter import BulgeKinematicInverter


# ---------------------------------------------------------------------------
# Main Inversion Block
# ---------------------------------------------------------------------------

def main() -> None:
    """Run combined non-parametric kinematic inversion."""
    # 0. Dynamically Extract Parameters from our Saved Parametric Fit
    param_path = os.path.join(
        project_root, "results/inversion/parametric_fit_results_obs_h25c20.npz"
    )
    if not os.path.exists(param_path):
        raise FileNotFoundError(
            f"Parametric fit results not found at {param_path}. "
            "Please run fit_obs_parametric_h25c20.py first."
        )

    print(f"Loading parametric fit results from {param_path}...")
    param_data = np.load(param_path)
    p_fitted = jnp.array(param_data["params"])
    sigma_omega = float(param_data.get("sigma_omega", 0.51))  # Dynamic load!

    # Extract parametric parameters
    v0_str, y0_str = p_fitted[0], p_fitted[1]
    sig_i0_x, sig_i1_x, sig_i0_y, sig_i1_y, sig_i0_z, sig_i1_z = p_fitted[2:8]
    h0_r = p_fitted[8:11]
    h0_z = p_fitted[11:14]
    C_par_r, C_perp_r = p_fitted[14], p_fitted[15]
    C_par_z, C_perp_z = p_fitted[16], p_fitted[17]
    omega_fit = p_fitted[18]

    # Extra parameters saved in res.x indices 19 and 20:
    fixed_alpha = float(p_fitted[19])  # 18.43 deg
    raw_f_bulge = float(p_fitted[20])  # 0.50
    fixed_log_f_bulge = float(np.log(raw_f_bulge / (1.0 - raw_f_bulge)))  # logit

    solar = {
        "R0": 8.178,
        "U_SUN": 12.9,
        "V_LSR": 233.4,
        "V_SUN_REL": 12.2,
        "W_SUN": 7.78,
        "bar_angle_rad": jnp.radians(fixed_alpha),
        "x_scale": 1.0,
        "y_scale": 1.0,
        "z_scale": 1.0,
    }

    print(f"Loaded fitted omega: {omega_fit:.2f} km/s/kpc")

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
    inverter_v = BulgeKinematicInverter(
        grid_axes, solar, n_bins=n_bins_v, kl_weight=0.0, curv_weight=1.0
    )
    inverter_b = BulgeKinematicInverter(
        grid_axes, solar, n_bins=1, kl_weight=0.0, curv_weight=1.0
    )
    inverter_g = BulgeKinematicInverter(
        grid_axes, solar, n_bins=1, kl_weight=0.0, curv_weight=1.0
    )

    # 3. Construct Parametric Prior Grids
    X, Y, Z = jnp.meshgrid(grid_axes[0], grid_axes[1], grid_axes[2], indexing="ij")
    ax, ay, az = jnp.abs(X), jnp.abs(Y), jnp.abs(Z)

    s_r = (
        ((ax / h0_r[0]) ** C_perp_r + (ay / h0_r[1]) ** C_perp_r) ** (C_par_r / C_perp_r)
        + (az / h0_r[2]) ** C_par_r
    ) ** (1 / C_par_r)
    s_z = (
        ((ax / h0_z[0]) ** C_perp_z + (ay / h0_z[1]) ** C_perp_z) ** (C_par_z / C_perp_z)
        + (az / h0_z[2]) ** C_par_z
    ) ** (1 / C_par_z)

    # Streaming velocity prior in first octant
    ux_prior = v0_str * (1.0 - jnp.exp(-((Y / y0_str) ** 2)))
    uy_prior = jnp.zeros_like(X)

    # Dispersion priors
    sig_x_prior = sig_i0_x + sig_i1_x * jnp.exp(-s_r)
    sig_y_prior = sig_i0_y + sig_i1_y * jnp.exp(-s_r)
    sig_z_prior = sig_i0_z + sig_i1_z * jnp.exp(-s_z)

    # 4. Custom Anisotropic KL Divergence Penalty
    @jit
    def custom_kl_divergence_penalty(grid_params):
        ux, uy, log_L11, L21, log_L22, log_L33 = (
            grid_params[..., 0],
            grid_params[..., 1],
            grid_params[..., 2],
            grid_params[..., 3],
            grid_params[..., 4],
            grid_params[..., 5],
        )

        L11_sq = jnp.exp(2 * log_L11)
        L22_sq = jnp.exp(2 * log_L22)
        L33_sq = jnp.exp(2 * log_L33)
        L21_sq = L21**2

        tr_term = (
            L11_sq / (sig_x_prior**2)
            + (L21_sq + L22_sq) / (sig_y_prior**2)
            + L33_sq / (sig_z_prior**2)
        )
        mu_term = (ux - ux_prior) ** 2 / (sig_x_prior**2) + (uy - uy_prior) ** 2 / (
            sig_y_prior**2
        )

        log_det_prior = 2.0 * (
            jnp.log(sig_x_prior) + jnp.log(sig_y_prior) + jnp.log(sig_z_prior)
        )
        log_det_sigma = 2.0 * (log_L11 + log_L22 + log_L33)

        kl = 0.5 * (tr_term + mu_term - 3.0 + log_det_prior - log_det_sigma)
        return jnp.mean(kl)

    # 5. Masked Non-Parametric NLL Helper
    def get_masked_grid_nll(
        inverter,
        grid_params,
        omega,
        meta_batch,
        obs_mu_pm=None,
        obs_cov_pm=None,
        obs_mu_err_pm=None,
        obs_mu_rv=None,
        obs_var_rv=None,
        obs_mu_err_rv=None,
        mask=None,
        rv_weight=1.0,
    ):
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
            lambda m: inverter.predict_pixel(
                grid_params, omega, m, log_f_bulge=fixed_log_f_bulge
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

    # 6. Joint Loss Function
    kl_weight = 1.0
    curv_weight = 1.0

    @jit
    def joint_loss_fn(params):
        grid_params, omega = params

        # Proper motion term
        nll_pm = get_masked_grid_nll(
            inverter_v,
            grid_params,
            omega,
            meta_v,
            obs_mu_pm=obs_mu_pm_v,
            obs_cov_pm=obs_cov_pm_v,
            obs_mu_err_pm=obs_mu_err_v,
            mask=mask_v,
        )

        # BRAVA RV term
        nll_rv_b = get_masked_grid_nll(
            inverter_b,
            grid_params,
            omega,
            meta_b,
            obs_mu_rv=obs_mu_rv_b,
            obs_var_rv=obs_var_rv_b,
            obs_mu_err_rv=obs_mu_err_b,
            mask=mask_b,
        )

        # GIBS RV term
        nll_rv_g = get_masked_grid_nll(
            inverter_g,
            grid_params,
            omega,
            meta_g,
            obs_mu_rv=obs_mu_rv_g,
            obs_var_rv=obs_var_rv_g,
            obs_mu_err_rv=obs_mu_err_g,
            mask=mask_g,
        )

        # Curvature Regularization
        curv_reg = 0.0
        for p in range(6):
            field = grid_params[..., p]
            curv_reg += (
                jnp.mean(jnp.diff(field, n=2, axis=0) ** 2)
                + jnp.mean(jnp.diff(field, n=2, axis=1) ** 2)
                + jnp.mean(jnp.diff(field, n=2, axis=2) ** 2)
            )

        # KL Penalty (Custom anisotropic)
        kl = custom_kl_divergence_penalty(grid_params)

        # Omega penalty relative to the parametric fit omega
        om_pen = 0.5 * ((omega - omega_fit) / sigma_omega) ** 2

        total_loss = (
            nll_pm
            + nll_rv_b
            + nll_rv_g
            + kl_weight * kl
            + curv_weight * curv_reg
            + om_pen
        )
        return total_loss, (
            nll_pm,
            nll_rv_b,
            nll_rv_g,
            kl_weight * kl,
            curv_weight * curv_reg,
            om_pen,
        )

    val_and_grad_fn = jit(value_and_grad(joint_loss_fn, has_aux=True))

    # 7. Initialize Grid Exactly from the Fitted Parametric Model Prior
    print("\nInitializing grid from the optimized parametric prior...")
    grid_shape = inverter_v.shape + (6,)
    current_grid = np.zeros(grid_shape)

    current_grid[..., 0] = np.array(ux_prior)
    current_grid[..., 1] = np.array(uy_prior)
    current_grid[..., 2] = np.log(np.array(sig_x_prior))
    current_grid[..., 3] = np.zeros(inverter_v.shape)  # L21 = 0
    current_grid[..., 4] = np.log(np.array(sig_y_prior))
    current_grid[..., 5] = np.log(np.array(sig_z_prior))

    # Add small perturbation to break symmetry
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    current_grid[..., 0] += np.array(jax.random.normal(k1, inverter_v.shape) * 1.0)
    current_grid[..., 1] += np.array(jax.random.normal(k2, inverter_v.shape) * 1.0)
    current_grid[..., 3] += np.array(jax.random.normal(k4, inverter_v.shape) * 0.05)

    initial_omega = 45.0

    print(
        f"\n--- Combined OBSERVED DATA Non-Parametric Optimization: VIRAC ({np.sum(mask_v)} pix) + BRAVA ({np.sum(mask_b)} pix) + GIBS ({np.sum(mask_g)} pix) ---"
    )
    print(f"Alpha dynamically fixed at: {fixed_alpha:.2f} deg")
    print(f"R0 dynamically set at: {solar['R0']:.3f} kpc")
    print(
        f"Scales dynamically set at: x={solar['x_scale']:.4f}, y={solar['y_scale']:.4f}, z={solar['z_scale']:.4f}"
    )
    print(f"f_bulge fixed at: {raw_f_bulge:.4f} (logit={fixed_log_f_bulge:.4f})")

    def scipy_objective(x_flat):
        grid_flat = x_flat[:-1]
        omega = x_flat[-1]
        grid_jnp = jnp.array(grid_flat).reshape(grid_shape)

        (loss, aux), grads = val_and_grad_fn((grid_jnp, omega))
        grid_grads, omega_grad = grads
        nll_pm, nll_rv_b, nll_rv_g, kl, curv, om_pen = aux

        if not hasattr(scipy_objective, "n_calls"):
            scipy_objective.n_calls = 0
        if scipy_objective.n_calls % 10 == 0:
            print(
                f"  Iter {scipy_objective.n_calls:4d}: Loss = {loss:.4f} [PM_V={nll_pm:.2f}, RV_B={nll_rv_b:.2f}, RV_G={nll_rv_g:.2f}, KL={kl:.2f}, Curv={curv:.2f}, OmP={om_pen:.2f}], Om={omega:.2f}"
            )
        scipy_objective.n_calls += 1

        return float(loss), np.concatenate(
            [np.array(grid_grads).flatten(), [float(omega_grad)]]
        ).astype(np.float64)

    # 8. Define bounds
    x0 = np.concatenate([current_grid.flatten(), [initial_omega]])

    grid_bounds = [
        (-400, 400),
        (-400, 400),
        (1.6, 6.0),
        (-400, 400),
        (1.6, 6.0),
        (1.6, 6.0),
    ] * np.prod(grid_shape[:-1])
    omega_bounds = [(0, 100.0)]
    bounds = grid_bounds + omega_bounds

    print("Starting L-BFGS-B optimization...")
    max_iter_val = 5 if os.environ.get("FAST_TEST") == "1" else 10000
    res = minimize(
        scipy_objective,
        x0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": max_iter_val, "ftol": 1e-9, "gtol": 1e-9},
    )
    print("\nOptimization Complete.")
    print(f"Success: {res.success}")
    print(f"Status: {res.message}")
    print(f"Total iterations: {res.nit}")

    # 9. Save Results
    final_params = np.array(res.x[:-1]).reshape(grid_shape)
    final_omega = float(res.x[-1])
    output_path = os.path.join(
        project_root, "results/inversion/fit_results_obs_combined_h25c20.npz"
    )
    np.savez(
        output_path,
        grid=final_params,
        omega=final_omega,
        axes_x=grid_axes[0],
        axes_y=grid_axes[1],
        axes_z=grid_axes[2],
        p_parametric=np.array(p_fitted),
    )
    print(f"\nOptimization Complete. Results saved to {output_path}")
    print(f"Final Omega: {final_omega:.2f} km/s/kpc")


if __name__ == "__main__":
    main()
