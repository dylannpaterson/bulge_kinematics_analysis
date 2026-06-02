"""
Profile likelihood scan for Omega in the parametric model.

Holds all other parameters fixed at their best-fit values and evaluates
the parametric NLL as a function of Omega alone. Fits a parabola to the
minimum region to extract a 1-sigma error bar via the observed Fisher
information (second derivative of the NLL).

Saves omega_best and sigma_omega back into the parametric fit results file.
"""
from __future__ import annotations

import os
import pickle
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import vmap
from scipy.optimize import minimize_scalar

jax.config.update("jax_enable_x64", True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../shared")))
from inversion_config import get_inverter_grid_axes
from jax_kinematic_inverter import BulgeKinematicInverter


def main() -> None:
    """Evaluate profile likelihood for Omega and compute uncertainty."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    param_path = os.path.join(
        project_root, "results/inversion/parametric_fit_results_obs_h25c20.npz"
    )
    if not os.path.exists(param_path):
        raise FileNotFoundError(f"Parametric fit results not found at {param_path}")

    print("Loading parametric fit results...")
    param_data = np.load(param_path)
    params_all = param_data["params"]
    names = [
        n.decode() if isinstance(n, bytes) else str(n) for n in param_data["names"]
    ]

    p_kosh = jnp.array(
        params_all[
            [
                i
                for i, n in enumerate(names)
                if n not in ("omega", "alpha", "f_bulge", "sigmoid_f_bulge")
            ]
        ]
    )
    omega_best = float(params_all[names.index("omega")])
    alpha_deg = float(params_all[names.index("alpha")])
    raw_f = float(params_all[names.index("f_bulge")])
    log_f = float(np.log(raw_f / (1.0 - raw_f)))
    alpha_rad = jnp.radians(alpha_deg)

    print(f"Best-fit Omega = {omega_best:.4f} km/s/kpc,  alpha = {alpha_deg:.2f} deg")

    solar = {
        "R0": 8.178,
        "U_SUN": 12.9,
        "V_LSR": 233.4,
        "V_SUN_REL": 12.2,
        "W_SUN": 7.78,
        "bar_angle_rad": alpha_rad,
        "x_scale": 1.0,
        "y_scale": 1.0,
        "z_scale": 1.0,
    }

    # ── Load pre-computed metadata & observations ──────────────────────
    print("Loading data and metadata...")

    data_v = np.load(
        os.path.join(project_root, "results/reduction/virac2_obs_aligned_h25c20.npz")
    )
    obs_mu_pm_v = jnp.array(data_v["mu_pm"])
    obs_cov_pm_v = (
        jnp.array(data_v["cov_pm"]) + jnp.eye(2)[None, None, :, :] * (0.01**2)
    )
    obs_mu_err_v = jnp.array(data_v["mu_err"])
    mask_v = jnp.array(data_v["mask"]) & (jnp.abs(data_v["b"]) >= 2.0)
    with open(
        os.path.join(
            project_root, "results/inversion/huston25c20_metadata_science_binned.pkl"
        ),
        "rb",
    ) as f:
        meta_v = pickle.load(f)
    n_bins_v = obs_mu_pm_v.shape[1]

    data_b = np.load(
        os.path.join(project_root, "results/reduction/brava_obs_aligned_h25c20.npz")
    )
    obs_mu_rv_b = jnp.array(data_b["mu_rv"])
    obs_var_rv_b = jnp.array(data_b["var_rv"]) + 1.0
    mask_b = jnp.array(data_b["mask"])
    with open(
        os.path.join(
            project_root, "results/inversion/huston25c20_brava_obs_metadata.pkl"
        ),
        "rb",
    ) as f:
        meta_b = pickle.load(f)

    data_g = np.load(
        os.path.join(project_root, "results/reduction/gibs_obs_aligned_h25c20.npz")
    )
    obs_mu_rv_g = jnp.array(data_g["mu_rv"])
    obs_var_rv_g = jnp.array(data_g["var_rv"]) + 1.0
    mask_g = jnp.array(data_g["mask"])
    with open(
        os.path.join(
            project_root, "results/inversion/huston25c20_gibs_metadata_science.pkl"
        ),
        "rb",
    ) as f:
        meta_g = pickle.load(f)

    # ── Setup inverters ────────────────────────────────────────────────
    grid_axes = get_inverter_grid_axes()
    inv_v = BulgeKinematicInverter(
        grid_axes, solar, n_bins=n_bins_v, kl_weight=0.0, curv_weight=0.0
    )
    inv_b = BulgeKinematicInverter(
        grid_axes, solar, n_bins=1, kl_weight=0.0, curv_weight=0.0
    )
    inv_g = BulgeKinematicInverter(
        grid_axes, solar, n_bins=1, kl_weight=0.0, curv_weight=0.0
    )

    # Build masked meta axes
    meta_axes_v = {
        k: (0 if (jnp.ndim(v) > 0 and v.shape[0] == obs_mu_pm_v.shape[0]) else None)
        for k, v in meta_v.items()
    }
    n_pix_b = len(meta_b["l"])
    meta_axes_b = {
        k: (0 if (jnp.ndim(v) > 0 and v.shape[0] == n_pix_b) else None)
        for k, v in meta_b.items()
    }
    n_pix_g = len(meta_g["l"])
    meta_axes_g = {
        k: (0 if (jnp.ndim(v) > 0 and v.shape[0] == n_pix_g) else None)
        for k, v in meta_g.items()
    }

    # ── Define profile NLL as a function of omega only ─────────────────
    @jax.jit
    def profile_nll(omega_scalar):
        omega = jnp.array(omega_scalar)
        alpha = alpha_rad  # fixed

        # --- VIRAC PM loss ---
        preds_v = vmap(
            lambda m: inv_v.predict_parametric(
                p_kosh, omega, alpha, m, log_f_bulge=log_f
            ),
            in_axes=(meta_axes_v,),
        )(meta_v)
        pred_mu_pm_v = preds_v[0]  # (N_pix, N_bins, 2)
        pred_cov_pm_v = preds_v[1]  # (N_pix, N_bins, 2, 2)

        def pm_nll_pixel(pred_mu, pred_cov, obs_mu, obs_cov, obs_err, mask):
            combined_cov = pred_cov + obs_cov
            diff = pred_mu - obs_mu

            # 2x2 Gaussian NLL
            def bin_nll(k):
                C = combined_cov[k]
                d = diff[k]
                det = C[0, 0] * C[1, 1] - C[0, 1] * C[1, 0]
                det = jnp.maximum(det, 1e-20)
                inv_C = (
                    jnp.array([[C[1, 1], -C[0, 1]], [-C[1, 0], C[0, 0]]]) / det
                )
                return 0.5 * (jnp.log(det) + d @ inv_C @ d)

            total = jnp.sum(vmap(bin_nll)(jnp.arange(pred_mu.shape[0])))
            return jnp.where(mask, total, 0.0)

        nll_pm = jnp.sum(
            vmap(pm_nll_pixel)(
                pred_mu_pm_v,
                pred_cov_pm_v,
                obs_mu_pm_v,
                obs_cov_pm_v,
                obs_mu_err_v,
                mask_v,
            )
        )

        # --- BRAVA RV loss ---
        preds_b = vmap(
            lambda m: inv_b.predict_parametric(
                p_kosh, omega, alpha, m, log_f_bulge=log_f
            ),
            in_axes=(meta_axes_b,),
        )(meta_b)
        pred_mu_rv_b = preds_b[2][:, 0]
        pred_var_rv_b = preds_b[3][:, 0]

        def rv_nll(pred_mu, pred_var, obs_mu, obs_var, mask):
            total_var = pred_var + obs_var + 1e-6
            nll = 0.5 * (jnp.log(total_var) + (obs_mu - pred_mu) ** 2 / total_var)
            return jnp.where(mask, nll, 0.0)

        nll_rv_b = jnp.sum(
            vmap(rv_nll)(
                pred_mu_rv_b,
                pred_var_rv_b,
                obs_mu_rv_b.flatten(),
                obs_var_rv_b.flatten(),
                mask_b,
            )
        )

        # --- GIBS RV loss ---
        preds_g = vmap(
            lambda m: inv_g.predict_parametric(
                p_kosh, omega, alpha, m, log_f_bulge=log_f
            ),
            in_axes=(meta_axes_g,),
        )(meta_g)
        pred_mu_rv_g = preds_g[2][:, 0]
        pred_var_rv_g = preds_g[3][:, 0]

        nll_rv_g = jnp.sum(
            vmap(rv_nll)(
                pred_mu_rv_g,
                pred_var_rv_g,
                obs_mu_rv_g.flatten(),
                obs_var_rv_g.flatten(),
                mask_g,
            )
        )

        return nll_pm + nll_rv_b + nll_rv_g

    # ── Scan omega profile ─────────────────────────────────────────────
    print(f"\nCompiling profile NLL... (first call is slow)")
    nll0 = float(profile_nll(omega_best))
    print(f"NLL at omega_best={omega_best:.3f}: {nll0:.4f}")

    # Scan ±10 km/s/kpc in fine steps
    print("Scanning omega profile likelihood...")
    n_scan = 41
    omega_lo, omega_hi = omega_best - 10.0, omega_best + 10.0
    omega_scan = np.linspace(omega_lo, omega_hi, n_scan)
    nll_scan = np.array([float(profile_nll(float(w))) for w in omega_scan])

    delta_nll = nll_scan - nll_scan.min()
    print("\nomega   delta_NLL")
    for w, dn in zip(omega_scan, delta_nll):
        print(f"  {w:6.2f}  {dn:8.3f}")

    # ── Fit parabola near minimum to extract curvature ─────────────────
    close = delta_nll < 5.0
    if close.sum() < 3:
        close = delta_nll < delta_nll[np.argsort(delta_nll)[:5]].max()

    coeffs = np.polyfit(omega_scan[close], nll_scan[close], 2)
    a, b, c = coeffs
    omega_parabola = -b / (2 * a)
    sigma_omega = 1.0 / np.sqrt(2.0 * a)

    print(f"\n── Profile Likelihood Results ──────────────────────────")
    print(f"  Parabola minimum : Omega = {omega_parabola:.3f} km/s/kpc")
    print(f"  Curvature (a)    : {a:.6f}")
    print(f"  1-sigma          : {sigma_omega:.3f} km/s/kpc")
    print(f"  Best-fit Omega   : {omega_best:.3f} ± {sigma_omega:.3f} km/s/kpc")

    # ── Save sigma_omega back to the parametric results file ───────────
    new_data = dict(param_data)
    new_data["sigma_omega"] = np.array(sigma_omega)
    new_data["omega_parabola_min"] = np.array(omega_parabola)
    np.savez(param_path, **new_data)
    print(f"\nSaved sigma_omega={sigma_omega:.3f} to {param_path}")

    # ── Suggest updated NP prior ────────────────────────────────────────
    print(f"\n── Suggested NP model omega prior ──────────────────────")
    print(f"  omega_p = 0.5 * ((omega - {omega_best:.2f}) / {sigma_omega:.2f})**2")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        omega_scan, delta_nll, "o-", color="#4488ff", lw=1.5, ms=4, label="Profile NLL"
    )
    omega_fine = np.linspace(omega_scan[close][0], omega_scan[close][-1], 200)
    ax.plot(
        omega_fine,
        np.polyval(coeffs, omega_fine) - nll_scan.min(),
        "--",
        color="#ff9944",
        lw=2,
        label="Parabola fit",
    )
    ax.axvline(
        omega_best,
        color="white",
        linestyle=":",
        lw=1.0,
        label=f"Best fit Ω={omega_best:.2f}",
    )
    ax.axvline(
        omega_parabola,
        color="#ff4488",
        linestyle="--",
        lw=1.5,
        label=f"Parabola min Ω={omega_parabola:.2f}",
    )
    ax.axhline(0.5, color="#88ff88", linestyle=":", lw=1.0, label="1σ (ΔNLL=0.5)")
    ax.fill_betweenx(
        [0, 0.5],
        omega_parabola - sigma_omega,
        omega_parabola + sigma_omega,
        alpha=0.15,
        color="#88ff88",
    )
    ax.set_xlabel(r"$\Omega_{\rm bar}$ [km/s/kpc]", color="white")
    ax.set_ylabel(r"$\Delta \mathcal{L}$ (profile NLL)", color="white")
    ax.set_title(
        "Profile Likelihood for Pattern Speed\n"
        rf"$\Omega = {omega_best:.2f} \pm {sigma_omega:.2f}$ km/s/kpc (parametric model)",
        color="white",
    )
    ax.legend(facecolor="#111116", edgecolor="#444455", labelcolor="white")
    ax.set_facecolor("#111116")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444455")
    ax.set_ylim(bottom=-0.05)
    ax.set_xlim(omega_lo, omega_hi)
    fig.patch.set_facecolor("#111116")
    out_plots_dir = os.path.join(project_root, "results/inversion")
    os.makedirs(out_plots_dir, exist_ok=True)
    out_path = os.path.join(out_plots_dir, "omega_profile_likelihood.png")
    fig.savefig(out_path, dpi=200, facecolor="#111116")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
