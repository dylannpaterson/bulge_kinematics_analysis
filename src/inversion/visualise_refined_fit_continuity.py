"""
Visualization diagnostics for the refined non-parametric kinematic inversion (with continuity).

Generates binned proper-motion component maps, dispersion maps, correlation maps, and
radial velocity normalized residual histograms, comparing the VVV, BRAVA, and GIBS
observational datasets against the refined non-parametric kinematic grid.
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

jax.config.update("jax_enable_x64", True)

# Append paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../shared")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../synthpop")))

import synthpop
from inversion_config import get_inverter_grid_axes
from jax_kinematic_inverter import BulgeKinematicInverter


def main() -> None:
    """Generate diagnostic maps and plots for the refined non-parametric inversion."""
    if os.environ.get("FAST_TEST") == "1":
        print("FAST_TEST mode active: skipping complex 2D contour plotting.")
        return
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

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
    alpha_deg = float(p_fitted[19])  # 18.79 deg
    raw_f_bulge = float(p_fitted[20])  # 0.50
    log_f_bulge = float(np.log(raw_f_bulge / (1.0 - raw_f_bulge)))  # logit
    alpha_rad = np.radians(alpha_deg)

    # 1. Load Refined Fit Results
    fit_path = os.path.join(
        project_root, "results/inversion/fit_results_refined_continuity.npz"
    )
    if not os.path.exists(fit_path):
        raise FileNotFoundError(
            f"Refined fit results not found at {fit_path}. "
            "Please run the refinement script first."
        )

    print(f"Loading refined fit results from {fit_path}...")
    fit_res = np.load(fit_path)
    grid_params = jnp.array(fit_res["grid"])
    omega = float(fit_res["omega"])
    grid_axes = [fit_res["axes_x"], fit_res["axes_y"], fit_res["axes_z"]]

    print(f"Loaded refined grid parameters of shape {grid_params.shape}")
    print(
        f"Pattern Speed Omega: {omega:.2f} km/s/kpc, Bar Angle: {alpha_deg:.2f} deg, f_bulge: {raw_f_bulge:.2f}"
    )

    # 2. Setup Inverter
    solar = {
        "R0": 8.178,
        "U_SUN": 12.9,
        "V_LSR": 233.4,
        "V_SUN_REL": 12.2,
        "W_SUN": 7.78,
        "bar_angle_rad": jnp.radians(17.85655), 
        "x_scale": 1.13442, "y_scale": 1.15296, "z_scale": 1.10504
    }

    # --- VIRAC2 (PMs) ---
    print("Loading aligned VIRAC2 observed data...")
    real_v = np.load(
        os.path.join(project_root, "results/reduction/virac2_obs_aligned_h25c20.npz")
    )
    obs_mu_pm_v = real_v["mu_pm"]
    obs_cov_pm_v = real_v["cov_pm"]
    obs_mu_err_v = real_v["mu_err"]
    l_v, b_v = real_v["l"], real_v["b"]
    mask_v = real_v["mask"]

    with open(
        os.path.join(
            project_root, "results/inversion/huston25c20_metadata_science_binned.pkl"
        ),
        "rb",
    ) as f:
        meta_v = pickle.load(f)

    inverter_v = BulgeKinematicInverter(grid_axes, solar, n_bins=obs_mu_pm_v.shape[1])
    n_pix_v = len(l_v)
    meta_axes = {
        k: (0 if (jnp.ndim(v) > 0 and v.shape[0] == n_pix_v) else None)
        for k, v in meta_v.items()
    }

    print("Predicting VIRAC2 proper motions from the refined grid...")
    pred_v = vmap(
        lambda m: inverter_v.predict_pixel(
            grid_params, omega, m, log_f_bulge=log_f_bulge, return_components=True
        ),
        in_axes=(meta_axes,),
    )(meta_v)
    pred_mu_pm_v = np.array(pred_v[0])  # (N_pix, N_bins, 2)
    pred_cov_pm_v = np.array(pred_v[1])  # (N_pix, N_bins, 2, 2)
    f_bulge_v = np.array(pred_v[6])  # (N_pix, N_bins)

    # --- BRAVA (RVs) ---
    print("Loading BRAVA observed data...")
    real_b = np.load(
        os.path.join(project_root, "results/reduction/brava_obs_aligned_h25c20.npz")
    )
    obs_mu_rv_b = real_b["mu_rv"].flatten()
    mask_b = real_b["mask"].flatten()
    obs_err_rv_b = (
        real_b["mu_err"].flatten()
        if "mu_err" in real_b
        else np.ones_like(obs_mu_rv_b) * 10.0
    )

    with open(
        os.path.join(
            project_root, "results/inversion/huston25c20_brava_obs_metadata.pkl"
        ),
        "rb",
    ) as f:
        meta_b = pickle.load(f)
    inverter_b = BulgeKinematicInverter(grid_axes, solar, n_bins=1)
    n_pix_b = len(meta_b["l"])
    meta_axes_b = {
        k: (0 if (jnp.ndim(v) > 0 and v.shape[0] == n_pix_b) else None)
        for k, v in meta_b.items()
    }
    pred_b = vmap(
        lambda m: inverter_b.predict_pixel(
            grid_params, omega, m, log_f_bulge=log_f_bulge
        ),
        in_axes=(meta_axes_b,),
    )(meta_b)
    pred_mu_rv_b = np.array(pred_b[2])[:, 0].flatten()  # (N_pix,)

    # --- GIBS (RVs) ---
    print("Loading GIBS observed data...")
    real_g = np.load(
        os.path.join(project_root, "results/reduction/gibs_obs_aligned_h25c20.npz")
    )
    obs_mu_rv_g = real_g["mu_rv"].flatten()
    mask_g = real_g["mask"].flatten()
    obs_err_rv_g = (
        real_g["mu_err"].flatten()
        if "mu_err" in real_g
        else np.ones_like(obs_mu_rv_g) * 10.0
    )

    with open(
        os.path.join(
            project_root, "results/inversion/huston25c20_gibs_metadata_science.pkl"
        ),
        "rb",
    ) as f:
        meta_g = pickle.load(f)
    inverter_g = BulgeKinematicInverter(grid_axes, solar, n_bins=1)
    n_pix_g = len(meta_g["l"])
    meta_axes_g = {
        k: (0 if (jnp.ndim(v) > 0 and v.shape[0] == n_pix_g) else None)
        for k, v in meta_g.items()
    }
    pred_g = vmap(
        lambda m: inverter_g.predict_pixel(
            grid_params, omega, m, log_f_bulge=log_f_bulge
        ),
        in_axes=(meta_axes_g,),
    )(meta_g)
    pred_mu_rv_g = np.array(pred_g[2])[:, 0].flatten()  # (N_pix,)

    # 3. Plotting Setup
    k_indices = [2, 7, 12, 17]  # selected slices
    mag_bins = np.linspace(12.0, 14.0, 21)

    unique_l = np.unique(l_v)
    unique_b = np.unique(b_v)
    L_grid, B_grid = np.meshgrid(unique_l, unique_b)

    def plot_component(comp_idx, comp_name, filename):
        fig, axes = plt.subplots(
            4,
            len(k_indices),
            figsize=(5.5 * len(k_indices), 17),
            sharex=True,
            sharey=True,
        )

        for i, k_idx in enumerate(k_indices):
            m = mask_v.reshape(len(unique_b), len(unique_l))
            obs_data = obs_mu_pm_v[:, k_idx, comp_idx].reshape(
                len(unique_b), len(unique_l)
            )
            obs_err = obs_mu_err_v[:, k_idx, comp_idx].reshape(
                len(unique_b), len(unique_l)
            )
            model_data = pred_mu_pm_v[:, k_idx, comp_idx].reshape(
                len(unique_b), len(unique_l)
            )
            f_bulge_data = f_bulge_v[:, k_idx].reshape(len(unique_b), len(unique_l))

            # Mask low latitude zone
            valid_mask = np.abs(B_grid) >= 2.0
            valid_for_limits = (
                (np.abs(B_grid) >= 2.0) & m & (obs_err < 1.0) & (B_grid <= 5.0)
            )

            obs_masked = np.where(valid_mask, obs_data, np.nan)
            model_masked = np.where(valid_mask, model_data, np.nan)

            obs_for_limits = np.where(valid_for_limits, obs_data, np.nan)
            vmin = np.nanpercentile(obs_for_limits, 1)
            vmax = np.nanpercentile(obs_for_limits, 99)
            if np.isnan(vmin) or np.isnan(vmax) or vmin == vmax:
                vmin, vmax = -10.0, 10.0

            levels = np.linspace(vmin, vmax, 10)

            # Row 0: DATA
            ax_data = axes[0, i]
            im_obs = ax_data.pcolormesh(
                L_grid,
                B_grid,
                obs_masked,
                cmap="viridis",
                shading="auto",
                vmin=vmin,
                vmax=vmax,
                alpha=0.85,
            )
            cb_obs = plt.colorbar(im_obs, ax=ax_data)
            cb_obs.set_label(fr"$\mu_{comp_name}$ (Obs) [mas/yr]", fontsize=9)

            contours_obs = ax_data.contour(
                L_grid, B_grid, model_masked, levels=levels, colors="black", linewidths=1.0
            )
            ax_data.clabel(contours_obs, inline=True, fontsize=8, fmt="%.1f")

            ax_data.set_title(
                fr"$K_s \in [{mag_bins[k_idx]:.1f}, {mag_bins[k_idx+1]:.1f}]$"
            )
            if i == 0:
                ax_data.set_ylabel(r"$b$ [deg]")
            ax_data.set_xlim(unique_l.max(), unique_l.min())
            ax_data.set_ylim(unique_b.min(), 5.0)
            ax_data.set_aspect("equal")

            # Row 1: MODEL
            ax_model = axes[1, i]
            im_model = ax_model.pcolormesh(
                L_grid,
                B_grid,
                model_masked,
                cmap="viridis",
                shading="auto",
                vmin=vmin,
                vmax=vmax,
                alpha=0.85,
            )
            cb_model = plt.colorbar(im_model, ax=ax_model)
            cb_model.set_label(fr"$\mu_{comp_name}$ (Model) [mas/yr]", fontsize=9)

            contours_model = ax_model.contour(
                L_grid, B_grid, model_masked, levels=levels, colors="black", linewidths=1.0
            )
            ax_model.clabel(contours_model, inline=True, fontsize=8, fmt="%.1f")
            if i == 0:
                ax_model.set_ylabel(r"$b$ [deg]")
            ax_model.set_ylim(unique_b.min(), 5.0)
            ax_model.set_aspect("equal")

            # Row 2: NORMALIZED RESIDUAL
            ax_res = axes[2, i]
            res = (model_data - obs_data) / (obs_err + 1e-10)
            res_masked = np.where(valid_mask & m & (obs_err < 1.0), res, np.nan)

            im_res = ax_res.pcolormesh(
                L_grid, B_grid, res_masked, cmap="RdBu_r", shading="auto", vmin=-5, vmax=5
            )
            cb_res = plt.colorbar(im_res, ax=ax_res)
            cb_res.set_label(r"(Model - Obs) / $\sigma_{Obs}$", fontsize=9)

            if i == 0:
                ax_res.set_ylabel(r"$b$ [deg]")
            ax_res.set_ylim(unique_b.min(), 5.0)
            ax_res.set_aspect("equal")

            # Row 3: BULGE FRACTION
            ax_frac = axes[3, i]
            frac_masked = np.where(valid_mask, f_bulge_data, np.nan)
            im_frac = ax_frac.pcolormesh(
                L_grid,
                B_grid,
                frac_masked,
                cmap="plasma",
                shading="auto",
                vmin=0.0,
                vmax=1.0,
                alpha=0.85,
            )
            cb_frac = plt.colorbar(im_frac, ax=ax_frac)
            cb_frac.set_label(r"$f_{\rm bulge}$", fontsize=9)

            frac_levels = np.array([0.2, 0.4, 0.6, 0.8])
            contours_frac = ax_frac.contour(
                L_grid,
                B_grid,
                frac_masked,
                levels=frac_levels,
                colors="white",
                linewidths=1.0,
            )
            ax_frac.clabel(contours_frac, inline=True, fontsize=8, fmt="%.1f")

            ax_frac.set_xlabel(r"$l$ [deg]")
            if i == 0:
                ax_frac.set_ylabel(r"$b$ [deg]")
            ax_frac.set_xlim(unique_l.max(), unique_l.min())
            ax_frac.set_ylim(unique_b.min(), 5.0)
            ax_frac.set_aspect("equal")

        plt.suptitle(
            fr"H25C20 Observed Data Refined Fit (with Continuity): Proper Motion $\mu_{comp_name}$",
            fontsize=16,
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(filename, dpi=200)
        plt.close()
        print(f"Saved {filename}")

    def plot_covariance_metric(metric_type, metric_name, filename):
        fig, axes = plt.subplots(
            4,
            len(k_indices),
            figsize=(5.5 * len(k_indices), 17),
            sharex=True,
            sharey=True,
        )

        for i, k_idx in enumerate(k_indices):
            m = mask_v.reshape(len(unique_b), len(unique_l))
            obs_err = obs_mu_err_v[:, k_idx, 0].reshape(len(unique_b), len(unique_l))

            if metric_type == "sig_l":
                obs_data = np.sqrt(
                    np.clip(obs_cov_pm_v[:, k_idx, 0, 0], 1e-10, None)
                ).reshape(len(unique_b), len(unique_l))
                model_data = np.sqrt(
                    np.clip(pred_cov_pm_v[:, k_idx, 0, 0], 1e-10, None)
                ).reshape(len(unique_b), len(unique_l))
                cmap = "viridis"
                label_suffix = r"$\sigma_{\mu_l}$ [mas/yr]"
                fmt = "%.2f"
            elif metric_type == "sig_b":
                obs_data = np.sqrt(
                    np.clip(obs_cov_pm_v[:, k_idx, 1, 1], 1e-10, None)
                ).reshape(len(unique_b), len(unique_l))
                model_data = np.sqrt(
                    np.clip(pred_cov_pm_v[:, k_idx, 1, 1], 1e-10, None)
                ).reshape(len(unique_b), len(unique_l))
                cmap = "viridis"
                label_suffix = r"$\sigma_{\mu_b}$ [mas/yr]"
                fmt = "%.2f"
            elif metric_type == "corr":
                obs_var_l = obs_cov_pm_v[:, k_idx, 0, 0]
                obs_var_b = obs_cov_pm_v[:, k_idx, 1, 1]
                obs_data = (
                    obs_cov_pm_v[:, k_idx, 0, 1]
                    / (np.sqrt(np.clip(obs_var_l * obs_var_b, 1e-10, None)) + 1e-10)
                ).reshape(len(unique_b), len(unique_l))
                obs_data = np.clip(obs_data, -1.0, 1.0)

                model_var_l = pred_cov_pm_v[:, k_idx, 0, 0]
                model_var_b = pred_cov_pm_v[:, k_idx, 1, 1]
                model_data = (
                    pred_cov_pm_v[:, k_idx, 0, 1]
                    / (
                        np.sqrt(np.clip(model_var_l * model_var_b, 1e-10, None))
                        + 1e-10
                    )
                ).reshape(len(unique_b), len(unique_l))
                model_data = np.clip(model_data, -1.0, 1.0)
                cmap = "RdBu_r"
                label_suffix = r"$\rho_{\mu_l, \mu_b}$"
                fmt = "%.2f"

            valid_mask = np.abs(B_grid) >= 2.0
            valid_for_limits = (
                (np.abs(B_grid) >= 2.0) & m & (obs_err < 1.0) & (B_grid <= 5.0)
            )

            obs_masked = np.where(valid_mask, obs_data, np.nan)
            model_masked = np.where(valid_mask, model_data, np.nan)
            obs_for_limits = np.where(valid_for_limits, obs_data, np.nan)

            if metric_type == "corr":
                max_val = np.nanpercentile(np.abs(obs_for_limits), 98)
                if np.isnan(max_val) or max_val == 0:
                    max_val = 0.5
                vmin, vmax = -max_val, max_val
            else:
                vmin = np.nanpercentile(obs_for_limits, 1)
                vmax = np.nanpercentile(obs_for_limits, 99)
                if np.isnan(vmin) or np.isnan(vmax) or vmin == vmax:
                    vmin, vmax = 0.0, 5.0

            levels = np.linspace(vmin, vmax, 10)

            # Row 0: DATA
            ax_data = axes[0, i]
            im_obs = ax_data.pcolormesh(
                L_grid,
                B_grid,
                obs_masked,
                cmap=cmap,
                shading="auto",
                vmin=vmin,
                vmax=vmax,
                alpha=0.85,
            )
            cb_obs = plt.colorbar(im_obs, ax=ax_data)
            cb_obs.set_label(fr"{label_suffix} (Obs)", fontsize=9)

            contours_obs = ax_data.contour(
                L_grid, B_grid, model_masked, levels=levels, colors="black", linewidths=1.0
            )
            ax_data.clabel(contours_obs, inline=True, fontsize=8, fmt=fmt)

            ax_data.set_title(
                fr"$K_s \in [{mag_bins[k_idx]:.1f}, {mag_bins[k_idx+1]:.1f}]$"
            )
            if i == 0:
                ax_data.set_ylabel(r"$b$ [deg]")
            ax_data.set_xlim(unique_l.max(), unique_l.min())
            ax_data.set_ylim(unique_b.min(), 5.0)
            ax_data.set_aspect("equal")

            # Row 1: MODEL
            ax_model = axes[1, i]
            im_model = ax_model.pcolormesh(
                L_grid,
                B_grid,
                model_masked,
                cmap=cmap,
                shading="auto",
                vmin=vmin,
                vmax=vmax,
                alpha=0.85,
            )
            cb_model = plt.colorbar(im_model, ax=ax_model)
            cb_model.set_label(fr"{label_suffix} (Model)", fontsize=9)

            contours_model = ax_model.contour(
                L_grid, B_grid, model_masked, levels=levels, colors="black", linewidths=1.0
            )
            ax_model.clabel(contours_model, inline=True, fontsize=8, fmt=fmt)
            if i == 0:
                ax_model.set_ylabel(r"$b$ [deg]")
            ax_model.set_ylim(unique_b.min(), 5.0)
            ax_model.set_aspect("equal")

            # Row 2: NORMALIZED RESIDUAL
            ax_res = axes[2, i]
            res = (model_data - obs_data) / (obs_err + 1e-10)
            res_masked = np.where(valid_mask & m & (obs_err < 1.0), res, np.nan)

            im_res = ax_res.pcolormesh(
                L_grid, B_grid, res_masked, cmap="RdBu_r", shading="auto", vmin=-5, vmax=5
            )
            cb_res = plt.colorbar(im_res, ax=ax_res)
            cb_res.set_label(fr"$\Delta$ {label_suffix} / $\sigma$", fontsize=9)
            if i == 0:
                ax_res.set_ylabel(r"$b$ [deg]")
            ax_res.set_ylim(unique_b.min(), 5.0)
            ax_res.set_aspect("equal")

            # Row 3: BULGE FRACTION
            ax_frac = axes[3, i]
            frac_data = f_bulge_v[:, k_idx].reshape(len(unique_b), len(unique_l))
            frac_masked = np.where(valid_mask, frac_data, np.nan)
            im_frac = ax_frac.pcolormesh(
                L_grid,
                B_grid,
                frac_masked,
                cmap="plasma",
                shading="auto",
                vmin=0.0,
                vmax=1.0,
                alpha=0.85,
            )
            cb_frac = plt.colorbar(im_frac, ax=ax_frac)
            cb_frac.set_label(r"$f_{\rm bulge}$", fontsize=9)

            frac_levels = np.array([0.2, 0.4, 0.6, 0.8])
            contours_frac = ax_frac.contour(
                L_grid,
                B_grid,
                frac_masked,
                levels=frac_levels,
                colors="white",
                linewidths=1.0,
            )
            ax_frac.clabel(contours_frac, inline=True, fontsize=8, fmt="%.1f")

            ax_frac.set_xlabel(r"$l$ [deg]")
            if i == 0:
                ax_frac.set_ylabel(r"$b$ [deg]")
            ax_frac.set_xlim(unique_l.max(), unique_l.min())
            ax_frac.set_ylim(unique_b.min(), 5.0)
            ax_frac.set_aspect("equal")

        plt.suptitle(
            fr"H25C20 Observed Data Refined Fit (with Continuity): Proper Motion {metric_name}",
            fontsize=16,
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(filename, dpi=200)
        plt.close()
        print(f"Saved {filename}")

    # Generate Mean and Covariance plots
    print("Generating proper motion component plots...")
    out_dir = os.path.join(project_root, "results/inversion")
    os.makedirs(out_dir, exist_ok=True)

    plot_component(0, "l", os.path.join(out_dir, "refined_fit_mu_l_h25c20.png"))
    plot_component(1, "b", os.path.join(out_dir, "refined_fit_mu_b_h25c20.png"))

    print("Generating dispersion and correlation plots...")
    plot_covariance_metric(
        "sig_l",
        "Dispersion l",
        os.path.join(out_dir, "refined_fit_sigma_l_h25c20.png"),
    )
    plot_covariance_metric(
        "sig_b",
        "Dispersion b",
        os.path.join(out_dir, "refined_fit_sigma_b_h25c20.png"),
    )
    plot_covariance_metric(
        "corr",
        "Correlation",
        os.path.join(out_dir, "refined_fit_correlation_h25c20.png"),
    )

    # 4. Plot Radial Velocity Residual Histograms (BRAVA and GIBS)
    print("Plotting RV residual histograms...")
    # BRAVA Residuals
    valid_b = mask_b & (~np.isnan(obs_mu_rv_b)) & (~np.isnan(pred_mu_rv_b))
    res_b = (obs_mu_rv_b - pred_mu_rv_b)[valid_b]
    norm_res_b = res_b / (obs_err_rv_b[valid_b] + 1e-10)

    # GIBS Residuals
    valid_g = mask_g & (~np.isnan(obs_mu_rv_g)) & (~np.isnan(pred_mu_rv_g))
    res_g = (obs_mu_rv_g - pred_mu_rv_g)[valid_g]
    norm_res_g = res_g / (obs_err_rv_g[valid_g] + 1e-10)

    plt.figure(figsize=(10, 4.5))

    plt.subplot(1, 2, 1)
    plt.hist(
        norm_res_b,
        bins=15,
        range=(-5, 5),
        color="crimson",
        edgecolor="black",
        alpha=0.7,
    )
    plt.axvline(0, color="black", linestyle="--", linewidth=1.5)
    plt.xlabel(
        r"$v_{los}$ Norm. Residual (Obs - Model) / $\sigma_{obs}$", fontsize=10
    )
    plt.ylabel("Count", fontsize=10)
    plt.title(
        f"BRAVA RV Norm. Residuals (N={len(res_b)})", fontsize=11, fontweight="bold"
    )
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(
        norm_res_g,
        bins=15,
        range=(-5, 5),
        color="royalblue",
        edgecolor="black",
        alpha=0.7,
    )
    plt.axvline(0, color="black", linestyle="--", linewidth=1.5)
    plt.xlabel(
        r"$v_{los}$ Norm. Residual (Obs - Model) / $\sigma_{obs}$", fontsize=10
    )
    plt.ylabel("Count", fontsize=10)
    plt.title(
        f"GIBS RV Norm. Residuals (N={len(res_g)})", fontsize=11, fontweight="bold"
    )
    plt.grid(True, alpha=0.3)

    plt.suptitle(
        "H25C20 Observed Data Refined Fit (with Continuity): Radial Velocity Norm. Residuals",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "refined_fit_rv_residuals_h25c20.png"), dpi=200
    )
    plt.close()
    print(
        f"Saved {os.path.join(out_dir, 'refined_fit_rv_residuals_h25c20.png')}"
    )

    print("\nRefined non-parametric visualization diagnostics complete!")


if __name__ == "__main__":
    main()
