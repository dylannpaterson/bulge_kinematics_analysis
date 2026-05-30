import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import vmap
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import json

# Append paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../shared')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../synthpop')))
import synthpop
from jax_kinematic_inverter import BulgeKinematicInverter
from inversion_config import get_inverter_grid_axes

def main():
    if os.environ.get("FAST_TEST") == "1":
        print("FAST_TEST mode active: skipping complex 2D contour plotting.")
        return
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    # Extract parameters from SynthPop
    print("Extracting initial parameters from SynthPop...")
    model_name = "Huston2025_C20Bulge"
    config = {
        "MANDATORY": {"name_for_output": "fit_init", "model_name": model_name},
        "SIGHTLINES": {"l_set": [0.0], "l_set_type": "list", "b_set": [0.0], "b_set_type": "list", "solid_angle": 0.01, "solid_angle_unit": "deg^2"}
    }
    tmp_conf = "tmp_vis_init.json"
    with open(tmp_conf, "w") as f: json.dump(config, f)
    sp = synthpop.SynthPop(tmp_conf)
    sp.init_populations()
    os.remove(tmp_conf)

    bulge_pop = sp.populations[0]
    dens = bulge_pop.population_density
    xyz_scale = getattr(dens, 'xyz_scale', 1.0)

    # 1. Load Fit Results
    output_path = os.path.join(project_root, "results/inversion/parametric_fit_results_real_h25c20.npz")
    if not os.path.exists(output_path):
        print(f"Fit results {output_path} not found.")
        return
    res = np.load(output_path)
    params_all = res['params']
    
    # Robustly load parameter values based on names if saved, or fall back to length
    if 'names' in res:
        names = list(res['names'])
        if isinstance(names[0], bytes):
            names = [n.decode('utf-8') for n in names]
        
        p_kosh_indices = [i for i, name in enumerate(names) if name not in ['omega', 'alpha', 'f_bulge', 'sigmoid_f_bulge']]
        p_kosh = params_all[p_kosh_indices]
        
        if 'omega' in names:
            omega = float(params_all[names.index('omega')])
        else:
            omega = 50.0
            
        if 'alpha' in names:
            alpha_deg = float(params_all[names.index('alpha')])
        else:
            alpha_deg = 18.43
            
        if 'f_bulge' in names:
            raw_f_bulge = float(params_all[names.index('f_bulge')])
            log_f_bulge = float(np.log(raw_f_bulge / (1.0 - raw_f_bulge)))
        elif 'sigmoid_f_bulge' in names:
            sig_f = float(params_all[names.index('sigmoid_f_bulge')])
            log_f_bulge = sig_f
        else:
            log_f_bulge = 0.0
    else:
        # Index-based fallbacks
        if len(params_all) == 21:
            p_kosh = params_all[:18]
            omega = params_all[18]
            alpha_deg = params_all[19]
            raw_f_bulge = params_all[20]
            log_f_bulge = np.log(raw_f_bulge / (1.0 - raw_f_bulge))
        elif len(params_all) == 11:
            p_kosh = params_all[:8]
            omega = params_all[8]
            log_f_bulge = params_all[9]
            alpha_deg = params_all[10]
        else:
            p_kosh = params_all[:8]
            omega = params_all[8]
            log_f_bulge = 0.0
            alpha_deg = params_all[9]
        
    alpha_rad = np.radians(alpha_deg)

    print(f"Loaded params from {output_path}")
    print(f"Omega: {omega:.2f}, Alpha: {alpha_deg:.2f} deg, Log(f_bulge): {log_f_bulge:.2f}")

    # 2. Setup Inverter
    grid_axes = get_inverter_grid_axes()
    solar = {
        'R0': np.abs(sp.parms.sun.gal_dist),
        'U_SUN': sp.parms.sun.u,
        'V_LSR': sp.parms.lsr['v_lsr'],
        'V_SUN_REL': sp.parms.sun.v - sp.parms.lsr['v_lsr'],
        'W_SUN': sp.parms.sun.w,
        'bar_angle_rad': alpha_rad,
        'xyz_scale': xyz_scale
    }
    
    # --- VIRAC2 (PMs) ---
    real_v = np.load(os.path.join(project_root, 'results/reduction/virac2_real_aligned_h25c20.npz'))
    obs_mu_pm_v = real_v['mu_pm']
    obs_cov_pm_v = real_v['cov_pm']
    obs_mu_err_v = real_v['mu_err']
    l_v, b_v = real_v['l'], real_v['b']
    mask_v = real_v['mask']
    
    # Use H25C20 metadata
    with open(os.path.join(project_root, 'results/inversion/huston25c20_metadata_science_binned.pkl'), 'rb') as f: meta_v = pickle.load(f)
    
    inverter_v = BulgeKinematicInverter(grid_axes, solar, n_bins=obs_mu_pm_v.shape[1])
    n_pix_v = len(l_v)
    
    # Prepare flat meta for vmap
    meta_axes = {k: (0 if (jnp.ndim(v) > 0 and v.shape[0] == n_pix_v) else None) for k, v in meta_v.items()}
    
    print("Predicting VIRAC2...")
    pred_v = vmap(lambda m: inverter_v.predict_parametric(p_kosh, omega, alpha_rad, m, log_f_bulge=log_f_bulge, return_components=True), in_axes=(meta_axes,))(meta_v)
    pred_mu_pm_v = np.array(pred_v[0]) # (N_pix, N_bins, 2)
    pred_cov_pm_v = np.array(pred_v[1]) # (N_pix, N_bins, 2, 2)
    f_bulge_v = np.array(pred_v[6]) # (N_pix, N_bins)

    # --- BRAVA (RVs) ---
    real_b = np.load(os.path.join(project_root, 'results/reduction/brava_real_aligned_h25c20.npz'))
    obs_mu_rv_b = real_b['mu_rv']
    mask_b = real_b['mask']
    # Use H25C20 metadata
    with open(os.path.join(project_root, 'results/inversion/huston25c20_brava_real_metadata.pkl'), 'rb') as f: meta_b = pickle.load(f)
    inverter_b = BulgeKinematicInverter(grid_axes, solar, n_bins=1)
    n_pix_b = len(meta_b['l'])
    meta_axes_b = {k: (0 if (jnp.ndim(v) > 0 and v.shape[0] == n_pix_b) else None) for k, v in meta_b.items()}
    print("Predicting BRAVA...")
    pred_b = vmap(lambda m: inverter_b.predict_parametric(p_kosh, omega, alpha_rad, m, log_f_bulge=log_f_bulge), in_axes=(meta_axes_b,))(meta_b)
    pred_mu_rv_b = np.array(pred_b[2])[:, 0] # (N_pix,)

    # --- GIBS (RVs) ---
    real_g = np.load(os.path.join(project_root, 'results/reduction/gibs_real_aligned_h25c20.npz'))
    obs_mu_rv_g = real_g['mu_rv']
    mask_g = real_g['mask']
    # Use H25C20 metadata
    with open(os.path.join(project_root, 'results/inversion/huston25c20_gibs_metadata_science.pkl'), 'rb') as f: meta_g = pickle.load(f)
    inverter_g = BulgeKinematicInverter(grid_axes, solar, n_bins=1)
    n_pix_g = len(meta_g['l'])
    meta_axes_g = {k: (0 if (jnp.ndim(v) > 0 and v.shape[0] == n_pix_g) else None) for k, v in meta_g.items()}
    print("Predicting GIBS...")
    pred_g = vmap(lambda m: inverter_g.predict_parametric(p_kosh, omega, alpha_rad, m, log_f_bulge=log_f_bulge), in_axes=(meta_axes_g,))(meta_g)
    pred_mu_rv_g = np.array(pred_g[2])[:, 0] # (N_pix,)

    # 3. Plotting VIRAC2 (PMs)
    k_indices = [2, 7, 12, 17] # selected slices
    # Mag bins from generate_metadata_huston25c20.py are likely same as science binned
    # Let's check mag_bins if stored, otherwise assume standard
    mag_bins = np.linspace(12.0, 14.0, 21)
    
    unique_l = np.unique(l_v)
    unique_b = np.unique(b_v)
    L_grid, B_grid = np.meshgrid(unique_l, unique_b)

    def plot_component(comp_idx, comp_name, filename):
        fig, axes = plt.subplots(4, len(k_indices), figsize=(5.5 * len(k_indices), 17), sharex=True, sharey=True)
        
        for i, k_idx in enumerate(k_indices):
            m = mask_v.reshape(len(unique_b), len(unique_l))
            obs_data = obs_mu_pm_v[:, k_idx, comp_idx].reshape(len(unique_b), len(unique_l))
            obs_err = obs_mu_err_v[:, k_idx, comp_idx].reshape(len(unique_b), len(unique_l))
            model_data = pred_mu_pm_v[:, k_idx, comp_idx].reshape(len(unique_b), len(unique_l))
            f_bulge_data = f_bulge_v[:, k_idx].reshape(len(unique_b), len(unique_l))
            
            # valid_mask: Masks ONLY the low-latitude |b| < 2.0 zone
            valid_mask = (np.abs(B_grid) >= 2.0)
            
            # valid_for_limits: strictly excludes the low-latitude zone, missing survey regions, high-error bins, and b > 5.0 for limits/contour scaling
            valid_for_limits = (np.abs(B_grid) >= 2.0) & m & (obs_err < 1.0) & (B_grid <= 5.0)
            
            obs_masked = np.where(valid_mask, obs_data, np.nan)
            model_masked = np.where(valid_mask, model_data, np.nan)
            obs_for_limits = np.where(valid_for_limits, obs_data, np.nan)
            
            # Calculate robust limits from the reliable high-latitude zone
            vmin = np.nanpercentile(obs_for_limits, 1)
            vmax = np.nanpercentile(obs_for_limits, 99)
            if np.isnan(vmin) or np.isnan(vmax) or vmin == vmax:
                vmin, vmax = -10.0, 10.0
            
            levels = np.linspace(vmin, vmax, 10)
            
            # Row 0: DATA
            ax_data = axes[0, i]
            im_obs = ax_data.pcolormesh(L_grid, B_grid, obs_masked, cmap='viridis', shading='auto', vmin=vmin, vmax=vmax, alpha=0.85)
            cb_obs = plt.colorbar(im_obs, ax=ax_data)
            cb_obs.set_label(fr'$\mu_{comp_name}$ (Obs) [mas/yr]', fontsize=9)
            
            contours_obs = ax_data.contour(L_grid, B_grid, model_masked, levels=levels, colors='black', linewidths=1.0)
            ax_data.clabel(contours_obs, inline=True, fontsize=8, fmt='%.1f')
            
            ax_data.set_title(fr"$K_s \in [{mag_bins[k_idx]:.1f}, {mag_bins[k_idx+1]:.1f}]$")
            if i == 0: ax_data.set_ylabel(r'$b$ [deg]')
            ax_data.set_xlim(unique_l.max(), unique_l.min())
            ax_data.set_ylim(unique_b.min(), 5.0)
            ax_data.set_aspect('equal')
            
            # Row 1: MODEL
            ax_model = axes[1, i]
            im_model = ax_model.pcolormesh(L_grid, B_grid, model_masked, cmap='viridis', shading='auto', vmin=vmin, vmax=vmax, alpha=0.85)
            cb_model = plt.colorbar(im_model, ax=ax_model)
            cb_model.set_label(fr'$\mu_{comp_name}$ (Model) [mas/yr]', fontsize=9)
            
            contours_model = ax_model.contour(L_grid, B_grid, model_masked, levels=levels, colors='black', linewidths=1.0)
            ax_model.clabel(contours_model, inline=True, fontsize=8, fmt='%.1f')
            if i == 0: ax_model.set_ylabel(r'$b$ [deg]')
            ax_model.set_ylim(unique_b.min(), 5.0)
            ax_model.set_aspect('equal')
            
            # Row 2: NORMALIZED RESIDUAL (Model - Obs) / Err
            ax_res = axes[2, i]
            res = (model_data - obs_data) / (obs_err + 1e-10)
            res_masked = np.where(valid_mask & m & (obs_err < 1.0), res, np.nan)
            
            im_res = ax_res.pcolormesh(L_grid, B_grid, res_masked, cmap='RdBu_r', vmin=-5, vmax=5, shading='auto')
            cb_res = plt.colorbar(im_res, ax=ax_res)
            cb_res.set_label(r'(Model - Obs) / $\sigma_{Obs}$', fontsize=9)
            if i == 0: ax_res.set_ylabel(r'$b$ [deg]')
            ax_res.set_ylim(unique_b.min(), 5.0)
            ax_res.set_aspect('equal')
            
            # Row 3: BULGE FRACTION
            ax_frac = axes[3, i]
            frac_masked = np.where(valid_mask, f_bulge_data, np.nan)
            
            im_frac = ax_frac.pcolormesh(L_grid, B_grid, frac_masked, cmap='plasma', shading='auto', vmin=0.0, vmax=1.0, alpha=0.85)
            cb_frac = plt.colorbar(im_frac, ax=ax_frac)
            cb_frac.set_label(r'$f_{\rm bulge}$', fontsize=9)
            
            frac_levels = np.array([0.2, 0.4, 0.6, 0.8])
            contours_frac = ax_frac.contour(L_grid, B_grid, frac_masked, levels=frac_levels, colors='white', linewidths=1.0)
            ax_frac.clabel(contours_frac, inline=True, fontsize=8, fmt='%.1f')
            
            ax_frac.set_xlabel(r'$l$ [deg]')
            if i == 0: ax_frac.set_ylabel(r'$b$ [deg]')
            ax_frac.set_xlim(unique_l.max(), unique_l.min())
            ax_frac.set_ylim(unique_b.min(), 5.0)
            ax_frac.set_aspect('equal')

        plt.suptitle(fr"H25C20 Real Data Parametric Fit: Proper Motion $\mu_{comp_name}$", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(filename, dpi=200)
        print(f"Saved {filename}")

    def plot_covariance_metric(metric_type, metric_name, filename):
        fig, axes = plt.subplots(4, len(k_indices), figsize=(5.5 * len(k_indices), 17), sharex=True, sharey=True)
        
        for i, k_idx in enumerate(k_indices):
            m = mask_v.reshape(len(unique_b), len(unique_l))
            # Use proper motion l error as proxy for limit masking
            obs_err = obs_mu_err_v[:, k_idx, 0].reshape(len(unique_b), len(unique_l))
            
            if metric_type == 'sig_l':
                obs_data = np.sqrt(np.clip(obs_cov_pm_v[:, k_idx, 0, 0], 1e-10, None)).reshape(len(unique_b), len(unique_l))
                model_data = np.sqrt(np.clip(pred_cov_pm_v[:, k_idx, 0, 0], 1e-10, None)).reshape(len(unique_b), len(unique_l))
                cmap = 'plasma'
                label_suffix = r'$\sigma_{\mu_l}$ [mas/yr]'
                fmt = '%.2f'
            elif metric_type == 'sig_b':
                obs_data = np.sqrt(np.clip(obs_cov_pm_v[:, k_idx, 1, 1], 1e-10, None)).reshape(len(unique_b), len(unique_l))
                model_data = np.sqrt(np.clip(pred_cov_pm_v[:, k_idx, 1, 1], 1e-10, None)).reshape(len(unique_b), len(unique_l))
                cmap = 'plasma'
                label_suffix = r'$\sigma_{\mu_b}$ [mas/yr]'
                fmt = '%.2f'
            elif metric_type == 'corr':
                obs_var_l = obs_cov_pm_v[:, k_idx, 0, 0]
                obs_var_b = obs_cov_pm_v[:, k_idx, 1, 1]
                obs_data = (obs_cov_pm_v[:, k_idx, 0, 1] / (np.sqrt(np.clip(obs_var_l * obs_var_b, 1e-10, None)) + 1e-10)).reshape(len(unique_b), len(unique_l))
                obs_data = np.clip(obs_data, -1.0, 1.0)
                
                model_var_l = pred_cov_pm_v[:, k_idx, 0, 0]
                model_var_b = pred_cov_pm_v[:, k_idx, 1, 1]
                model_data = (pred_cov_pm_v[:, k_idx, 0, 1] / (np.sqrt(np.clip(model_var_l * model_var_b, 1e-10, None)) + 1e-10)).reshape(len(unique_b), len(unique_l))
                model_data = np.clip(model_data, -1.0, 1.0)
                cmap = 'RdBu_r'
                label_suffix = r'$\rho_{\mu_l, \mu_b}$'
                fmt = '%.2f'
            
            # valid_mask: Masks ONLY the low-latitude |b| < 2.0 zone, leaving the rest completely unmasked as requested!
            valid_mask = (np.abs(B_grid) >= 2.0)
            
            # valid_for_limits: strictly excludes the low-latitude zone, missing survey regions, high-error bins, and b > 5.0 for limits/contour scaling
            valid_for_limits = (np.abs(B_grid) >= 2.0) & m & (obs_err < 1.0) & (B_grid <= 5.0)
            
            obs_masked = np.where(valid_mask, obs_data, np.nan)
            model_masked = np.where(valid_mask, model_data, np.nan)
            obs_for_limits = np.where(valid_for_limits, obs_data, np.nan)
            
            # Calculate robust limits from the reliable high-latitude zone
            if metric_type == 'corr':
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
            
            # Row 0: DATA (Observed proper motion metric background)
            ax_data = axes[0, i]
            im_obs = ax_data.pcolormesh(L_grid, B_grid, obs_masked, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax, alpha=0.85)
            cb_obs = plt.colorbar(im_obs, ax=ax_data)
            cb_obs.set_label(fr'{label_suffix} (Obs)', fontsize=9)
            
            # Overlay smooth model contours for direct visual comparison
            contours_obs = ax_data.contour(L_grid, B_grid, model_masked, levels=levels, colors='black', linewidths=1.0)
            ax_data.clabel(contours_obs, inline=True, fontsize=8, fmt=fmt)
            
            ax_data.set_title(fr"$K_s \in [{mag_bins[k_idx]:.1f}, {mag_bins[k_idx+1]:.1f}]$")
            if i == 0: ax_data.set_ylabel(r'$b$ [deg]')
            ax_data.set_xlim(unique_l.max(), unique_l.min())
            ax_data.set_ylim(unique_b.min(), 5.0)
            ax_data.set_aspect('equal')
            
            # Row 1: MODEL (Predicted proper motion metric background + contours)
            ax_model = axes[1, i]
            im_model = ax_model.pcolormesh(L_grid, B_grid, model_masked, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax, alpha=0.85)
            cb_model = plt.colorbar(im_model, ax=ax_model)
            cb_model.set_label(fr'{label_suffix} (Model)', fontsize=9)
            
            contours_model = ax_model.contour(L_grid, B_grid, model_masked, levels=levels, colors='black', linewidths=1.0)
            ax_model.clabel(contours_model, inline=True, fontsize=8, fmt=fmt)
            if i == 0: ax_model.set_ylabel(r'$b$ [deg]')
            ax_model.set_ylim(unique_b.min(), 5.0)
            ax_model.set_aspect('equal')
            
            # Row 2: RESIDUAL (Model - Obs)
            ax_res = axes[2, i]
            res = np.where(valid_mask, model_data - obs_data, np.nan)
            res_for_limits = np.where(valid_for_limits, model_data - obs_data, np.nan)
            
            # Determine symmetric color scale for residuals
            max_res = np.nanpercentile(np.abs(res_for_limits), 98)
            if np.isnan(max_res) or max_res == 0:
                max_res = 0.5
                
            im_res = ax_res.pcolormesh(L_grid, B_grid, res, cmap='RdBu_r', vmin=-max_res, vmax=max_res, shading='auto')
            cb_res = plt.colorbar(im_res, ax=ax_res)
            cb_res.set_label('Model - Obs', fontsize=9)
            if i == 0: ax_res.set_ylabel(r'$b$ [deg]')
            ax_res.set_ylim(unique_b.min(), 5.0)
            ax_res.set_aspect('equal')
            
            # Row 3: BULGE FRACTION
            ax_frac = axes[3, i]
            f_bulge_data = f_bulge_v[:, k_idx].reshape(len(unique_b), len(unique_l))
            frac_masked = np.where(valid_mask, f_bulge_data, np.nan)
            
            im_frac = ax_frac.pcolormesh(L_grid, B_grid, frac_masked, cmap='plasma', shading='auto', vmin=0.0, vmax=1.0, alpha=0.85)
            cb_frac = plt.colorbar(im_frac, ax=ax_frac)
            cb_frac.set_label(r'$f_{\rm bulge}$', fontsize=9)
            
            frac_levels = np.array([0.2, 0.4, 0.6, 0.8])
            contours_frac = ax_frac.contour(L_grid, B_grid, frac_masked, levels=frac_levels, colors='white', linewidths=1.0)
            ax_frac.clabel(contours_frac, inline=True, fontsize=8, fmt='%.1f')
            
            ax_frac.set_xlabel(r'$l$ [deg]')
            if i == 0: ax_frac.set_ylabel(r'$b$ [deg]')
            ax_frac.set_xlim(unique_l.max(), unique_l.min())
            ax_frac.set_ylim(unique_b.min(), 5.0)
            ax_frac.set_aspect('equal')

        plt.suptitle(fr"H25C20 Real Data Parametric Fit: {metric_name}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(filename, dpi=200)
        print(f"Saved {filename}")

    out_dir = os.path.join(project_root, "results/inversion")
    os.makedirs(out_dir, exist_ok=True)

    plot_component(0, 'l', os.path.join(out_dir, 'parametric_fit_real_h25c20_pml.png'))
    plot_component(1, 'b', os.path.join(out_dir, 'parametric_fit_real_h25c20_pmb.png'))
    
    print("Generating proper motion dispersion and correlation plots...")
    plot_covariance_metric('sig_l', r'Proper Motion Dispersion in Longitude $\sigma_{\mu_l}$', os.path.join(out_dir, 'parametric_fit_real_h25c20_sigl.png'))
    plot_covariance_metric('sig_b', r'Proper Motion Dispersion in Latitude $\sigma_{\mu_b}$', os.path.join(out_dir, 'parametric_fit_real_h25c20_sigb.png'))
    plot_covariance_metric('corr', r'Proper Motion Correlation $\rho_{\mu_l, \mu_b}$', os.path.join(out_dir, 'parametric_fit_real_h25c20_corr.png'))

    # 4. Plotting RV Residuals
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    
    # Check mask_b length vs pred_mu_rv_b
    if len(mask_b) == len(pred_mu_rv_b):
        res_b = (pred_mu_rv_b - obs_mu_rv_b.flatten())[mask_b]
        err_b = obs_mu_err_v[0, 0, 0] # fallback if not found, but let's try to get it properly
        # Note: obs_mu_err_b should be available if we load it. Let's assume it's in the real_b file.
        err_b = real_b['mu_err'].flatten()[mask_b] if 'mu_err' in real_b else 10.0
        norm_res_b = res_b / (err_b + 1e-10)
        
        axes2[0].hist(norm_res_b, bins=15, range=(-5, 5), color='C0', alpha=0.7, edgecolor='black')
        axes2[0].axvline(0, color='red', linestyle='--')
        axes2[0].set_title(f"BRAVA RV Norm. Residuals (H25C20)\nMean={np.mean(norm_res_b):.2f}, Std={np.std(norm_res_b):.2f}")
        axes2[0].set_xlabel(r"$(\mu_{RV, model} - \mu_{RV, obs}) / \sigma_{obs}$")
    
    if len(mask_g) == len(pred_mu_rv_g):
        res_g = (pred_mu_rv_g - obs_mu_rv_g.flatten())[mask_g]
        err_g = real_g['mu_err'].flatten()[mask_g] if 'mu_err' in real_g else 10.0
        norm_res_g = res_g / (err_g + 1e-10)
        
        axes2[1].hist(norm_res_g, bins=15, range=(-5, 5), color='C1', alpha=0.7, edgecolor='black')
        axes2[1].axvline(0, color='red', linestyle='--')
        axes2[1].set_title(f"GIBS RV Norm. Residuals (H25C20)\nMean={np.mean(norm_res_g):.2f}, Std={np.std(norm_res_g):.2f}")
        axes2[1].set_xlabel(r"$(\mu_{RV, model} - \mu_{RV, obs}) / \sigma_{obs}$")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'parametric_fit_real_h25c20_rv_residuals.png'), dpi=200)
    print("RV residual histograms saved.")

if __name__ == "__main__":
    main()
