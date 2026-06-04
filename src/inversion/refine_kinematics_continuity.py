"""
Refined non-parametric kinematic inversion with continuity equation penalty.

Starts from the results of fit_obs_combined_h25c20.py and performs a second 
optimization pass including an analytical divergence penalty del . (rho * u) = 0.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, value_and_grad, vmap, jacfwd
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

# Set up project root imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(project_root, "src/shared"))
sys.path.append(os.path.join(project_root, "src/inversion"))
sys.path.append(os.path.join(project_root, "synthpop"))

import synthpop
from inversion_config import get_inverter_grid_axes
from jax_kinematic_inverter import BulgeKinematicInverter

# ---------------------------------------------------------------------------
# 1. Differentiable Density Grid Precomputation
# ---------------------------------------------------------------------------

def precompute_density_grid(grid_axes):
    model_name = "Huston2025_C20Bulge"
    config = {
        "MANDATORY": {"name_for_output": "dens_tmp", "model_name": model_name},
        "SIGHTLINES": {
            "l_set": [0.0], "l_set_type": "list", "b_set": [0.0], "b_set_type": "list",
            "solid_angle": 0.01, "solid_angle_unit": "deg^2",
        },
    }
    tmp_conf = "tmp_dens_pre.json"
    with open(tmp_conf, "w") as f:
        json.dump(config, f)
    sp = synthpop.SynthPop(tmp_conf)
    sp.init_populations()
    os.remove(tmp_conf)
    
    bulge_pop = sp.populations[0]
    trans = bulge_pop.coord_trans
    
    X, Y, Z = jnp.meshgrid(grid_axes[0], grid_axes[1], grid_axes[2], indexing="ij")
    r, phi, z_sp = trans.xyz_to_rphiz(X.flatten(), Y.flatten(), Z.flatten())
    rho = bulge_pop.population_density.density(r, phi, z_sp).reshape(X.shape)
    
    if bulge_pop.population_density.density_unit == "number":
        # Extract mean mass from SynthPop provider
        from synthpop_moment_provider import SynthPopDiskMomentProvider
        provider = SynthPopDiskMomentProvider(sp)
        avg_m = provider.mass_mag_maps[bulge_pop.name]['mean_mass']
        rho *= avg_m
        
    return rho

# ---------------------------------------------------------------------------
# 2. Divergence Penalty Helper
# ---------------------------------------------------------------------------

@jit
def interpolate_density(x, y, z, axes, rho_grid):
    ix = (jnp.abs(x) - axes[0][0]) / (axes[0][1] - axes[0][0])
    iy = (jnp.abs(y) - axes[1][0]) / (axes[1][1] - axes[1][0])
    iz = (jnp.abs(z) - axes[2][0]) / (axes[2][1] - axes[2][0])
    coords = jnp.stack([ix, iy, iz])
    return jax.scipy.ndimage.map_coordinates(rho_grid, coords.reshape(3,1), order=1, mode='nearest').squeeze()

def get_flux_logic(inverter, density_grid, grid_axes_jnp):
    def mass_flux_vector(pos, grid_params, omega):
        x, y, z = pos[0], pos[1], pos[2]
        rho = interpolate_density(x, y, z, grid_axes_jnp, density_grid)
        p_chol = inverter.interpolate_octant(x, y, z, grid_params)
        ux, uy = p_chol[0], p_chol[1]
        ux_rot = ux * jnp.sign(y)
        uy_rot = uy * jnp.sign(x)
        return jnp.array([rho * ux_rot, rho * uy_rot, 0.0])
    return mass_flux_vector

@functools.partial(jit, static_argnums=(2,))
def compute_divergence_penalty(grid_params, omega, flux_fn, collocation_points):
    def single_point_divergence(pos):
        J_matrix = jacfwd(flux_fn, argnums=0)(pos, grid_params, omega)
        return jnp.trace(J_matrix)
    divs = vmap(single_point_divergence)(collocation_points)
    return jnp.mean(divs**2)

# ---------------------------------------------------------------------------
# 3. Masked Grid NLL (Copy from fit_obs_combined_h25c20.py)
# ---------------------------------------------------------------------------

def get_masked_grid_nll(
    inverter, grid_params, omega, meta_batch, log_f_bulge,
    obs_mu_pm=None, obs_cov_pm=None, obs_mu_err_pm=None,
    obs_mu_rv=None, obs_var_rv=None, obs_mu_err_rv=None,
    mask=None, rv_weight=1.0,
):
    n_pixels = jnp.size(meta_batch["l"])
    def flatten_spatial(x):
        if x is None: return None
        return x.reshape((n_pixels,) + x.shape[len(meta_batch["l"].shape) :])

    flat_meta = {k: (v if k == "d" else flatten_spatial(v)) for k, v in meta_batch.items()}
    flat_obs_mu_pm = flatten_spatial(obs_mu_pm)
    flat_obs_cov_pm = flatten_spatial(obs_cov_pm)
    flat_obs_mu_err_pm = flatten_spatial(obs_mu_err_pm)
    flat_obs_mu_rv = flatten_spatial(obs_mu_rv)
    flat_obs_var_rv = flatten_spatial(obs_var_rv)
    flat_obs_mu_err_rv = flatten_spatial(obs_mu_err_rv)
    flat_mask = mask.flatten()

    meta_axes = {k: (0 if (jnp.ndim(v) > 0 and v.shape[0] == n_pixels) else None) for k, v in flat_meta.items()}
    preds = vmap(lambda m: inverter.predict_pixel(grid_params, omega, m, log_f_bulge=log_f_bulge), in_axes=(meta_axes,))(flat_meta)
    pred_mu_pm, pred_cov_pm, pred_mu_rv, pred_var_rv = preds

    nll = 0.0
    if flat_obs_mu_pm is not None:
        diff_pm = flat_obs_mu_pm - pred_mu_pm
        def bin_loss_pm(i, k):
            C_pred = pred_cov_pm[i, k] + jnp.eye(2) * 1e-6
            inv_C = jnp.linalg.inv(C_pred)
            _, logdet = jnp.linalg.slogdet(C_pred)
            n_eff = jnp.mean(jnp.diag(flat_obs_cov_pm[i, k]) / (flat_obs_mu_err_pm[i, k] ** 2 + 1e-10)) if flat_obs_mu_err_pm is not None else 100.0
            n_eff = jnp.clip(n_eff, 1.0, 1e6)
            return n_eff * (logdet + jnp.trace(inv_C @ flat_obs_cov_pm[i, k]) + diff_pm[i, k].T @ inv_C @ diff_pm[i, k])
        pixel_losses = vmap(lambda i: jnp.sum(vmap(lambda k: bin_loss_pm(i, k))(jnp.arange(inverter.n_bins))))(jnp.arange(n_pixels))
        nll += jnp.sum(pixel_losses * flat_mask) / (jnp.sum(flat_mask) + 1e-10)

    if flat_obs_mu_rv is not None:
        diff_rv = flat_obs_mu_rv - pred_mu_rv
        def bin_loss_rv(i, k):
            var_pred = pred_var_rv[i, k] + 1e-6
            n_eff = (flat_obs_var_rv[i, k] / (flat_obs_mu_err_rv[i, k] ** 2 + 1e-10)) if flat_obs_mu_err_rv is not None else 100.0
            n_eff = jnp.clip(n_eff, 1.0, 1e6)
            return n_eff * (jnp.log(var_pred) + (flat_obs_var_rv[i, k] + (diff_rv[i, k] ** 2)) / var_pred)
        pixel_losses = vmap(lambda i: jnp.sum(vmap(lambda k: bin_loss_rv(i, k))(jnp.arange(inverter.n_bins))))(jnp.arange(n_pixels))
        nll += rv_weight * jnp.sum(pixel_losses * flat_mask) / (jnp.sum(flat_mask) + 1e-10)
    return nll

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main() -> None:
    """Run combined non-parametric kinematic inversion with continuity."""
    # 0. Load Previous NP Fit Results
    input_path = os.path.join(project_root, "results/inversion/fit_results_obs_combined_h25c20.npz")
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Run standard inversion first.")
        return
    print(f"Loading previous fit from {input_path}...")
    prev_data = np.load(input_path)
    grid_start = jnp.array(prev_data['grid'])
    omega_start = float(prev_data['omega'])
    p_parametric = jnp.array(prev_data['p_parametric'])

    # 1. Setup Fixed Parameters
    v0_str, y0_str = p_parametric[0], p_parametric[1]
    sig_i0_x, sig_i1_x, sig_i0_y, sig_i1_y, sig_i0_z, sig_i1_z = p_parametric[2:8]
    h0_r = p_parametric[8:11]
    h0_z = p_parametric[11:14]
    C_par_r, C_perp_r = p_parametric[14], p_parametric[15]
    C_par_z, C_perp_z = p_parametric[16], p_parametric[17]
    omega_fit = p_parametric[19]
    sigma_omega = 0.51
    fixed_alpha = float(p_parametric[20])
    raw_f_bulge = float(p_parametric[21])
    fixed_log_f_bulge = float(np.log(raw_f_bulge / (1.0 - raw_f_bulge)))

    solar = {
        "R0": 8.178, "U_SUN": 12.9, "V_LSR": 233.4, "V_SUN_REL": 12.2, "W_SUN": 7.78,
        "bar_angle_rad": jnp.radians(fixed_alpha),
        "x_scale": 1.13442, "y_scale": 1.15296, "z_scale": 1.10504,
    }

    # 2. Load Observed Data
    print("Loading observed data...")
    data_v = np.load(os.path.join(project_root, "results/reduction/virac2_obs_aligned_h25c20.npz"))
    obs_mu_pm_v, obs_cov_pm_v, obs_mu_err_v, mask_v = jnp.array(data_v["mu_pm"]), jnp.array(data_v["cov_pm"]), jnp.array(data_v["mu_err"]), jnp.array(data_v["mask"])
    mask_v = mask_v & (jnp.abs(data_v["b"]) >= 2.0)
    with open(os.path.join(project_root, "results/inversion/huston25c20_metadata_science_binned.pkl"), "rb") as f: meta_v = pickle.load(f)
    
    data_b = np.load(os.path.join(project_root, "results/reduction/brava_obs_aligned_h25c20.npz"))
    obs_mu_rv_b, obs_var_rv_b, obs_mu_err_b, mask_b = jnp.array(data_b["mu_rv"]), jnp.array(data_b["var_rv"]), jnp.array(data_b["mu_err"]), jnp.array(data_b["mask"])
    with open(os.path.join(project_root, "results/inversion/huston25c20_brava_obs_metadata.pkl"), "rb") as f: meta_b = pickle.load(f)
    
    data_g = np.load(os.path.join(project_root, "results/reduction/gibs_obs_aligned_h25c20.npz"))
    obs_mu_rv_g, obs_var_rv_g, obs_mu_err_g, mask_g = jnp.array(data_g["mu_rv"]), jnp.array(data_g["var_rv"]), jnp.array(data_g["mu_err"]), jnp.array(data_g["mask"])
    with open(os.path.join(project_root, "results/inversion/huston25c20_gibs_metadata_science.pkl"), "rb") as f: meta_g = pickle.load(f)

    # 3. Setup Inverters
    grid_axes = [jnp.array(prev_data['axes_x']), jnp.array(prev_data['axes_y']), jnp.array(prev_data['axes_z'])]
    inverter_v = BulgeKinematicInverter(grid_axes, solar, n_bins=obs_mu_pm_v.shape[1])
    inverter_b = BulgeKinematicInverter(grid_axes, solar, n_bins=1)
    inverter_g = BulgeKinematicInverter(grid_axes, solar, n_bins=1)

    # 4. Construct Parametric Prior Grids
    X, Y, Z = jnp.meshgrid(grid_axes[0], grid_axes[1], grid_axes[2], indexing="ij")
    ax, ay, az = jnp.abs(X), jnp.abs(Y), jnp.abs(Z)
    s_r = (((ax/h0_r[0])**C_perp_r + (ay/h0_r[1])**C_perp_r)**(C_par_r/C_perp_r) + (az/h0_r[2])**C_par_r)**(1/C_par_r)
    s_z = (((ax/h0_z[0])**C_perp_z + (ay/h0_z[1])**C_perp_z)**(C_par_z/C_perp_z) + (az/h0_z[2])**C_par_z)**(1/C_par_z)
    ux_prior, uy_prior = v0_str * (1.0 - jnp.exp(-((Y / y0_str) ** 2))), jnp.zeros_like(X)
    sig_x_prior, sig_y_prior, sig_z_prior = sig_i0_x + sig_i1_x * jnp.exp(-s_r), sig_i0_y + sig_i1_y * jnp.exp(-s_r), sig_i0_z + sig_i1_z * jnp.exp(-s_z)

    # 5. Precompute Continuity components
    print("Precomputing differentiable density and collocation points...")
    density_grid = precompute_density_grid(grid_axes)
    N_COLL = 500
    key = jax.random.PRNGKey(42)
    cp_x = jax.random.uniform(key, (N_COLL,), minval=-3.5, maxval=3.5)
    cp_y = jax.random.uniform(jax.random.split(key)[1], (N_COLL,), minval=-1.8, maxval=1.8)
    cp_z = jax.random.uniform(jax.random.split(key)[1], (N_COLL,), minval=-0.8, maxval=0.8)
    collocation_points = jnp.stack([cp_x, cp_y, cp_z], axis=1)
    flux_fn = get_flux_logic(inverter_v, density_grid, grid_axes)

    # 6. Joint Loss Function
    lambda_div = 1.0e-18 # Significantly lowered to be comparable to NLL (~10^5) vs DivSq (~10^19)
    kl_weight, curv_weight = 1.0, 1.0

    @jit
    def custom_kl_divergence_penalty(grid_params):
        ux, uy, log_L11, L21, log_L22, L31, L32, log_L33 = [grid_params[..., i] for i in range(8)]
        L11_sq, L22_sq, L33_sq, L21_sq, L31_sq, L32_sq = jnp.exp(2 * log_L11), jnp.exp(2 * log_L22), jnp.exp(2 * log_L33), L21**2, L31**2, L32**2
        tr_term = L11_sq / (sig_x_prior**2) + (L21_sq + L22_sq) / (sig_y_prior**2) + (L31_sq + L32_sq + L33_sq) / (sig_z_prior**2)
        mu_term = (ux - ux_prior) ** 2 / (sig_x_prior**2) + (uy - uy_prior) ** 2 / (sig_y_prior**2)
        log_det_p, log_det_s = 2.0 * (jnp.log(sig_x_prior) + jnp.log(sig_y_prior) + jnp.log(sig_z_prior)), 2.0 * (log_L11 + log_L22 + log_L33)
        return jnp.mean(0.5 * (tr_term + mu_term - 3.0 + log_det_p - log_det_s))

    @jit
    def joint_loss_fn(params):
        grid_params, omega = params
        nll_v = get_masked_grid_nll(inverter_v, grid_params, omega, meta_v, fixed_log_f_bulge, obs_mu_pm=obs_mu_pm_v, obs_cov_pm=obs_cov_pm_v, obs_mu_err_pm=obs_mu_err_v, mask=mask_v)
        nll_b = get_masked_grid_nll(inverter_b, grid_params, omega, meta_b, fixed_log_f_bulge, obs_mu_rv=obs_mu_rv_b, obs_var_rv=obs_var_rv_b, obs_mu_err_rv=obs_mu_err_b, mask=mask_b)
        nll_g = get_masked_grid_nll(inverter_g, grid_params, omega, meta_g, fixed_log_f_bulge, obs_mu_rv=obs_mu_rv_g, obs_var_rv=obs_var_rv_g, obs_mu_err_rv=obs_mu_err_g, mask=mask_g)
        
        curv_reg = 0.0
        for p in range(8):
            field = grid_params[..., p]
            curv_reg += (jnp.mean(jnp.diff(field, n=2, axis=0) ** 2) + jnp.mean(jnp.diff(field, n=2, axis=1) ** 2) + jnp.mean(jnp.diff(field, n=2, axis=2) ** 2))

        kl = custom_kl_divergence_penalty(grid_params)
        sigma_omega = 5.0
        om_pen = 0.5 * ((omega - omega_fit) / sigma_omega) ** 2
        div_pen = compute_divergence_penalty(grid_params, omega, flux_fn, collocation_points)
        
        total_loss = nll_v + nll_b + nll_g + kl_weight * kl + curv_weight * curv_reg + om_pen + lambda_div * div_pen
        return total_loss, (nll_v + nll_b + nll_g, div_pen)

    val_and_grad_fn = jit(value_and_grad(joint_loss_fn, has_aux=True))

    def scipy_objective(x_flat):
        grid_jnp, omega = jnp.array(x_flat[:-1]).reshape(grid_start.shape), x_flat[-1]
        (loss, aux), grads = val_and_grad_fn((grid_jnp, omega))
        if not hasattr(scipy_objective, 'n'): scipy_objective.n = 0
        if scipy_objective.n % 5 == 0:
            print(f"  Iter {scipy_objective.n:3d}: Loss={loss:.4f} [NLL={aux[0]:.2f}, DivSq={aux[1]:.2e}], Om={omega:.2f}")
        scipy_objective.n += 1
        return float(loss), np.concatenate([np.array(grads[0]).flatten(), [float(grads[1])]]).astype(np.float64)

    grid_bounds = [
        (-400, 400),  # ux
        (-400, 400),  # uy
        (1.6, 6.0),   # log_L11
        (-400, 400),  # L21
        (1.6, 6.0),   # log_L22
        (-400, 400),  # L31
        (-400, 400),  # L32
        (1.6, 6.0),   # log_L33
    ] * np.prod(grid_start.shape[:-1])
    omega_bounds = [(0, 100.0)]
    bounds = grid_bounds + omega_bounds

    print(f"Starting Refined Optimization with continuity constraint (lambda={lambda_div})...")
    res = minimize(
        scipy_objective,
        np.concatenate([np.array(grid_start).flatten(), [omega_start]]),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 10000, "ftol": 1e-10}
    )

    out_path = os.path.join(project_root, "results/inversion/fit_results_refined_continuity.npz")
    np.savez(out_path, grid=np.array(res.x[:-1]).reshape(grid_start.shape), omega=float(res.x[-1]), axes_x=grid_axes[0], axes_y=grid_axes[1], axes_z=grid_axes[2], p_parametric=np.array(p_parametric))
    print(f"\nRefinement Complete. Results saved to {out_path}")

if __name__ == "__main__":
    main()
