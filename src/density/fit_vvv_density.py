import os
import sys

# 1. Set thread-limiting variables to prevent CPU thrashing in parallel workers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"

# 2. Force 'fork' start method for clean, fast parallel workers
import multiprocessing as mp
try:
    mp.set_start_method('fork', force=True)
except Exception:
    pass

import json
import time
import numpy as np
import astropy.io.fits as fits
import emcee
import corner
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Add necessary paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)
# Add synthpop path
synthpop_root = os.path.abspath(os.path.join(project_root, '../synthpop'))
if synthpop_root not in sys.path:
    sys.path.append(synthpop_root)

import synthpop
from src.shared.synthpop_moment_provider import SynthPopDiskMomentProvider
from src.shared.inversion_config import get_base_synthpop_config

# -----------------------------------------------------------------------------
# PHASE 1: PRECOMPUTATION
# -----------------------------------------------------------------------------
class VVVStarCountsProvider(SynthPopDiskMomentProvider):
    def _precompute_mass_mag_mappings(self):
        print("  Running custom high-resolution VVV precomputation...")
        for pop in self.populations_all:
            is_bulge = 'bulge' in pop.name.lower()
            
            # Bulge stars are far away, so faint stars are completely out of VVV selection
            # We can use a different mass and magnitude grid for the bulge vs the disk
            if is_bulge:
                m_grid_base = np.geomspace(0.15, 100.0, 5000)
            else:
                m_grid_base = np.geomspace(0.08, 100.0, 10000)
                
            dm_base = np.gradient(m_grid_base)

            z_low = getattr(pop.metallicity, 'lower', getattr(pop.metallicity, 'low_bound', -2.0))
            z_high = getattr(pop.metallicity, 'upper', getattr(pop.metallicity, 'high_bound', 0.5))
            z_steps = np.unique(np.linspace(z_low, z_high, 40))

            if hasattr(pop.age, 'age_value'):
                a_steps = np.array([pop.age.age_value])
            else:
                a_low = getattr(pop.age, 'lower', getattr(pop.age, 'low_bound', 0.1))
                a_high = getattr(pop.age, 'upper', getattr(pop.age, 'high_bound', 12.0))
                n_a = 1 if is_bulge else 10
                a_steps = np.unique(np.linspace(a_low, a_high, n_a))

            M, Z, A = np.meshgrid(m_grid_base, z_steps, a_steps, indexing='ij')
            DM, _, _ = np.meshgrid(dm_base, z_steps, a_steps, indexing='ij')
            m_flat, z_flat, a_flat = M.flatten(), Z.flatten(), A.flatten()
            dm_flat = DM.flatten()

            w_z = pop.metallicity.likelyhood_distribution(z_steps)
            w_a = pop.age.likelyhood_distribution(a_steps) if hasattr(pop.age, 'likelyhood_distribution') else np.ones_like(a_steps)
            grid_norm = np.sum(w_z) * np.sum(w_a)
            if grid_norm <= 0: grid_norm = 1.0

            _, WZ, WA = np.meshgrid(m_grid_base, w_z, w_a, indexing='ij')
            weights = (pop.imf.imf(m_flat) * dm_flat * WZ.flatten() * WA.flatten()) / grid_norm

            try:
                req_props = {'2MASS_J', '2MASS_Ks', 'phase'}
                props, _, _ = pop.evolution.get_evolved_props(m_flat, z_flat, a_flat, req_props)
                if isinstance(props, tuple): props = props[0]

                mk_abs = np.array(props['2MASS_Ks'], dtype=float)
                mj_abs = np.array(props['2MASS_J'], dtype=float)

                if is_bulge:
                    valid = ~np.isnan(mk_abs) & ~np.isnan(mj_abs) & \
                            (mj_abs - mk_abs > 0.4) & (mj_abs - mk_abs < 1.0) & \
                            (mk_abs > -7.0) & (mk_abs < 5.0)
                else:
                    valid = ~np.isnan(mk_abs) & ~np.isnan(mj_abs) & \
                            (mj_abs - mk_abs > 0.4) & (mj_abs - mk_abs < 1.0) & \
                            (mk_abs > -7.0) & (mk_abs < 15.0)

                av_mass_corr = getattr(pop, 'av_mass_corr', None)
                if av_mass_corr is None:
                    try:
                        if not hasattr(pop, 'position') or pop.position.l_deg is None:
                            pop.set_position(0.0, 0.0, 0.01, 'deg^2')
                        av_mass_corr = pop.get_mass_loss_for_option(pop.lost_mass_option)
                    except:
                        av_mass_corr = 1.0
                    pop.av_mass_corr = av_mass_corr

                total_num_grid = np.sum(weights)
                total_mass_grid = np.sum(m_flat * weights)
                norm = total_mass_grid * av_mass_corr
                mean_mass_per_star = norm / total_num_grid if total_num_grid > 0 else 1.0

                v_idx = np.where(valid)[0]
                target = mk_abs[v_idx] 
                sort_idx = np.argsort(target)

                self.mass_mag_maps[pop.name] = {
                    'Mk': target[sort_idx],
                    'Mj': mj_abs[v_idx][sort_idx],
                    'cum_imf_num': np.cumsum(weights[v_idx][sort_idx]),
                    'total_num_norm': total_num_grid,
                    'total_mass_norm': norm,
                    'mean_mass': mean_mass_per_star
                }
            except Exception as e:
                print(f"Error mapping {pop.name}: {e}")

def run_precomputation():
    print("====================================================================")
    print("--- PHASE 1: Precomputing VVV Star Counts ---")
    print("====================================================================")
    
    # 1. Setup SynthPop model
    model_name = "Huston2025_C20Bulge"
    config = get_base_synthpop_config(model_name, "fit", solid_angle=0.04)

    tmp_conf = "tmp_fit_config.json"
    with open(tmp_conf, "w") as f:
        json.dump(config, f)
    sp = synthpop.SynthPop(tmp_conf)
    sp.init_populations()

    R0 = np.abs(sp.parms.sun.gal_dist)
    Z0 = sp.parms.sun.z
    theta_tilt = sp.parms.sun.theta

    # 2. VVV FITS data and Ks Observational Errors
    fits_path = os.path.join(project_root, 'data/vvvdaophot_completed_masked_mlferr_newzp.fits')
    with fits.open(fits_path) as hdul:
        header = hdul[0].header
        obs_data = hdul[0].data 

    err_fits_path = os.path.join(project_root, 'data/vvvdaophot_kserr.fits')
    with fits.open(err_fits_path) as hdul_err:
        err_data = hdul_err[0].data

    naxis1 = header['NAXIS1']
    naxis2 = header['NAXIS2']
    naxis3 = header['NAXIS3']

    l_arr = header['CRVAL1'] + (np.arange(1, naxis1 + 1) - header['CRPIX1']) * header['CDELT1']
    b_arr = header['CRVAL2'] + (np.arange(1, naxis2 + 1) - header['CRPIX2']) * header['CDELT2']
    ks_arr = header['CRVAL3'] + (np.arange(1, naxis3 + 1) - header['CRPIX3']) * header['CDELT3']

    # Filter to Ks range 12-14 for observations
    ks_mask_obs = (ks_arr >= 12.0) & (ks_arr <= 14.0)
    ks_arr_obs = ks_arr[ks_mask_obs]
    n_ks_obs = len(ks_arr_obs)

    # Extended true Ks range 11.8-14.2 for convolutions to capture spill-over
    ks_mask_true = (ks_arr >= 11.8) & (ks_arr <= 14.2)
    ks_arr_true = ks_arr[ks_mask_true]
    n_ks_true = len(ks_arr_true)

    obs_data_obs = obs_data[ks_mask_obs, :, :]
    sightline_mask = ~np.all(obs_data_obs == 0, axis=0)
    l_grid, b_grid = np.meshgrid(l_arr, b_arr)
    active_indices = np.where(sightline_mask.flatten())[0]
    if os.environ.get("FAST_TEST") == "1":
        active_indices = active_indices[:2]
        print(f"FAST_TEST mode active: Only using first 2 active sightlines: {active_indices}")
    l_active = l_grid.flatten()[active_indices]
    b_active = b_grid.flatten()[active_indices]
    n_active = len(l_active)
    
    obs_active = obs_data_obs.transpose(1, 2, 0).reshape(-1, n_ks_obs)[active_indices]
    err_active_true = err_data[ks_mask_true, :, :].transpose(1, 2, 0).reshape(-1, n_ks_true)[active_indices]
    
    # Clean NaNs in err_active_true
    median_err = np.nanmedian(err_active_true)
    if np.isnan(median_err):
        median_err = 0.033
    err_active_true = np.where(np.isnan(err_active_true), median_err, err_active_true)

    d_ks = header['CDELT3']
    mag_bins_true = np.concatenate(([ks_arr_true[0] - d_ks/2], ks_arr_true + d_ks/2))

    # 3. Initialize Provider on true mag bins
    provider = VVVStarCountsProvider(sp, mag_bins=mag_bins_true, band_name='2MASS_Ks')

    # Define separate high-resolution grids for disk and bulge
    distances_disk = np.linspace(0.01, 25.0, 512)
    n_dist_disk = len(distances_disk)
    d_dist_disk = distances_disk[1] - distances_disk[0]
    s_sq_disk = distances_disk**2
    dist_mod_disk = 5 * np.log10(distances_disk * 1000.0) - 5
    shift_all_disk = dist_mod_disk[None, :]

    distances_bulge = np.linspace(3.0, 13.0, 256)
    n_dist_bulge = len(distances_bulge)
    d_dist_bulge = distances_bulge[1] - distances_bulge[0]
    s_sq_bulge = distances_bulge**2
    dist_mod_bulge = 5 * np.log10(distances_bulge * 1000.0) - 5
    shift_all_bulge = dist_mod_bulge[None, :]

    solid_angle_sr = np.abs(header['CDELT1'] * header['CDELT2']) * (np.pi / 180.0)**2

    # 3b. Compute Gaussian convolution matrix C_conv of shape (n_active, n_ks_obs, n_ks_true)
    # Broadens the model LF to account for observational error in Ks
    from scipy.special import erf
    mag_bins_obs = np.concatenate(([ks_arr_obs[0] - d_ks/2], ks_arr_obs + d_ks/2))
    
    # diff arrays: (n_ks_obs, n_ks_true)
    diff_upper = mag_bins_obs[1:, None] - ks_arr_true[None, :]   # (n_ks_obs, n_ks_true)
    diff_lower = mag_bins_obs[:-1, None] - ks_arr_true[None, :]  # (n_ks_obs, n_ks_true)

    # sigma_val: (n_active, n_ks_true)
    sigma_val = err_active_true  # (n_active, n_ks_true)
    # Clip to avoid divide-by-zero
    sigma_val = np.clip(sigma_val, 1e-4, None)

    # Broadcast to (n_active, n_ks_obs, n_ks_true)
    val_upper = diff_upper[None, :, :] / (sigma_val[:, None, :] * np.sqrt(2.0))
    val_lower = diff_lower[None, :, :] / (sigma_val[:, None, :] * np.sqrt(2.0))

    C_conv = 0.5 * (erf(val_upper) - erf(val_lower))  # (n_active, n_ks_obs, n_ks_true)

    # 4. Precompute Arrays
    print("\n--- Precomputing Data with Dual Disk/Bulge Grids & Obs Errors ---")
    t0 = time.time()
    
    pred_disk_counts = np.zeros((n_active, n_ks_true), dtype=np.float32)
    bulge_S_bins = np.zeros((n_active, n_dist_bulge, n_ks_true), dtype=np.float32)
    r_sp_all = np.zeros((n_active, n_dist_bulge), dtype=np.float32)
    phi_sp_all = np.zeros((n_active, n_dist_bulge), dtype=np.float32)
    z_sp_all = np.zeros((n_active, n_dist_bulge), dtype=np.float32)

    bulge_pop = sp.populations[0]
    batch_size = 500

    for start in range(0, n_active, batch_size):
        end = min(start + batch_size, n_active)
        curr_l = l_active[start:end]
        curr_b = b_active[start:end]
        n_curr = end - start

        # A. Bulge Coordinates and Bulge Selection Fractions (evaluated on distances_bulge)
        L_mesh_b, D_mesh_b = np.meshgrid(curr_l, distances_bulge, indexing='ij')
        B_mesh_b, _ = np.meshgrid(curr_b, distances_bulge, indexing='ij')
        x_gc_b, y_gc_b, z_gc_b = provider.coord_trans.dlb_to_xyz(D_mesh_b.flatten(), L_mesh_b.flatten(), B_mesh_b.flatten())
        r_sp_b, phi_sp_b, z_sp_b = provider.coord_trans.xyz_to_rphiz(x_gc_b, y_gc_b, z_gc_b)
        
        r_sp_all[start:end] = r_sp_b.reshape(n_curr, n_dist_bulge).astype(np.float32)
        phi_sp_all[start:end] = phi_sp_b.reshape(n_curr, n_dist_bulge).astype(np.float32)
        z_sp_all[start:end] = z_sp_b.reshape(n_curr, n_dist_bulge).astype(np.float32)

        # Bulge Selection Fractions (Pre-weighted)
        S_chunk_b = provider.get_selection_fractions(bulge_pop.name, distances_bulge, curr_l, curr_b, use_number=True, shift_all=shift_all_bulge)
        S_trans_b = S_chunk_b.transpose(1, 2, 0) # (n_curr, n_dist_bulge, n_ks_true)
        S_weighted_b = S_trans_b * (solid_angle_sr * s_sq_bulge[None, :, None] * d_dist_bulge)
        bulge_S_bins[start:end] = S_weighted_b.astype(np.float32)

        # B. Disk Coordinates and Disk Counts (evaluated on distances_disk)
        L_mesh_d, D_mesh_d = np.meshgrid(curr_l, distances_disk, indexing='ij')
        B_mesh_d, _ = np.meshgrid(curr_b, distances_disk, indexing='ij')
        x_gc_d, y_gc_d, z_gc_d = provider.coord_trans.dlb_to_xyz(D_mesh_d.flatten(), L_mesh_d.flatten(), B_mesh_d.flatten())
        r_sp_d, phi_sp_d, z_sp_d = provider.coord_trans.xyz_to_rphiz(x_gc_d, y_gc_d, z_gc_d)

        sum_rb = np.zeros((n_curr, n_dist_disk, n_ks_true))
        for pop in provider.populations_disk:
            rho_raw = pop.population_density.density(r_sp_d, phi_sp_d, z_sp_d).reshape(n_curr, n_dist_disk)
            if pop.population_density.density_unit == 'mass':
                rho_raw /= provider.mass_mag_maps.get(pop.name)['mean_mass']
            S_bins = provider.get_selection_fractions(pop.name, distances_disk, curr_l, curr_b, use_number=True, shift_all=shift_all_disk)
            sum_rb += (rho_raw[None, :, :] * S_bins).transpose(1, 2, 0)
        pred_disk_counts[start:end] = (solid_angle_sr * np.sum(sum_rb * s_sq_disk[None, :, None], axis=1) * d_dist_disk).astype(np.float32)

        print(f"  Processed {end}/{n_active} sightlines...", end='\r')

    # 5. Extract Bulge Density Parameters
    dens_model = bulge_pop.population_density
    bulge_params = {
        'n0_init': dens_model.n0_scale,
        'x_init': dens_model.x_scale,
        'y_init': dens_model.y_scale,
        'z_init': dens_model.z_scale,
        'bar_ang_init_rad': dens_model.bar_ang
    }

    # 6. Save State
    cache_path = os.path.join(project_root, 'results/density/mcmc_state_cache.npz')
    np.savez(
        cache_path,
        obs_active=obs_active,
        pred_disk_counts=pred_disk_counts,
        bulge_S_bins_weighted=bulge_S_bins,
        r_sp_all=r_sp_all,
        phi_sp_all=phi_sp_all,
        z_sp_all=z_sp_all,
        distances=distances_bulge,
        ks_arr=ks_arr_obs,
        l_arr=l_arr,
        b_arr=b_arr,
        active_indices=active_indices,
        sightline_mask=sightline_mask,
        C_conv=C_conv,
        bulge_params=np.array([
            bulge_params['n0_init'], 
            bulge_params['bar_ang_init_rad'], 
            bulge_params['x_init'],
            bulge_params['y_init'],
            bulge_params['z_init']
        ]),
        global_params=np.array([R0, Z0, theta_tilt])
    )
    print(f"\nPhase 1 Complete. Saved MCMC state to {cache_path} in {time.time() - t0:.2f} s")
    return cache_path


# -----------------------------------------------------------------------------
# LIKELIHOOD FUNCTION
# -----------------------------------------------------------------------------
def _pure_density(r, phi_rad, z, n0_scale, bar_ang_rad, x_scale, y_scale, z_scale, C20_PARAMS):
    p = C20_PARAMS
    alpha_rad = p['alpha'] * np.pi / 180.0
    
    x_gc = r * np.cos(phi_rad)
    y_gc = r * np.sin(phi_rad)
    
    X_bar = x_gc * np.cos(bar_ang_rad) + y_gc * np.sin(bar_ang_rad)
    Y_bar = -x_gc * np.sin(bar_ang_rad) + y_gc * np.cos(bar_ang_rad)
    Z_bar = z
    
    X_bar_scaled = X_bar / x_scale
    Y_bar_scaled = Y_bar / y_scale
    Z_bar_scaled = Z_bar / z_scale
    
    x_fake = X_bar_scaled * np.cos(alpha_rad) - Y_bar_scaled * np.sin(alpha_rad)
    y_fake = X_bar_scaled * np.sin(alpha_rad) + Y_bar_scaled * np.cos(alpha_rad)
    z_fake = Z_bar_scaled
    
    x_coleman = x_fake + 8.0 - p['dx']
    y_coleman = y_fake - p['dy']
    z_coleman = z_fake - p['dz']

    from coleman_bulge_density import bulge_density_model
    rho = bulge_density_model.evaluate_symmetric(x=x_coleman, y=y_coleman, z=z_coleman)
    return np.array(rho) * n0_scale / (x_scale * y_scale * z_scale)


# -----------------------------------------------------------------------------
# MULTIPROCESSING WORKER GLOBALS & SAMPLER FUNCTIONS
# -----------------------------------------------------------------------------
data_dict_global = None

def init_worker(cache_path):
    global data_dict_global
    
    # Delayed import of JAX-dependent libraries inside the child process
    from fit_vvv_density import _pure_density
    from coleman_bulge_density import bulge_density_model
    
    # Load cache directly from disk once per worker process during spawn
    data = np.load(cache_path)
    
    obs_active = data['obs_active']
    pred_disk_counts = data['pred_disk_counts']
    bulge_S_bins_weighted = data['bulge_S_bins_weighted']
    r_sp_all = data['r_sp_all']
    phi_sp_all = data['phi_sp_all']
    z_sp_all = data['z_sp_all']
    distances = data['distances']
    dist_mask_bulge = (distances >= 3.0)

    n_active, n_dist = r_sp_all.shape
    
    dist_mask_bulge_2d = np.broadcast_to(dist_mask_bulge, (n_active, n_dist))
    active_mask = np.any(bulge_S_bins_weighted > 2e-10, axis=-1) & dist_mask_bulge_2d
    
    data_dict_global = {
        'C20_PARAMS': bulge_density_model.sym_params,
        'obs_active': obs_active,
        'pred_disk_counts': pred_disk_counts,
        'bulge_S_bins_weighted': bulge_S_bins_weighted,
        'C_conv': data['C_conv'],
        'active_mask': active_mask,
        'r_active': r_sp_all[active_mask],
        'phi_active': phi_sp_all[active_mask],
        'z_active': z_sp_all[active_mask],
        'n_active': n_active,
        'n_dist': n_dist,
        '_pure_density_func': _pure_density
    }

def log_probability_worker(params):
    global data_dict_global
    n0, alpha, xs, ys, zs = params
    if not (1e-3 <= n0 <= 10.0 and 0.0 <= alpha <= 90.0 and 0.1 <= xs <= 3.0 and 0.1 <= ys <= 3.0 and 0.1 <= zs <= 3.0):
        return -np.inf

    bar_ang_rad = alpha * np.pi / 180.0
    
    rho_active = data_dict_global['_pure_density_func'](
        data_dict_global['r_active'], data_dict_global['phi_active'], data_dict_global['z_active'], 
        float(n0), float(bar_ang_rad), float(xs), float(ys), float(zs), data_dict_global['C20_PARAMS']
    )
    
    rho_raw = np.zeros((data_dict_global['n_active'], data_dict_global['n_dist']))
    rho_raw[data_dict_global['active_mask']] = rho_active
    
    pred_bulge_true = np.einsum('ij,ijk->ik', rho_raw, data_dict_global['bulge_S_bins_weighted'])
    pred_true = data_dict_global['pred_disk_counts'] + pred_bulge_true
    
    # Broaden the true magnitudes to account for observational error
    pred = np.einsum('ijk,ik->ij', data_dict_global['C_conv'], pred_true)

    eps = 1e-10
    return np.sum(data_dict_global['obs_active'] * np.log(pred + eps) - pred)


# -----------------------------------------------------------------------------
# PHASE 1.5: L-BFGS OPTIMIZATION
# -----------------------------------------------------------------------------
def run_lbfgs_phase(cache_path, init_angle=None):
    print("\n====================================================================")
    print("--- PHASE 1.5: Running L-BFGS Optimization ---")
    print("====================================================================")
    
    t0 = time.time()
    init_worker(cache_path) # Initialize global data in main process
    
    data = np.load(cache_path)
    n0_init, bar_ang_init_rad, x_init, y_init, z_init = data['bulge_params']
    R0, Z0, theta_tilt = data['global_params']
    active_indices = data['active_indices']
    
    start_angle = init_angle if init_angle is not None else np.degrees(bar_ang_init_rad)
    p0 = np.array([n0_init, start_angle, x_init, y_init, z_init])
    p_names = ['n0_scale', 'bar_angle', 'x_scale', 'y_scale', 'z_scale']
    bounds = [(1e-3, 10.0), (0.0, 90.0), (0.1, 3.0), (0.1, 3.0), (0.1, 3.0)]
    
    def nll(params):
        lp = log_probability_worker(params)
        if not np.isfinite(lp):
            return 1e10
        return -lp
        
    print(f"Starting L-BFGS optimization from p0: {p0}")
    res = minimize(nll, p0, method='L-BFGS-B', bounds=bounds, options={'disp': True, 'maxiter': 200})
    
    print(f"\nL-BFGS Optimization Complete in {time.time() - t0:.2f} s")
    print(f"Success: {res.success}")
    print(f"Message: {res.message}")
    print(f"Final NLL: {res.fun:.2f}")
    print("Best Fit Parameters:")
    for i, name in enumerate(p_names):
        print(f"  {name:<15} : {res.x[i]:.5f}")
        
    json_path = os.path.join(project_root, 'results/density/vvv_fit_results.json')
    
    results_dict = {
        'model_name': "Huston2025_C20Bulge",
        'R0_kpc': float(R0),
        'Z0_kpc': float(Z0),
        'theta_tilt_deg': float(np.degrees(theta_tilt)),
        'n_active_sightlines': int(len(active_indices)),
        'lbfgs_success': bool(res.success),
        'lbfgs_nll': float(res.fun),
        'best_fit_parameters': {
            p_names[i]: {'val': float(res.x[i])} for i in range(len(p_names))
        }
    }
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=4)
    print(f"L-BFGS results saved to {json_path}")
    return res.x


# -----------------------------------------------------------------------------
# PHASE 2: MCMC SAMPLING
# -----------------------------------------------------------------------------
def run_mcmc_phase(cache_path):
    print("\n====================================================================")
    print("--- PHASE 2: Running MCMC ---")
    print("====================================================================")
    
    t0 = time.time()
    
    # 1. Load Precomputed metadata required in parent process
    print("Loading cached metadata...")
    data = np.load(cache_path)
    
    obs_active = data['obs_active']
    pred_disk_counts = data['pred_disk_counts']
    bulge_S_bins_weighted = data['bulge_S_bins_weighted']
    r_sp_all = data['r_sp_all']
    phi_sp_all = data['phi_sp_all']
    z_sp_all = data['z_sp_all']
    distances = data['distances']
    dist_mask_bulge = (distances >= 3.0)
    
    ks_arr = data['ks_arr']
    l_arr = data['l_arr']
    b_arr = data['b_arr']
    active_indices = data['active_indices']
    sightline_mask = data['sightline_mask']
    
    n0_init, bar_ang_init_rad, x_init, y_init, z_init = data['bulge_params']
    R0, Z0, theta_tilt = data['global_params']

    # Starting parameters for MCMC
    p_names = ['n0_scale', 'bar_angle', 'x_scale', 'y_scale', 'z_scale']
    p0 = np.array([1.4, 18.0, 1.1, 1.1, 1.1])  # default fallback
    
    # Load from L-BFGS results if available
    lbfgs_json = os.path.join(project_root, 'results/density/vvv_fit_results.json')
    if os.path.exists(lbfgs_json):
        try:
            with open(lbfgs_json, 'r') as f:
                res = json.load(f)
            if 'best_fit_parameters' in res and 'n0_scale' in res['best_fit_parameters']:
                p_lbfgs = res['best_fit_parameters']
                p0 = np.array([
                    p_lbfgs['n0_scale']['val'],
                    p_lbfgs['bar_angle']['val'],
                    p_lbfgs.get('x_scale', {'val': x_init})['val'],
                    p_lbfgs.get('y_scale', {'val': y_init})['val'],
                    p_lbfgs.get('z_scale', {'val': z_init})['val']
                ])
                print(f"Loaded L-BFGS best-fit parameters for MCMC initialization: {p0}")
        except Exception as e:
            print(f"Could not load L-BFGS fit results: {e}")

    is_fast = os.environ.get("FAST_TEST") == "1"
    n_walkers = 8 if is_fast else 16
    n_params = len(p0)
    
    pos = p0 + 1e-4 * np.random.randn(n_walkers, n_params)
    
    # Run parallel EnsembleSampler across workers
    from multiprocessing import Pool
    n_workers = 2 if is_fast else 6
    print(f"Starting parallel MCMC with {n_walkers} walkers using {n_workers} workers...")
    
    with Pool(processes=n_workers, initializer=init_worker, initargs=(cache_path,)) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, n_params, log_probability_worker, pool=pool)
        
        burn_in_steps = 5 if is_fast else 500
        print(f"Running burn-in ({burn_in_steps} steps)...")
        state = sampler.run_mcmc(pos, burn_in_steps, progress=True)
        sampler.reset()
        
        prod_steps = 5 if is_fast else 1000
        print(f"Running production ({prod_steps} steps)...")
        sampler.run_mcmc(state, prod_steps, progress=True)
        
    print(f"\nMCMC sampling complete in {time.time() - t0:.2f} seconds.")
    
    # Save Results
    samples = sampler.get_chain(flat=True)
    log_probs = sampler.get_log_prob(flat=True)
    percentiles = np.percentile(samples, [16, 50, 84], axis=0)
    x_opt = percentiles[1]
    
    mcmc_samples_path = os.path.join(project_root, 'results/density/vvv_mcmc_samples.npz')
    np.savez(mcmc_samples_path, samples=samples, deviance=-2*log_probs)
    
    print("Generating corner plot...")
    plots_dir = os.path.join(project_root, 'results/density')
    os.makedirs(plots_dir, exist_ok=True)
    fig = corner.corner(samples, labels=p_names, truths=x_opt, quantiles=[0.16, 0.5, 0.84], show_titles=True)
    fig.savefig(os.path.join(plots_dir, 'corner_plot.png'))
    plt.close()

    print("\n====================================================================")
    print("--- MCMC Parameter Results ---")
    for i in range(n_params):
        mcmc = percentiles[:, i]
        q = np.diff(mcmc)
        print(f"  {p_names[i]:<15} : {mcmc[1]:.5f} (+{q[1]:.5f}, -{q[0]:.5f})")
    print("====================================================================")
    
    # Update JSON
    json_path = os.path.join(project_root, 'results/density/vvv_fit_results.json')
    results_dict = {
        'model_name': "Huston2025_C20Bulge",
        'R0_kpc': float(R0),
        'Z0_kpc': float(Z0),
        'theta_tilt_deg': float(np.degrees(theta_tilt)),
        'n_active_sightlines': int(len(active_indices)),
        'mcmc_success': True,
        'best_fit_parameters': {
            p_names[i]: {
                'median': float(percentiles[1, i]),
                'p16': float(percentiles[0, i]),
                'p84': float(percentiles[2, i])
            } for i in range(n_params)
        }
    }
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=4)
    print(f"JSON fit summary saved to {json_path}")
    
    # Regenerate Plots (safely import JAX here after worker processes have exited!)
    print("\nGenerating best-fit plots and residuals...")
    from coleman_bulge_density import bulge_density_model
    
    n_active, n_dist = r_sp_all.shape
    dist_mask_bulge_2d = np.broadcast_to(dist_mask_bulge, (n_active, n_dist))
    active_mask = np.any(bulge_S_bins_weighted > 2e-10, axis=-1) & dist_mask_bulge_2d
    
    C_conv = data['C_conv']

    def get_pred_counts(params):
        n0, alpha, xs, ys, zs = params
        bar_ang_rad = alpha * np.pi / 180.0
        
        rho_active = _pure_density(r_sp_all[active_mask], phi_sp_all[active_mask], z_sp_all[active_mask], 
                                   float(n0), float(bar_ang_rad), float(xs), float(ys), float(zs), bulge_density_model.sym_params)
        rho_raw = np.zeros((n_active, n_dist))
        rho_raw[active_mask] = rho_active
        pred_bulge_true = np.einsum('ij,ijk->ik', rho_raw, bulge_S_bins_weighted)
        pred_true = pred_disk_counts + pred_bulge_true
        return np.einsum('ijk,ik->ij', C_conv, pred_true)
        
    best_pred = get_pred_counts(x_opt)
    
    obs_map = np.zeros((75, 100))
    pred_map = np.zeros((75, 100))
    obs_map.flat[active_indices] = np.sum(obs_active, axis=1)
    pred_map.flat[active_indices] = np.sum(best_pred, axis=1)
    
    residuals = obs_map - pred_map
    obs_map[~sightline_mask] = np.nan
    pred_map[~sightline_mask] = np.nan
    residuals[~sightline_mask] = np.nan

    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(obs_map, origin='lower', extent=[l_arr.max(), l_arr.min(), b_arr.min(), b_arr.max()], cmap='viridis', aspect='equal')
    plt.colorbar(label='Total Observed Star Counts')
    plt.xlabel('Galactic Longitude l (deg)')
    plt.ylabel('Galactic Latitude b (deg)')
    plt.title('VVV Star Counts (Observed)')
    
    plt.subplot(1, 3, 2)
    plt.imshow(pred_map, origin='lower', extent=[l_arr.max(), l_arr.min(), b_arr.min(), b_arr.max()], cmap='viridis', aspect='equal')
    plt.colorbar(label='Total Predicted Star Counts')
    plt.xlabel('Galactic Longitude l (deg)')
    plt.title('Best-Fit Model (Predicted)')
    
    plt.subplot(1, 3, 3)
    plt.imshow(residuals, origin='lower', extent=[l_arr.max(), l_arr.min(), b_arr.min(), b_arr.max()], cmap='coolwarm', aspect='equal')
    plt.colorbar(label='Residuals (Observed - Predicted)')
    plt.xlabel('Galactic Longitude l (deg)')
    plt.title('Residuals Map')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'best_fit_residuals_2d.png'), dpi=150)
    plt.close()
    print("MCMC Phase Complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lbfgs', action='store_true', help='Run L-BFGS phase from cache')
    parser.add_argument('--mcmc', action='store_true', help='Run MCMC phase from cache')
    parser.add_argument('--init_angle', type=float, default=None, help='Initial bar angle for L-BFGS (degrees)')
    args = parser.parse_args()

    cache_path = os.path.join(project_root, 'results/density/mcmc_state_cache.npz')

    if not args.lbfgs and not args.mcmc:
        cache_file = run_precomputation()
        print(f"\nTo run L-BFGS, execute:\n  python3 fit_vvv_density.py --lbfgs")
        print(f"To run MCMC, execute:\n  python3 fit_vvv_density.py --mcmc")
        
    if args.lbfgs:
        run_lbfgs_phase(cache_path, init_angle=args.init_angle)
        
    if args.mcmc:
        run_mcmc_phase(cache_path)
