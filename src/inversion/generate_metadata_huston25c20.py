import jax
import jax.numpy as jnp
import numpy as np
import sys
import os
import json
import pickle
import pandas as pd

# Append paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../shared')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../synthpop')))
import synthpop

from synthpop_moment_provider import SynthPopDiskMomentProvider
from jax_kinematic_inverter import BulgeKinematicInverter
from inversion_config import get_distance_grid, BATCH_SIZE, get_inverter_grid_axes, get_base_synthpop_config

def generate_metadata_batched(model_name, l_obs, b_obs, mag_bins, output_cache, band_name='2MASS_Ks'):
    n_bins = len(mag_bins) - 1
    n_pix = len(l_obs)
    distances = get_distance_grid()
    n_dist = len(distances)

    print(f"--- Generating Metadata Cache for {model_name} ({n_pix} pixels) ---")
    
    # 1. Setup SynthPop
    config = get_base_synthpop_config(model_name, "meta_gen", solid_angle=0.01)
        
    tmp_conf = f"tmp_meta_{model_name}_{n_bins}.json"
    with open(tmp_conf, "w") as f: json.dump(config, f)
    sp = synthpop.SynthPop(tmp_conf)
    sp.init_populations()
    os.remove(tmp_conf)

    bulge_pop = sp.populations[0]
    xyz_scale = getattr(bulge_pop.population_density, 'xyz_scale', 1.0)
    print(f"  Dynamically extracted xyz_scale: {xyz_scale}")
    
    # FIX: Koshimoto2021Bulge unit bug
    if hasattr(bulge_pop.kinematics, "y0_stream"):
        if bulge_pop.kinematics.y0_stream > 10.0:
            print(f"  Fixing y0_stream units: {bulge_pop.kinematics.y0_stream} pc -> {bulge_pop.kinematics.y0_stream/1000.0} kpc")
            bulge_pop.kinematics.y0_stream /= 1000.0

    # MONKEYPATCH: Fix bar angle inconsistency
    density_alpha = bulge_pop.population_density.bar_ang
    if hasattr(bulge_pop.kinematics, "bar_ang"):
        if not np.isclose(bulge_pop.kinematics.bar_ang, density_alpha):
            bulge_pop.kinematics.bar_ang = density_alpha

    # 2. Pre-allocate Arrays
    print("  Pre-allocating arrays...")
    x_bar_all = np.zeros((n_pix, n_dist))
    y_bar_all = np.zeros((n_pix, n_dist))
    z_bar_all = np.zeros((n_pix, n_dist))
    rho_b_all = np.zeros((n_pix, n_dist, n_bins))
    rho_d_all = np.zeros((n_pix, n_dist, n_bins))
    mu_d_pm_all = np.zeros((n_pix, n_dist, n_bins, 2))
    cov_d_pm_all = np.zeros((n_pix, n_dist, n_bins, 2, 2))
    mu_d_rv_all = np.zeros((n_pix, n_dist, n_bins))
    var_d_rv_all = np.zeros((n_pix, n_dist, n_bins))

    # 3. Setup Moment Provider and Inverter
    provider = SynthPopDiskMomentProvider(sp, mag_bins=mag_bins, band_name=band_name)
    v_lsr = sp.parms.lsr['v_lsr'] if isinstance(sp.parms.lsr, dict) else sp.parms.lsr.v_lsr
    solar = {
        'R0': np.abs(sp.parms.sun.gal_dist),
        'U_SUN': sp.parms.sun.u,
        'V_LSR': v_lsr,
        'V_SUN_REL': sp.parms.sun.v - v_lsr,
        'W_SUN': sp.parms.sun.w,
        'bar_angle_rad': density_alpha,
        'xyz_scale': xyz_scale
    }
    grid_axes = get_inverter_grid_axes()
    inverter = BulgeKinematicInverter(grid_axes, solar, n_bins=n_bins)

    # 4. Process in Large Batches
    batch_size = BATCH_SIZE
    for start in range(0, n_pix, batch_size):
        end = min(start + batch_size, n_pix)
        curr_l, curr_b = l_obs[start:end], b_obs[start:end]
        n_curr = end - start
        
        print(f"  Processing Batch {start}-{end}...")
        
        # A. Coordinates (Vectorized)
        pos_bar = jax.vmap(jax.vmap(lambda l, b, d: inverter.galactic_to_bar(l, b, d), in_axes=(None, None, 0)), in_axes=(0, 0, None))(curr_l, curr_b, distances)
        x_bar_all[start:end] = pos_bar[:, :, 0]
        y_bar_all[start:end] = pos_bar[:, :, 1]
        z_bar_all[start:end] = pos_bar[:, :, 2]
        
        # B. Combined Disk Moments (Vectorized)
        m_pm, c_pm, m_rv, v_rv, r_d = provider.get_combined_disk_moments_binned(curr_l, curr_b, distances, use_2d_extinction=True)
        mu_d_pm_all[start:end] = m_pm
        cov_d_pm_all[start:end] = c_pm
        mu_d_rv_all[start:end] = m_rv
        var_d_rv_all[start:end] = v_rv
        rho_d_all[start:end] = r_d
        
        # C. Bulge Density (Vectorized)
        L_grid, D_grid = np.meshgrid(curr_l, distances, indexing='ij')
        B_grid, _ = np.meshgrid(curr_b, distances, indexing='ij')
        x_gc, y_gc, z_gc = provider.coord_trans.dlb_to_xyz(D_grid.flatten(), L_grid.flatten(), B_grid.flatten())
        r_sp, phi_sp, z_sp = provider.coord_trans.xyz_to_rphiz(x_gc, y_gc, z_gc)
        
        rb_raw = bulge_pop.population_density.density(r_sp, phi_sp, z_sp).reshape(n_curr, n_dist)
        
        if bulge_pop.population_density.density_unit == 'number':
             norm = provider.mass_mag_maps[bulge_pop.name]['mean_mass']
             rb_raw *= norm
        
        S_bins = provider.get_selection_fractions(bulge_pop.name, distances, curr_l, curr_b, use_number=False, use_2d_extinction=True)
        rho_b_all[start:end] = np.transpose(rb_raw[None, :, :] * S_bins, (1, 2, 0))

    # 5. Final Batch Formatting
    print("  Finalizing metadata cache...")
    meta_batch = {
        'l': jnp.array(l_obs), 'b': jnp.array(b_obs), 'd': jnp.array(distances),
        'x_bar': jnp.array(x_bar_all), 'y_bar': jnp.array(y_bar_all), 'z_bar': jnp.array(z_bar_all),
        'rho_b': jnp.array(rho_b_all), 'rho_d': jnp.array(rho_d_all),
        'mu_d_pm': jnp.array(mu_d_pm_all), 'cov_d_pm': jnp.array(cov_d_pm_all),
        'mu_d_rv': jnp.array(mu_d_rv_all), 'var_d_rv': jnp.array(var_d_rv_all)
    }
    
    with open(output_cache, 'wb') as f:
        pickle.dump(meta_batch, f)
    print(f"--- Metadata Cache Saved to {output_cache} ---")

def parse_gibs_data(gibs_path):
    """
    Parses GIBS_full.txt to extract coordinates and kinematic moments.
    """
    lines = []
    with open(gibs_path, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#') and '|' in line:
                parts = line.strip().split('|')
                if len(parts) >= 4:
                    try:
                        field_parts = parts[0].strip().split()
                        if not field_parts:
                            continue
                        field = field_parts[0]
                        coords = parts[2].strip().split()
                        glon = float(coords[0])
                        if glon > 180:
                            glon -= 360
                        glat = float(coords[1])
                        rv = float(parts[3].strip())
                        lines.append({
                            'field': field,
                            'l': glon,
                            'b': glat,
                            'rv': rv
                        })
                    except ValueError:
                        continue

    df = pd.DataFrame(lines)
    grouped = df.groupby('field').agg(
        l_center=('l', 'mean'),
        b_center=('b', 'mean'),
        rv_mean=('rv', 'mean'),
        rv_disp=('rv', lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0),
        count=('rv', 'count')
    ).reset_index()
    return grouped

def main():
    model_name = "Huston2025_C20Bulge"
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    is_fast = os.environ.get("FAST_TEST") == "1"
    
    # --- 1. VIRAC2 ---
    mock_file = os.path.join(project_root, 'data/huston2025_theoretical_mock_binned.npz')
    if os.path.exists(mock_file):
        mock_data = np.load(mock_file)
        l_arr = mock_data['l']
        b_arr = mock_data['b']
        if is_fast:
            l_arr = l_arr[:5]
            b_arr = b_arr[:5]
            print(f"FAST_TEST mode active: Only generating 5 metadata pixels for VIRAC2")
        generate_metadata_batched(model_name, l_arr, b_arr, mock_data['mag_bins'], os.path.join(project_root, 'results/inversion/huston25c20_metadata_science_binned.pkl'))

    # --- 2. BRAVA ---
    brava_csv = os.path.join(project_root, 'data/brava_paper_results.csv')
    if os.path.exists(brava_csv):
        df_b = pd.read_csv(brava_csv)
        l_arr = df_b['l'].values
        b_arr = df_b['b'].values
        if is_fast:
            l_arr = l_arr[:5]
            b_arr = b_arr[:5]
            print(f"FAST_TEST mode active: Only generating 5 metadata pixels for BRAVA")
        generate_metadata_batched(model_name, l_arr, b_arr, np.array([8.2, 9.25]), os.path.join(project_root, 'results/inversion/huston25c20_brava_real_metadata.pkl'))

    # --- 3. GIBS ---
    gibs_txt = os.path.join(project_root, 'data/GIBS_full.txt')
    if os.path.exists(gibs_txt):
        gibs_stats = parse_gibs_data(gibs_txt)
        l_arr = gibs_stats['l_center'].values
        b_arr = gibs_stats['b_center'].values
        if is_fast:
            l_arr = l_arr[:5]
            b_arr = b_arr[:5]
            print(f"FAST_TEST mode active: Only generating 5 metadata pixels for GIBS")
        generate_metadata_batched(model_name, l_arr, b_arr, np.array([10.0, 16.0]), os.path.join(project_root, 'results/inversion/huston25c20_gibs_metadata_science.pkl'), band_name='2MASS_J')

if __name__ == "__main__":
    main()
