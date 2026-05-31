"""
Distance posterior P(D | mu) for Roman GBTDS fields using the fitted JAX model.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import json
import jax

# Set up project root imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(project_root, "src/inversion"))
sys.path.append(os.path.join(project_root, "src/shared"))
sys.path.append(os.path.join(project_root, "synthpop"))

from jax_kinematic_inverter import BulgeKinematicInverter
from inversion_config import get_inverter_grid_axes, get_distance_grid, get_base_synthpop_config
import synthpop
from synthpop_moment_provider import SynthPopDiskMomentProvider

def main():
    # 1. Load best-fit results
    res_path = os.path.join(project_root, 'results/inversion/fit_results_obs_combined_h25c20.npz')
    if not os.path.exists(res_path):
        print(f"Error: Fit results not found at {res_path}")
        return
    
    data = np.load(res_path, allow_pickle=True)
    grid_params = jnp.array(data['grid'])
    omega = float(data['omega'])
    
    # 2. Setup Roman Field Metadata
    l_roman, b_roman = 0.5, -1.4
    distances = get_distance_grid()
    mag_bins = np.array([10.0, 16.0]) # Science-like bin
    n_bins = len(mag_bins) - 1
    
    print(f"--- Generating Metadata for Roman Field (l={l_roman}, b={b_roman}) ---")
    model_name = "Huston2025_C20Bulge"
    config = get_base_synthpop_config(model_name, "roman_test", l_set=[l_roman], b_set=[b_roman], solid_angle=0.01)
    
    # Temporary config for synthpop
    tmp_conf = "tmp_roman.json"
    with open(tmp_conf, "w") as f: json.dump(config, f)
    sp = synthpop.SynthPop(tmp_conf)
    sp.init_populations()
    os.remove(tmp_conf)
    
    bulge_pop = sp.populations[0]
    provider = SynthPopDiskMomentProvider(sp, mag_bins=mag_bins)
    v_lsr = sp.parms.lsr['v_lsr'] if isinstance(sp.parms.lsr, dict) else sp.parms.lsr.v_lsr
    solar = {
        'R0': np.abs(sp.parms.sun.gal_dist),
        'U_SUN': sp.parms.sun.u,
        'V_LSR': v_lsr,
        'V_SUN_REL': sp.parms.sun.v - v_lsr,
        'W_SUN': sp.parms.sun.w,
        'bar_angle_rad': bulge_pop.population_density.bar_ang,
    }
    
    inverter = BulgeKinematicInverter(get_inverter_grid_axes(), solar, n_bins=n_bins)
    
    # Generate the LOS metadata
    pos_bar = inverter.galactic_to_bar(l_roman, b_roman, distances)
    m_pm, c_pm, m_rv, v_rv, r_d = provider.get_combined_disk_moments_binned(np.array([l_roman]), np.array([b_roman]), distances, use_2d_extinction=True)
    
    # Bulge density
    l_rad, b_rad = np.radians(l_roman), np.radians(b_roman)
    xh = distances * np.cos(b_rad) * np.cos(l_rad)
    yh = distances * np.cos(b_rad) * np.sin(l_rad)
    zh = distances * np.sin(b_rad)
    
    # H_pos for coordinate shift
    R0 = solar['R0']
    Z0 = 0.0
    theta = np.arcsin(Z0 / np.sqrt(R0**2 + Z0**2))
    ct, st = np.cos(theta), np.sin(theta)
    H_pos = np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]])
    
    xyz_gc = H_pos @ np.stack([xh - R0, yh, zh])
    r_sp, phi_sp, z_sp = provider.coord_trans.xyz_to_rphiz(xyz_gc[0], xyz_gc[1], xyz_gc[2])
    rb_raw = bulge_pop.population_density.density(r_sp, phi_sp, z_sp)
    if bulge_pop.population_density.density_unit == 'number':
        rb_raw *= provider.mass_mag_maps[bulge_pop.name]['mean_mass']
    
    S_bins = provider.get_selection_fractions(bulge_pop.name, distances, np.array([l_roman]), np.array([b_roman]), use_number=False, use_2d_extinction=True)
    rho_b = np.transpose(rb_raw[None, :] * S_bins, (1, 0)) # (n_dist, n_bins)
    
    pixel_meta = {
        'l': l_roman, 'b': b_roman, 'd': jnp.array(distances),
        'x_bar': jnp.array(pos_bar[0]), 'y_bar': jnp.array(pos_bar[1]), 'z_bar': jnp.array(pos_bar[2]),
        'rho_b': jnp.array(rho_b), 'rho_d': jnp.array(r_d[0]),
        'mu_d_pm': jnp.array(m_pm[0]), 'cov_d_pm': jnp.array(c_pm[0])
    }
    
    # 3. Compute Posteriors for several mu
    obs_mus = [
        jnp.array([-5.0, -0.2]), # Bar-like
        jnp.array([-1.0, 0.0]),  # Disk-like
        jnp.array([0.0, 0.0]),   # Zero
        jnp.array([-8.0, -1.0])  # Fast bar
    ]
    
    plt.figure(figsize=(10, 6))
    for mu in obs_mus:
        post = inverter.compute_distance_posterior(grid_params, omega, pixel_meta, mu)
        plt.plot(distances, post, label=f"$\mu = ({mu[0]}, {mu[1]})$ mas/yr")
    
    # Also plot the Prior P(D)
    wb = rho_b[:, 0] * distances**2
    wd = r_d[0, :, 0] * distances**2
    prior = (wb + wd)
    prior /= prior.sum()
    plt.plot(distances, prior, 'k--', alpha=0.5, label='Prior $P(D)$ (normalized)')
    
    plt.axvline(8.15, color='gray', linestyle=':', label='$R_0 = 8.15$ kpc')
    plt.xlabel("Distance [kpc]")
    plt.ylabel("$P(D | \mu)$ [normalized]")
    plt.title(f"Roman Distance Posterior at $(l, b) = ({l_roman}^\circ, {b_roman}^\circ)$")
    plt.legend()
    plt.xlim(4, 12)
    plt.grid(alpha=0.3)
    
    out_plot = os.path.join(project_root, 'results/inversion/roman_distance_posterior.png')
    plt.savefig(out_plot, dpi=300)
    print(f"Plot saved to {out_plot}")

if __name__ == "__main__":
    main()
