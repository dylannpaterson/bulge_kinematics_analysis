import numpy as np
import os
import sys
import json
import gc
import agama

# Set Agama units (kpc, Msun, km/s)
agama.setUnits(length=1, mass=1, velocity=1) 
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# Append paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(project_root, 'src/shared'))
sys.path.append(os.path.abspath(os.path.join(project_root, '../synthpop')))

import synthpop
from inversion_config import get_inverter_grid_axes

# ─────────────────────────────────────────────────────────
# 1. Load SynthPop density & build potential
# ─────────────────────────────────────────────────────────

def get_synthpop_model(model_name="Besancon_Coleman2020_symmetric"):
    config = {
        "MANDATORY": {"name_for_output": "m2m_tmp", "model_name": model_name},
        "SIGHTLINES": {"l_set": [0.0], "l_set_type": "list", "b_set": [0.0], "b_set_type": "list", 
                       "solid_angle": 0.01, "solid_angle_unit": "deg^2"}
    }
    tmp_conf = f"tmp_m2m_pot_{model_name}.json"
    with open(tmp_conf, "w") as f:
        json.dump(config, f)
    
    sp_instance = synthpop.SynthPop(tmp_conf)
    sp_instance.init_populations()
    os.remove(tmp_conf)
    
    bar_angle_rad = 0
    for pop in sp_instance.populations:
        if hasattr(pop.population_density, 'bar_ang'):
            bar_angle_rad = pop.population_density.bar_ang
            break

    def total_density(pos):
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        r = np.sqrt(x**2 + y**2)
        phi_bar = np.arctan2(y, x)
        phi_sp = phi_bar + bar_angle_rad
        
        rho_total = np.zeros(len(pos))
        for pop in sp_instance.populations:
            dens = pop.population_density.density(r, phi_sp, z)
            if pop.population_density.density_unit == 'number':
                 avg_m = pop.imf.average_mass(min_mass=pop.min_mass, max_mass=pop.max_mass)
                 corr = getattr(pop.pop_params, 'av_mass_corr', 1.0)
                 if corr is None: corr = 1.0
                 dens *= (avg_m * corr)
            rho_total += dens
        return rho_total

    def bulge_only_density(pos):
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        r = np.sqrt(x**2 + y**2)
        phi_bar = np.arctan2(y, x)
        phi_sp = phi_bar + bar_angle_rad
        
        rho_bulge = np.zeros(len(pos))
        for pop in sp_instance.populations:
            if 'bulge' in pop.name.lower():
                dens = pop.population_density.density(r, phi_sp, z)
                if pop.population_density.density_unit == 'number':
                     avg_m = pop.imf.average_mass(min_mass=pop.min_mass, max_mass=pop.max_mass)
                     corr = getattr(pop.pop_params, 'av_mass_corr', 1.0)
                     if corr is None: corr = 1.0
                     dens *= (avg_m * corr)
                rho_bulge += dens
        return rho_bulge

    return total_density, bulge_only_density

# ─────────────────────────────────────────────────────────
# 2. Sample Orbit Initial Conditions
# ─────────────────────────────────────────────────────────

def sample_initial_conditions(n_stars, potential, sampling_dens_func):
    """
    Samples particles in the triaxial potential from the bulge density.
    """
    print(f"Sampling {n_stars:,} particles...")
    ics = np.zeros((n_stars, 6), dtype=np.float32)
    n_filled = 0
    DENS_PEAK = 1e9
    VR_DISP, VT_FRAC, VT_DISP, VZ_DISP = 160.0, 0.4, 160.0, 100.0
    
    while n_filled < n_stars:
        n_batch = min((n_stars - n_filled) * 15, 1000000)
        xs = np.random.uniform(-4.5, 4.5, n_batch)
        ys = np.random.uniform(-4.5, 4.5, n_batch)
        zs = np.random.uniform(-1.8, 1.8, n_batch)
        
        pos = np.column_stack([xs, ys, zs])
        dens_vals = sampling_dens_func(pos)
        
        keep = np.random.uniform(0, DENS_PEAK, n_batch) < dens_vals
        xf, yf, zf = xs[keep], ys[keep], zs[keep]
        if len(xf) == 0: continue
        
        rf = np.maximum(np.sqrt(xf**2 + yf**2), 0.01)
        forces = potential.force(np.column_stack([xf, yf, zf]))
        vc = np.sqrt(rf * np.abs((forces[:, 0] * xf + forces[:, 1] * yf) / rf))
        v_esc = np.sqrt(np.maximum(-2.0 * potential.potential(np.column_stack([xf, yf, zf])), 0.0))
        
        vr = np.random.normal(0, VR_DISP, len(xf))
        vt = np.random.normal(VT_FRAC * vc, VT_DISP, len(xf))
        vz = np.random.normal(0, VZ_DISP, len(xf))
        
        bound = np.sqrt(vr**2 + vt**2 + vz**2) < (v_esc * 0.95)
        n_to_copy = min(np.sum(bound), n_stars - n_filled)
        if n_to_copy == 0: continue
        
        cos_p, sin_p = xf[bound][:n_to_copy] / rf[bound][:n_to_copy], yf[bound][:n_to_copy] / rf[bound][:n_to_copy]
        vr_b, vt_b, vz_b = vr[bound][:n_to_copy], vt[bound][:n_to_copy], vz[bound][:n_to_copy]
        
        ics[n_filled:n_filled+n_to_copy] = np.column_stack([
            xf[bound][:n_to_copy], yf[bound][:n_to_copy], zf[bound][:n_to_copy], 
            vr_b*cos_p - vt_b*sin_p, vr_b*sin_p + vt_b*cos_p, vz_b
        ])
        n_filled += n_to_copy
        
    return ics

# ─────────────────────────────────────────────────────────
# 3. Trilinear Interpolation of NP Fit Kinematics
# ─────────────────────────────────────────────────────────

def get_np_target_kinematics(grid_params, axes_x, axes_y, axes_z, xc, yc, zc):
    """
    Interpolates mean velocity and dispersions in the 1st octant.
    """
    NX, NY, NZ = grid_params.shape[:3]
    
    # Enforce bounds
    ix = np.clip(np.searchsorted(axes_x, xc) - 1, 0, NX - 2)
    iy = np.clip(np.searchsorted(axes_y, yc) - 1, 0, NY - 2)
    iz = np.clip(np.searchsorted(axes_z, zc) - 1, 0, NZ - 2)
    
    tx = (xc - axes_x[ix]) / (axes_x[ix+1] - axes_x[ix] + 1e-10)
    ty = (yc - axes_y[iy]) / (axes_y[iy+1] - axes_y[iy] + 1e-10)
    tz = (zc - axes_z[iz]) / (axes_z[iz+1] - axes_z[iz] + 1e-10)
    
    # Trilinear interpolation
    interp = np.zeros(6)
    for p in range(6):
        val_000 = grid_params[ix,   iy,   iz,   p]
        val_100 = grid_params[ix+1, iy,   iz,   p]
        val_010 = grid_params[ix,   iy+1, iz,   p]
        val_110 = grid_params[ix+1, iy+1, iz,   p]
        val_001 = grid_params[ix,   iy,   iz+1, p]
        val_101 = grid_params[ix+1, iy,   iz+1, p]
        val_011 = grid_params[ix,   iy+1, iz+1, p]
        val_111 = grid_params[ix+1, iy+1, iz+1, p]
        
        interp[p] = (
            (1-tx)*(1-ty)*(1-tz)*val_000 +
            tx*(1-ty)*(1-tz)*val_100 +
            (1-tx)*ty*(1-tz)*val_010 +
            tx*ty*(1-tz)*val_110 +
            (1-tx)*(1-ty)*tz*val_001 +
            tx*(1-ty)*tz*val_101 +
            (1-tx)*ty*tz*val_011 +
            tx*ty*tz*val_111
        )
        
    ux, uy, log_L11, L21, log_L22, log_L33 = interp
    L11, L22, L33 = np.exp(log_L11), np.exp(log_L22), np.exp(log_L33)
    
    sxx = L11**2
    syy = L21**2 + L22**2
    szz = L33**2
    
    return ux, uy, sxx, syy, szz

# ─────────────────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────────────────

def main():
    fit_path = os.path.join(project_root, "results/inversion/fit_results_real_combined_h25c20.npz")
    if not os.path.exists(fit_path):
        print(f"Error: {fit_path} not found.")
        return
        
    print("Loading non-parametric fit results...")
    fit_res = np.load(fit_path)
    grid_params = fit_res['grid']
    omega = float(fit_res['omega'])
    axes_x = fit_res['axes_x']
    axes_y = fit_res['axes_y']
    axes_z = fit_res['axes_z']
    
    # 1. Build Triaxial Multipole Potential in Agama
    print("Loading SynthPop densities & building potential...")
    total_density, bulge_density = get_synthpop_model("Besancon_Coleman2020_symmetric")
    
    print("Generating Agama triaxial multipole potential...")
    full_pot = agama.Potential(type='Multipole', density=total_density, lmax=16, symmetry='triaxial', 
                               gridsizeR=100, rmin=0.01, rmax=50.0)
                               
    # 2. Setup Coarse Target Grid (1st Octant only)
    # Define coarser 3D bin axes for stable statistics
    nx_bins, ny_bins, nz_bins = 12, 12, 6
    x_bins = np.linspace(0.0, 4.5, nx_bins + 1)
    y_bins = np.linspace(0.0, 4.5, ny_bins + 1)
    z_bins = np.linspace(0.0, 1.8, nz_bins + 1)
    
    xc = 0.5 * (x_bins[1:] + x_bins[:-1])
    yc = 0.5 * (y_bins[1:] + y_bins[:-1])
    zc = 0.5 * (z_bins[1:] + z_bins[:-1])
    
    n_cells = nx_bins * ny_bins * nz_bins
    
    # Compute bin volumes
    dx_val = x_bins[1] - x_bins[0]
    dy_val = y_bins[1] - y_bins[0]
    dz_val = z_bins[1] - z_bins[0]
    vol_cell = dx_val * dy_val * dz_val
    
    # Setup target matrices
    target_mass = np.zeros((nx_bins, ny_bins, nz_bins))
    target_ux = np.zeros((nx_bins, ny_bins, nz_bins))
    target_uy = np.zeros((nx_bins, ny_bins, nz_bins))
    target_sxx = np.zeros((nx_bins, ny_bins, nz_bins))
    target_syy = np.zeros((nx_bins, ny_bins, nz_bins))
    target_szz = np.zeros((nx_bins, ny_bins, nz_bins))
    
    print("Evaluating target profiles on coarse grid...")
    for ix in range(nx_bins):
        for iy in range(ny_bins):
            for iz in range(nz_bins):
                x_val, y_val, z_val = xc[ix], yc[iy], zc[iz]
                
                # Mass density at cell center
                rho_val = bulge_density(np.array([[x_val, y_val, z_val]]))[0]
                target_mass[ix, iy, iz] = rho_val * vol_cell
                
                # Kinematics interpolated from NP model
                ux_val, uy_val, sxx_val, syy_val, szz_val = get_np_target_kinematics(
                    grid_params, axes_x, axes_y, axes_z, x_val, y_val, z_val
                )
                target_ux[ix, iy, iz] = ux_val
                target_uy[ix, iy, iz] = uy_val
                target_sxx[ix, iy, iz] = sxx_val
                target_syy[ix, iy, iz] = syy_val
                target_szz[ix, iy, iz] = szz_val
                
    # Flatten targets for simplified matrix math
    t_mass = target_mass.flatten()
    t_ux = target_ux.flatten()
    t_uy = target_uy.flatten()
    t_sxx = target_sxx.flatten()
    t_syy = target_syy.flatten()
    t_szz = target_szz.flatten()
    
    # 3. Integrate Orbits
    is_fast = os.environ.get("FAST_TEST") == "1"
    n_stars = 100 if is_fast else 40000
    nsnap = 50
    total_time = 3.0 # ~ 3 Gyr
    
    ics = sample_initial_conditions(n_stars, full_pot, bulge_density)
    
    print("Integrating orbits in Agama potential...")
    trajs_out = agama.orbit(potential=full_pot, ic=ics, time=total_time, trajsize=nsnap, Omega=omega)
    
    # Orbit arrays to build response matrix
    # Matrix element: cell index c, particle p
    orb_mass = np.zeros((n_cells, n_stars))
    orb_ux = np.zeros((n_cells, n_stars))
    orb_uy = np.zeros((n_cells, n_stars))
    orb_vxx = np.zeros((n_cells, n_stars))
    orb_vyy = np.zeros((n_cells, n_stars))
    orb_vzz = np.zeros((n_cells, n_stars))
    
    print("Projecting trajectories onto 1st-octant coarse grid...")
    for p in range(n_stars):
        traj = trajs_out[p][1] # shape (nsnap, 6)
        x_bar, y_bar, z_bar = traj[:, 0], traj[:, 1], traj[:, 2]
        vx_bar, vy_bar, vz_bar = traj[:, 3], traj[:, 4], traj[:, 5]
        
        # Enforce octant symmetries
        xo = np.abs(x_bar)
        yo = np.abs(y_bar)
        zo = np.abs(z_bar)
        
        u_xo = (vx_bar - omega * y_bar) * np.sign(y_bar)
        u_yo = (vy_bar + omega * x_bar) * np.sign(x_bar)
        
        vxx_o = (vx_bar - omega * y_bar)**2
        vyy_o = (vy_bar + omega * x_bar)**2
        vzz_o = vz_bar**2
        
        # Find cell indices
        ix = np.digitize(xo, x_bins) - 1
        iy = np.digitize(yo, y_bins) - 1
        iz = np.digitize(zo, z_bins) - 1
        
        mask = (ix >= 0) & (ix < nx_bins) & (iy >= 0) & (iy < ny_bins) & (iz >= 0) & (iz < nz_bins)
        
        if np.sum(mask) == 0:
            continue
            
        c_indices = ix[mask] * (ny_bins * nz_bins) + iy[mask] * nz_bins + iz[mask]
        
        # Aggregate properties
        unique_c, counts = np.unique(c_indices, return_counts=True)
        for uc, cnt in zip(unique_c, counts):
            c_mask = c_indices == uc
            # Mass contribution
            orb_mass[uc, p] = cnt / nsnap
            
            # Momentum / Energy contribution
            orb_ux[uc, p] = np.mean(u_xo[mask][c_mask]) * (cnt / nsnap)
            orb_uy[uc, p] = np.mean(u_yo[mask][c_mask]) * (cnt / nsnap)
            orb_vxx[uc, p] = np.mean(vxx_o[mask][c_mask]) * (cnt / nsnap)
            orb_vyy[uc, p] = np.mean(vyy_o[mask][c_mask]) * (cnt / nsnap)
            orb_vzz[uc, p] = np.mean(vzz_o[mask][c_mask]) * (cnt / nsnap)
            
    # 4. Made-to-Measure Weight Refinement Solver
    # Objective: Minimize mismatch of density, momentum, and energy
    print("\n--- M2M Particle Weight Optimization ---")
    weights = np.full(n_stars, np.sum(t_mass) / n_stars) # Initial uniform weights
    
    n_iters = 5 if is_fast else 300
    learning_rate = 0.05
    epsilon = 1e-8
    v_char = 150.0 # characteristic velocity scale for normalization
    
    for i in range(n_iters):
        # Current predictions
        m_pred = orb_mass.dot(weights)
        ux_pred = orb_ux.dot(weights) / (m_pred + epsilon)
        uy_pred = orb_uy.dot(weights) / (m_pred + epsilon)
        vxx_pred = orb_vxx.dot(weights) / (m_pred + epsilon)
        vyy_pred = orb_vyy.dot(weights) / (m_pred + epsilon)
        vzz_pred = orb_vzz.dot(weights) / (m_pred + epsilon)
        
        # Mismatch residuals
        delta_m = m_pred - t_mass
        delta_ux = (ux_pred - t_ux) * m_pred
        delta_uy = (uy_pred - t_uy) * m_pred
        delta_vxx = (vxx_pred - (t_sxx + t_ux**2)) * m_pred
        delta_vyy = (vyy_pred - (t_syy + t_uy**2)) * m_pred
        delta_vzz = (vzz_pred - t_szz) * m_pred
        
        # Compute gradient
        grad_w = np.zeros(n_stars)
        
        # 1. Density gradient (weight = 1.0)
        grad_w += orb_mass.T.dot(delta_m / (t_mass + 1e-4))
        
        # 2. Velocity gradients (weight = 0.5)
        grad_w += 0.5 * orb_ux.T.dot(delta_ux / (t_mass * v_char**2 + 1e-2))
        grad_w += 0.5 * orb_uy.T.dot(delta_uy / (t_mass * v_char**2 + 1e-2))
        
        # 3. Dispersion/Energy gradients (weight = 0.2)
        grad_w += 0.2 * orb_vxx.T.dot(delta_vxx / (t_mass * v_char**4 + 1e-2))
        grad_w += 0.2 * orb_vyy.T.dot(delta_vyy / (t_mass * v_char**4 + 1e-2))
        grad_w += 0.2 * orb_vzz.T.dot(delta_vzz / (t_mass * v_char**4 + 1e-2))
        
        # Weight update preserving positivity
        weights *= np.exp(-learning_rate * np.clip(grad_w, -10.0, 10.0))
        
        # Renormalize to conserve total mass
        weights *= np.sum(t_mass) / np.sum(orb_mass.dot(weights))
        
        if i % 30 == 0 or i == n_iters - 1:
            # Print current chi2
            chi2_m = np.sum(delta_m**2 / (t_mass + 1e-6))
            chi2_u = np.sum((ux_pred - t_ux)**2 * (t_mass > 1e-2))
            chi2_s = np.sum((vxx_pred - ux_pred**2 - t_sxx)**2 * (t_mass > 1e-2))
            print(f"  Iter {i:3d}: Chi2_Mass = {chi2_m:.2e} | Chi2_Vel = {chi2_u:.2e} | Chi2_Disp = {chi2_s:.2e}")
            
    print("\nM2M Fit Complete!")
    
    # 5. Save results
    output_path = os.path.join(project_root, "results/m2m/np_m2m_fit_results.npz")
    np.savez_compressed(output_path, 
                        weights=weights.astype(np.float32), ics=ics,
                        x_bins=x_bins, y_bins=y_bins, z_bins=z_bins,
                        target_mass=target_mass, target_ux=target_ux, target_uy=target_uy,
                        target_sxx=target_sxx, target_syy=target_syy, target_szz=target_szz,
                        orb_mass=orb_mass, orb_ux=orb_ux, orb_uy=orb_uy,
                        orb_vxx=orb_vxx, orb_vyy=orb_vyy, orb_vzz=orb_vzz)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
