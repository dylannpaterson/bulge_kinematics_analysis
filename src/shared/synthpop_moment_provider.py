import numpy as np
import sys
import os

class SynthPopDiskMomentProvider:
    # Global cache for mass-mag mappings to avoid redundant precomputation
    _global_mapping_cache = {}

    def __init__(self, sp_instance, mag_bins=None, mag_range=None, phase_filter=None, color_filter=None, band_name='2MASS_Ks'):
        self.sp = sp_instance
        self.phase_filter = phase_filter
        self.color_filter = color_filter
        self.band_name = band_name
        if mag_bins is not None:
            self.mag_bins = mag_bins
            self.n_bins = len(mag_bins) - 1
        elif mag_range is not None:
            self.mag_bins = np.array([mag_range[0], mag_range[1]])
            self.n_bins = 1
        else:
            self.mag_bins = np.array([12.0, 14.0])
            self.n_bins = 1
            
        self.populations_all = sp_instance.populations
        self.populations_disk = [pop for pop in sp_instance.populations if 'bulge' not in pop.name.lower()]
        self.coord_trans = sp_instance.populations[0].coord_trans
        self.mass_mag_maps = {}
        self.mass_mag_grid_maps = {}
        self._precompute_mass_mag_mappings()

    def _precompute_mass_mag_mappings(self):
        """High-resolution vectorized mapping with direct integration normalization."""
        for pop in self.populations_all:
            cache_key = (pop.name, self.band_name, str(self.phase_filter))
            if cache_key in SynthPopDiskMomentProvider._global_mapping_cache:
                self.mass_mag_maps[pop.name] = SynthPopDiskMomentProvider._global_mapping_cache[cache_key]['global']
                continue

            # 1. Grids (UPDATED FOR RED CLUMP RESOLUTION AND PERFORMANCE)
            m_min = 0.08 # STRICTLY 0.08 for all populations
            
            # Use geomspace for better IMF integration
            # 100,000 points provides the required resolution for Red Clump peaks
            m_grid_base = np.geomspace(m_min, 100.0, 100000)
            dm_base = np.gradient(m_grid_base) 

            # Detect single-value distributions to avoid redundant grid points
            z_low = getattr(pop.metallicity, 'lower', getattr(pop.metallicity, 'low_bound', -2.0))
            z_high = getattr(pop.metallicity, 'upper', getattr(pop.metallicity, 'high_bound', 0.5))
            if np.isclose(z_low, z_high):
                z_steps = np.array([z_low])
            else:
                z_steps = np.linspace(z_low, z_high, 30)

            a_low = getattr(pop.age, 'lower', getattr(pop.age, 'low_bound', 10.0))
            a_high = getattr(pop.age, 'upper', getattr(pop.age, 'high_bound', 10.0))
            if np.isclose(a_low, a_high):
                a_steps = np.array([a_low])
            else:
                a_steps = np.unique(np.linspace(a_low, a_high, 3))

            # 2. Cartesian Mesh (Vectorized)
            M, Z, A = np.meshgrid(m_grid_base, z_steps, a_steps, indexing='ij')
            DM, _, _ = np.meshgrid(dm_base, z_steps, a_steps, indexing='ij')
            m_flat, z_flat, a_flat = M.flatten(), Z.flatten(), A.flatten()
            dm_flat = DM.flatten()
            
            # Weights from PDF
            w_z = pop.metallicity.likelyhood_distribution(z_steps)
            w_a = pop.age.likelyhood_distribution(a_steps) if hasattr(pop.age, 'likelyhood_distribution') else np.ones_like(a_steps)
            grid_norm = np.sum(w_z) * np.sum(w_a)
            if grid_norm <= 0: grid_norm = 1.0
            
            # Broadcast weights to the full grid
            _, WZ, WA = np.meshgrid(m_grid_base, w_z, w_a, indexing='ij')
            weights = (pop.imf.imf(m_flat) * dm_flat * WZ.flatten() * WA.flatten()) / grid_norm

            # 3. Evolution (Single large vectorized call)
            try:
                req_props = {'2MASS_J', '2MASS_Ks', 'phase'}
                props, _, _ = pop.evolution.get_evolved_props(m_flat, z_flat, a_flat, req_props)
                if isinstance(props, tuple): props = props[0]
                
                mk_abs = np.array(props['2MASS_Ks'], dtype=float)
                mj_abs = np.array(props['2MASS_J'], dtype=float)
                phase = np.array(props['phase'], dtype=float)
                
                # 4. Filters & Normalization
                valid = ~np.isnan(mk_abs) & ~np.isnan(mj_abs) & (mj_abs - mk_abs > 0.4)
                if self.phase_filter is not None:
                    valid &= np.isclose(phase, self.phase_filter, atol=0.1)
                
                # Mass loss correction
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
                
                # norm is PRESENT-DAY mass
                norm = total_mass_grid * av_mass_corr 
                
                # mean_mass MUST be present-day mass per star to account for remnants
                mean_mass_per_star = norm / total_num_grid if total_num_grid > 0 else 1.0
                
                v_idx = np.where(valid)[0]
                target = mj_abs[v_idx] if 'J' in self.band_name else mk_abs[v_idx]
                sort_idx = np.argsort(target)
                
                self.mass_mag_maps[pop.name] = {
                    'Mk': mk_abs[v_idx][sort_idx], 
                    'Mj': mj_abs[v_idx][sort_idx],
                    'cum_imf_num': np.cumsum(weights[v_idx][sort_idx]),
                    'total_num_norm': total_num_grid,
                    'total_mass_norm': norm,
                    'mean_mass': mean_mass_per_star,
                    'is_rc': (phase[v_idx][sort_idx] >= 3.0) & (phase[v_idx][sort_idx] < 4.0)
                }
                
                SynthPopDiskMomentProvider._global_mapping_cache[cache_key] = {
                    'global': self.mass_mag_maps[pop.name]
                }
            except Exception as e:
                import traceback
                print(f"Error mapping {pop.name}: {e}")
                traceback.print_exc()

    def get_selection_fractions(self, pop_name, d_kpc_array, l_deg, b_deg, use_number=False, use_2d_extinction=False, mapping=None, shift_all=None):
        """Vectorized selection fractions using binary search (searchsorted)."""
        if mapping is None:
            mapping = self.mass_mag_maps.get(pop_name)
        
        l_arr, b_arr = np.atleast_1d(l_deg), np.atleast_1d(b_deg)
        n_pix = len(l_arr)
        n_dist = len(d_kpc_array)

        if mapping is None or 'Mk' not in mapping or len(mapping['Mk']) == 0: 
            return np.zeros((self.n_bins, n_pix, n_dist))
        
        if shift_all is None:
            ext = self.populations_all[0].extinction
            
            # 1. Fully Vectorized Extinction Call
            L_grid, D_grid = np.meshgrid(l_arr, d_kpc_array, indexing='ij')
            B_grid, _ = np.meshgrid(b_arr, d_kpc_array, indexing='ij')
            
            _, ext_dict_all = ext.get_extinctions(L_grid.flatten(), B_grid.flatten(), D_grid.flatten())
            a_v_all = ext_dict_all[self.band_name].reshape(n_pix, n_dist)
            
            # 2D Extinction (asymptotic value at large distance)
            if use_2d_extinction:
                from surot_extinction import get_surot_A_Ks
                a_v_2d = get_surot_A_Ks(l_arr, b_arr) # (n_pix,)
            else:
                a_v_2d = None

            # Distance modulus shift
            dist_mod = 5 * np.log10(d_kpc_array * 1000) - 5
            
            if use_2d_extinction:
                shift_all = dist_mod[None, :] + (a_v_all - a_v_2d[:, None])
            else:
                shift_all = dist_mod[None, :] + a_v_all
        
        # 2. Fast Fraction Calculation using Binary Search
        abs_mag = mapping['Mj'] if 'J' in self.band_name else mapping['Mk']
        cum_pad = np.concatenate(([0.0], mapping['cum_imf_num']))
        norm_val = mapping['total_num_norm'] if use_number else mapping['total_mass_norm']
        
        s_bins = np.zeros((self.n_bins, n_pix, n_dist))
        
        for k in range(self.n_bins):
            # Target absolute magnitude bounds based on bin and shift
            lower_abs = self.mag_bins[k] - shift_all
            upper_abs = self.mag_bins[k+1] - shift_all
            
            # Binary search for the indices (Shape: n_pix, n_dist)
            left_idx = np.searchsorted(abs_mag, lower_abs, side='left')
            right_idx = np.searchsorted(abs_mag, upper_abs, side='right')
            
            s_bins[k, :, :] = (cum_pad[right_idx] - cum_pad[left_idx]) / norm_val
                    
        return s_bins

    def get_pop_moments_gc(self, pop, x_arr, y_arr, z_arr, age_override=None):
        """Fully vectorized analytic retrieval or fast MC estimation."""
        n_points = len(x_arr)
        kin = pop.kinematics
        kin_class = kin.__class__.__name__

        # 1. Analytic Branch for Cylindrical Disks (Standard and Koshimoto2021)
        if hasattr(kin, 'sigma_u') and hasattr(kin, 'sigma_v'):
            r, phi, z = self.coord_trans.xyz_to_rphiz(x_arr, y_arr, z_arr)
            lsr = self.sp.parms.lsr
            v_lsr = lsr['v_lsr'] if isinstance(lsr, dict) else lsr.v_lsr
            lsr_dist = abs(self.sp.parms.sun.gal_dist)
            v_grad = getattr(kin, 'v_lsr_grad', 0.0)
            
            v_phi = (v_lsr + v_grad * (r - lsr_dist))
            mu = np.zeros((n_points, 3))
            # x = -r cos(phi) -> vx = v_phi sin(phi)
            # y =  r sin(phi) -> vy = v_phi cos(phi)
            mu[:, 0] = v_phi * np.sin(phi)
            mu[:, 1] = v_phi * np.cos(phi)
            
            sig_u, sig_v, sig_w = kin.sigma_u, kin.sigma_v, kin.sigma_w
            cp, sp = np.cos(phi), np.sin(phi)
            
            cov = np.zeros((n_points, 3, 3))
            cov[:, 0, 0] = cp**2 * sig_u**2 + sp**2 * sig_v**2
            cov[:, 1, 1] = sp**2 * sig_u**2 + cp**2 * sig_v**2
            # Cov(vx, vy) = sin(phi)cos(phi) * (sigma_phi^2 - sigma_r^2)
            cov[:, 0, 1] = cov[:, 1, 0] = cp * sp * (sig_v**2 - sig_u**2)
            cov[:, 2, 2] = sig_w**2
            return mu, cov

        elif kin_class == 'Koshimoto2021Disk':
            r, phi, z = self.coord_trans.xyz_to_rphiz(x_arr, y_arr, z_arr)
            # Parameters from Koshimoto+21
            T_min, T_max = 0.01, 10.0
            r_sun = abs(self.sp.parms.sun.gal_dist)
            
            age = age_override if age_override is not None else kin.pop_age
            age_factor_r = ((age + T_min) / (T_max + T_min))**kin.beta_r
            age_factor_z = ((age + T_min) / (T_max + T_min))**kin.beta_z
            
            sig_r = kin.sigma_r_sun * age_factor_r * np.exp(-(r - r_sun) / kin.R_sigma_r)
            sig_z = kin.sigma_z_sun * age_factor_z * np.exp(-(r - r_sun) / kin.R_sigma_z)
            sig_phi = sig_r * 0.707 # Epicycle approximation
            
            # Rotation curve mean
            v_c = np.interp(r, kin.rot_curve[0], kin.rot_curve[1])
            # Account for vertical height reduction
            v_phi = v_c / (1.0 + 0.0374 * np.abs(z)**1.34)
            
            # Apply Asymmetric Drift (Lag) based on radial dispersion
            # Jeans equation approximation: Va ≈ (sigma_r^2 / 2*Vc) * [r/R_d + 2*r/R_sigma_r - 0.5]
            # R_d is roughly 2.6 kpc for this disk model.
            R_d = 2.6
            v_lag = (sig_r**2) / (2.0 * v_c + 1e-10) * (r / R_d + 2.0 * r / kin.R_sigma_r - 0.5)
            v_phi = np.maximum(v_phi - v_lag, 0.0)
            
            mu = np.zeros((n_points, 3))
            mu[:, 0] = v_phi * np.sin(phi)
            mu[:, 1] = v_phi * np.cos(phi)
            
            cp, sp = np.cos(phi), np.sin(phi)
            cov = np.zeros((n_points, 3, 3))
            cov[:, 0, 0] = cp**2 * sig_r**2 + sp**2 * sig_phi**2
            cov[:, 1, 1] = sp**2 * sig_r**2 + cp**2 * sig_phi**2
            cov[:, 0, 1] = cov[:, 1, 0] = cp * sp * (sig_phi**2 - sig_r**2)
            cov[:, 2, 2] = sig_z**2
            return mu, cov

        # 2. Analytic Branch for Koshimoto2021Bulge (Bar Frame)
        elif kin_class == 'Koshimoto2021Bulge':
            alpha = kin.bar_ang
            
            # Rotate to bar frame (Matches SynthPop)
            xp = x_arr * np.cos(alpha) - y_arr * np.sin(alpha)
            yp = x_arr * np.sin(alpha) + y_arr * np.cos(alpha)
            R = np.sqrt(x_arr**2 + y_arr**2)
            
            # Stream velocity (Matches SynthPop sign logic)
            y0_stream = kin.y0_stream / 1000.0 if kin.y0_stream > 10.0 else kin.y0_stream
            v_x_stream = kin.v0_stream * (1 - np.exp(-(yp / y0_stream)**2)) * np.sign(yp)
            
            # Solid body velocity
            v_sb = kin.omega_p * R

            # Calculate velocities in bar frame
            vxp = v_sb * (yp / (R + 1e-10)) + v_x_stream
            vyp = v_sb * (-xp / (R + 1e-10))
            
            # Convert back into Galactic frame (rot = -alpha)
            rot = -alpha
            cr, sr = np.cos(rot), np.sin(rot)
            mu = np.zeros((n_points, 3))
            mu[:, 0] = vxp * cr - vyp * sr
            mu[:, 1] = vxp * sr + vyp * cr
            mu[:, 2] = 0.0
            
            def get_sig(i):
                x0, y0, z0 = kin.h0_r if i < 2 else kin.h0_z
                cp, cq = (kin.C_par_r, kin.C_perp_r) if i < 2 else (kin.C_par_z, kin.C_perp_z)
                rs = (((np.abs(xp)/x0)**cq + (np.abs(yp)/y0)**cq)**(cp/cq) + (np.abs(z_arr)/z0)**cp)**(1/cp)
                return kin.sigma_i0[i] + kin.sigma_i1[i] * np.exp(-rs)

            sx, sy, sz = get_sig(0), get_sig(1), get_sig(2)
            cov = np.zeros((n_points, 3, 3))
            # Rotate covariance to Galactic frame
            cov[:, 0, 0] = cr**2 * sx**2 + sr**2 * sy**2
            cov[:, 1, 1] = sr**2 * sx**2 + cr**2 * sy**2
            cov[:, 0, 1] = cov[:, 1, 0] = cr * sr * (sx**2 - sy**2) 
            cov[:, 2, 2] = sz**2
            return mu, cov

        elif kin_class == 'VelocityGradient':
            mu = np.zeros((n_points, 3))
            mu[:, 0] = kin.u_mean + kin.u_grad * (x_arr - kin.sun.x)
            mu[:, 1] = kin.v_mean + kin.v_grad * (y_arr - kin.sun.y)
            mu[:, 2] = kin.w_mean + kin.w_grad * (z_arr - kin.sun.z)
            cov = np.zeros((n_points, 3, 3))
            cov[:, 0, 0], cov[:, 1, 1], cov[:, 2, 2] = kin.sigma_u**2, kin.sigma_v**2, kin.sigma_w**2
            return mu, cov

        else:
            # Vectorized MC Fallback (only for classes not covered above, like NSD grid)
            # Reduced samples and vectorized to prevent total freeze
            n_mc = 100
            xi = np.repeat(x_arr, n_mc)
            yi = np.repeat(y_arr, n_mc)
            zi = np.repeat(z_arr, n_mc)
            vxi, vyi, vzi = kin.draw_random_velocity(xi, yi, zi)
            v_samples = np.stack([vxi, vyi, vzi], axis=1).reshape(n_points, n_mc, 3)
            mu = np.mean(v_samples, axis=1)
            diff = v_samples - mu[:, None, :]
            cov = np.einsum('nij,nik->njk', diff, diff) / (n_mc - 1)
            return mu, cov

    def get_combined_disk_moments_binned(self, l_deg, b_deg, distances, use_2d_extinction=False, use_number=True):
        """Vectorized mixture moments across pixels and distances with correct volume weighting."""
        l_arr, b_arr = np.atleast_1d(l_deg), np.atleast_1d(b_deg)
        n_pix = len(l_arr)
        n_dist = len(distances)
        
        # Integration weights include the s^2 volume factor
        s_sq = distances**2
        
        mu_pm = np.zeros((n_pix, n_dist, self.n_bins, 2))
        cov_pm = np.zeros((n_pix, n_dist, self.n_bins, 2, 2))
        mu_rv = np.zeros((n_pix, n_dist, self.n_bins))
        var_rv = np.zeros((n_pix, n_dist, self.n_bins))
        sum_rho = np.zeros((n_pix, n_dist, self.n_bins))
        
        v_sun = np.array([self.sp.parms.sun.u, self.sp.parms.sun.v, self.sp.parms.sun.w])
        l_rad, b_rad = np.radians(l_arr), np.radians(b_arr)
        
        xh_d = np.cos(b_rad)*np.cos(l_rad)
        yh_d = np.cos(b_rad)*np.sin(l_rad)
        zh_d = np.sin(b_rad)
        dxy_d = np.maximum(np.sqrt(xh_d**2 + yh_d**2), 0.001)
        
        R_lb = np.zeros((n_pix, 2, 3))
        R_lb[:, 0, 0], R_lb[:, 0, 1] = -yh_d/dxy_d, xh_d/dxy_d
        R_lb[:, 1, 0], R_lb[:, 1, 1] = -(xh_d*zh_d)/dxy_d, -(yh_d*zh_d)/dxy_d
        R_lb[:, 1, 2] = dxy_d
        
        R_los = np.stack([xh_d, yh_d, zh_d], axis=1)
        scale = 0.2108 / distances

        L_grid, D_grid = np.meshgrid(l_arr, distances, indexing='ij')
        B_grid, _ = np.meshgrid(b_arr, distances, indexing='ij')
        
        x_gc, y_gc, z_gc = self.coord_trans.dlb_to_xyz(D_grid.flatten(), L_grid.flatten(), B_grid.flatten())
        r_sp, phi_sp, z_sp = self.coord_trans.xyz_to_rphiz(x_gc, y_gc, z_gc)

        # Precompute extinction shift once for all disk populations
        ext = self.populations_all[0].extinction
        _, ext_dict_all = ext.get_extinctions(L_grid.flatten(), B_grid.flatten(), D_grid.flatten())
        a_v_all = ext_dict_all[self.band_name].reshape(n_pix, n_dist)
        
        if use_2d_extinction:
            from surot_extinction import get_surot_A_Ks
            a_v_2d = get_surot_A_Ks(l_arr, b_arr) # (n_pix,)
        else:
            a_v_2d = None

        dist_mod = 5 * np.log10(distances * 1000) - 5
        
        if use_2d_extinction:
            shift_all_precomputed = dist_mod[None, :] + (a_v_all - a_v_2d[:, None])
        else:
            shift_all_precomputed = dist_mod[None, :] + a_v_all

        v_sun = np.array([self.sp.parms.sun.u, self.sp.parms.sun.v, self.sp.parms.sun.w])
        
        # Tilt correction matrix H (from coordinates_transformation.py)
        # Accounts for the angle between the Sun-GC line and the Galactic plane
        theta = self.sp.parms.sun.theta
        ct, st = np.cos(theta), np.sin(theta)
        H = np.array([
            [ct, 0, -st],
            [0, 1, 0],
            [st, 0, ct]
        ])

        for pop in self.populations_disk:
            rho_raw = pop.population_density.density(r_sp, phi_sp, z_sp).reshape(n_pix, n_dist)
            if use_number:
                if pop.population_density.density_unit == 'mass':
                    mapping = self.mass_mag_maps.get(pop.name)
                    rho_raw /= mapping['mean_mass']
            else:
                if pop.population_density.density_unit == 'number':
                     mapping = self.mass_mag_maps.get(pop.name)
                     rho_raw *= mapping['mean_mass']
            
            S_bins = self.get_selection_fractions(pop.name, distances, l_arr, b_arr, use_number=use_number, use_2d_extinction=use_2d_extinction, shift_all=shift_all_precomputed)
            rb_pop = (rho_raw[None, :, :] * S_bins).transpose(1, 2, 0)

            mu_uvw, cov_uvw = self.get_pop_moments_gc(pop, x_gc, y_gc, z_gc)
            mu_uvw = mu_uvw.reshape(n_pix, n_dist, 3)
            cov_uvw = cov_uvw.reshape(n_pix, n_dist, 3, 3)
            
            # Correct for motion of the sun and APPLY TILT
            # uvw_rel = H @ (v_star - v_sun)
            diff_v = mu_uvw - v_sun[None, None, :]
            diff_v_tilted = np.einsum('ij,pnj->pni', H, diff_v)
            
            # Rotate covariance too
            cov_uvw_tilted = np.einsum('ij,pnjk,lk->pnil', H, cov_uvw, H)
            
            m_pm = np.einsum('pij,pnj->pni', R_lb, diff_v_tilted) * scale[None, :, None]
            c_pm = np.einsum('pij,pnjk,plk->pnil', R_lb, cov_uvw_tilted, R_lb) * (scale[None, :, None, None]**2)
            m_rv = np.einsum('pj,pnj->pn', R_los, diff_v_tilted)
            v_rv = np.einsum('pj,pnjk,pk->pn', R_los, cov_uvw_tilted, R_los)
            
            sum_rho += rb_pop
            mu_pm += m_pm[:, :, None, :] * rb_pop[:, :, :, None]
            mu_sq_pm = m_pm[:, :, None, :, None] * m_pm[:, :, None, None, :]
            cov_pm += (c_pm[:, :, None, :, :] + mu_sq_pm) * rb_pop[:, :, :, None, None]
            
            mu_rv += m_rv[:, :, None] * rb_pop
            var_rv += (v_rv[:, :, None] + m_rv[:, :, None]**2) * rb_pop

        valid = sum_rho > 1e-25
        mu_pm_final = np.zeros_like(mu_pm)
        cov_pm_final = np.zeros_like(cov_pm)
        mu_rv_final = np.zeros_like(mu_rv)
        var_rv_final = np.zeros_like(var_rv)
        
        mu_pm_final[valid] = mu_pm[valid] / sum_rho[valid][:, None]
        mu_rv_final[valid] = mu_rv[valid] / sum_rho[valid]
        
        mu_sq_final = mu_pm_final[:, :, :, :, None] * mu_pm_final[:, :, :, None, :]
        cov_pm_final[valid] = (cov_pm[valid] / sum_rho[valid][:, None, None]) - mu_sq_final[valid]
        var_rv_final[valid] = (var_rv[valid] / sum_rho[valid]) - mu_rv_final[valid]**2
        
        cov_pm_final[~valid] = np.eye(2) * 10.0
        var_rv_final[~valid] = 100.0
        
        if n_pix == 1:
            return mu_pm_final[0], cov_pm_final[0], mu_rv_final[0], var_rv_final[0], sum_rho[0]
        return mu_pm_final, cov_pm_final, mu_rv_final, var_rv_final, sum_rho
