import numpy as np
import sys
import os

class SynthPopTheoreticalMoments:
    def __init__(self, sp_instance, mag_bins, phase_filter=None, color_filter=None, band_name='2MASS_Ks'):
        self.sp = sp_instance
        self.mag_bins = mag_bins
        self.n_bins = len(mag_bins) - 1
        self.populations = sp_instance.populations
        self.coord_trans = sp_instance.populations[0].coord_trans
        
        from synthpop_moment_provider import SynthPopDiskMomentProvider
        self.provider = SynthPopDiskMomentProvider(sp_instance, mag_bins, phase_filter=phase_filter, color_filter=color_filter, band_name=band_name)

    def get_pop_moments_gc(self, pop, x_arr, y_arr, z_arr):
        """Vectorized analytic moment retrieval or MC estimation."""
        return self.provider.get_pop_moments_gc(pop, x_arr, y_arr, z_arr)

    def get_theoretical_los_moments_binned(self, l_deg, b_deg, distances, use_2d_extinction=False):
        """Analytical LOS integration using vectorized provider."""
        v_sun_gc = np.array([self.sp.parms.sun.u, self.sp.parms.sun.v, self.sp.parms.sun.w])
        
        # Tilt correction matrix H (from coordinates_transformation.py)
        # Accounts for the angle between the Sun-GC line and the Galactic plane
        theta = self.sp.parms.sun.theta
        ct, st = np.cos(theta), np.sin(theta)
        H = np.array([
            [ct, 0, -st],
            [0, 1, 0],
            [st, 0, ct]
        ])

        l_arr, b_arr = np.atleast_1d(l_deg), np.atleast_1d(b_deg)
        n_pix = len(l_arr)
        n_dist = len(distances)
        
        l_rad, b_rad = np.radians(l_arr), np.radians(b_arr)
        xh_d, yh_d, zh_d = np.cos(b_rad)*np.cos(l_rad), np.cos(b_rad)*np.sin(l_rad), np.sin(b_rad)
        
        # Hel -> LB Basis
        dxy_d = np.maximum(np.sqrt(xh_d**2 + yh_d**2), 0.001)
        
        R_lb = np.zeros((n_pix, 2, 3))
        R_lb[:, 0, 0], R_lb[:, 0, 1] = -yh_d/dxy_d, xh_d/dxy_d
        R_lb[:, 1, 0], R_lb[:, 1, 1], R_lb[:, 1, 2] = -(xh_d*zh_d)/dxy_d, -(yh_d*zh_d)/dxy_d, dxy_d
        
        R_los = np.stack([xh_d, yh_d, zh_d], axis=1) # (n_pix, 3)
        scale = 0.2108 / distances # (n_dist,)

        # 1. Vectorized Coordinates
        L_grid, D_grid = np.meshgrid(l_arr, distances, indexing='ij')
        B_grid, _ = np.meshgrid(b_arr, distances, indexing='ij')
        x_gc, y_gc, z_gc = self.coord_trans.dlb_to_xyz(D_grid.flatten(), L_grid.flatten(), B_grid.flatten())
        r_sp, phi_sp, z_sp = self.coord_trans.xyz_to_rphiz(x_gc, y_gc, z_gc)

        # Precompute extinction shift once for all populations
        ext = self.populations[0].extinction
        _, ext_dict_all = ext.get_extinctions(L_grid.flatten(), B_grid.flatten(), D_grid.flatten())
        a_v_all = ext_dict_all[self.provider.band_name].reshape(n_pix, n_dist)
        
        if use_2d_extinction:
            from surot_extinction import get_surot_A_Ks
            a_v_2d = get_surot_A_Ks(l_arr, b_arr) # (n_pix,)
            dist_mod = 5 * np.log10(distances * 1000) - 5
            shift_all_precomputed = dist_mod[None, :] + (a_v_all - a_v_2d[:, None])
        else:
            dist_mod = 5 * np.log10(distances * 1000) - 5
            shift_all_precomputed = dist_mod[None, :] + a_v_all

        # 2. Vectorized Mixture components per population
        sum_rho_k = np.zeros((n_pix, n_dist, self.n_bins))
        sum_mu_k_pm = np.zeros((n_pix, n_dist, self.n_bins, 2))
        sum_mu_k_rv = np.zeros((n_pix, n_dist, self.n_bins))
        sum_cov_k_pm = np.zeros((n_pix, n_dist, self.n_bins, 2, 2))
        sum_var_k_rv = np.zeros((n_pix, n_dist, self.n_bins))
        
        pop_moments = [] # (mu_pm_pop, mu_rv_pop, rho_bins, mu_uvw)

        for pop in self.populations:
            rho_raw = pop.population_density.density(r_sp, phi_sp, z_sp).reshape(n_pix, n_dist)
            
            if pop.population_density.density_unit == 'number':
                 mapping = self.provider.mass_mag_maps.get(pop.name)
                 rho_raw *= mapping['mean_mass']
            
            S_bins = self.provider.get_selection_fractions(pop.name, distances, l_arr, b_arr, use_number=False, use_2d_extinction=use_2d_extinction, shift_all=shift_all_precomputed) 
            rho_bins = rho_raw[:, :, None] * S_bins.transpose(1, 2, 0) # (n_pix, n_dist, n_bins)
            
            mu_uvw, coop_uvw = self.get_pop_moments_gc(pop, x_gc, y_gc, z_gc)
            mu_uvw = mu_uvw.reshape(n_pix, n_dist, 3)
            coop_uvw = coop_uvw.reshape(n_pix, n_dist, 3, 3)
            
            # Project with TILT H
            # Correct for motion of the sun: diff = (v_star - v_sun)
            diff_v = mu_uvw - v_sun_gc[None, None, :]
            # Apply tilt: v_rel_tilted = H @ diff_v
            diff_v_t = np.einsum('ij,pnj->pni', H, diff_v)
            # Apply tilt to covariance: cov_t = H @ cov @ H^T
            coop_uvw_t = np.einsum('ij,pnjk,lk->pnil', H, coop_uvw, H)

            # PMs
            mu_pm_pop = np.einsum('pij,pnj->pni', R_lb, diff_v_t) * scale[None, :, None]
            cov_pm_pop = np.einsum('pij,pnjk,plk->pnil', R_lb, coop_uvw_t, R_lb) * (scale[None, :, None, None]**2)
            
            # RVs
            mu_rv_pop = np.einsum('pj,pnj->pn', R_los, diff_v_t)
            var_rv_pop = np.einsum('pj,pnjk,pk->pn', R_los, coop_uvw_t, R_los)
            
            sum_rho_k += rho_bins
            sum_mu_k_pm += mu_pm_pop[:, :, None, :] * rho_bins[:, :, :, None]
            sum_mu_k_rv += mu_rv_pop[:, :, None] * rho_bins
            sum_cov_k_pm += cov_pm_pop[:, :, None, :, :] * rho_bins[:, :, :, None, None]
            sum_var_k_rv += var_rv_pop[:, :, None] * rho_bins
            pop_moments.append((mu_pm_pop, mu_rv_pop, rho_bins, mu_uvw))

        # 3. Law of Total Variance across populations at each distance d
        mu_d_k_pm = np.zeros((n_pix, n_dist, self.n_bins, 2))
        mu_d_k_rv = np.zeros((n_pix, n_dist, self.n_bins))
        cov_d_k_pm = np.zeros((n_pix, n_dist, self.n_bins, 2, 2))
        var_d_k_rv = np.zeros((n_pix, n_dist, self.n_bins))
        
        valid_d_k = sum_rho_k > 1e-25
        mu_d_k_pm[valid_d_k] = sum_mu_k_pm[valid_d_k] / sum_rho_k[valid_d_k][:, None]
        mu_d_k_rv[valid_d_k] = sum_mu_k_rv[valid_d_k] / sum_rho_k[valid_d_k]
        
        e_var_d_k_pm = np.zeros_like(cov_d_k_pm)
        e_var_d_k_rv = np.zeros_like(var_d_k_rv)
        e_var_d_k_pm[valid_d_k] = sum_cov_k_pm[valid_d_k] / sum_rho_k[valid_d_k][:, None, None]
        e_var_d_k_rv[valid_d_k] = sum_var_k_rv[valid_d_k] / sum_rho_k[valid_d_k]
        
        v_mu_d_k_pm = np.zeros_like(cov_d_k_pm)
        v_mu_d_k_rv = np.zeros_like(var_d_k_rv)
        for mu_pm_pop, mu_rv_pop, rb_pop, _ in pop_moments:
            diff_pm = mu_pm_pop[:, :, None, :] - mu_d_k_pm
            v_mu_d_k_pm += rb_pop[:, :, :, None, None] * np.einsum('pnki,pnkj->pnkij', diff_pm, diff_pm)
            diff_rv = mu_rv_pop[:, :, None] - mu_d_k_rv
            v_mu_d_k_rv += rb_pop * (diff_rv**2)
            
        cov_d_k_pm[valid_d_k] = e_var_d_k_pm[valid_d_k] + (v_mu_d_k_pm[valid_d_k] / (sum_rho_k[valid_d_k][:, None, None] + 1e-25))
        var_d_k_rv[valid_d_k] = e_var_d_k_rv[valid_d_k] + (v_mu_d_k_rv[valid_d_k] / (sum_rho_k[valid_d_k] + 1e-25))

        # 4. Final LOS integration over distances per bin
        final_mu_pm = np.zeros((n_pix, self.n_bins, 2))
        final_mu_rv = np.zeros((n_pix, self.n_bins))
        final_cov_pm = np.zeros((n_pix, self.n_bins, 2, 2))
        final_var_rv = np.zeros((n_pix, self.n_bins))
        
        weights_d_k = sum_rho_k * (distances[None, :, None]**2)
        total_w_k = np.sum(weights_d_k, axis=1) + 1e-25
        
        for p in range(n_pix):
            for k in range(self.n_bins):
                if total_w_k[p, k] > 1e-20:
                    final_mu_pm[p, k] = np.sum(mu_d_k_pm[p, :, k, :] * weights_d_k[p, :, k, None], axis=0) / total_w_k[p, k]
                    final_mu_rv[p, k] = np.sum(mu_d_k_rv[p, :, k] * weights_d_k[p, :, k], axis=0) / total_w_k[p, k]
                    e_var_bin_pm = np.sum(cov_d_k_pm[p, :, k, :, :] * weights_d_k[p, :, k, None, None], axis=0) / total_w_k[p, k]
                    e_var_bin_rv = np.sum(var_d_k_rv[p, :, k] * weights_d_k[p, :, k], axis=0) / total_w_k[p, k]
                    diff_los_pm = mu_d_k_pm[p, :, k, :] - final_mu_pm[p, k]
                    v_mu_los_pm = np.sum(weights_d_k[p, :, k, None, None] * np.einsum('ni,nj->nij', diff_los_pm, diff_los_pm), axis=0) / total_w_k[p, k]
                    diff_los_rv = mu_d_k_rv[p, :, k] - final_mu_rv[p, k]
                    v_mu_los_rv = np.sum(weights_d_k[p, :, k] * (diff_los_rv**2), axis=0) / total_w_k[p, k]
                    final_cov_pm[p, k] = e_var_bin_pm + v_mu_los_pm
                    final_var_rv[p, k] = e_var_bin_rv + v_mu_los_rv
                else:
                    final_cov_pm[p, k] = np.eye(2) * 10.0
                    final_var_rv[p, k] = 100.0**2


        return final_mu_pm, final_cov_pm, final_mu_rv, final_var_rv, total_w_k
