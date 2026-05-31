"""
Full-population (bulge + disk) theoretical LOS kinematic moments.

Wraps :class:`SynthPopDiskMomentProvider` and adds the bulge population,
computing the Law-of-Total-Variance mixture moments at each distance and
then integrating along the line of sight weighted by d² density.
"""
from __future__ import annotations

import numpy as np

from synthpop_moment_provider import SynthPopDiskMomentProvider


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SynthPopTheoreticalMoments:
    """
    Compute theoretical LOS kinematic moments for all SynthPop populations.

    Parameters
    ----------
    sp_instance : synthpop.SynthPop
        An initialised SynthPop object.
    mag_bins : array_like
        K_s apparent-magnitude bin edges.
    phase_filter : float, optional
        Restrict to stars with this evolutionary phase (±0.1).
    color_filter : float, optional
        Reserved for future colour-based filtering.
    band_name : str, optional
        SynthPop photometric band identifier.  Default is ``'2MASS_Ks'``.
    """

    def __init__(
        self,
        sp_instance,
        mag_bins,
        phase_filter=None,
        color_filter=None,
        band_name: str = "2MASS_Ks",
    ) -> None:
        self.sp = sp_instance
        self.mag_bins = np.asarray(mag_bins)
        self.n_bins = len(self.mag_bins) - 1
        self.populations = sp_instance.populations
        self.coord_trans = sp_instance.populations[0].coord_trans
        self.provider = SynthPopDiskMomentProvider(
            sp_instance,
            mag_bins,
            phase_filter=phase_filter,
            color_filter=color_filter,
            band_name=band_name,
        )

    def get_pop_moments_gc(
        self,
        pop,
        x_arr: np.ndarray,
        y_arr: np.ndarray,
        z_arr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Delegate to the provider's analytic/MC moment retrieval.

        Parameters
        ----------
        pop : synthpop population object
        x_arr, y_arr, z_arr : np.ndarray, shape (N,)
            Galactocentric Cartesian positions in kpc.

        Returns
        -------
        mu : np.ndarray, shape (N, 3)
        cov : np.ndarray, shape (N, 3, 3)
        """
        return self.provider.get_pop_moments_gc(pop, x_arr, y_arr, z_arr)

    def get_theoretical_los_moments_binned(
        self,
        l_deg: np.ndarray,
        b_deg: np.ndarray,
        distances: np.ndarray,
        use_2d_extinction: bool = False,
    ) -> tuple:
        """
        Compute theoretical LOS-integrated kinematic moments for all populations.

        Applies the Law of Total Variance across populations at each distance
        shell, then integrates over distance weighted by d² density.

        Parameters
        ----------
        l_deg : array_like, shape (n_pix,)
            Galactic longitudes in degrees.
        b_deg : array_like, shape (n_pix,)
            Galactic latitudes in degrees.
        distances : np.ndarray, shape (n_dist,)
            Distance grid in kpc.
        use_2d_extinction : bool, optional
            Subtract 2-D asymptotic extinction.  Default False.

        Returns
        -------
        mu_pm : np.ndarray, shape (n_pix, n_bins, 2)
        cov_pm : np.ndarray, shape (n_pix, n_bins, 2, 2)
        mu_rv : np.ndarray, shape (n_pix, n_bins)
        var_rv : np.ndarray, shape (n_pix, n_bins)
        total_w_k : np.ndarray, shape (n_pix, n_bins)
            Total LOS weight per pixel and bin (for diagnostics).
        """
        v_sun_gc = np.array([
            self.sp.parms.sun.u,
            self.sp.parms.sun.v,
            self.sp.parms.sun.w,
        ])

        # Galactic tilt correction matrix.
        theta = self.sp.parms.sun.theta
        ct, st = np.cos(theta), np.sin(theta)
        H = np.array([[ct, 0, -st], [0, 1, 0], [st, 0, ct]])

        l_arr = np.atleast_1d(l_deg)
        b_arr = np.atleast_1d(b_deg)
        n_pix = len(l_arr)
        n_dist = len(distances)

        l_rad = np.radians(l_arr)
        b_rad = np.radians(b_arr)
        xh_d = np.cos(b_rad) * np.cos(l_rad)
        yh_d = np.cos(b_rad) * np.sin(l_rad)
        zh_d = np.sin(b_rad)
        dxy_d = np.maximum(np.sqrt(xh_d**2 + yh_d**2), 0.001)

        R_lb = np.zeros((n_pix, 2, 3))
        R_lb[:, 0, 0] = -yh_d / dxy_d
        R_lb[:, 0, 1] = xh_d / dxy_d
        R_lb[:, 1, 0] = -(xh_d * zh_d) / dxy_d
        R_lb[:, 1, 1] = -(yh_d * zh_d) / dxy_d
        R_lb[:, 1, 2] = dxy_d
        R_los = np.stack([xh_d, yh_d, zh_d], axis=1)
        scale = 0.2108 / distances

        L_grid, D_grid = np.meshgrid(l_arr, distances, indexing="ij")
        B_grid, _ = np.meshgrid(b_arr, distances, indexing="ij")
        x_gc, y_gc, z_gc = self.coord_trans.dlb_to_xyz(
            D_grid.flatten(), L_grid.flatten(), B_grid.flatten()
        )
        r_sp, phi_sp, z_sp = self.coord_trans.xyz_to_rphiz(x_gc, y_gc, z_gc)

        ext = self.populations[0].extinction
        _, ext_dict = ext.get_extinctions(
            L_grid.flatten(), B_grid.flatten(), D_grid.flatten()
        )
        a_v_all = ext_dict[self.provider.band_name].reshape(n_pix, n_dist)
        dist_mod = 5 * np.log10(distances * 1000) - 5

        if use_2d_extinction:
            from surot_extinction import get_surot_A_Ks
            a_v_2d = get_surot_A_Ks(l_arr, b_arr)
            shift_all = dist_mod[None, :] + (a_v_all - a_v_2d[:, None])
        else:
            shift_all = dist_mod[None, :] + a_v_all

        # Accumulate density-weighted moments across all populations.
        sum_rho_k = np.zeros((n_pix, n_dist, self.n_bins))
        sum_mu_k_pm = np.zeros((n_pix, n_dist, self.n_bins, 2))
        sum_mu_k_rv = np.zeros((n_pix, n_dist, self.n_bins))
        sum_cov_k_pm = np.zeros((n_pix, n_dist, self.n_bins, 2, 2))
        sum_var_k_rv = np.zeros((n_pix, n_dist, self.n_bins))
        pop_moments: list = []

        for pop in self.populations:
            rho_raw = pop.population_density.density(r_sp, phi_sp, z_sp).reshape(
                n_pix, n_dist
            )
            if pop.population_density.density_unit == "number":
                rho_raw *= self.provider.mass_mag_maps[pop.name]["mean_mass"]

            S_bins = self.provider.get_selection_fractions(
                pop.name, distances, l_arr, b_arr,
                use_number=False,
                use_2d_extinction=use_2d_extinction,
                shift_all=shift_all,
            )
            rho_bins = rho_raw[:, :, None] * S_bins.transpose(1, 2, 0)

            mu_uvw, cov_uvw = self.get_pop_moments_gc(pop, x_gc, y_gc, z_gc)
            mu_uvw = mu_uvw.reshape(n_pix, n_dist, 3)
            cov_uvw = cov_uvw.reshape(n_pix, n_dist, 3, 3)

            diff_v_t = np.einsum("ij,pnj->pni", H, mu_uvw - v_sun_gc[None, None, :])
            cov_uvw_t = np.einsum("ij,pnjk,lk->pnil", H, cov_uvw, H)

            mu_pm_pop = (
                np.einsum("pij,pnj->pni", R_lb, diff_v_t) * scale[None, :, None]
            )
            cov_pm_pop = (
                np.einsum("pij,pnjk,plk->pnil", R_lb, cov_uvw_t, R_lb)
                * scale[None, :, None, None] ** 2
            )
            mu_rv_pop = np.einsum("pj,pnj->pn", R_los, diff_v_t)
            var_rv_pop = np.einsum("pj,pnjk,pk->pn", R_los, cov_uvw_t, R_los)

            sum_rho_k += rho_bins
            sum_mu_k_pm += mu_pm_pop[:, :, None, :] * rho_bins[:, :, :, None]
            sum_mu_k_rv += mu_rv_pop[:, :, None] * rho_bins
            sum_cov_k_pm += cov_pm_pop[:, :, None, :, :] * rho_bins[:, :, :, None, None]
            sum_var_k_rv += var_rv_pop[:, :, None] * rho_bins
            pop_moments.append((mu_pm_pop, mu_rv_pop, rho_bins, mu_uvw))

        # ---------------------------------------------------------------------------
        # Law of Total Variance across populations at each distance shell
        # ---------------------------------------------------------------------------
        valid_dk = sum_rho_k > 1e-25

        mu_dk_pm = np.zeros((n_pix, n_dist, self.n_bins, 2))
        mu_dk_rv = np.zeros((n_pix, n_dist, self.n_bins))
        cov_dk_pm = np.zeros((n_pix, n_dist, self.n_bins, 2, 2))
        var_dk_rv = np.zeros((n_pix, n_dist, self.n_bins))

        mu_dk_pm[valid_dk] = sum_mu_k_pm[valid_dk] / sum_rho_k[valid_dk, None]
        mu_dk_rv[valid_dk] = sum_mu_k_rv[valid_dk] / sum_rho_k[valid_dk]

        e_var_pm = np.zeros_like(cov_dk_pm)
        e_var_rv = np.zeros_like(var_dk_rv)
        e_var_pm[valid_dk] = sum_cov_k_pm[valid_dk] / sum_rho_k[valid_dk, None, None]
        e_var_rv[valid_dk] = sum_var_k_rv[valid_dk] / sum_rho_k[valid_dk]

        v_mu_pm = np.zeros_like(cov_dk_pm)
        v_mu_rv = np.zeros_like(var_dk_rv)
        for mu_pm_pop, mu_rv_pop, rb_pop, _ in pop_moments:
            d_pm = mu_pm_pop[:, :, None, :] - mu_dk_pm
            v_mu_pm += rb_pop[:, :, :, None, None] * np.einsum(
                "pnki,pnkj->pnkij", d_pm, d_pm
            )
            d_rv = mu_rv_pop[:, :, None] - mu_dk_rv
            v_mu_rv += rb_pop * d_rv**2

        cov_dk_pm[valid_dk] = (
            e_var_pm[valid_dk]
            + v_mu_pm[valid_dk] / (sum_rho_k[valid_dk, None, None] + 1e-25)
        )
        var_dk_rv[valid_dk] = (
            e_var_rv[valid_dk]
            + v_mu_rv[valid_dk] / (sum_rho_k[valid_dk] + 1e-25)
        )

        # ---------------------------------------------------------------------------
        # LOS integration over distances
        # ---------------------------------------------------------------------------
        weights_dk = sum_rho_k * distances[None, :, None] ** 2
        total_w_k = np.sum(weights_dk, axis=1) + 1e-25

        final_mu_pm = np.zeros((n_pix, self.n_bins, 2))
        final_mu_rv = np.zeros((n_pix, self.n_bins))
        final_cov_pm = np.zeros((n_pix, self.n_bins, 2, 2))
        final_var_rv = np.zeros((n_pix, self.n_bins))

        for p in range(n_pix):
            for k in range(self.n_bins):
                if total_w_k[p, k] <= 1e-20:
                    final_cov_pm[p, k] = np.eye(2) * 10.0
                    final_var_rv[p, k] = 100.0**2
                    continue

                w = weights_dk[p, :, k]
                wt = total_w_k[p, k]

                final_mu_pm[p, k] = np.sum(mu_dk_pm[p, :, k, :] * w[:, None], axis=0) / wt
                final_mu_rv[p, k] = np.sum(mu_dk_rv[p, :, k] * w) / wt

                e_var_bin_pm = np.sum(
                    cov_dk_pm[p, :, k, :, :] * w[:, None, None], axis=0
                ) / wt
                e_var_bin_rv = np.sum(var_dk_rv[p, :, k] * w) / wt

                d_los_pm = mu_dk_pm[p, :, k, :] - final_mu_pm[p, k]
                v_mu_los_pm = np.sum(
                    w[:, None, None] * np.einsum("ni,nj->nij", d_los_pm, d_los_pm),
                    axis=0,
                ) / wt
                d_los_rv = mu_dk_rv[p, :, k] - final_mu_rv[p, k]
                v_mu_los_rv = np.sum(w * d_los_rv**2) / wt

                final_cov_pm[p, k] = e_var_bin_pm + v_mu_los_pm
                final_var_rv[p, k] = e_var_bin_rv + v_mu_los_rv

        return final_mu_pm, final_cov_pm, final_mu_rv, final_var_rv, total_w_k
