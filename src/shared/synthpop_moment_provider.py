"""
SynthPop disk moment provider for LOS kinematic integration.

Precomputes mass-magnitude mappings for each stellar population and exposes
vectorised methods to compute selection fractions and weighted disk kinematic
moments across pixels and distance slices.
"""
from __future__ import annotations

import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SynthPopDiskMomentProvider:
    """
    Vectorised disk kinematic moment provider backed by SynthPop populations.

    On construction, builds mass-to-apparent-magnitude mappings for each
    population using a fine mass grid and vectorised stellar evolution calls.
    These mappings are cached at the class level so multiple instances that
    share the same population can reuse the precomputed tables.

    Parameters
    ----------
    sp_instance : synthpop.SynthPop
        An initialised SynthPop object whose populations will be queried.
    mag_bins : array_like, optional
        K_s apparent-magnitude bin edges.  Takes priority over ``mag_range``.
    mag_range : tuple of float, optional
        (faint, bright) limits used to define a single magnitude bin when
        ``mag_bins`` is not provided.
    phase_filter : float, optional
        If set, only stars with ``phase == phase_filter`` (±0.1) contribute.
    color_filter : float, optional
        Reserved for future colour-based filtering (not yet implemented).
    band_name : str, optional
        SynthPop photometric band identifier.  Default is ``'2MASS_Ks'``.
    """

    # Class-level cache shared across all instances.
    _global_mapping_cache: dict = {}

    def __init__(
        self,
        sp_instance,
        mag_bins=None,
        mag_range=None,
        phase_filter=None,
        color_filter=None,
        band_name: str = "2MASS_Ks",
    ) -> None:
        self.sp = sp_instance
        self.phase_filter = phase_filter
        self.color_filter = color_filter
        self.band_name = band_name

        if mag_bins is not None:
            self.mag_bins = np.asarray(mag_bins)
            self.n_bins = len(self.mag_bins) - 1
        elif mag_range is not None:
            self.mag_bins = np.array([mag_range[0], mag_range[1]])
            self.n_bins = 1
        else:
            self.mag_bins = np.array([12.0, 14.0])
            self.n_bins = 1

        self.populations_all = sp_instance.populations
        self.populations_disk = [
            p for p in sp_instance.populations if "bulge" not in p.name.lower()
        ]
        self.coord_trans = sp_instance.populations[0].coord_trans
        self.mass_mag_maps: dict = {}
        self.mass_mag_grid_maps: dict = {}
        self._precompute_mass_mag_mappings()

    # ---------------------------------------------------------------------------
    # Internal: mass-magnitude precomputation
    # ---------------------------------------------------------------------------

    def _precompute_mass_mag_mappings(self) -> None:
        """Build or retrieve cached mass-to-magnitude tables for every population."""
        for pop in self.populations_all:
            cache_key = (pop.name, self.band_name, str(self.phase_filter))
            if cache_key in SynthPopDiskMomentProvider._global_mapping_cache:
                self.mass_mag_maps[pop.name] = (
                    SynthPopDiskMomentProvider._global_mapping_cache[cache_key]["global"]
                )
                continue

            # Fine mass grid — geomspace gives better IMF integration.
            m_min = 0.08
            is_fast = os.environ.get("FAST_TEST") == "1"
            n_m = 5000 if is_fast else 100_000
            m_grid = np.geomspace(m_min, 100.0, n_m)
            dm = np.gradient(m_grid)

            z_low = getattr(pop.metallicity, "lower",
                            getattr(pop.metallicity, "low_bound", -2.0))
            z_high = getattr(pop.metallicity, "upper",
                             getattr(pop.metallicity, "high_bound", 0.5))
            n_z = 5 if is_fast else 30
            z_steps = (
                np.array([z_low]) if np.isclose(z_low, z_high)
                else np.linspace(z_low, z_high, n_z)
            )

            a_low = getattr(pop.age, "lower", getattr(pop.age, "low_bound", 10.0))
            a_high = getattr(pop.age, "upper", getattr(pop.age, "high_bound", 10.0))
            n_a = 1 if is_fast else 3
            a_steps = (
                np.array([a_low]) if np.isclose(a_low, a_high)
                else np.unique(np.linspace(a_low, a_high, n_a))
            )

            M, Z, A = np.meshgrid(m_grid, z_steps, a_steps, indexing="ij")
            DM, _, _ = np.meshgrid(dm, z_steps, a_steps, indexing="ij")
            m_flat = M.flatten()
            z_flat = Z.flatten()
            a_flat = A.flatten()
            dm_flat = DM.flatten()

            w_z = pop.metallicity.likelyhood_distribution(z_steps)
            w_a = (
                pop.age.likelyhood_distribution(a_steps)
                if hasattr(pop.age, "likelyhood_distribution")
                else np.ones_like(a_steps)
            )
            grid_norm = max(np.sum(w_z) * np.sum(w_a), 1e-300)

            _, WZ, WA = np.meshgrid(m_grid, w_z, w_a, indexing="ij")
            weights = pop.imf.imf(m_flat) * dm_flat * WZ.flatten() * WA.flatten() / grid_norm

            try:
                req_props = {"2MASS_J", "2MASS_Ks", "phase"}
                props, _, _ = pop.evolution.get_evolved_props(
                    m_flat, z_flat, a_flat, req_props
                )
                if isinstance(props, tuple):
                    props = props[0]

                mk_abs = np.array(props["2MASS_Ks"], dtype=float)
                mj_abs = np.array(props["2MASS_J"], dtype=float)
                phase = np.array(props["phase"], dtype=float)

                valid = (
                    ~np.isnan(mk_abs)
                    & ~np.isnan(mj_abs)
                    & (mj_abs - mk_abs > 0.4)
                )
                if self.phase_filter is not None:
                    valid &= np.isclose(phase, self.phase_filter, atol=0.1)

                # Present-day mass normalisation (accounts for stellar remnants).
                av_mass_corr = getattr(pop, "av_mass_corr", None)
                if av_mass_corr is None:
                    try:
                        if not hasattr(pop, "position") or pop.position.l_deg is None:
                            pop.set_position(0.0, 0.0, 0.01, "deg^2")
                        av_mass_corr = pop.get_mass_loss_for_option(pop.lost_mass_option)
                    except Exception:
                        av_mass_corr = 1.0
                    pop.av_mass_corr = av_mass_corr

                total_num = np.sum(weights)
                total_mass = np.sum(m_flat * weights)
                norm = total_mass * av_mass_corr
                mean_mass = norm / total_num if total_num > 0 else 1.0

                v_idx = np.where(valid)[0]
                target = mj_abs[v_idx] if "J" in self.band_name else mk_abs[v_idx]
                sort_idx = np.argsort(target)

                mapping = {
                    "Mk": mk_abs[v_idx][sort_idx],
                    "Mj": mj_abs[v_idx][sort_idx],
                    "cum_imf_num": np.cumsum(weights[v_idx][sort_idx]),
                    "total_num_norm": total_num,
                    "total_mass_norm": norm,
                    "mean_mass": mean_mass,
                    "is_rc": (
                        (phase[v_idx][sort_idx] >= 3.0)
                        & (phase[v_idx][sort_idx] < 4.0)
                    ),
                }
                self.mass_mag_maps[pop.name] = mapping
                SynthPopDiskMomentProvider._global_mapping_cache[cache_key] = {
                    "global": mapping
                }

            except Exception as e:
                import traceback
                print(f"Error mapping population '{pop.name}': {e}")
                traceback.print_exc()

    # ---------------------------------------------------------------------------
    # Selection fractions
    # ---------------------------------------------------------------------------

    def get_selection_fractions(
        self,
        pop_name: str,
        d_kpc_array: np.ndarray,
        l_deg: np.ndarray,
        b_deg: np.ndarray,
        use_number: bool = False,
        use_2d_extinction: bool = False,
        mapping: dict | None = None,
        shift_all: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute the fraction of stellar mass (or number) in each magnitude bin.

        Uses binary search on the precomputed cumulative IMF array to make the
        computation O(log N) per pixel-distance pair.

        Parameters
        ----------
        pop_name : str
            Population identifier key in ``self.mass_mag_maps``.
        d_kpc_array : np.ndarray, shape (n_dist,)
            Distance grid in kpc.
        l_deg : array_like, shape (n_pix,)
            Galactic longitudes in degrees.
        b_deg : array_like, shape (n_pix,)
            Galactic latitudes in degrees.
        use_number : bool, optional
            If True normalise by number density; otherwise by mass.  Default False.
        use_2d_extinction : bool, optional
            If True subtract the 2-D asymptotic extinction before computing
            distance modulus shifts.  Default False.
        mapping : dict, optional
            Pre-fetched mass-mag mapping.  Looked up from cache if None.
        shift_all : np.ndarray, shape (n_pix, n_dist), optional
            Pre-computed (distance modulus + extinction) shifts.  Computed
            internally if None.

        Returns
        -------
        s_bins : np.ndarray, shape (n_bins, n_pix, n_dist)
            Selection fractions for each magnitude bin.
        """
        if mapping is None:
            mapping = self.mass_mag_maps.get(pop_name)

        l_arr = np.atleast_1d(l_deg)
        b_arr = np.atleast_1d(b_deg)
        n_pix = len(l_arr)
        n_dist = len(d_kpc_array)

        if mapping is None or "Mk" not in mapping or len(mapping["Mk"]) == 0:
            return np.zeros((self.n_bins, n_pix, n_dist))

        if shift_all is None:
            ext = self.populations_all[0].extinction
            L_grid, D_grid = np.meshgrid(l_arr, d_kpc_array, indexing="ij")
            B_grid, _ = np.meshgrid(b_arr, d_kpc_array, indexing="ij")
            _, ext_dict = ext.get_extinctions(
                L_grid.flatten(), B_grid.flatten(), D_grid.flatten()
            )
            a_v_all = ext_dict[self.band_name].reshape(n_pix, n_dist)

            dist_mod = 5 * np.log10(d_kpc_array * 1000) - 5

            if use_2d_extinction:
                from surot_extinction import get_surot_A_Ks
                a_v_2d = get_surot_A_Ks(l_arr, b_arr)
                shift_all = dist_mod[None, :] + (a_v_all - a_v_2d[:, None])
            else:
                shift_all = dist_mod[None, :] + a_v_all

        abs_mag = mapping["Mj"] if "J" in self.band_name else mapping["Mk"]
        cum_pad = np.concatenate(([0.0], mapping["cum_imf_num"]))
        norm_val = (
            mapping["total_num_norm"] if use_number else mapping["total_mass_norm"]
        )

        s_bins = np.zeros((self.n_bins, n_pix, n_dist))
        for k in range(self.n_bins):
            lower_abs = self.mag_bins[k] - shift_all
            upper_abs = self.mag_bins[k + 1] - shift_all
            left_idx = np.searchsorted(abs_mag, lower_abs, side="left")
            right_idx = np.searchsorted(abs_mag, upper_abs, side="right")
            s_bins[k] = (cum_pad[right_idx] - cum_pad[left_idx]) / norm_val

        return s_bins

    # ---------------------------------------------------------------------------
    # Per-population kinematic moments (Galactocentric frame)
    # ---------------------------------------------------------------------------

    def get_pop_moments_gc(
        self,
        pop,
        x_arr: np.ndarray,
        y_arr: np.ndarray,
        z_arr: np.ndarray,
        age_override: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return mean velocity and covariance for a single population at 3-D positions.

        Dispatches to an analytic branch if the kinematic class supports it,
        otherwise falls back to a vectorised Monte Carlo estimator.

        Parameters
        ----------
        pop : synthpop population object
        x_arr, y_arr, z_arr : np.ndarray, shape (N,)
            Galactocentric Cartesian positions in kpc.
        age_override : float, optional
            Override the population age for disk dispersion scaling.

        Returns
        -------
        mu : np.ndarray, shape (N, 3)
            Mean (vx, vy, vz) in km/s (Galactocentric).
        cov : np.ndarray, shape (N, 3, 3)
            Velocity covariance matrix in (km/s)².
        """
        n_pts = len(x_arr)
        kin = pop.kinematics
        kin_class = kin.__class__.__name__

        # --- Analytic: generic cylindrical disk ---
        if hasattr(kin, "sigma_u") and hasattr(kin, "sigma_v"):
            r, phi, z = self.coord_trans.xyz_to_rphiz(x_arr, y_arr, z_arr)
            lsr = self.sp.parms.lsr
            v_lsr = lsr["v_lsr"] if isinstance(lsr, dict) else lsr.v_lsr
            lsr_dist = abs(self.sp.parms.sun.gal_dist)
            v_grad = getattr(kin, "v_lsr_grad", 0.0)

            v_phi = v_lsr + v_grad * (r - lsr_dist)
            mu = np.zeros((n_pts, 3))
            mu[:, 0] = v_phi * np.sin(phi)
            mu[:, 1] = v_phi * np.cos(phi)

            sig_u, sig_v, sig_w = kin.sigma_u, kin.sigma_v, kin.sigma_w
            cp, sp = np.cos(phi), np.sin(phi)
            cov = np.zeros((n_pts, 3, 3))
            cov[:, 0, 0] = cp**2 * sig_u**2 + sp**2 * sig_v**2
            cov[:, 1, 1] = sp**2 * sig_u**2 + cp**2 * sig_v**2
            cov[:, 0, 1] = cov[:, 1, 0] = cp * sp * (sig_v**2 - sig_u**2)
            cov[:, 2, 2] = sig_w**2
            return mu, cov

        # --- Analytic: Koshimoto 2021 disk with asymmetric drift ---
        if kin_class == "Koshimoto2021Disk":
            r, phi, z = self.coord_trans.xyz_to_rphiz(x_arr, y_arr, z_arr)
            T_min, T_max = 0.01, 10.0
            r_sun = abs(self.sp.parms.sun.gal_dist)
            age = age_override if age_override is not None else kin.pop_age

            age_factor_r = ((age + T_min) / (T_max + T_min)) ** kin.beta_r
            age_factor_z = ((age + T_min) / (T_max + T_min)) ** kin.beta_z
            sig_r = kin.sigma_r_sun * age_factor_r * np.exp(-(r - r_sun) / kin.R_sigma_r)
            sig_z = kin.sigma_z_sun * age_factor_z * np.exp(-(r - r_sun) / kin.R_sigma_z)
            sig_phi = sig_r * 0.707  # epicycle approximation

            v_c = np.interp(r, kin.rot_curve[0], kin.rot_curve[1])
            v_phi = v_c / (1.0 + 0.0374 * np.abs(z) ** 1.34)

            R_d = 2.6  # approximate thin-disk scale length (kpc)
            v_lag = (sig_r**2) / (2.0 * v_c + 1e-10) * (
                r / R_d + 2.0 * r / kin.R_sigma_r - 0.5
            )
            v_phi = np.maximum(v_phi - v_lag, 0.0)

            mu = np.zeros((n_pts, 3))
            mu[:, 0] = v_phi * np.sin(phi)
            mu[:, 1] = v_phi * np.cos(phi)

            cp, sp = np.cos(phi), np.sin(phi)
            cov = np.zeros((n_pts, 3, 3))
            cov[:, 0, 0] = cp**2 * sig_r**2 + sp**2 * sig_phi**2
            cov[:, 1, 1] = sp**2 * sig_r**2 + cp**2 * sig_phi**2
            cov[:, 0, 1] = cov[:, 1, 0] = cp * sp * (sig_phi**2 - sig_r**2)
            cov[:, 2, 2] = sig_z**2
            return mu, cov

        # --- Analytic: Koshimoto 2021 bulge (bar frame) ---
        if kin_class == "Koshimoto2021Bulge":
            alpha = kin.bar_ang
            xp = x_arr * np.cos(alpha) - y_arr * np.sin(alpha)
            yp = x_arr * np.sin(alpha) + y_arr * np.cos(alpha)
            R = np.sqrt(x_arr**2 + y_arr**2)

            y0_stream = kin.y0_stream / 1000.0 if kin.y0_stream > 10.0 else kin.y0_stream
            v_x_stream = (
                kin.v0_stream * (1 - np.exp(-(yp / y0_stream) ** 2)) * np.sign(yp)
            )
            v_sb = kin.omega_p * R

            vxp = v_sb * (yp / (R + 1e-10)) + v_x_stream
            vyp = v_sb * (-xp / (R + 1e-10))

            cr, sr = np.cos(-alpha), np.sin(-alpha)
            mu = np.zeros((n_pts, 3))
            mu[:, 0] = vxp * cr - vyp * sr
            mu[:, 1] = vxp * sr + vyp * cr

            def _sigma(i):
                x0, y0, z0 = kin.h0_r if i < 2 else kin.h0_z
                cp_c, cq_c = (
                    (kin.C_par_r, kin.C_perp_r) if i < 2
                    else (kin.C_par_z, kin.C_perp_z)
                )
                rs = (
                    ((np.abs(xp) / x0) ** cq_c + (np.abs(yp) / y0) ** cq_c)
                    ** (cp_c / cq_c)
                    + (np.abs(z_arr) / z0) ** cp_c
                ) ** (1 / cp_c)
                return kin.sigma_i0[i] + kin.sigma_i1[i] * np.exp(-rs)

            sx, sy, sz = _sigma(0), _sigma(1), _sigma(2)
            cov = np.zeros((n_pts, 3, 3))
            cov[:, 0, 0] = cr**2 * sx**2 + sr**2 * sy**2
            cov[:, 1, 1] = sr**2 * sx**2 + cr**2 * sy**2
            cov[:, 0, 1] = cov[:, 1, 0] = cr * sr * (sx**2 - sy**2)
            cov[:, 2, 2] = sz**2
            return mu, cov

        # --- Analytic: linear velocity gradient model ---
        if kin_class == "VelocityGradient":
            mu = np.zeros((n_pts, 3))
            mu[:, 0] = kin.u_mean + kin.u_grad * (x_arr - kin.sun.x)
            mu[:, 1] = kin.v_mean + kin.v_grad * (y_arr - kin.sun.y)
            mu[:, 2] = kin.w_mean + kin.w_grad * (z_arr - kin.sun.z)
            cov = np.zeros((n_pts, 3, 3))
            cov[:, 0, 0] = kin.sigma_u**2
            cov[:, 1, 1] = kin.sigma_v**2
            cov[:, 2, 2] = kin.sigma_w**2
            return mu, cov

        # --- Fallback: vectorised Monte Carlo ---
        n_mc = 100
        xi = np.repeat(x_arr, n_mc)
        yi = np.repeat(y_arr, n_mc)
        zi = np.repeat(z_arr, n_mc)
        vxi, vyi, vzi = kin.draw_random_velocity(xi, yi, zi)
        v_samples = np.stack([vxi, vyi, vzi], axis=1).reshape(n_pts, n_mc, 3)
        mu = np.mean(v_samples, axis=1)
        diff = v_samples - mu[:, None, :]
        cov = np.einsum("nij,nik->njk", diff, diff) / (n_mc - 1)
        return mu, cov

    # ---------------------------------------------------------------------------
    # Disk mixture moments (binned by magnitude, per pixel and distance)
    # ---------------------------------------------------------------------------

    def get_combined_disk_moments_binned(
        self,
        l_deg: np.ndarray,
        b_deg: np.ndarray,
        distances: np.ndarray,
        use_2d_extinction: bool = False,
        use_number: bool = True,
    ) -> tuple:
        """
        Compute LOS-integrated disk kinematic moments across pixels and distances.

        Combines all disk populations using density-weighted mixture moments
        (Law of Total Variance) in each magnitude bin and distance slice.

        Parameters
        ----------
        l_deg : array_like, shape (n_pix,)
            Galactic longitudes in degrees.
        b_deg : array_like, shape (n_pix,)
            Galactic latitudes in degrees.
        distances : np.ndarray, shape (n_dist,)
            Distance grid in kpc.
        use_2d_extinction : bool, optional
            Subtract 2-D asymptotic extinction before computing shifts.
        use_number : bool, optional
            Normalise densities by number (True) or mass (False).

        Returns
        -------
        mu_pm : np.ndarray, shape (n_pix, n_dist, n_bins, 2)
            Weighted mean proper motions in (l, b) — mas/yr.
        cov_pm : np.ndarray, shape (n_pix, n_dist, n_bins, 2, 2)
            Proper motion covariance matrices.
        mu_rv : np.ndarray, shape (n_pix, n_dist, n_bins)
            Weighted mean radial velocities — km/s.
        var_rv : np.ndarray, shape (n_pix, n_dist, n_bins)
            Radial velocity variance — (km/s)².
        sum_rho : np.ndarray, shape (n_pix, n_dist, n_bins)
            Total weighted density used for normalisation.
        """
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

        # Precompute extinction shifts for all disk populations at once.
        ext = self.populations_all[0].extinction
        _, ext_dict = ext.get_extinctions(
            L_grid.flatten(), B_grid.flatten(), D_grid.flatten()
        )
        a_v_all = ext_dict[self.band_name].reshape(n_pix, n_dist)
        dist_mod = 5 * np.log10(distances * 1000) - 5

        if use_2d_extinction:
            from surot_extinction import get_surot_A_Ks
            a_v_2d = get_surot_A_Ks(l_arr, b_arr)
            shift_all = dist_mod[None, :] + (a_v_all - a_v_2d[:, None])
        else:
            shift_all = dist_mod[None, :] + a_v_all

        # Solar motion and galactic tilt correction matrix H.
        v_sun = np.array([
            self.sp.parms.sun.u,
            self.sp.parms.sun.v,
            self.sp.parms.sun.w,
        ])
        theta = self.sp.parms.sun.theta
        ct, st = np.cos(theta), np.sin(theta)
        H = np.array([[ct, 0, -st], [0, 1, 0], [st, 0, ct]])

        mu_pm = np.zeros((n_pix, n_dist, self.n_bins, 2))
        cov_pm = np.zeros((n_pix, n_dist, self.n_bins, 2, 2))
        mu_rv = np.zeros((n_pix, n_dist, self.n_bins))
        var_rv = np.zeros((n_pix, n_dist, self.n_bins))
        sum_rho = np.zeros((n_pix, n_dist, self.n_bins))

        for pop in self.populations_disk:
            rho_raw = pop.population_density.density(r_sp, phi_sp, z_sp).reshape(
                n_pix, n_dist
            )
            if use_number:
                if pop.population_density.density_unit == "mass":
                    rho_raw /= self.mass_mag_maps[pop.name]["mean_mass"]
            else:
                if pop.population_density.density_unit == "number":
                    rho_raw *= self.mass_mag_maps[pop.name]["mean_mass"]

            S_bins = self.get_selection_fractions(
                pop.name, distances, l_arr, b_arr,
                use_number=use_number,
                use_2d_extinction=use_2d_extinction,
                shift_all=shift_all,
            )
            # Shape: (n_pix, n_dist, n_bins)
            rb_pop = (rho_raw[None, :, :] * S_bins).transpose(1, 2, 0)

            mu_uvw, cov_uvw = self.get_pop_moments_gc(pop, x_gc, y_gc, z_gc)
            mu_uvw = mu_uvw.reshape(n_pix, n_dist, 3)
            cov_uvw = cov_uvw.reshape(n_pix, n_dist, 3, 3)

            diff_v_t = np.einsum("ij,pnj->pni", H, mu_uvw - v_sun[None, None, :])
            cov_uvw_t = np.einsum("ij,pnjk,lk->pnil", H, cov_uvw, H)

            m_pm = np.einsum("pij,pnj->pni", R_lb, diff_v_t) * scale[None, :, None]
            c_pm = (
                np.einsum("pij,pnjk,plk->pnil", R_lb, cov_uvw_t, R_lb)
                * scale[None, :, None, None] ** 2
            )
            m_rv = np.einsum("pj,pnj->pn", R_los, diff_v_t)
            v_rv = np.einsum("pj,pnjk,pk->pn", R_los, cov_uvw_t, R_los)

            sum_rho += rb_pop
            mu_pm += m_pm[:, :, None, :] * rb_pop[:, :, :, None]
            mu_sq_pm = m_pm[:, :, None, :, None] * m_pm[:, :, None, None, :]
            cov_pm += (c_pm[:, :, None, :, :] + mu_sq_pm) * rb_pop[:, :, :, None, None]
            mu_rv += m_rv[:, :, None] * rb_pop
            var_rv += (v_rv[:, :, None] + m_rv[:, :, None] ** 2) * rb_pop

        valid = sum_rho > 1e-25
        mu_pm_out = np.zeros_like(mu_pm)
        cov_pm_out = np.zeros_like(cov_pm)
        mu_rv_out = np.zeros_like(mu_rv)
        var_rv_out = np.zeros_like(var_rv)

        mu_pm_out[valid] = mu_pm[valid] / sum_rho[valid, None]
        mu_rv_out[valid] = mu_rv[valid] / sum_rho[valid]
        mu_sq_out = mu_pm_out[:, :, :, :, None] * mu_pm_out[:, :, :, None, :]
        cov_pm_out[valid] = (
            cov_pm[valid] / sum_rho[valid, None, None]
        ) - mu_sq_out[valid]
        var_rv_out[valid] = var_rv[valid] / sum_rho[valid] - mu_rv_out[valid] ** 2
        cov_pm_out[~valid] = np.eye(2) * 10.0
        var_rv_out[~valid] = 100.0

        if n_pix == 1:
            return (
                mu_pm_out[0], cov_pm_out[0],
                mu_rv_out[0], var_rv_out[0],
                sum_rho[0],
            )
        return mu_pm_out, cov_pm_out, mu_rv_out, var_rv_out, sum_rho
