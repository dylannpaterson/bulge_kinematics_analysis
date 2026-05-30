import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax.tree_util import register_pytree_node_class
import numpy as np

@register_pytree_node_class
class BulgeKinematicInverter:
    def __init__(self, grid_axes, solar_params, n_bins=20, kl_weight=0.1, curv_weight=1.0):
        self.grid_axes = [jnp.array(ax) for ax in grid_axes]
        self.solar_params = solar_params
        self.shape = (len(grid_axes[0]), len(grid_axes[1]), len(grid_axes[2]))
        self.n_bins = n_bins
        self.kl_weight = kl_weight
        self.curv_weight = curv_weight

    def tree_flatten(self):
        children = (self.grid_axes, self.solar_params)
        aux_data = {'n_bins': self.n_bins, 'kl_weight': self.kl_weight, 'curv_weight': self.curv_weight}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def get_projection_matrix(self, l_deg, b_deg, alpha=None):
        if alpha is None:
            alpha = self.solar_params['bar_angle_rad']
        R0 = self.solar_params['R0']
        Z0 = self.solar_params.get('Z0', 0.0)
        
        # 1. Coordinate Trigonometry
        l_rad, b_rad = jnp.radians(l_deg), jnp.radians(b_deg)
        sl, cl = jnp.sin(l_rad), jnp.cos(l_rad)
        sb, cb = jnp.sin(b_rad), jnp.cos(b_rad)
        
        # 2. Tilt correction matrix H (Matches SynthPop coordinate_transformation.py)
        theta = jnp.arcsin(Z0 / jnp.sqrt(R0**2 + Z0**2))
        ct, st = jnp.cos(theta), jnp.sin(theta)
        # H = [[ct, 0, -st], [0, 1, 0], [st, 0, ct]]
        H = jnp.array([
            [ct, 0.0, -st],
            [0.0, 1.0, 0.0],
            [st, 0.0, ct]
        ])
        
        # 3. Basis matrix helio Cartesian -> lb (Bovy 2011)
        R_hel = jnp.array([
            [-sl, cl, 0.0],
            [-cl*sb, -sl*sb, cb]
        ])
        R_rad = jnp.array([cl*cb, sl*cb, sb])
        
        # 4. Rotation matrix Bar Frame -> Untilted Galactic Frame
        cos_a, sin_a = jnp.cos(alpha), jnp.sin(alpha)
        R_back = jnp.array([
            [cos_a, sin_a, 0.0],
            [-sin_a, cos_a, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # 5. Combined projection matrix: P = R_hel @ H @ R_back
        P_pm = R_hel @ H @ R_back
        P_rad = R_rad @ H @ R_back
        
        # 6. Heliocentric Correction (Sun Velocity in Untilted Frame)
        v_sun_gc = jnp.array([
            self.solar_params['U_SUN'], 
            self.solar_params['V_LSR'] + self.solar_params['V_SUN_REL'], 
            self.solar_params['W_SUN']
        ])
        
        # Offset is -R_hel @ H @ v_sun_gc
        v_off_pm = -R_hel @ H @ v_sun_gc
        v_off_rad = -R_rad @ H @ v_sun_gc
        
        return P_pm, P_rad, v_off_pm, v_off_rad

    def galactic_to_bar(self, l_deg, b_deg, d_kpc):
        R0 = self.solar_params['R0']
        Z0 = self.solar_params.get('Z0', 0.0)
        alpha = self.solar_params['bar_angle_rad']
        
        l_rad, b_rad = jnp.radians(l_deg), jnp.radians(b_deg)
        xh = d_kpc * jnp.cos(b_rad) * jnp.cos(l_rad)
        yh = d_kpc * jnp.cos(b_rad) * jnp.sin(l_rad)
        zh = d_kpc * jnp.sin(b_rad)
        
        # 1. Shift to GC (Untilted)
        hc = jnp.stack([xh - R0, yh, zh])
        
        # 2. APPLY TILT H_pos (Matches SynthPop dlb_to_xyz)
        # H_pos = [[ct, 0, st], [0, 1, 0], [-st, 0, ct]]
        theta = jnp.arcsin(Z0 / jnp.sqrt(R0**2 + Z0**2))
        ct, st = jnp.cos(theta), jnp.sin(theta)
        H_pos = jnp.array([
            [ct, 0.0, st],
            [0.0, 1.0, 0.0],
            [-st, 0.0, ct]
        ])
        xyz_gc = H_pos @ hc
        
        # 3. Rotate to Bar frame
        cos_a, sin_a = jnp.cos(alpha), jnp.sin(alpha)
        xb = xyz_gc[0] * cos_a - xyz_gc[1] * sin_a
        yb = xyz_gc[0] * sin_a + xyz_gc[1] * cos_a
        return jnp.stack([xb, yb, xyz_gc[2]])

    def apply_symmetries(self, x, y, z, p_oct):
        ux, uy, sxx, syy, szz, sxy = p_oct
        return jnp.stack([ux * jnp.sign(y), uy * jnp.sign(x), sxx, syy, szz, sxy * jnp.sign(x) * jnp.sign(y)])

    def interpolate_octant(self, x, y, z, grid_params):
        ix = (jnp.abs(x) - self.grid_axes[0][0]) / (self.grid_axes[0][1] - self.grid_axes[0][0])
        iy = (jnp.abs(y) - self.grid_axes[1][0]) / (self.grid_axes[1][1] - self.grid_axes[1][0])
        iz = (jnp.abs(z) - self.grid_axes[2][0]) / (self.grid_axes[2][1] - self.grid_axes[2][0])
        coords = jnp.stack([ix, iy, iz])
        def interp_field(data):
            return jax.scipy.ndimage.map_coordinates(data, coords.reshape(3,1), order=1, mode='nearest').squeeze()
        return vmap(interp_field, in_axes=-1)(grid_params)

    def integrate_mixed_bin(self, k, f_bulge, rb_sb, rd_sb, mu_b_pm_s, cov_b_pm_s, mu_b_rv_s, var_b_rv_s, mu_d_pm_sb, cov_d_pm_sb, mu_d_rv_sb, var_d_rv_sb, dist_sq, return_components=False):
        # Surgical optimization: Vectorized integration to avoid inner vmap over LOS points
        wb = f_bulge * rb_sb[..., k] * dist_sq
        wd = (1.0 - f_bulge) * rd_sb[..., k] * dist_sq
        wt = wb + wd
        W = jnp.sum(wt) + 1e-10
        
        # Proper Motion
        mu_d = mu_d_pm_sb[:, k]
        cov_d = cov_d_pm_sb[:, k]
        
        sum_mu = jnp.sum(mu_b_pm_s * wb[:, None] + mu_d * wd[:, None], axis=0)
        final_mu_pm = sum_mu / W
        
        sum_cov = jnp.sum(cov_b_pm_s * wb[:, None, None] + cov_d * wd[:, None, None], axis=0)
        sum_cov += jnp.einsum('ni,nj,n->ij', mu_b_pm_s, mu_b_pm_s, wb)
        sum_cov += jnp.einsum('ni,nj,n->ij', mu_d, mu_d, wd)
        final_cov_pm = sum_cov / W - jnp.outer(final_mu_pm, final_mu_pm)
        
        # Numerical stability: Force symmetry and positivity
        final_cov_pm = (final_cov_pm + final_cov_pm.T) / 2.0 + jnp.eye(2) * 1e-8
        
        # Radial Velocity
        mu_rv_d = mu_d_rv_sb[:, k]
        var_rv_d = var_d_rv_sb[:, k]
        
        sum_mu_rv = jnp.sum(mu_b_rv_s * wb + mu_rv_d * wd)
        final_mu_rv = sum_mu_rv / W
        
        sum_var_rv = jnp.sum((var_b_rv_s + mu_b_rv_s**2) * wb + (var_rv_d + mu_rv_d**2) * wd)
        final_var_rv = jnp.maximum(sum_var_rv / W - final_mu_rv**2, 1e-8)
        
        if return_components:
            sum_wb = jnp.sum(wb) + 1e-25
            sum_wd = jnp.sum(wd) + 1e-25
            mu_b_pm_bin = jnp.sum(mu_b_pm_s * wb[:, None], axis=0) / sum_wb
            mu_d_pm_bin = jnp.sum(mu_d_pm_sb[:, k] * wd[:, None], axis=0) / sum_wd
            f_bulge_bin = sum_wb / (sum_wb + sum_wd + 1e-25)
            return final_mu_pm, final_cov_pm, final_mu_rv, final_var_rv, mu_b_pm_bin, mu_d_pm_bin, f_bulge_bin

        return final_mu_pm, final_cov_pm, final_mu_rv, final_var_rv

    def predict_pixel(self, grid_params, omega, pixel_meta, log_f_bulge=0.0, return_components=False):
        f_bulge = jax.nn.sigmoid(log_f_bulge)
        xb, yb, zb = pixel_meta['x_bar'], pixel_meta['y_bar'], pixel_meta['z_bar']
        l_deg, b_deg, dist = pixel_meta['l'], pixel_meta['b'], pixel_meta['d']
        
        # Optimized: Move projection matrix out of LOS loop
        P_pm, P_rad, v_off_pm, v_off_rad = self.get_projection_matrix(l_deg, b_deg)
        
        def body(i):
            p_chol = self.interpolate_octant(xb[i], yb[i], zb[i], grid_params)
            ux, uy, log_L11, L21, log_L22, log_L33 = p_chol
            L11, L22, L33 = jnp.exp(log_L11), jnp.exp(log_L22), jnp.exp(log_L33)
            sxx, sxy, syy, szz = L11**2, L11*L21, L21**2 + L22**2, L33**2
            p_3d = self.apply_symmetries(xb[i], yb[i], zb[i], jnp.stack([ux, uy, sxx, syy, szz, sxy]))
            
            scale = 0.2108 / dist[i]
            # Add Pattern Speed Rotation dynamically: v_bar = v_stream + omega x r
            ux_total = p_3d[0] + omega * yb[i]
            uy_total = p_3d[1] - omega * xb[i]
            mu_b_pm = (P_pm @ jnp.stack([ux_total, uy_total, 0.0]) + v_off_pm) * scale
            c1 = jnp.stack([p_3d[2], p_3d[5], 0.0])
            c2 = jnp.stack([p_3d[5], p_3d[3], 0.0])
            c3 = jnp.stack([0.0, 0.0, p_3d[4]])
            cov_3d = jnp.stack([c1, c2, c3])
            cov_b_pm = (P_pm @ cov_3d @ P_pm.T) * (scale**2)
            mu_b_rv = P_rad @ jnp.stack([ux_total, uy_total, 0.0]) + v_off_rad
            var_b_rv = P_rad @ cov_3d @ P_rad.T
            return mu_b_pm, cov_b_pm, mu_b_rv, var_b_rv
            
        mu_b_pm_s, cov_b_pm_s, mu_b_rv_s, var_b_rv_s = vmap(body)(jnp.arange(len(xb)))
        rb_sb, rd_sb = jnp.atleast_2d(pixel_meta['rho_b']), jnp.atleast_2d(pixel_meta['rho_d'])
        mu_d_pm_sb = jnp.atleast_3d(pixel_meta['mu_d_pm'])
        cov_d_pm_sb = pixel_meta['cov_d_pm']
        mu_d_rv_sb = jnp.atleast_2d(pixel_meta.get('mu_d_rv', jnp.zeros((len(xb), self.n_bins))))
        var_d_rv_sb = jnp.atleast_2d(pixel_meta.get('var_d_rv', jnp.ones((len(xb), self.n_bins)) * 100.0))
        dist_sq = dist**2
        return vmap(lambda k: self.integrate_mixed_bin(k, f_bulge, rb_sb, rd_sb, mu_b_pm_s, cov_b_pm_s, mu_b_rv_s, var_b_rv_s, mu_d_pm_sb, cov_d_pm_sb, mu_d_rv_sb, var_d_rv_sb, dist_sq, return_components))(jnp.arange(self.n_bins))

    def predict_parametric(self, p_kosh, omega, alpha_rad, pixel_meta, log_f_bulge=0.0, return_components=False):
        f_bulge = jax.nn.sigmoid(log_f_bulge)
        v0_str, y0_str = p_kosh[0], p_kosh[1]
        sig_i0_x, sig_i1_x, sig_i0_y, sig_i1_y, sig_i0_z, sig_i1_z = p_kosh[2:8]
        h0_r = p_kosh[8:11]
        h0_z = p_kosh[11:14]
        C_par_r, C_perp_r = p_kosh[14], p_kosh[15]
        C_par_z, C_perp_z = p_kosh[16], p_kosh[17]
        
        dist = pixel_meta['d']
        l_deg, b_deg = pixel_meta['l'], pixel_meta['b']
        xb, yb, zb = pixel_meta['x_bar'], pixel_meta['y_bar'], pixel_meta['z_bar']
        
        # Optimized: Move projection matrix out of LOS loop
        P_pm, P_rad, v_off_pm, v_off_rad = self.get_projection_matrix(l_deg, b_deg, alpha_rad)
        
        ax, ay, az = jnp.abs(xb), jnp.abs(yb), jnp.abs(zb)
        s_r = (((ax/h0_r[0])**C_perp_r + (ay/h0_r[1])**C_perp_r)**(C_par_r/C_perp_r) + (az/h0_r[2])**C_par_r)**(1/C_par_r)
        s_z = (((ax/h0_z[0])**C_perp_z + (ay/h0_z[1])**C_perp_z)**(C_par_z/C_perp_z) + (az/h0_z[2])**C_par_z)**(1/C_par_z)
        
        ux_int = v0_str * (1.0 - jnp.exp(-(yb / y0_str)**2)) * jnp.sign(yb)
        v_bar_s = jnp.stack([ux_int + omega * yb, -omega * xb, jnp.zeros_like(xb)], axis=-1)
        
        sig_x = sig_i0_x + sig_i1_x * jnp.exp(-s_r)
        sig_y = sig_i0_y + sig_i1_y * jnp.exp(-s_r)
        sig_z = sig_i0_z + sig_i1_z * jnp.exp(-s_z)
        
        scale = 0.2108 / dist
        mu_b_pm_s = (v_bar_s @ P_pm.T + v_off_pm) * scale[:, None]
        mu_b_rv_s = v_bar_s @ P_rad + v_off_rad
        
        # Covariance projection
        c11 = (sig_x**2 * P_pm[0,0]**2 + sig_y**2 * P_pm[0,1]**2 + sig_z**2 * P_pm[0,2]**2) * scale**2
        c12 = (sig_x**2 * P_pm[0,0]*P_pm[1,0] + sig_y**2 * P_pm[0,1]*P_pm[1,1] + sig_z**2 * P_pm[0,2]*P_pm[1,2]) * scale**2
        c22 = (sig_x**2 * P_pm[1,0]**2 + sig_y**2 * P_pm[1,1]**2 + sig_z**2 * P_pm[1,2]**2) * scale**2
        cov_b_pm_s = jnp.stack([jnp.stack([c11, c12], axis=-1), jnp.stack([c12, c22], axis=-1)], axis=-2)
        
        var_b_rv_s = (sig_x**2 * P_rad[0]**2 + sig_y**2 * P_rad[1]**2 + sig_z**2 * P_rad[2]**2)
            
        rb_sb, rd_sb = jnp.atleast_2d(pixel_meta['rho_b']), jnp.atleast_2d(pixel_meta['rho_d'])
        mu_d_pm_sb = jnp.atleast_3d(pixel_meta['mu_d_pm'])
        cov_d_pm_sb = pixel_meta['cov_d_pm']
        mu_d_rv_sb = jnp.atleast_2d(pixel_meta.get('mu_d_rv', jnp.zeros((len(xb), self.n_bins))))
        var_d_rv_sb = jnp.atleast_2d(pixel_meta.get('var_d_rv', jnp.ones((len(xb), self.n_bins)) * 100.0))
        dist_sq = dist**2
        return vmap(lambda k: self.integrate_mixed_bin(k, f_bulge, rb_sb, rd_sb, mu_b_pm_s, cov_b_pm_s, mu_b_rv_s, var_b_rv_s, mu_d_pm_sb, cov_d_pm_sb, mu_d_rv_sb, var_d_rv_sb, dist_sq, return_components))(jnp.arange(self.n_bins))

    def parametric_loss(self, params, meta_batch, obs_mu_pm=None, obs_cov_pm=None, obs_mu_err_pm=None, obs_mu_rv=None, obs_var_rv=None, obs_mu_err_rv=None, rv_weight=1.0):
        if len(params) == 4:
            p_kosh, omega, alpha_deg, log_f_bulge = params
        else:
            p_kosh, omega, alpha_deg = params
            log_f_bulge = 0.0
            
        omega = jnp.squeeze(omega)
        alpha_deg = jnp.squeeze(alpha_deg)
        alpha_rad = jnp.radians(alpha_deg)
        spatial_shape = meta_batch['l'].shape
        spatial_ndim = len(spatial_shape)
        n_pixels = jnp.size(meta_batch['l'])
        
        def flatten_spatial(x):
            if x is None: return None
            return x.reshape((n_pixels,) + x.shape[spatial_ndim:])

        flat_meta = {k: (v if k == 'd' else flatten_spatial(v)) for k, v in meta_batch.items()}
        flat_obs_mu_pm = flatten_spatial(obs_mu_pm)
        flat_obs_cov_pm = flatten_spatial(obs_cov_pm)
        flat_obs_mu_rv = flatten_spatial(obs_mu_rv)
        flat_obs_var_rv = flatten_spatial(obs_var_rv)

        meta_axes = {k: (0 if (jnp.ndim(v) > 0 and v.shape[0] == n_pixels) else None) for k, v in flat_meta.items()}
        preds = vmap(lambda m: self.predict_parametric(p_kosh, omega, alpha_rad, m, log_f_bulge=log_f_bulge), in_axes=(meta_axes,))(flat_meta)
        pred_mu_pm, pred_cov_pm, pred_mu_rv, pred_var_rv = preds
        
        nll = 0.0
        if flat_obs_mu_pm is not None:
            diff_pm = flat_obs_mu_pm - pred_mu_pm
            def bin_loss_pm(i, k):
                C_pred = pred_cov_pm[i, k] + jnp.eye(2) * 1e-6
                inv_C = jnp.linalg.inv(C_pred)
                _, logdet = jnp.linalg.slogdet(C_pred)
                return 100.0 * (logdet + jnp.trace(inv_C @ flat_obs_cov_pm[i, k]) + diff_pm[i, k].T @ inv_C @ diff_pm[i, k])
            pixel_losses = vmap(lambda i: jnp.sum(vmap(lambda k: bin_loss_pm(i, k))(jnp.arange(self.n_bins))))(jnp.arange(n_pixels))
            nll += jnp.mean(pixel_losses)
            
        if flat_obs_mu_rv is not None:
            diff_rv = flat_obs_mu_rv - pred_mu_rv
            def bin_loss_rv(i, k):
                var_pred = pred_var_rv[i, k] + 1e-6
                return 100.0 * (jnp.log(var_pred) + (flat_obs_var_rv[i, k] + (diff_rv[i, k]**2)) / var_pred)
            pixel_losses = vmap(lambda i: jnp.sum(vmap(lambda k: bin_loss_rv(i, k))(jnp.arange(self.n_bins))))(jnp.arange(n_pixels))
            nll += rv_weight * jnp.mean(pixel_losses)
        return nll

    def kl_divergence_penalty(self, grid_params):
        X, Y, _ = jnp.meshgrid(self.grid_axes[0], self.grid_axes[1], self.grid_axes[2], indexing='ij')
        v_str_prior = 50.0 * (1.0 - jnp.exp(-(Y / 0.34)**2))
        R = jnp.sqrt(X**2 + Y**2)
        sigma_prior = 130.0 * jnp.exp(-R / 1.5) + 60.0
        ux, uy, log_L11, L21, log_L22, log_L33 = grid_params[..., 0], grid_params[..., 1], grid_params[..., 2], grid_params[..., 3], grid_params[..., 4], grid_params[..., 5]
        L11_sq, L22_sq, L33_sq, L21_sq = jnp.exp(2 * log_L11), jnp.exp(2 * log_L22), jnp.exp(2 * log_L33), L21**2
        tr_sigma = L11_sq + L21_sq + L22_sq + L33_sq
        mu_sq = (ux - v_str_prior)**2 + uy**2
        log_det_sigma = 2 * (log_L11 + log_L22 + log_L33)
        kl = 0.5 * ((tr_sigma + mu_sq) / (sigma_prior**2) - 3 + 6 * jnp.log(sigma_prior) - log_det_sigma)
        return jnp.mean(kl)

    def loss_function(self, params, meta_batch, obs_mu_pm=None, obs_cov_pm=None, obs_mu_rv=None, obs_var_rv=None, rv_weight=1.0):
        grid_params, omega = params
        omega = jnp.squeeze(omega)
        n_pixels = jnp.size(meta_batch['l'])
        
        def flatten_spatial(x):
            if x is None: return None
            return x.reshape((n_pixels,) + x.shape[len(meta_batch['l'].shape):])

        flat_meta = {k: (v if k == 'd' else flatten_spatial(v)) for k, v in meta_batch.items()}
        flat_obs_mu_pm = flatten_spatial(obs_mu_pm)
        flat_obs_cov_pm = flatten_spatial(obs_cov_pm)
        flat_obs_mu_rv = flatten_spatial(obs_mu_rv)
        flat_obs_var_rv = flatten_spatial(obs_var_rv)

        meta_axes = {k: (0 if (jnp.ndim(v) > 0 and v.shape[0] == n_pixels) else None) for k, v in flat_meta.items()}
        preds = vmap(lambda m: self.predict_pixel(grid_params, omega, m), in_axes=(meta_axes,))(flat_meta)
        pred_mu_pm, pred_cov_pm, pred_mu_rv, pred_var_rv = preds

        nll_pm = 0.0
        if flat_obs_mu_pm is not None:
            diff_pm = flat_obs_mu_pm - pred_mu_pm
            def bin_loss_pm(i, k):
                C_pred = pred_cov_pm[i, k] + jnp.eye(2) * 1e-6
                inv_C = jnp.linalg.inv(C_pred)
                _, logdet = jnp.linalg.slogdet(C_pred)
                return 100.0 * (logdet + jnp.trace(inv_C @ flat_obs_cov_pm[i, k]) + diff_pm[i, k].T @ inv_C @ diff_pm[i, k])
            nll_pm = jnp.mean(vmap(lambda i: jnp.sum(vmap(lambda k: bin_loss_pm(i, k))(jnp.arange(self.n_bins))))(jnp.arange(n_pixels)))
            
        nll_rv = 0.0
        if flat_obs_mu_rv is not None:
            diff_rv = flat_obs_mu_rv - pred_mu_rv
            def bin_loss_rv(i, k):
                var_pred = pred_var_rv[i, k] + 1e-6
                return 100.0 * (jnp.log(var_pred) + (flat_obs_var_rv[i, k] + (diff_rv[i, k]**2)) / var_pred)
            nll_rv = jnp.mean(vmap(lambda i: jnp.sum(vmap(lambda k: bin_loss_rv(i, k))(jnp.arange(self.n_bins))))(jnp.arange(n_pixels)))
        
        curv_reg = 0.0
        for p in range(6):
            field = grid_params[..., p]
            curv_reg += jnp.mean(jnp.diff(field, n=2, axis=0)**2) + jnp.mean(jnp.diff(field, n=2, axis=1)**2) + jnp.mean(jnp.diff(field, n=2, axis=2)**2)
            
        kl_p = self.kl_divergence_penalty(grid_params)
        omega_p = 0.5 * ((omega - 37.67) / 0.53)**2
        total_loss = nll_pm + rv_weight * nll_rv + self.curv_weight * curv_reg + self.kl_weight * kl_p + omega_p
        return total_loss, (nll_pm, nll_rv, self.kl_weight * kl_p, self.curv_weight * curv_reg, omega_p)

if __name__ == "__main__":
    print("Binned NLL Inverter Ready.")
