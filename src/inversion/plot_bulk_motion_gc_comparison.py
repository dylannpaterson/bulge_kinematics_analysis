import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os
import sys

# Limit CPU threading
os.environ["OMP_NUM_THREADS"] = "1"

# Set up project root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

sys.path.append(os.path.join(project_root, 'src/shared'))
from inversion_config import get_inverter_grid_axes


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def compute_parametric_bar(v0_str, y0_str, omega, X_bar, Y_bar):
    ux = v0_str * (1.0 - np.exp(-(Y_bar / y0_str) ** 2)) * np.sign(Y_bar)
    vx = ux + omega * Y_bar
    vy = -omega * X_bar
    return vx, vy


def bar_to_gc(vx_bar, vy_bar, alpha_rad):
    """Rotate bar-frame velocities into GC frame (inverse / transpose rotation)."""
    vx_gc = vx_bar * np.cos(alpha_rad) + vy_bar * np.sin(alpha_rad)
    vy_gc = -vx_bar * np.sin(alpha_rad) + vy_bar * np.cos(alpha_rad)
    return vx_gc, vy_gc


def compute_parametric_gc(v0_str, y0_str, omega, alpha_rad, X_bar, Y_bar):
    vx_b, vy_b = compute_parametric_bar(v0_str, y0_str, omega, X_bar, Y_bar)
    return bar_to_gc(vx_b, vy_b, alpha_rad)


def compute_nonparametric_gc(grid_params, omega, alpha_rad, axes_x, axes_y, axes_z, X_bar, Y_bar):
    NX, NY, NZ = grid_params.shape[:3]
    n_dense = X_bar.shape[0]
    vx_gc_all = np.zeros((n_dense, n_dense, NZ))
    vy_gc_all = np.zeros((n_dense, n_dense, NZ))

    for iz in range(NZ):
        ux = np.zeros((n_dense, n_dense))
        uy = np.zeros((n_dense, n_dense))
        for i in range(n_dense):
            for j in range(n_dense):
                x_val, y_val = X_bar[i, j], Y_bar[i, j]
                xo, yo = np.abs(x_val), np.abs(y_val)
                ix = np.clip(np.searchsorted(axes_x, xo) - 1, 0, NX - 2)
                iy = np.clip(np.searchsorted(axes_y, yo) - 1, 0, NY - 2)
                tx = (xo - axes_x[ix]) / (axes_x[ix+1] - axes_x[ix] + 1e-10)
                ty = (yo - axes_y[iy]) / (axes_y[iy+1] - axes_y[iy] + 1e-10)

                ux_oct = ((1-tx)*(1-ty)*grid_params[ix,   iy,   iz, 0]
                        +    tx *(1-ty)*grid_params[ix+1, iy,   iz, 0]
                        + (1-tx)*   ty *grid_params[ix,   iy+1, iz, 0]
                        +    tx *   ty *grid_params[ix+1, iy+1, iz, 0])
                uy_oct = ((1-tx)*(1-ty)*grid_params[ix,   iy,   iz, 1]
                        +    tx *(1-ty)*grid_params[ix+1, iy,   iz, 1]
                        + (1-tx)*   ty *grid_params[ix,   iy+1, iz, 1]
                        +    tx *   ty *grid_params[ix+1, iy+1, iz, 1])

                ux[i, j] = ux_oct * np.sign(y_val)
                uy[i, j] = uy_oct * np.sign(x_val)

        vx_total_bar = ux + omega * Y_bar
        vy_total_bar = vy + omega * X_bar # Wait, is this -omega or +omega? Let's check original. In original it was: vy_total_bar = uy - omega * X_bar. Let's make sure it matches original line 65!
        # Ah, yes! In original line 65: vy_total_bar = uy - omega * X_bar.
        vy_total_bar = uy - omega * X_bar
        vx_gc_all[:, :, iz], vy_gc_all[:, :, iz] = bar_to_gc(vx_total_bar, vy_total_bar, alpha_rad)

    return vx_gc_all, vy_gc_all


def mirror_z(V):
    return np.concatenate([V[:, :, ::-1][:, :, :-1], V], axis=2)


def project_planes(VX, VY):
    return (np.nanmean(VX, axis=2), np.nanmean(VY, axis=2),
            np.nanmean(VX, axis=1), np.nanmean(VY, axis=1),
            np.nanmean(VX, axis=0), np.nanmean(VY, axis=0))


def interpolate_to_gc_grid(vx_gc_all, vy_gc_all, X_gc, Y_gc, xi, yi, NZ):
    n_gc = len(xi)
    VXI = np.zeros((n_gc, n_gc, NZ))
    VYI = np.zeros((n_gc, n_gc, NZ))
    pts = np.stack([X_gc.flatten(), Y_gc.flatten()], axis=-1)
    grid_xi, grid_yi = np.meshgrid(xi, yi, indexing='ij')
    for k in range(NZ):
        VXI[:, :, k] = griddata(pts, vx_gc_all[:, :, k].flatten(), (grid_xi, grid_yi), method='cubic')
        VYI[:, :, k] = griddata(pts, vy_gc_all[:, :, k].flatten(), (grid_xi, grid_yi), method='cubic')
    return VXI, VYI, grid_xi, grid_yi


# ─────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────

VMAX = 220
CMAP = 'inferno'
DARK = '#111116'
GRID_C = '#444455'


def style_ax(ax):
    ax.set_facecolor(DARK)
    ax.tick_params(colors='white')
    ax.grid(True, color=GRID_C, linestyle=':', alpha=0.4)


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main():
    if os.environ.get("FAST_TEST") == "1":
        print("FAST_TEST mode active: skipping complex 2D contour plotting.")
        return
    fit_path   = os.path.join(project_root, "results/inversion/fit_results_real_combined_h25c20.npz")
    param_path = os.path.join(project_root, "results/inversion/parametric_fit_results_real_h25c20.npz")

    if not os.path.exists(fit_path) or not os.path.exists(param_path):
        print("Required fit files not found.")
        return

    print("Loading fit results...")
    fit_res     = np.load(fit_path)
    grid_params = fit_res['grid']
    omega_np    = float(fit_res['omega'])
    axes_x      = fit_res['axes_x']
    axes_y      = fit_res['axes_y']
    axes_z      = fit_res['axes_z']

    param_data  = np.load(param_path)
    params      = param_data['params']
    names       = [n.decode() if isinstance(n, bytes) else str(n) for n in param_data['names']]
    v0_str      = float(params[names.index('v0_str')])
    y0_str      = float(params[names.index('y0_str')])
    omega_p     = float(params[names.index('omega')])
    alpha_deg   = float(params[names.index('alpha')])
    alpha_rad   = np.radians(alpha_deg)

    print(f"Parametric:     Omega={omega_p:.2f}, v0_str={v0_str:.2f}")
    print(f"Non-Parametric: Omega={omega_np:.2f}, alpha={alpha_deg:.2f} deg")

    x_max = float(axes_x[-1])
    y_max = float(axes_y[-1])
    z_lim = float(axes_z[-1])
    NZ    = len(axes_z)

    # Dense bar-frame grid (source for both models)
    n_dense = 100
    x_dense = np.linspace(-x_max, x_max, n_dense)
    y_dense = np.linspace(-y_max, y_max, n_dense)
    X_bar, Y_bar = np.meshgrid(x_dense, y_dense, indexing='ij')

    # GC uniform target grid
    gc_lim = 4.5
    n_gc   = 100
    xi = np.linspace(-gc_lim, gc_lim, n_gc)
    yi = np.linspace(-gc_lim, gc_lim, n_gc)

    # GC coords of bar-frame source points (same for both models)
    X_gc = X_bar * np.cos(alpha_rad) + Y_bar * np.sin(alpha_rad)
    Y_gc = -X_bar * np.sin(alpha_rad) + Y_bar * np.cos(alpha_rad)

    axes_z_full = np.concatenate([-axes_z[::-1][:-1], axes_z])

    # ── Parametric GC field ──────────────────────────────
    print("Computing parametric GC velocities...")
    vx_p, vy_p = compute_parametric_gc(v0_str, y0_str, omega_p, alpha_rad, X_bar, Y_bar)
    VXP = np.broadcast_to(vx_p[:, :, None], (n_dense, n_dense, NZ)).copy()
    VYP = np.broadcast_to(vy_p[:, :, None], (n_dense, n_dense, NZ)).copy()
    VXI_p, VYI_p, grid_xi, grid_yi = interpolate_to_gc_grid(VXP, VYP, X_gc, Y_gc, xi, yi, NZ)
    VXI_p_full = mirror_z(VXI_p)
    VYI_p_full = mirror_z(VYI_p)
    vx_xy_p, vy_xy_p, vx_xz_p, _, vx_yz_p, vy_yz_p = project_planes(VXI_p_full, VYI_p_full)

    # ── Non-parametric GC field ──────────────────────────
    print("Computing non-parametric GC velocities (interpolation in progress)...")
    vx_n_bar, vy_n_bar = compute_nonparametric_gc(
        grid_params, omega_np, alpha_rad, axes_x, axes_y, axes_z, X_bar, Y_bar)
    VXI_n, VYI_n, _, _ = interpolate_to_gc_grid(vx_n_bar, vy_n_bar, X_gc, Y_gc, xi, yi, NZ)
    VXI_n_full = mirror_z(VXI_n)
    VYI_n_full = mirror_z(VYI_n)
    vx_xy_n, vy_xy_n, vx_xz_n, _, vx_yz_n, vy_yz_n = project_planes(VXI_n_full, VYI_n_full)

    # Grid coords for XZ/YZ panels
    grid_x_xz, grid_z_xz = np.meshgrid(xi, axes_z_full, indexing='ij')
    grid_y_yz, grid_z_yz = np.meshgrid(yi, axes_z_full, indexing='ij')

    # ── 2-row plot ────────────────────────────────────────
    print("Plotting GC-frame 2-row comparison dashboard...")
    fig, axs = plt.subplots(2, 3, figsize=(22, 13), facecolor=DARK)
    fig.subplots_adjust(hspace=0.08, wspace=0.18, right=0.92)

    skip, skip_z = 8, 4

    row_data = [
        (f"Parametric  ($\\Omega={omega_p:.1f}$ km/s/kpc)",
         vx_xy_p, vy_xy_p, vx_xz_p, vx_yz_p, vy_yz_p),
        (f"Non-Parametric  ($\\Omega={omega_np:.1f}$ km/s/kpc)",
         vx_xy_n, vy_xy_n, vx_xz_n, vx_yz_n, vy_yz_n),
    ]

    last_im = None
    for row, (label, vx_xy, vy_xy, vx_xz, vx_yz, vy_yz) in enumerate(row_data):
        for ax in axs[row]:
            style_ax(ax)

        # --- XY ---
        ax = axs[row, 0]
        v_mag = np.sqrt(vx_xy**2 + vy_xy**2)
        im = ax.pcolormesh(grid_xi, grid_yi, v_mag, cmap=CMAP, shading='auto',
                           vmin=0, vmax=VMAX, alpha=0.85)
        ax.streamplot(xi, yi, vx_xy.T, vy_xy.T, color='#88ccff',
                      linewidth=1.0, density=1.3, arrowstyle='->', arrowsize=1.0)
        ax.quiver(grid_xi[::skip, ::skip], grid_yi[::skip, ::skip],
                  vx_xy[::skip, ::skip], vy_xy[::skip, ::skip],
                  color='white', scale=1200, width=0.003, headwidth=4)
        # Bar major axis in GC frame
        r_bar = np.linspace(-3.5, 3.5, 100)
        ax.plot(r_bar * np.cos(alpha_rad), -r_bar * np.sin(alpha_rad),
                color='#ff9944', linestyle='--', linewidth=1.3, alpha=0.7,
                label=f'Bar ({alpha_deg:.1f}°)')
        ax.plot(0, 0, 'wo', markersize=5)
        ax.set_xlim(-gc_lim, gc_lim)
        ax.set_ylim(-gc_lim, gc_lim)
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x_{\rm GC}$ [kpc]', color='white')
        ax.set_ylabel(r'$y_{\rm GC}$ [kpc]', color='white')
        ax.legend(facecolor=DARK, edgecolor=GRID_C, labelcolor='white',
                  loc='upper right', fontsize=8)

        # --- XZ ---
        ax = axs[row, 1]
        im = ax.pcolormesh(grid_x_xz, grid_z_xz, np.sqrt(vx_xz**2),
                           cmap=CMAP, shading='auto', vmin=0, vmax=VMAX, alpha=0.85)
        ax.quiver(grid_x_xz[::skip, ::skip_z], grid_z_xz[::skip, ::skip_z],
                  vx_xz[::skip, ::skip_z], np.zeros_like(vx_xz[::skip, ::skip_z]),
                  color='white', scale=500, width=0.003, headwidth=4)
        ax.axhline(0, color='#aaaacc', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.set_xlim(-gc_lim, gc_lim)
        ax.set_ylim(-z_lim, z_lim)
        ax.set_xlabel(r'$x_{\rm GC}$ [kpc]', color='white')
        ax.set_ylabel(r'$z_{\rm GC}$ [kpc]', color='white')

        # --- YZ ---
        ax = axs[row, 2]
        im = ax.pcolormesh(grid_y_yz, grid_z_yz, np.sqrt(vy_yz**2),
                           cmap=CMAP, shading='auto', vmin=0, vmax=VMAX, alpha=0.85)
        ax.quiver(grid_y_yz[::skip, ::skip_z], grid_z_yz[::skip, ::skip_z],
                  vy_yz[::skip, ::skip_z], np.zeros_like(vy_yz[::skip, ::skip_z]),
                  color='white', scale=300, width=0.003, headwidth=4)
        ax.axhline(0, color='#aaaacc', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.set_xlim(-gc_lim, gc_lim)
        ax.set_ylim(-z_lim, z_lim)
        ax.set_xlabel(r'$y_{\rm GC}$ [kpc]', color='white')
        ax.set_ylabel(r'$z_{\rm GC}$ [kpc]', color='white')

        last_im = im
        axs[row, 0].text(-0.22, 0.5, label, transform=axs[row, 0].transAxes,
                         color='white', fontsize=11, fontweight='bold',
                         va='center', ha='left', rotation=90)

    # Column titles
    axs[0, 0].set_title("Galactic Plane View  (Z-Averaged)\n"
                         r"$|v|$ with streamlines & bar axis",
                         color='white', fontsize=12, fontweight='bold')
    axs[0, 1].set_title("Side Profile  ($y_{\\ rm GC}$-Averaged)\n"
                         r"$v_{x_{\rm GC}}$ streaming component",
                         color='white', fontsize=12, fontweight='bold')
    axs[0, 2].set_title("Front Profile  ($x_{\\ rm GC}$-Averaged)\n"
                         r"$v_{y_{\rm GC}}$ rotational component",
                         color='white', fontsize=12, fontweight='bold')

    # Shared colorbar
    cbar_ax = fig.add_axes([0.94, 0.12, 0.015, 0.76])
    cb = fig.colorbar(last_im, cax=cbar_ax)
    cb.set_label('Stellar Bulk Velocity Magnitude [km/s]', color='white', fontsize=10, labelpad=10)
    cb.ax.yaxis.set_tick_params(colors='white')

    plt.suptitle(
        r"Stellar Bulk Motion in the GC Frame — Parametric vs Non-Parametric"
        "\n(H25C20 Fit, Bar angle = " + f"{alpha_deg:.1f}°)",
        color='white', fontsize=14, fontweight='bold', y=0.995
    )

    out_dir = os.path.join(project_root, "results/inversion")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'bulk_motion_gc_comparison.png')
    plt.savefig(out, dpi=200, facecolor=DARK)
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
