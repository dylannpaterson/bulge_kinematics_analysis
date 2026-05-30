import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../shared')))
from inversion_config import get_inverter_grid_axes


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def compute_parametric_bar(v0_str, y0_str, omega, X_bar, Y_bar):
    """Analytic parametric model in the bar rotating frame."""
    ux = v0_str * (1.0 - np.exp(-(Y_bar / y0_str) ** 2)) * np.sign(Y_bar)
    vx = ux + omega * Y_bar
    vy = -omega * X_bar
    return vx, vy


def compute_nonparametric_bar(grid_params, omega, axes_x, axes_y, axes_z, X_bar, Y_bar):
    """
    Reconstruct full 4-quadrant velocity field in the bar frame from the
    non-parametric grid, using the same bilinear interpolation and symmetry
    rules as jax_kinematic_inverter.
    Returns vx_bar_all, vy_bar_all of shape (n_dense, n_dense, NZ).
    """
    NX, NY, NZ = grid_params.shape[:3]
    n_dense = X_bar.shape[0]
    vx_all = np.zeros((n_dense, n_dense, NZ))
    vy_all = np.zeros((n_dense, n_dense, NZ))

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

        vx_all[:, :, iz] = ux + omega * Y_bar
        vy_all[:, :, iz] = uy - omega * X_bar

    return vx_all, vy_all


def mirror_z(V, axes_z):
    """Mirror a (nx, ny, NZ) array across z=0, returning (nx, ny, 2*NZ-1)."""
    return np.concatenate([V[:, :, ::-1][:, :, :-1], V], axis=2)


def project_planes(VX_full, VY_full):
    vx_xy = np.nanmean(VX_full, axis=2)
    vy_xy = np.nanmean(VY_full, axis=2)
    vx_xz = np.nanmean(VX_full, axis=1)
    vy_xz = np.nanmean(VY_full, axis=1)
    vx_yz = np.nanmean(VX_full, axis=0)
    vy_yz = np.nanmean(VY_full, axis=0)
    return vx_xy, vy_xy, vx_xz, vy_xz, vx_yz, vy_yz


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


def panel_xy(ax, X, Y, vx, vy, x_dense, y_dense, alpha_deg, skip, extra_label=''):
    v_mag = np.sqrt(vx**2 + vy**2)
    im = ax.pcolormesh(X, Y, v_mag, cmap=CMAP, shading='auto', vmin=0, vmax=VMAX, alpha=0.85)
    ax.streamplot(x_dense, y_dense, vx.T, vy.T,
                  color='#88ccff', linewidth=1.0, density=1.3,
                  arrowstyle='->', arrowsize=1.0)
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              vx[::skip, ::skip], vy[::skip, ::skip],
              color='white', scale=1200, width=0.003, headwidth=4)
    ax.axhline(0, color='#ff9944', linestyle='--', linewidth=1.2, alpha=0.6, label='Bar Major Axis')
    ax.axvline(0, color='#aaaacc', linestyle=':', linewidth=0.8, alpha=0.4)
    ax.plot(0, 0, 'wo', markersize=5)
    ax.set_aspect('equal')
    ax.legend(facecolor=DARK, edgecolor=GRID_C, labelcolor='white', loc='upper right', fontsize=8)
    return im


def panel_xz(ax, grid_x, grid_z, vx, axes_z_full, skip, skip_z):
    v_mag = np.sqrt(vx**2 + np.zeros_like(vx)**2)
    im = ax.pcolormesh(grid_x, grid_z, np.sqrt(vx**2), cmap=CMAP, shading='auto',
                       vmin=0, vmax=VMAX, alpha=0.85)
    ax.quiver(grid_x[::skip, ::skip_z], grid_z[::skip, ::skip_z],
              vx[::skip, ::skip_z], np.zeros_like(vx[::skip, ::skip_z]),
              color='white', scale=500, width=0.003, headwidth=4)
    ax.axhline(0, color='#aaaacc', linestyle=':', linewidth=0.8, alpha=0.5)
    return im


def panel_yz(ax, grid_y, grid_z, vy, skip, skip_z):
    im = ax.pcolormesh(grid_y, grid_z, np.sqrt(vy**2), cmap=CMAP, shading='auto',
                       vmin=0, vmax=VMAX, alpha=0.85)
    ax.quiver(grid_y[::skip, ::skip_z], grid_z[::skip, ::skip_z],
              vy[::skip, ::skip_z], np.zeros_like(vy[::skip, ::skip_z]),
              color='white', scale=300, width=0.003, headwidth=4)
    ax.axhline(0, color='#aaaacc', linestyle=':', linewidth=0.8, alpha=0.5)
    return im


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main():
    if os.environ.get("FAST_TEST") == "1":
        print("FAST_TEST mode active: skipping complex 2D contour plotting.")
        return
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
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

    print(f"Parametric: Omega={omega_p:.2f}, v0_str={v0_str:.2f}, y0_str={y0_str:.4f}")
    print(f"Non-Parametric: Omega={omega_np:.2f}, Bar Angle={alpha_deg:.2f} deg")

    x_max = float(axes_x[-1])
    y_max = float(axes_y[-1])
    z_lim = float(axes_z[-1])

    n_dense = 100
    x_dense = np.linspace(-x_max, x_max, n_dense)
    y_dense = np.linspace(-y_max, y_max, n_dense)
    X_bar, Y_bar = np.meshgrid(x_dense, y_dense, indexing='ij')

    axes_z_full = np.concatenate([-axes_z[::-1][:-1], axes_z])

    # --- Parametric velocities ---
    print("Computing parametric model velocities...")
    vx_p, vy_p = compute_parametric_bar(v0_str, y0_str, omega_p, X_bar, Y_bar)
    # Tile across z (parametric model doesn't depend on z for bulk velocity)
    NZ = len(axes_z)
    VXP = np.broadcast_to(vx_p[:, :, None], (n_dense, n_dense, NZ)).copy()
    VYP = np.broadcast_to(vy_p[:, :, None], (n_dense, n_dense, NZ)).copy()
    VXP_full = mirror_z(VXP, axes_z)
    VYP_full = mirror_z(VYP, axes_z)
    vx_xy_p, vy_xy_p, vx_xz_p, vy_xz_p, vx_yz_p, vy_yz_p = project_planes(VXP_full, VYP_full)

    # --- Non-parametric velocities ---
    print("Computing non-parametric model velocities (this may take a moment)...")
    vx_np_3d, vy_np_3d = compute_nonparametric_bar(
        grid_params, omega_np, axes_x, axes_y, axes_z, X_bar, Y_bar)
    VXN_full = mirror_z(vx_np_3d, axes_z)
    VYN_full = mirror_z(vy_np_3d, axes_z)
    vx_xy_n, vy_xy_n, vx_xz_n, vy_xz_n, vx_yz_n, vy_yz_n = project_planes(VXN_full, VYN_full)

    # --- Grid coords for XZ/YZ panels ---
    grid_x_xz, grid_z_xz = np.meshgrid(x_dense, axes_z_full, indexing='ij')
    grid_y_yz, grid_z_yz = np.meshgrid(y_dense, axes_z_full, indexing='ij')

    # ─── Plot ────────────────────────────────────────────
    print("Plotting 2-row comparison dashboard...")
    fig, axs = plt.subplots(2, 3, figsize=(22, 13), facecolor=DARK)
    fig.subplots_adjust(hspace=0.08, wspace=0.18, right=0.92)

    skip, skip_z = 8, 4
    row_labels = [f"Parametric  ($\\Omega={omega_p:.1f}$ km/s/kpc)",
                  f"Non-Parametric  ($\\Omega={omega_np:.1f}$ km/s/kpc)"]

    data_rows = [
        (vx_xy_p, vy_xy_p, vx_xz_p, vy_xz_p, vx_yz_p, vy_yz_p),
        (vx_xy_n, vy_xy_n, vx_xz_n, vy_xz_n, vx_yz_n, vy_yz_n),
    ]

    last_im = None
    for row, (label, (vxx, vyy, vxz, vyz_x, vxyz, vyyz)) in enumerate(zip(row_labels, data_rows)):
        for ax in axs[row]:
            style_ax(ax)

        # Col 0: XY bar plane
        im = panel_xy(axs[row, 0], X_bar, Y_bar, vxx, vyy,
                      x_dense, y_dense, alpha_deg, skip)
        axs[row, 0].set_xlim(-x_max, x_max)
        axs[row, 0].set_ylim(-y_max, y_max)
        axs[row, 0].set_xlabel(r'$x_{\rm bar}$ [kpc]', color='white')
        axs[row, 0].set_ylabel(r'$y_{\rm bar}$ [kpc]', color='white')

        # Col 1: XZ side
        im = panel_xz(axs[row, 1], grid_x_xz, grid_z_xz, vxz, axes_z_full, skip, skip_z)
        axs[row, 1].set_xlim(-x_max, x_max)
        axs[row, 1].set_ylim(-z_lim, z_lim)
        axs[row, 1].set_xlabel(r'$x_{\rm bar}$ [kpc]', color='white')
        axs[row, 1].set_ylabel(r'$z$ [kpc]', color='white')

        # Col 2: YZ front
        im = panel_yz(axs[row, 2], grid_y_yz, grid_z_yz, vyyz, skip, skip_z)
        axs[row, 2].set_xlim(-y_max, y_max)
        axs[row, 2].set_ylim(-z_lim, z_lim)
        axs[row, 2].set_xlabel(r'$y_{\rm bar}$ [kpc]', color='white')
        axs[row, 2].set_ylabel(r'$z$ [kpc]', color='white')

        last_im = im

        # Row label on left spine
        axs[row, 0].text(-0.22, 0.5, label, transform=axs[row, 0].transAxes,
                         color='white', fontsize=11, fontweight='bold',
                         va='center', ha='left', rotation=90)

    # Column titles (only on top row)
    axs[0, 0].set_title("Bar Plane View  (Z-Averaged)\n"
                         r"$|v|$ with streamlines", color='white', fontsize=12, fontweight='bold')
    axs[0, 1].set_title("Side Profile  ($y$-Averaged)\n"
                         r"Streaming $v_x$ along Bar Major Axis",
                         color='white', fontsize=12, fontweight='bold')
    axs[0, 2].set_title("Front Profile  ($x$-Averaged)\n"
                         r"Rotation $v_y$ $\perp$ Bar",
                         color='white', fontsize=12, fontweight='bold')

    # Shared colorbar
    cbar_ax = fig.add_axes([0.94, 0.12, 0.015, 0.76])
    cb = fig.colorbar(last_im, cax=cbar_ax)
    cb.set_label('Stellar Bulk Velocity Magnitude [km/s]', color='white', fontsize=10, labelpad=10)
    cb.ax.yaxis.set_tick_params(colors='white')

    plt.suptitle(
        r"Stellar Bulk Motion in the Rotating Bar Frame — Parametric vs Non-Parametric"
        "\n(H25C20 Fit, Bar angle = " + f"{alpha_deg:.1f}°)",
        color='white', fontsize=14, fontweight='bold', y=0.995
    )

    out_dir = os.path.join(project_root, "results/inversion")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'bulk_motion_bar_comparison.png')
    plt.savefig(out, dpi=200, facecolor=DARK)
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
