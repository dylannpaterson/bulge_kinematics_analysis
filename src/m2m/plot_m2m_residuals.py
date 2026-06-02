"""
Plot Made-to-Measure (M2M) mass and velocity residuals.

This script computes and visualizes the spatial and statistical residuals
between the optimized M2M orbit model and the Non-Parametric (NP) target
density and velocity fields.
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Define project root and append to system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(project_root, 'src/shared'))


# ---------------------------------------------------------------------------
# Plotting Utilities
# ---------------------------------------------------------------------------

def style_ax(ax: plt.Axes, face_color: str = '#111116', grid_color: str = '#444455') -> None:
    """
    Apply a premium dark style to a matplotlib axes object.

    Parameters
    ----------
    ax : plt.Axes
        The axes object to style.
    face_color : str, optional
        The background color of the plot area. Default is '#111116'.
    grid_color : str, optional
        The color of the grid lines. Default is '#444455'.
    """
    ax.set_facecolor(face_color)
    ax.grid(True, color=grid_color, linestyle=':', alpha=0.4)
    ax.tick_params(colors='white', which='both')
    for spine in ax.spines.values():
        spine.set_color(grid_color)
        spine.set_linewidth(0.8)


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Main routine to load data, compute residuals, and generate diagnostic plots.
    """
    if os.environ.get("FAST_TEST") == "1":
        print("FAST_TEST mode active: skipping complex M2M residuals plotting.")
        return

    fit_path = os.path.join(project_root, "results/m2m/np_m2m_fit_results.npz")
    if not os.path.exists(fit_path):
        print(f"Error: {fit_path} not found. Please run validate_np_with_m2m.py first.")
        return

    print("Loading M2M fit results...")
    data = np.load(fit_path)
    weights = data['weights']

    # Grid coordinates
    x_bins = data['x_bins']
    y_bins = data['y_bins']
    z_bins = data['z_bins']

    xc = 0.5 * (x_bins[1:] + x_bins[:-1])
    yc = 0.5 * (y_bins[1:] + y_bins[:-1])
    zc = 0.5 * (z_bins[1:] + z_bins[:-1])

    nx, ny, nz = len(xc), len(yc), len(zc)

    # Target constraints
    t_mass = data['target_mass']
    t_ux = data['target_ux']
    t_uy = data['target_uy']

    # Orbit matrices
    orb_mass = data['orb_mass']
    orb_ux = data['orb_ux']
    orb_uy = data['orb_uy']

    # Compute predictions
    print("Computing M2M model predictions...")
    m_pred = orb_mass.dot(weights).reshape((nx, ny, nz))

    epsilon = 1e-8
    ux_pred = (orb_ux.dot(weights) / (orb_mass.dot(weights) + epsilon)).reshape((nx, ny, nz))
    uy_pred = (orb_uy.dot(weights) / (orb_mass.dot(weights) + epsilon)).reshape((nx, ny, nz))

    # Projections (averages along specific dimensions)
    # Mass projections
    t_m_xy = np.nanmean(t_mass, axis=2)
    p_m_xy = np.nanmean(m_pred, axis=2)

    t_m_xz = np.nanmean(t_mass, axis=1)
    p_m_xz = np.nanmean(m_pred, axis=1)

    t_m_yz = np.nanmean(t_mass, axis=0)
    p_m_yz = np.nanmean(m_pred, axis=0)

    # Fractional mass residuals on projected planes
    res_m_xy = (p_m_xy - t_m_xy) / (t_m_xy + epsilon)
    res_m_xz = (p_m_xz - t_m_xz) / (t_m_xz + epsilon)
    res_m_yz = (p_m_yz - t_m_yz) / (t_m_yz + epsilon)

    # Mask very low density bins to prevent high-frequency noise at boundaries
    mask_xy = t_m_xy > 1e-5
    mask_xz = t_m_xz > 1e-5
    mask_yz = t_m_yz > 1e-5

    res_m_xy = np.where(mask_xy, res_m_xy, np.nan)
    res_m_xz = np.where(mask_xz, res_m_xz, np.nan)
    res_m_yz = np.where(mask_yz, res_m_yz, np.nan)

    # Velocity projections
    t_ux_xy = np.nanmean(t_ux, axis=2)
    t_uy_xy = np.nanmean(t_uy, axis=2)
    t_ux_xz = np.nanmean(t_ux, axis=1)
    t_uy_xz = np.nanmean(t_uy, axis=1)
    t_ux_yz = np.nanmean(t_ux, axis=0)
    t_uy_yz = np.nanmean(t_uy, axis=0)

    p_ux_xy = np.nanmean(ux_pred, axis=2)
    p_uy_xy = np.nanmean(uy_pred, axis=2)
    p_ux_xz = np.nanmean(ux_pred, axis=1)
    p_uy_xz = np.nanmean(uy_pred, axis=1)
    p_ux_yz = np.nanmean(ux_pred, axis=0)
    p_uy_yz = np.nanmean(uy_pred, axis=0)

    # Velocity residuals
    res_ux_xy = p_ux_xy - t_ux_xy
    res_uy_xy = p_uy_xy - t_uy_xy

    res_ux_xz = p_ux_xz - t_ux_xz
    res_uy_xz = p_uy_xz - t_uy_xz

    res_ux_yz = p_ux_yz - t_ux_yz
    res_uy_yz = p_uy_yz - t_uy_yz

    # ---------------------------------------------------------------------------
    # Plot 1: 3x3 Spatial Residual Maps
    # ---------------------------------------------------------------------------
    print("Generating spatial residuals dashboard...")
    DARK = '#111116'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'

    fig, axs = plt.subplots(3, 3, figsize=(18, 16), facecolor=DARK)
    fig.subplots_adjust(hspace=0.25, wspace=0.25, right=0.9)

    # Visual ranges (symmetric around 0)
    m_lim = 0.3  # +/- 30% relative density mismatch
    v_lim = 25.0  # +/- 25 km/s velocity mismatch

    # Row 1: Mass Residuals
    for ax in axs[0]:
        style_ax(ax)
    im0 = axs[0, 0].pcolormesh(xc, yc, res_m_xy.T, cmap='coolwarm', vmin=-m_lim, vmax=m_lim, shading='auto')
    axs[0, 0].set_title("XY Plane: Fractional Mass Residual", fontsize=11, fontweight='bold')
    axs[0, 0].set_ylabel("y (kpc)")

    axs[0, 1].pcolormesh(xc, zc, res_m_xz.T, cmap='coolwarm', vmin=-m_lim, vmax=m_lim, shading='auto')
    axs[0, 1].set_title("XZ Plane: Fractional Mass Residual", fontsize=11, fontweight='bold')
    axs[0, 1].set_ylabel("z (kpc)")

    axs[0, 2].pcolormesh(yc, zc, res_m_yz.T, cmap='coolwarm', vmin=-m_lim, vmax=m_lim, shading='auto')
    axs[0, 2].set_title("YZ Plane: Fractional Mass Residual", fontsize=11, fontweight='bold')
    axs[0, 2].set_ylabel("z (kpc)")

    # Row 2: ux Velocity Residuals
    for ax in axs[1]:
        style_ax(ax)
    im1 = axs[1, 0].pcolormesh(xc, yc, res_ux_xy.T, cmap='coolwarm', vmin=-v_lim, vmax=v_lim, shading='auto')
    axs[1, 0].set_title("XY Plane: ux Velocity Residual", fontsize=11, fontweight='bold')
    axs[1, 0].set_ylabel("y (kpc)")

    axs[1, 1].pcolormesh(xc, zc, res_ux_xz.T, cmap='coolwarm', vmin=-v_lim, vmax=v_lim, shading='auto')
    axs[1, 1].set_title("XZ Plane: ux Velocity Residual", fontsize=11, fontweight='bold')
    axs[1, 1].set_ylabel("z (kpc)")

    axs[1, 2].pcolormesh(yc, zc, res_ux_yz.T, cmap='coolwarm', vmin=-v_lim, vmax=v_lim, shading='auto')
    axs[1, 2].set_title("YZ Plane: ux Velocity Residual", fontsize=11, fontweight='bold')
    axs[1, 2].set_ylabel("z (kpc)")

    # Row 3: uy Velocity Residuals
    for ax in axs[2]:
        style_ax(ax)
    im2 = axs[2, 0].pcolormesh(xc, yc, res_uy_xy.T, cmap='coolwarm', vmin=-v_lim, vmax=v_lim, shading='auto')
    axs[2, 0].set_title("XY Plane: uy Velocity Residual", fontsize=11, fontweight='bold')
    axs[2, 0].set_xlabel("x (kpc)")
    axs[2, 0].set_ylabel("y (kpc)")

    axs[2, 1].pcolormesh(xc, zc, res_uy_xz.T, cmap='coolwarm', vmin=-v_lim, vmax=v_lim, shading='auto')
    axs[2, 1].set_title("XZ Plane: uy Velocity Residual", fontsize=11, fontweight='bold')
    axs[2, 1].set_xlabel("x (kpc)")
    axs[2, 1].set_ylabel("z (kpc)")

    axs[2, 2].pcolormesh(yc, zc, res_uy_yz.T, cmap='coolwarm', vmin=-v_lim, vmax=v_lim, shading='auto')
    axs[2, 2].set_title("YZ Plane: uy Velocity Residual", fontsize=11, fontweight='bold')
    axs[2, 2].set_xlabel("y (kpc)")
    axs[2, 2].set_ylabel("z (kpc)")

    # Add velocity residual quivers on the XY plane for physical directionality
    skip = 2
    axs[1, 0].quiver(
        xc[::skip], yc[::skip], res_ux_xy[::skip, ::skip].T, res_uy_xy[::skip, ::skip].T,
        color='white', scale=150, width=0.003, alpha=0.8
    )
    axs[2, 0].quiver(
        xc[::skip], yc[::skip], res_ux_xy[::skip, ::skip].T, res_uy_xy[::skip, ::skip].T,
        color='white', scale=150, width=0.003, alpha=0.8
    )

    # Colorbars on the right side
    cbar_ax_m = fig.add_axes([0.92, 0.68, 0.015, 0.2])
    cb_m = fig.colorbar(im0, cax=cbar_ax_m)
    cb_m.set_label('Fractional Mass Residual (M2M - Target) / Target', color='white', fontsize=9)
    cb_m.ax.yaxis.set_tick_params(colors='white')

    cbar_ax_vx = fig.add_axes([0.92, 0.39, 0.015, 0.2])
    cb_vx = fig.colorbar(im1, cax=cbar_ax_vx)
    cb_vx.set_label('ux Velocity Residual [km/s]', color='white', fontsize=9)
    cb_vx.ax.yaxis.set_tick_params(colors='white')

    cbar_ax_vy = fig.add_axes([0.92, 0.10, 0.015, 0.2])
    cb_vy = fig.colorbar(im2, cax=cbar_ax_vy)
    cb_vy.set_label('uy Velocity Residual [km/s]', color='white', fontsize=9)
    cb_vy.ax.yaxis.set_tick_params(colors='white')

    plt.suptitle("M2M vs Non-Parametric Model: Spatial Residual Diagnostics", fontsize=15, fontweight='bold', y=0.95)

    out_path_maps = os.path.join(project_root, "results/m2m/m2m_residuals_maps.png")
    plt.savefig(out_path_maps, dpi=200, facecolor=DARK)
    plt.close()
    print(f"Saved M2M residual maps to {out_path_maps}")

    # ---------------------------------------------------------------------------
    # Plot 2: 1x3 Residual Statistical Distributions
    # ---------------------------------------------------------------------------
    print("Generating statistical distributions...")
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), facecolor=DARK)
    fig.subplots_adjust(wspace=0.25, bottom=0.15, top=0.85)

    # Compute unrolled 3D residuals (for cells where target has significant mass)
    weight_mask = t_mass.flatten() > 1e5
    m_res_flat = ((m_pred - t_mass) / (t_mass + epsilon)).flatten()[weight_mask]
    ux_res_flat = (ux_pred - t_ux).flatten()[weight_mask]
    uy_res_flat = (uy_pred - t_uy).flatten()[weight_mask]

    # Panel 1: Fractional Mass Residual
    style_ax(axs[0])
    axs[0].hist(m_res_flat, bins=40, range=(-0.6, 0.6), color='#4682B4', alpha=0.85, edgecolor=DARK)
    med_m = np.nanmedian(m_res_flat)
    std_m = np.nanstd(m_res_flat)
    axs[0].axvline(0.0, color='gray', linestyle='--', alpha=0.7)
    axs[0].axvline(med_m, color='orange', linestyle='-', linewidth=1.5, label=f'Median: {med_m:+.3f}')
    axs[0].set_title("Fractional Mass Residual", fontsize=11, fontweight='bold')
    axs[0].set_xlabel("(M2M - Target) / Target")
    axs[0].set_ylabel("Number of Grid Bins")
    axs[0].legend(facecolor=DARK, edgecolor='#444455', labelcolor='white')
    axs[0].text(
        0.05, 0.95, f"Mean: {np.nanmean(m_res_flat):+.3f}\nStd Dev: {std_m:.3f}",
        transform=axs[0].transAxes, color='white', fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor=DARK, alpha=0.6, edgecolor='#444455')
    )

    # Panel 2: ux Velocity Residual
    style_ax(axs[1])
    axs[1].hist(ux_res_flat, bins=40, range=(-40, 40), color='#2E8B57', alpha=0.85, edgecolor=DARK)
    mean_ux = np.nanmean(ux_res_flat)
    std_ux = np.nanstd(ux_res_flat)
    axs[1].axvline(0.0, color='gray', linestyle='--', alpha=0.7)
    axs[1].axvline(mean_ux, color='orange', linestyle='-', linewidth=1.5, label=f'Mean: {mean_ux:+.1f} km/s')
    axs[1].set_title("ux Velocity Residual", fontsize=11, fontweight='bold')
    axs[1].set_xlabel("ux_M2M - ux_Target [km/s]")
    axs[1].set_ylabel("Number of Grid Bins")
    axs[1].legend(facecolor=DARK, edgecolor='#444455', labelcolor='white')
    axs[1].text(
        0.05, 0.95, f"Median: {np.nanmedian(ux_res_flat):+.1f} km/s\nStd Dev: {std_ux:.1f} km/s",
        transform=axs[1].transAxes, color='white', fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor=DARK, alpha=0.6, edgecolor='#444455')
    )

    # Panel 3: uy Velocity Residual
    style_ax(axs[2])
    axs[2].hist(uy_res_flat, bins=40, range=(-40, 40), color='#CD5C5C', alpha=0.85, edgecolor=DARK)
    mean_uy = np.nanmean(uy_res_flat)
    std_uy = np.nanstd(uy_res_flat)
    axs[2].axvline(0.0, color='gray', linestyle='--', alpha=0.7)
    axs[2].axvline(mean_uy, color='orange', linestyle='-', linewidth=1.5, label=f'Mean: {mean_uy:+.1f} km/s')
    axs[2].set_title("uy Velocity Residual", fontsize=11, fontweight='bold')
    axs[2].set_xlabel("uy_M2M - uy_Target [km/s]")
    axs[2].set_ylabel("Number of Grid Bins")
    axs[2].legend(facecolor=DARK, edgecolor='#444455', labelcolor='white')
    axs[2].text(
        0.05, 0.95, f"Median: {np.nanmedian(uy_res_flat):+.1f} km/s\nStd Dev: {std_uy:.1f} km/s",
        transform=axs[2].transAxes, color='white', fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor=DARK, alpha=0.6, edgecolor='#444455')
    )

    plt.suptitle("M2M vs Non-Parametric Model: Residual Statistical Distributions", fontsize=15, fontweight='bold', y=0.98)

    out_path_dist = os.path.join(project_root, "results/m2m/m2m_residuals_distributions.png")
    plt.savefig(out_path_dist, dpi=200, facecolor=DARK)
    plt.close()
    print(f"Saved M2M residual distributions to {out_path_dist}")


if __name__ == "__main__":
    main()
