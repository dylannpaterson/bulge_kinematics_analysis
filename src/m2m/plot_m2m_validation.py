import os
import sys
import numpy as np
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(project_root, 'src/shared'))

# ─────────────────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────────────────

def main():
    if os.environ.get("FAST_TEST") == "1":
        print("FAST_TEST mode active: skipping complex M2M validation plotting.")
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
    t_sxx = data['target_sxx']
    t_syy = data['target_syy']
    t_szz = data['target_szz']
    
    # Orbit matrices
    orb_mass = data['orb_mass']
    orb_ux = data['orb_ux']
    orb_uy = data['orb_uy']
    orb_vxx = data['orb_vxx']
    orb_vyy = data['orb_vyy']
    orb_vzz = data['orb_vzz']
    
    # Compute predictions
    print("Computing M2M model predictions...")
    m_pred = orb_mass.dot(weights).reshape((nx, ny, nz))
    
    m_pred_flat = m_pred.flatten()
    epsilon = 1e-8
    
    ux_pred = (orb_ux.dot(weights) / (orb_mass.dot(weights) + epsilon)).reshape((nx, ny, nz))
    uy_pred = (orb_uy.dot(weights) / (orb_mass.dot(weights) + epsilon)).reshape((nx, ny, nz))
    vxx_pred = (orb_vxx.dot(weights) / (orb_mass.dot(weights) + epsilon)).reshape((nx, ny, nz))
    vyy_pred = (orb_vyy.dot(weights) / (orb_mass.dot(weights) + epsilon)).reshape((nx, ny, nz))
    vzz_pred = (orb_vzz.dot(weights) / (orb_mass.dot(weights) + epsilon)).reshape((nx, ny, nz))
    
    sxx_pred = np.maximum(vxx_pred - ux_pred**2, 1e-4)
    syy_pred = np.maximum(vyy_pred - uy_pred**2, 1e-4)
    szz_pred = np.maximum(vzz_pred, 1e-4)
    
    # ─────────────────────────────────────────────────────────
    # Plotting (Premium Dark Theme)
    # ─────────────────────────────────────────────────────────
    DARK = '#111116'
    GRID_C = '#444455'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    
    def style_ax(ax):
        ax.set_facecolor(DARK)
        ax.grid(True, color=GRID_C, linestyle=':', alpha=0.4)
        
    print("Creating visualization plots...")
    
    # ─────────────────────────────────────────────────────────
    # 1. Kinematics Comparison Dashboard (Mean Velocities)
    # ─────────────────────────────────────────────────────────
    fig, axs = plt.subplots(2, 3, figsize=(18, 11), facecolor=DARK)
    fig.subplots_adjust(hspace=0.22, wspace=0.2)
    
    # Slices/Projections (average over Z for XY, Y for XZ, X for YZ)
    t_ux_xy = np.nanmean(t_ux, axis=2)
    t_uy_xy = np.nanmean(t_uy, axis=2)
    t_v_xy = np.sqrt(t_ux_xy**2 + t_uy_xy**2)
    
    p_ux_xy = np.nanmean(ux_pred, axis=2)
    p_uy_xy = np.nanmean(uy_pred, axis=2)
    p_v_xy = np.sqrt(p_ux_xy**2 + p_uy_xy**2)
    
    t_ux_xz = np.nanmean(t_ux, axis=1)
    p_ux_xz = np.nanmean(ux_pred, axis=1)
    
    t_uy_yz = np.nanmean(t_uy, axis=0)
    p_uy_yz = np.nanmean(uy_pred, axis=0)
    
    v_max = 200.0
    cmap = 'inferno'
    
    # Row 1: Target (NP Model)
    for ax in axs[0]: style_ax(ax)
    im0 = axs[0, 0].pcolormesh(xc, yc, t_v_xy.T, cmap=cmap, vmin=0, vmax=v_max, shading='auto')
    axs[0, 0].quiver(xc[::2], yc[::2], t_ux_xy[::2, ::2].T, t_uy_xy[::2, ::2].T, color='white', scale=1000, width=0.004)
    axs[0, 0].set_title("XY Plane: Target Bulk Velocity", fontsize=11, fontweight='bold')
    axs[0, 0].set_ylabel("y (kpc)")
    
    im1 = axs[0, 1].pcolormesh(xc, zc, np.abs(t_ux_xz).T, cmap=cmap, vmin=0, vmax=v_max, shading='auto')
    axs[0, 1].set_title("XZ Plane: Target |ux|", fontsize=11, fontweight='bold')
    axs[0, 1].set_ylabel("z (kpc)")
    
    im2 = axs[0, 2].pcolormesh(yc, zc, np.abs(t_uy_yz).T, cmap=cmap, vmin=0, vmax=v_max, shading='auto')
    axs[0, 2].set_title("YZ Plane: Target |uy|", fontsize=11, fontweight='bold')
    axs[0, 2].set_ylabel("z (kpc)")
    
    # Row 2: Model (M2M Fit)
    for ax in axs[1]: style_ax(ax)
    axs[1, 0].pcolormesh(xc, yc, p_v_xy.T, cmap=cmap, vmin=0, vmax=v_max, shading='auto')
    axs[1, 0].quiver(xc[::2], yc[::2], p_ux_xy[::2, ::2].T, p_uy_xy[::2, ::2].T, color='white', scale=1000, width=0.004)
    axs[1, 0].set_title("XY Plane: M2M Bulk Velocity", fontsize=11, fontweight='bold')
    axs[1, 0].set_xlabel("x (kpc)")
    axs[1, 0].set_ylabel("y (kpc)")
    
    axs[1, 1].pcolormesh(xc, zc, np.abs(p_ux_xz).T, cmap=cmap, vmin=0, vmax=v_max, shading='auto')
    axs[1, 1].set_title("XZ Plane: M2M |ux|", fontsize=11, fontweight='bold')
    axs[1, 1].set_xlabel("x (kpc)")
    axs[1, 1].set_ylabel("z (kpc)")
    
    axs[1, 2].pcolormesh(yc, zc, np.abs(p_uy_yz).T, cmap=cmap, vmin=0, vmax=v_max, shading='auto')
    axs[1, 2].set_title("YZ Plane: M2M |uy|", fontsize=11, fontweight='bold')
    axs[1, 2].set_xlabel("y (kpc)")
    axs[1, 2].set_ylabel("z (kpc)")
    
    # Label Row Headers on the left
    axs[0, 0].text(-0.25, 0.5, "NP Model (Target)", transform=axs[0, 0].transAxes,
                 color='white', fontsize=12, fontweight='bold', va='center', ha='left', rotation=90)
    axs[1, 0].text(-0.25, 0.5, "M2M Orbit Model", transform=axs[1, 0].transAxes,
                 color='white', fontsize=12, fontweight='bold', va='center', ha='left', rotation=90)
                 
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cb = fig.colorbar(im0, cax=cbar_ax)
    cb.set_label('Mean Velocity Scale [km/s]', color='white', fontsize=10)
    cb.ax.yaxis.set_tick_params(colors='white')
    
    plt.suptitle("Galactic Bulge Kinematics Validation: Non-Parametric Model vs M2M Orbits", fontsize=15, fontweight='bold', y=0.98)
    
    out_path_v = os.path.join(project_root, "results/m2m/m2m_validation_kinematics.png")
    plt.savefig(out_path_v, dpi=200, facecolor=DARK)
    plt.close()
    print(f"Saved kinematics validation plot to {out_path_v}")
    
    # ─────────────────────────────────────────────────────────
    # 2. Dispersions Comparison Dashboard
    # ─────────────────────────────────────────────────────────
    fig, axs = plt.subplots(2, 3, figsize=(18, 11), facecolor=DARK)
    fig.subplots_adjust(hspace=0.22, wspace=0.2)
    
    # Projections of Dispersions
    t_sx = np.sqrt(np.nanmean(t_sxx, axis=2))
    p_sx = np.sqrt(np.nanmean(sxx_pred, axis=2))
    
    t_sy = np.sqrt(np.nanmean(t_syy, axis=2))
    p_sy = np.sqrt(np.nanmean(syy_pred, axis=2))
    
    t_sz = np.sqrt(np.nanmean(t_szz, axis=2))
    p_sz = np.sqrt(np.nanmean(szz_pred, axis=2))
    
    s_max = 160.0
    cmap_s = 'plasma'
    
    # Row 1: Target
    for ax in axs[0]: style_ax(ax)
    im0_s = axs[0, 0].pcolormesh(xc, yc, t_sx.T, cmap=cmap_s, vmin=40, vmax=s_max, shading='auto')
    axs[0, 0].set_title("XY Plane: Target sigma_x", fontsize=11, fontweight='bold')
    axs[0, 0].set_ylabel("y (kpc)")
    
    axs[0, 1].pcolormesh(xc, yc, t_sy.T, cmap=cmap_s, vmin=40, vmax=s_max, shading='auto')
    axs[0, 1].set_title("XY Plane: Target sigma_y", fontsize=11, fontweight='bold')
    axs[0, 1].set_ylabel("y (kpc)")
    
    axs[0, 2].pcolormesh(xc, yc, t_sz.T, cmap=cmap_s, vmin=40, vmax=s_max, shading='auto')
    axs[0, 2].set_title("XY Plane: Target sigma_z", fontsize=11, fontweight='bold')
    axs[0, 2].set_ylabel("y (kpc)")
    
    # Row 2: M2M
    for ax in axs[1]: style_ax(ax)
    axs[1, 0].pcolormesh(xc, yc, p_sx.T, cmap=cmap_s, vmin=40, vmax=s_max, shading='auto')
    axs[1, 0].set_title("XY Plane: M2M sigma_x", fontsize=11, fontweight='bold')
    axs[1, 0].set_xlabel("x (kpc)")
    axs[1, 0].set_ylabel("y (kpc)")
    
    axs[1, 1].pcolormesh(xc, yc, p_sy.T, cmap=cmap_s, vmin=40, vmax=s_max, shading='auto')
    axs[1, 1].set_title("XY Plane: M2M sigma_y", fontsize=11, fontweight='bold')
    axs[1, 1].set_xlabel("x (kpc)")
    axs[1, 1].set_ylabel("y (kpc)")
    
    axs[1, 2].pcolormesh(xc, yc, p_sz.T, cmap=cmap_s, vmin=40, vmax=s_max, shading='auto')
    axs[1, 2].set_title("XY Plane: M2M sigma_z", fontsize=11, fontweight='bold')
    axs[1, 2].set_xlabel("x (kpc)")
    axs[1, 2].set_ylabel("y (kpc)")
    
    # Label Row Headers on the left
    axs[0, 0].text(-0.25, 0.5, "NP Model (Target)", transform=axs[0, 0].transAxes,
                 color='white', fontsize=12, fontweight='bold', va='center', ha='left', rotation=90)
    axs[1, 0].text(-0.25, 0.5, "M2M Orbit Model", transform=axs[1, 0].transAxes,
                 color='white', fontsize=12, fontweight='bold', va='center', ha='left', rotation=90)
                 
    cbar_ax_s = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cb_s = fig.colorbar(im0_s, cax=cbar_ax_s)
    cb_s.set_label('Velocity Dispersion [km/s]', color='white', fontsize=10)
    cb_s.ax.yaxis.set_tick_params(colors='white')
    
    plt.suptitle("Galactic Bulge Dispersions Validation: Non-Parametric Model vs M2M Orbits", fontsize=15, fontweight='bold', y=0.98)
    
    out_path_s = os.path.join(project_root, "results/m2m/m2m_validation_dispersions.png")
    plt.savefig(out_path_s, dpi=200, facecolor=DARK)
    plt.close()
    print(f"Saved dispersions validation plot to {out_path_s}")
    
    # ─────────────────────────────────────────────────────────
    # 3. Particle Weights distribution (Diagnostic)
    # ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=DARK)
    style_ax(ax)
    ax.hist(np.log10(weights + 1e-12), bins=50, color='teal', alpha=0.8, edgecolor=DARK)
    ax.set_title("M2M Particle Weight Distribution (Smoothness Check)")
    ax.set_xlabel("log10(Weight)")
    ax.set_ylabel("Number of Particles")
    
    out_path_w = os.path.join(project_root, "results/m2m/m2m_weight_entropy.png")
    plt.savefig(out_path_w, dpi=150, facecolor=DARK)
    plt.close()
    print(f"Saved weight entropy plot to {out_path_w}")

if __name__ == "__main__":
    main()
