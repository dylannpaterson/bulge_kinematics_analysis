import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import os
import sys
import json
import agama

# Set Agama units
agama.setUnits(length=1, mass=1, velocity=1)
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# Append paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../shared')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../synthpop')))

import synthpop

def get_synthpop_total_density(model_name="Besancon_Coleman2020_symmetric"):
    config = {
        "MANDATORY": {"name_for_output": "m2m_tmp", "model_name": model_name},
        "SIGHTLINES": {"l_set": [0.0], "l_set_type": "list", "b_set": [0.0], "b_set_type": "list", 
                       "solid_angle": 0.01, "solid_angle_unit": "deg^2"}
    }
    tmp_conf = f"tmp_anim_{model_name}.json"
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

    return total_density

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    fit_path = os.path.join(project_root, "results/m2m/np_m2m_fit_results.npz")
    if not os.path.exists(fit_path):
        print(f"Error: M2M fit results not found at {fit_path}. Please run validate_np_with_m2m.py first.")
        return
        
    print("Loading M2M fit results...")
    fit_res = np.load(fit_path)
    ics = fit_res['ics']
    weights = fit_res['weights']
    
    print("Loading SynthPop densities & building potential...")
    total_density = get_synthpop_total_density("Besancon_Coleman2020_symmetric")
    
    print("Generating Agama potential...")
    full_pot = agama.Potential(type='Multipole', density=total_density, lmax=16, symmetry='triaxial', 
                               gridsizeR=100, rmin=0.01, rmax=50.0)
                               
    # Pattern speed from NP model
    omega = 37.67 # km/s/kpc
    
    # 10x particles: 30,000
    is_fast = os.environ.get("FAST_TEST") == "1"
    n_sample = 100 if is_fast else 30000
    p_weights = weights / np.sum(weights)
    
    # Sample from ics
    sampled_idx = np.random.choice(len(ics), size=n_sample, p=p_weights, replace=False)
    sampled_ics = ics[sampled_idx]
    
    print(f"Integrating {n_sample} M2M-weighted orbits...")
    n_orbit_steps = 10 if is_fast else 400
    total_time = 0.4 # ~ 400 Myr
    trajs = agama.orbit(potential=full_pot, ic=sampled_ics, time=total_time, trajsize=n_orbit_steps, Omega=omega)
    
    # Coordinates of all orbits: shape (n_sample, n_orbit_steps, 6)
    coords = np.stack([trajs[p][1] for p in range(n_sample)])
    
    # Classify orbits to highlight the X-shape banana orbits in gold
    print("Classifying orbits to identify X-shape supporting peanut orbits...")
    is_gold = np.zeros(n_sample, dtype=bool)
    for p in range(n_sample):
        c_p = coords[p] # (n_orbit_steps, 6)
        x_max = np.max(np.abs(c_p[:, 0]))
        z_max = np.max(np.abs(c_p[:, 2]))
        
        # Classic banana orbit check: confined bulge stars with high vertical excursion
        # ratio z_max/x_max represents vertical thickness of peanut shape
        if (z_max >= 0.38) and (z_max <= 1.2) and (x_max >= 0.8) and (x_max <= 2.4):
            ratio = z_max / x_max
            if ratio >= 0.22 and ratio <= 0.65:
                is_gold[p] = True
                
    gold_idx = np.where(is_gold)[0]
    cyan_idx = np.where(~is_gold)[0]
    n_gold = len(gold_idx)
    n_cyan = len(cyan_idx)
    print(f"  Classified: {n_gold} gold X-shape orbits, {n_cyan} cyan standard orbits (Fraction = {n_gold/n_sample*100:.1f}%)")
    
    initial_phases = np.random.randint(0, n_orbit_steps, size=n_sample)
    
    # 2. Setup Animation (Premium Dark Theme)
    DARK = '#111116'
    GRID_C = '#444455'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), facecolor=DARK)
    
    def style_ax(ax, title, xlabel, ylabel, lim_val, z_lim=None):
        ax.set_facecolor(DARK)
        ax.grid(True, color=GRID_C, linestyle=':', alpha=0.4)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect('equal')
        ax.set_xlim(-lim_val, lim_val)
        if z_lim is not None:
            ax.set_ylim(-z_lim, z_lim)
        else:
            ax.set_ylim(-lim_val, lim_val)
            
    style_ax(axs[0], "Top-Down View (XY Plane)", "x_bar (kpc)", "y_bar (kpc)", 3.8)
    style_ax(axs[1], "Side-On peanut View (XZ Plane)", "x_bar (kpc)", "z_bar (kpc)", 3.8, z_lim=1.8)
    style_ax(axs[2], "Front-On Profile (YZ Plane)", "y_bar (kpc)", "z_bar (kpc)", 3.8, z_lim=1.8)
    
    # Fading trails (extremely low alpha and width to remain smooth for 30k stars)
    trails_cyan_xy, = axs[0].plot([], [], color='#00d8ff', alpha=0.008, linewidth=0.3)
    trails_cyan_xz, = axs[1].plot([], [], color='#00d8ff', alpha=0.008, linewidth=0.3)
    trails_cyan_yz, = axs[2].plot([], [], color='#00d8ff', alpha=0.008, linewidth=0.3)
    
    trails_gold_xy, = axs[0].plot([], [], color='#ffd700', alpha=0.015, linewidth=0.4)
    trails_gold_xz, = axs[1].plot([], [], color='#ffd700', alpha=0.015, linewidth=0.4)
    trails_gold_yz, = axs[2].plot([], [], color='#ffd700', alpha=0.015, linewidth=0.4)
    
    trails_magenta_xy, = axs[0].plot([], [], color='#ff33aa', alpha=0.015, linewidth=0.4)
    trails_magenta_xz, = axs[1].plot([], [], color='#ff33aa', alpha=0.015, linewidth=0.4)
    trails_magenta_yz, = axs[2].plot([], [], color='#ff33aa', alpha=0.015, linewidth=0.4)
    
    # Head dots (cyan, gold & magenta, tiny markersize for high resolution)
    dots_cyan_xy, = axs[0].plot([], [], 'o', color='#00d8ff', markersize=0.5, alpha=0.25)
    dots_cyan_xz, = axs[1].plot([], [], 'o', color='#00d8ff', markersize=0.5, alpha=0.25)
    dots_cyan_yz, = axs[2].plot([], [], 'o', color='#00d8ff', markersize=0.5, alpha=0.25)
    
    dots_gold_xy, = axs[0].plot([], [], 'o', color='#ffd700', markersize=0.6, alpha=0.45)
    dots_gold_xz, = axs[1].plot([], [], 'o', color='#ffd700', markersize=0.6, alpha=0.45)
    dots_gold_yz, = axs[2].plot([], [], 'o', color='#ffd700', markersize=0.6, alpha=0.45)
    
    dots_magenta_xy, = axs[0].plot([], [], 'o', color='#ff33aa', markersize=0.6, alpha=0.45)
    dots_magenta_xz, = axs[1].plot([], [], 'o', color='#ff33aa', markersize=0.6, alpha=0.45)
    dots_magenta_yz, = axs[2].plot([], [], 'o', color='#ff33aa', markersize=0.6, alpha=0.45)
    
    # Bar Outline (Stationary orange ellipse in the bar frame)
    bar_xy = Ellipse((0, 0), width=5.0, height=2.4, color='#ff9944', fill=False, linestyle='--', linewidth=1.2, alpha=0.7, label='Stationary Bar Outline')
    axs[0].add_artist(bar_xy)
    axs[0].legend(facecolor=DARK, edgecolor=GRID_C, loc='upper right', fontsize=9)
    
    # Guide lines for X-shape in XZ plane
    axs[1].plot([-2.0, 2.0], [-1.0, 1.0], color='#ff4444', linestyle=':', alpha=0.3, linewidth=1.0, label='X-shape Guidelights')
    axs[1].plot([-2.0, 2.0], [1.0, -1.0], color='#ff4444', linestyle=':', alpha=0.3, linewidth=1.0)
    axs[1].legend(facecolor=DARK, edgecolor=GRID_C, loc='upper right', fontsize=9)
    
    plt.suptitle("N-Body Stellar Flow in the Stationary Rotating Bar Frame\n(Gold/Magenta highlight prograde/retrograde X-shape peanut orbits)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    n_frames = 2 if is_fast else 200
    n_tail = 6 # trail length
    
    # Pre-allocate trail arrays with NaNs
    x_trail_cyan = np.empty((n_sample, n_tail + 1))
    y_trail_cyan = np.empty((n_sample, n_tail + 1))
    z_trail_cyan = np.empty((n_sample, n_tail + 1))
    x_trail_cyan[:, n_tail] = np.nan
    y_trail_cyan[:, n_tail] = np.nan
    z_trail_cyan[:, n_tail] = np.nan
    
    x_trail_gold = np.empty((n_sample, n_tail + 1))
    y_trail_gold = np.empty((n_sample, n_tail + 1))
    z_trail_gold = np.empty((n_sample, n_tail + 1))
    x_trail_gold[:, n_tail] = np.nan
    y_trail_gold[:, n_tail] = np.nan
    z_trail_gold[:, n_tail] = np.nan
    
    x_trail_magenta = np.empty((n_sample, n_tail + 1))
    y_trail_magenta = np.empty((n_sample, n_tail + 1))
    z_trail_magenta = np.empty((n_sample, n_tail + 1))
    x_trail_magenta[:, n_tail] = np.nan
    y_trail_magenta[:, n_tail] = np.nan
    z_trail_magenta[:, n_tail] = np.nan
    
    # Animation Init
    def init():
        trails_cyan_xy.set_data([], [])
        trails_cyan_xz.set_data([], [])
        trails_cyan_yz.set_data([], [])
        trails_gold_xy.set_data([], [])
        trails_gold_xz.set_data([], [])
        trails_gold_yz.set_data([], [])
        trails_magenta_xy.set_data([], [])
        trails_magenta_xz.set_data([], [])
        trails_magenta_yz.set_data([], [])
        dots_cyan_xy.set_data([], [])
        dots_cyan_xz.set_data([], [])
        dots_cyan_yz.set_data([], [])
        dots_gold_xy.set_data([], [])
        dots_gold_xz.set_data([], [])
        dots_gold_yz.set_data([], [])
        dots_magenta_xy.set_data([], [])
        dots_magenta_xz.set_data([], [])
        dots_magenta_yz.set_data([], [])
        return (trails_cyan_xy, trails_cyan_xz, trails_cyan_yz,
                trails_gold_xy, trails_gold_xz, trails_gold_yz,
                trails_magenta_xy, trails_magenta_xz, trails_magenta_yz,
                dots_cyan_xy, dots_cyan_xz, dots_cyan_yz,
                dots_gold_xy, dots_gold_xz, dots_gold_yz,
                dots_magenta_xy, dots_magenta_xz, dots_magenta_yz)
        
    # Animation Frame Update
    def update(frame):
        # Indices for current frame
        current_indices = (initial_phases[:, None] + frame) % n_orbit_steps
        trail_indices = (initial_phases[:, None] + frame - np.arange(n_tail)[None, :]) % n_orbit_steps
        
        # Coordinates and velocities in stationary bar frame: shape (n_sample, n_tail, 6)
        pos_trail_bar = coords[np.arange(n_sample)[:, None], trail_indices, :]
        
        # Current bar-frame velocity Vx (state at current frame is trail index 0)
        vx_bar_current = pos_trail_bar[:, 0, 3]
        
        xb_t = pos_trail_bar[:, :, 0]
        yb_t = pos_trail_bar[:, :, 1]
        zb_t = pos_trail_bar[:, :, 2]
        
        # Split into cyan, gold, and magenta
        is_cyan = ~is_gold
        is_gold_frame = is_gold & (vx_bar_current >= 0.0)
        is_magenta_frame = is_gold & (vx_bar_current < 0.0)
        
        cyan_idx = np.where(is_cyan)[0]
        gold_idx = np.where(is_gold_frame)[0]
        magenta_idx = np.where(is_magenta_frame)[0]
        
        n_cy = len(cyan_idx)
        n_go = len(gold_idx)
        n_ma = len(magenta_idx)
        
        # Fill trails
        x_trail_cyan[:n_cy, :n_tail] = xb_t[cyan_idx, :]
        y_trail_cyan[:n_cy, :n_tail] = yb_t[cyan_idx, :]
        z_trail_cyan[:n_cy, :n_tail] = zb_t[cyan_idx, :]
        
        x_trail_gold[:n_go, :n_tail] = xb_t[gold_idx, :]
        y_trail_gold[:n_go, :n_tail] = yb_t[gold_idx, :]
        z_trail_gold[:n_go, :n_tail] = zb_t[gold_idx, :]
        
        x_trail_magenta[:n_ma, :n_tail] = xb_t[magenta_idx, :]
        y_trail_magenta[:n_ma, :n_tail] = yb_t[magenta_idx, :]
        z_trail_magenta[:n_ma, :n_tail] = zb_t[magenta_idx, :]
        
        # Update cyan lines
        trails_cyan_xy.set_data(x_trail_cyan[:n_cy, :].flatten(), y_trail_cyan[:n_cy, :].flatten())
        trails_cyan_xz.set_data(x_trail_cyan[:n_cy, :].flatten(), z_trail_cyan[:n_cy, :].flatten())
        trails_cyan_yz.set_data(y_trail_cyan[:n_cy, :].flatten(), z_trail_cyan[:n_cy, :].flatten())
        
        # Update gold lines
        trails_gold_xy.set_data(x_trail_gold[:n_go, :].flatten(), y_trail_gold[:n_go, :].flatten())
        trails_gold_xz.set_data(x_trail_gold[:n_go, :].flatten(), z_trail_gold[:n_go, :].flatten())
        trails_gold_yz.set_data(y_trail_gold[:n_go, :].flatten(), z_trail_gold[:n_go, :].flatten())
        
        # Update magenta lines
        trails_magenta_xy.set_data(x_trail_magenta[:n_ma, :].flatten(), y_trail_magenta[:n_ma, :].flatten())
        trails_magenta_xz.set_data(x_trail_magenta[:n_ma, :].flatten(), z_trail_magenta[:n_ma, :].flatten())
        trails_magenta_yz.set_data(y_trail_magenta[:n_ma, :].flatten(), z_trail_magenta[:n_ma, :].flatten())
        
        # Head dots (index 0 represents current frame)
        dots_cyan_xy.set_data(xb_t[cyan_idx, 0], yb_t[cyan_idx, 0])
        dots_cyan_xz.set_data(xb_t[cyan_idx, 0], zb_t[cyan_idx, 0])
        dots_cyan_yz.set_data(yb_t[cyan_idx, 0], zb_t[cyan_idx, 0])
        
        dots_gold_xy.set_data(xb_t[gold_idx, 0], yb_t[gold_idx, 0])
        dots_gold_xz.set_data(xb_t[gold_idx, 0], zb_t[gold_idx, 0])
        dots_gold_yz.set_data(yb_t[gold_idx, 0], zb_t[gold_idx, 0])
        
        dots_magenta_xy.set_data(xb_t[magenta_idx, 0], yb_t[magenta_idx, 0])
        dots_magenta_xz.set_data(xb_t[magenta_idx, 0], zb_t[magenta_idx, 0])
        dots_magenta_yz.set_data(yb_t[magenta_idx, 0], zb_t[magenta_idx, 0])
        
        return (trails_cyan_xy, trails_cyan_xz, trails_cyan_yz,
                trails_gold_xy, trails_gold_xz, trails_gold_yz,
                trails_magenta_xy, trails_magenta_xz, trails_magenta_yz,
                dots_cyan_xy, dots_cyan_xz, dots_cyan_yz,
                dots_gold_xy, dots_gold_xz, dots_gold_yz,
                dots_magenta_xy, dots_magenta_xz, dots_magenta_yz)
        
    print("Generating stationary bar-frame animation frames...")
    anim = animation.FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True)
    
    out_dir = os.path.join(project_root, "results/m2m")
    os.makedirs(out_dir, exist_ok=True)
    out_gif = os.path.join(out_dir, "m2m_flow_animation_bar.gif")
    
    print("Saving stationary bar stellar flow animation as GIF (using Pillow)...")
    anim.save(out_gif, writer='pillow', fps=20, dpi=120)
    plt.close()
    
    print(f"Bar-frame flow animation successfully saved to {out_gif}!")

if __name__ == "__main__":
    main()
