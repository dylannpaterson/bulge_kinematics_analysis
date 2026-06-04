import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import os
import sys
import json
import agama

# Set Agama units (kpc, Msun, km/s)
agama.setUnits(length=1, mass=1, velocity=1)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Append paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../shared')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../synthpop')))

import synthpop

# ─────────────────────────────────────────────────────────
# 1. Load SynthPop density & build potential
# ─────────────────────────────────────────────────────────

def get_synthpop_total_density(model_name="Huston2025_C20Bulge"):
    config = {
        "MANDATORY": {"name_for_output": "m2m_tmp", "model_name": model_name},
        "SIGHTLINES": {"l_set": [0.0], "l_set_type": "list", "b_set": [0.0], "b_set_type": "list",
                       "solid_angle": 0.01, "solid_angle_unit": "deg^2"}
    }
    tmp_conf = f"tmp_flow_{model_name}.json"
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


# ─────────────────────────────────────────────────────────
# Marginal histogram helpers
# ─────────────────────────────────────────────────────────

N_HIST_BINS = 60

def make_step_data(values, bins, scale=1.0):
    """Return (x, y) for a step histogram line drawable via set_data."""
    counts, _ = np.histogram(values, bins=bins)
    if scale > 0:
        counts = counts / scale
    # Duplicate each bin edge / count to form a step waveform
    x = np.repeat(bins, 2)
    y = np.concatenate([[0], np.repeat(counts, 2), [0]])
    return x, y


def setup_panel(outer_gs_cell, fig, DARK, GRID_C,
                title, xlabel, ylabel, xlim, ylim):
    """
    Create a panel with equal-aspect main axis and perfectly aligned top and right
    marginal subplots using make_axes_locatable.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax_main = fig.add_subplot(outer_gs_cell)

    ax_main.set_facecolor(DARK)
    ax_main.grid(True, color=GRID_C, linestyle=':', alpha=0.4)
    ax_main.set_xlabel(xlabel, fontsize=9)
    ax_main.set_ylabel(ylabel, fontsize=9)
    ax_main.set_aspect('equal')
    ax_main.set_xlim(*xlim)
    ax_main.set_ylim(*ylim)

    # Use make_axes_locatable to append top and right axes to match ax_main's active area
    divider = make_axes_locatable(ax_main)
    ax_top = divider.append_axes("top", size="18%", pad=0.04, sharex=ax_main)
    ax_right = divider.append_axes("right", size="18%", pad=0.04, sharey=ax_main)

    for ax in (ax_top, ax_right):
        ax.set_facecolor(DARK)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Title above top histogram
    ax_top.set_title(title, fontsize=11, fontweight='bold', color='white', pad=4)
    ax_top.set_xlim(*xlim)
    ax_top.set_ylim(0, 1.15)

    ax_right.set_ylim(*ylim)
    ax_right.set_xlim(0, 1.15)

    return ax_main, ax_top, ax_right


def init_hist_lines(ax_top, ax_right, bins_x, bins_y, colors_alphas):
    """
    Create empty step Line2D objects in the histogram axes.
    colors_alphas: list of (color, alpha) per population.
    Returns (top_lines, right_lines) each a list of Line2D.
    """
    top_lines, right_lines = [], []
    for color, alpha in colors_alphas:
        tl, = ax_top.plot([], [], color=color, alpha=alpha, linewidth=1.0)
        rl, = ax_right.plot([], [], color=color, alpha=alpha, linewidth=1.0)
        # Fill variant (faint)
        tf, = ax_top.plot([], [], color=color, alpha=alpha * 0.35, linewidth=0)
        rf, = ax_right.plot([], [], color=color, alpha=alpha * 0.35, linewidth=0)
        ax_top.fill_between([], [], color=color, alpha=alpha * 0.15)
        ax_right.fill_betweenx([], [], color=color, alpha=alpha * 0.15)
        top_lines.append(tl)
        right_lines.append(rl)
    return top_lines, right_lines


# ─────────────────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────────────────

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    fit_path = os.path.join(project_root, "results/m2m/np_m2m_fit_results.npz")
    if not os.path.exists(fit_path):
        print(f"Error: {fit_path} not found. Please run validate_np_with_m2m.py first.")
        return

    print("Loading M2M fit results...")
    data = np.load(fit_path)
    weights = data['weights']
    ics = data['ics']

    # Load bar angle from parametric fit
    param_path = os.path.join(project_root, "results/inversion/parametric_fit_results_obs_h25c20.npz")
    if os.path.exists(param_path):
        param_data = np.load(param_path)
        p_fitted = param_data['params']
        alpha_deg = float(p_fitted[19])  # ~ 18.79 deg
    else:
        alpha_deg = 18.79  # Fallback

    alpha_0 = np.radians(alpha_deg)

    print("Loading SynthPop densities & building potential...")
    total_density = get_synthpop_total_density("Huston2025_C20Bulge")

    print("Generating Agama potential (stellar multipole + NFW DM halo)...")
    stellar_pot = agama.Potential(type='Multipole', density=total_density, lmax=16, symmetry='triaxial',
                                  gridsizeR=100, rmin=0.01, rmax=50.0)
    dm_pot = agama.Potential(type='NFW', mass=1.0e12, scaleRadius=16.0)
    full_pot = agama.Potential(stellar_pot, dm_pot)

    # Load pattern speed omega dynamically from the kinematic fit results
    refined_fit_path = os.path.join(project_root, "results/inversion/fit_results_refined_continuity.npz")
    combined_fit_path = os.path.join(project_root, "results/inversion/fit_results_obs_combined_h25c20.npz")
    if os.path.exists(refined_fit_path):
        fit_data = np.load(refined_fit_path)
        omega = float(fit_data['omega'])
        print(f"Loaded pattern speed Omega={omega:.2f} km/s/kpc dynamically from refined fit results.")
    elif os.path.exists(combined_fit_path):
        fit_data = np.load(combined_fit_path)
        omega = float(fit_data['omega'])
        print(f"Loaded pattern speed Omega={omega:.2f} km/s/kpc dynamically from combined fit results.")
    else:
        omega = 38.75  # Fallback
        print(f"Warning: Kinematic fit results not found. Using fallback Omega={omega:.2f} km/s/kpc.")

    # 2. Sample particles using the M2M weights
    is_fast = os.environ.get("FAST_TEST") == "1"
    n_sample = 100 if is_fast else 30000
    p_weights = weights / np.sum(weights)
    sampled_idx = np.random.choice(len(ics), size=n_sample, p=p_weights, replace=False)
    sampled_ics = ics[sampled_idx]

    print(f"Integrating {n_sample} M2M-weighted orbits...")
    n_orbit_steps = 10 if is_fast else 400
    total_time = 0.4  # ~ 400 Myr
    trajs = agama.orbit(potential=full_pot, ic=sampled_ics, time=total_time, trajsize=n_orbit_steps, Omega=-omega)

    # Coordinates of all orbits: shape (n_sample, n_orbit_steps, 6)
    coords = np.stack([trajs[p][1] for p in range(n_sample)])

    print("Classifying orbits to identify X-shape supporting peanut orbits...")
    is_gold = np.zeros(n_sample, dtype=bool)
    for p in range(n_sample):
        c_p = coords[p]  # (n_orbit_steps, 6)
        x_max = np.max(np.abs(c_p[:, 0]))
        z_max = np.max(np.abs(c_p[:, 2]))
        if (z_max >= 0.38) and (z_max <= 1.2) and (x_max >= 0.8) and (x_max <= 2.4):
            ratio = z_max / x_max
            if ratio >= 0.22 and ratio <= 0.65:
                is_gold[p] = True

    gold_idx = np.where(is_gold)[0]
    cyan_idx_base = np.where(~is_gold)[0]
    n_gold = len(gold_idx)
    n_cyan_base = len(cyan_idx_base)
    print(f"  Classified: {n_gold} gold X-shape orbits, {n_cyan_base} cyan standard orbits "
          f"(Fraction = {n_gold / n_sample * 100:.1f}%)")

    initial_phases = np.random.randint(0, n_orbit_steps, size=n_sample)

    # ─────────────────────────────────────────────────────────
    # 3. Setup Figure with marginal histogram layout
    # ─────────────────────────────────────────────────────────
    DARK = '#111116'
    GRID_C = '#444455'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'

    fig = plt.figure(figsize=(21, 8), facecolor=DARK)
    outer_gs = gridspec.GridSpec(1, 3, figure=fig, hspace=0.05, wspace=0.12,
                                 left=0.04, right=0.98, top=0.88, bottom=0.08)

    # Panel definitions: (title, xlabel, ylabel, xlim, ylim)
    LIM = 3.8
    ZLIM = 1.8
    panels = [
        ("GC Plane (XY)", "x (kpc) [Sun at x=−8.2]", "y (kpc)",   (-LIM, LIM),  (-LIM, LIM)),
        ("Side Profile (XZ)",  "x (kpc)",              "z (kpc)",   (-LIM, LIM),  (-ZLIM, ZLIM)),
        ("Front Profile (YZ)", "y (kpc)",              "z (kpc)",   (-LIM, LIM),  (-ZLIM, ZLIM)),
    ]

    axs_main, axs_top, axs_right = [], [], []
    for i, (title, xlabel, ylabel, xlim, ylim) in enumerate(panels):
        ax_m, ax_t, ax_r = setup_panel(outer_gs[i], fig, DARK, GRID_C,
                                        title, xlabel, ylabel, xlim, ylim)
        axs_main.append(ax_m)
        axs_top.append(ax_t)
        axs_right.append(ax_r)

    # Histogram bin edges
    hist_bins = [
        (np.linspace(-LIM,  LIM,  N_HIST_BINS + 1), np.linspace(-LIM,  LIM,  N_HIST_BINS + 1)),   # XY
        (np.linspace(-LIM,  LIM,  N_HIST_BINS + 1), np.linspace(-ZLIM, ZLIM, N_HIST_BINS + 1)),   # XZ
        (np.linspace(-LIM,  LIM,  N_HIST_BINS + 1), np.linspace(-ZLIM, ZLIM, N_HIST_BINS + 1)),   # YZ
    ]

    # Population colors (cyan, gold, magenta) with alpha
    pop_styles = [
        ('#00d8ff', 0.65),  # cyan
        ('#ffd700', 0.80),  # gold
        ('#ff33aa', 0.80),  # magenta
    ]

    # Create step histogram Line2D objects for each panel
    # top: x-marginal (horizontal axis = spatial coord)
    # right: y-marginal (vertical axis = spatial coord, plotted sideways)
    top_step_lines = []   # shape: [panel][pop] -> Line2D
    right_step_lines = []

    for pi in range(3):
        bins_x, bins_y = hist_bins[pi]
        t_panel, r_panel = [], []
        for color, alpha in pop_styles:
            tl, = axs_top[pi].plot([], [], color=color, alpha=alpha, linewidth=1.0, solid_capstyle='round')
            rl, = axs_right[pi].plot([], [], color=color, alpha=alpha, linewidth=1.0, solid_capstyle='round')
            t_panel.append(tl)
            r_panel.append(rl)
        top_step_lines.append(t_panel)
        right_step_lines.append(r_panel)

    # ─────────────────────────────────────────────────────────
    # Main scatter trails & dots (same as before)
    # ─────────────────────────────────────────────────────────
    trails_cyan_xy,  = axs_main[0].plot([], [], color='#00d8ff', alpha=0.008, linewidth=0.3)
    trails_cyan_xz,  = axs_main[1].plot([], [], color='#00d8ff', alpha=0.008, linewidth=0.3)
    trails_cyan_yz,  = axs_main[2].plot([], [], color='#00d8ff', alpha=0.008, linewidth=0.3)

    trails_gold_xy,  = axs_main[0].plot([], [], color='#ffd700', alpha=0.015, linewidth=0.4)
    trails_gold_xz,  = axs_main[1].plot([], [], color='#ffd700', alpha=0.015, linewidth=0.4)
    trails_gold_yz,  = axs_main[2].plot([], [], color='#ffd700', alpha=0.015, linewidth=0.4)

    trails_magenta_xy, = axs_main[0].plot([], [], color='#ff33aa', alpha=0.015, linewidth=0.4)
    trails_magenta_xz, = axs_main[1].plot([], [], color='#ff33aa', alpha=0.015, linewidth=0.4)
    trails_magenta_yz, = axs_main[2].plot([], [], color='#ff33aa', alpha=0.015, linewidth=0.4)

    dots_cyan_xy,  = axs_main[0].plot([], [], 'o', color='#00d8ff', markersize=0.5, alpha=0.25)
    dots_cyan_xz,  = axs_main[1].plot([], [], 'o', color='#00d8ff', markersize=0.5, alpha=0.25)
    dots_cyan_yz,  = axs_main[2].plot([], [], 'o', color='#00d8ff', markersize=0.5, alpha=0.25)

    dots_gold_xy,  = axs_main[0].plot([], [], 'o', color='#ffd700', markersize=0.6, alpha=0.45)
    dots_gold_xz,  = axs_main[1].plot([], [], 'o', color='#ffd700', markersize=0.6, alpha=0.45)
    dots_gold_yz,  = axs_main[2].plot([], [], 'o', color='#ffd700', markersize=0.6, alpha=0.45)

    dots_magenta_xy, = axs_main[0].plot([], [], 'o', color='#ff33aa', markersize=0.6, alpha=0.45)
    dots_magenta_xz, = axs_main[1].plot([], [], 'o', color='#ff33aa', markersize=0.6, alpha=0.45)
    dots_magenta_yz, = axs_main[2].plot([], [], 'o', color='#ff33aa', markersize=0.6, alpha=0.45)

    # Rotating bar ellipse removed as requested

    # Reference boundary circles in XZ, YZ
    for ax in axs_main[1:]:
        boundary = plt.Circle((0, 0), 2.5, color='#444455', fill=False, linestyle=':', alpha=0.4)
        ax.add_artist(boundary)

    plt.suptitle(
        f"Stellar Flow of M2M-Weighted Orbit Populations in the GC Frame ($\\Omega_p = {omega:.2f}$ km/s/kpc)\n"
        "(Gold/Magenta = prograde/retrograde X-shape peanut orbits; Marginals show per-population density)",
        fontsize=12, fontweight='bold', y=0.97, color='white'
    )

    # Dynamic time label
    time_text = axs_main[0].text(
        0.05, 0.95, '',
        transform=axs_main[0].transAxes,
        color='white', fontsize=10, fontweight='bold',
        bbox=dict(facecolor='black', alpha=0.6, edgecolor=GRID_C, boxstyle='round,pad=0.3')
    )

    # Trail buffers
    n_tail = 6

    def make_trail_buf():
        buf = np.empty((n_sample, n_tail + 1))
        buf[:, n_tail] = np.nan
        return buf

    x_trail_cyan    = make_trail_buf()
    y_trail_cyan    = make_trail_buf()
    z_trail_cyan    = make_trail_buf()
    x_trail_gold    = make_trail_buf()
    y_trail_gold    = make_trail_buf()
    z_trail_gold    = make_trail_buf()
    x_trail_magenta = make_trail_buf()
    y_trail_magenta = make_trail_buf()
    z_trail_magenta = make_trail_buf()

    # ─────────────────────────────────────────────────────────
    # Collect all blittable artists
    # ─────────────────────────────────────────────────────────
    scatter_artists = [
        trails_cyan_xy, trails_cyan_xz, trails_cyan_yz,
        trails_gold_xy, trails_gold_xz, trails_gold_yz,
        trails_magenta_xy, trails_magenta_xz, trails_magenta_yz,
        dots_cyan_xy, dots_cyan_xz, dots_cyan_yz,
        dots_gold_xy, dots_gold_xz, dots_gold_yz,
        dots_magenta_xy, dots_magenta_xz, dots_magenta_yz,
    ]
    hist_artists = [line for panel in top_step_lines + right_step_lines for line in panel]
    all_artists = scatter_artists + hist_artists + [time_text]

    def init():
        for art in scatter_artists:
            art.set_data([], [])
        for art in hist_artists:
            art.set_data([], [])
        time_text.set_text('')
        return all_artists

    # ─────────────────────────────────────────────────────────
    # Per-frame update
    # ─────────────────────────────────────────────────────────
    def update(frame):
        trail_indices = (initial_phases[:, None] + frame - np.arange(n_tail)[None, :]) % n_orbit_steps
        pos_trail_bar = coords[np.arange(n_sample)[:, None], trail_indices, :]
        vx_bar_current = pos_trail_bar[:, 0, 3]

        # Rotate into GC frame
        dt_frame = total_time / n_orbit_steps
        t_Myr = frame * dt_frame * 977.8
        time_text.set_text(f"t = {t_Myr:.1f} Myr")
        t_trail = (frame - np.arange(n_tail)) * dt_frame
        alpha_t = alpha_0 - omega * t_trail
        cos_a = np.cos(alpha_t)[None, :]
        sin_a = np.sin(alpha_t)[None, :]

        xb_t = pos_trail_bar[:, :, 0]
        yb_t = pos_trail_bar[:, :, 1]
        zb_t = pos_trail_bar[:, :, 2]

        x_gc = xb_t * cos_a - yb_t * sin_a
        y_gc = xb_t * sin_a + yb_t * cos_a
        z_gc = zb_t

        # Classify
        is_cyan_f    = ~is_gold
        is_gold_f    = is_gold & (vx_bar_current >= 0.0)
        is_magenta_f = is_gold & (vx_bar_current < 0.0)

        c_idx = np.where(is_cyan_f)[0]
        g_idx = np.where(is_gold_f)[0]
        m_idx = np.where(is_magenta_f)[0]
        n_cy, n_go, n_ma = len(c_idx), len(g_idx), len(m_idx)

        # Fill trail buffers
        x_trail_cyan[:n_cy, :n_tail]    = x_gc[c_idx, :]
        y_trail_cyan[:n_cy, :n_tail]    = y_gc[c_idx, :]
        z_trail_cyan[:n_cy, :n_tail]    = z_gc[c_idx, :]
        x_trail_gold[:n_go, :n_tail]    = x_gc[g_idx, :]
        y_trail_gold[:n_go, :n_tail]    = y_gc[g_idx, :]
        z_trail_gold[:n_go, :n_tail]    = z_gc[g_idx, :]
        x_trail_magenta[:n_ma, :n_tail] = x_gc[m_idx, :]
        y_trail_magenta[:n_ma, :n_tail] = y_gc[m_idx, :]
        z_trail_magenta[:n_ma, :n_tail] = z_gc[m_idx, :]

        # Update trail lines
        trails_cyan_xy.set_data(x_trail_cyan[:n_cy, :].flatten(), y_trail_cyan[:n_cy, :].flatten())
        trails_cyan_xz.set_data(x_trail_cyan[:n_cy, :].flatten(), z_trail_cyan[:n_cy, :].flatten())
        trails_cyan_yz.set_data(y_trail_cyan[:n_cy, :].flatten(), z_trail_cyan[:n_cy, :].flatten())
        trails_gold_xy.set_data(x_trail_gold[:n_go, :].flatten(), y_trail_gold[:n_go, :].flatten())
        trails_gold_xz.set_data(x_trail_gold[:n_go, :].flatten(), z_trail_gold[:n_go, :].flatten())
        trails_gold_yz.set_data(y_trail_gold[:n_go, :].flatten(), z_trail_gold[:n_go, :].flatten())
        trails_magenta_xy.set_data(x_trail_magenta[:n_ma, :].flatten(), y_trail_magenta[:n_ma, :].flatten())
        trails_magenta_xz.set_data(x_trail_magenta[:n_ma, :].flatten(), z_trail_magenta[:n_ma, :].flatten())
        trails_magenta_yz.set_data(y_trail_magenta[:n_ma, :].flatten(), z_trail_magenta[:n_ma, :].flatten())

        # Head dots (frame-current position = trail step 0)
        dots_cyan_xy.set_data(x_gc[c_idx, 0], y_gc[c_idx, 0])
        dots_cyan_xz.set_data(x_gc[c_idx, 0], z_gc[c_idx, 0])
        dots_cyan_yz.set_data(y_gc[c_idx, 0], z_gc[c_idx, 0])
        dots_gold_xy.set_data(x_gc[g_idx, 0], y_gc[g_idx, 0])
        dots_gold_xz.set_data(x_gc[g_idx, 0], z_gc[g_idx, 0])
        dots_gold_yz.set_data(y_gc[g_idx, 0], z_gc[g_idx, 0])
        dots_magenta_xy.set_data(x_gc[m_idx, 0], y_gc[m_idx, 0])
        dots_magenta_xz.set_data(x_gc[m_idx, 0], z_gc[m_idx, 0])
        dots_magenta_yz.set_data(y_gc[m_idx, 0], z_gc[m_idx, 0])

        # Rotating bar ellipse update removed

        # ── Marginal histograms ──────────────────────────────
        # Current head positions (frame-step 0) per population
        pop_x_data = [x_gc[c_idx, 0], x_gc[g_idx, 0], x_gc[m_idx, 0]]
        pop_y_data = [y_gc[c_idx, 0], y_gc[g_idx, 0], y_gc[m_idx, 0]]
        pop_z_data = [z_gc[c_idx, 0], z_gc[g_idx, 0], z_gc[m_idx, 0]]

        # Panel 0: XY → top=x-marginal, right=y-marginal
        # Panel 1: XZ → top=x-marginal, right=z-marginal
        # Panel 2: YZ → top=y-marginal, right=z-marginal
        panel_horiz = [pop_x_data, pop_x_data, pop_y_data]
        panel_vert  = [pop_y_data, pop_z_data, pop_z_data]

        for pi in range(3):
            bins_x, bins_y = hist_bins[pi]
            
            # Find global peak count for horizontal (top) histogram across all 3 pops
            peak_x = 0.0
            for pop_i in range(3):
                hx = panel_horiz[pi][pop_i]
                if len(hx) > 0:
                    counts_x, _ = np.histogram(hx, bins=bins_x)
                    if counts_x.max() > peak_x:
                        peak_x = counts_x.max()
            
            # Find global peak count for vertical (right) histogram across all 3 pops
            peak_y = 0.0
            for pop_i in range(3):
                hy = panel_vert[pi][pop_i]
                if len(hy) > 0:
                    counts_y, _ = np.histogram(hy, bins=bins_y)
                    if counts_y.max() > peak_y:
                        peak_y = counts_y.max()

            for pop_i in range(3):
                hx = panel_horiz[pi][pop_i]
                hy = panel_vert[pi][pop_i]

                if len(hx) > 1 and peak_x > 0:
                    sx, sy = make_step_data(hx, bins_x, scale=peak_x)
                    top_step_lines[pi][pop_i].set_data(sx, sy)
                else:
                    top_step_lines[pi][pop_i].set_data([], [])

                if len(hy) > 1 and peak_y > 0:
                    # Right histogram: counts along y-axis, so swap x/y
                    sy2, sy2_c = make_step_data(hy, bins_y, scale=peak_y)
                    right_step_lines[pi][pop_i].set_data(sy2_c, sy2)
                else:
                    right_step_lines[pi][pop_i].set_data([], [])

        return all_artists

    print("Generating M2M flow animation frames in GC frame...")
    n_frames = 2 if is_fast else 200
    anim = animation.FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True)

    out_dir = os.path.join(project_root, "results/m2m")
    os.makedirs(out_dir, exist_ok=True)
    out_gif = os.path.join(out_dir, "m2m_flow_animation.gif")

    print("Saving N-body stellar flow animation as GIF (using Pillow)...")
    anim.save(out_gif, writer='pillow', fps=20, dpi=120)
    plt.close()

    print(f"Flow animation successfully saved to {out_gif}!")


if __name__ == "__main__":
    main()
