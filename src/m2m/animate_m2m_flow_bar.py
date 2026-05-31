import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import os
import sys
import json
import agama

# Set Agama units
agama.setUnits(length=1, mass=1, velocity=1)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Append paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../shared')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../synthpop')))

import synthpop


def get_synthpop_total_density(model_name="Huston2025_C20Bulge"):
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


# ─────────────────────────────────────────────────────────
# Marginal histogram helpers (shared with GC frame script)
# ─────────────────────────────────────────────────────────

N_HIST_BINS = 60


def make_step_data(values, bins):
    """Return (x, y) for a blittable step histogram line."""
    counts, _ = np.histogram(values, bins=bins)
    peak = counts.max()
    if peak > 0:
        counts = counts / peak
    x = np.repeat(bins, 2)
    y = np.concatenate([[0], np.repeat(counts, 2), [0]])
    return x, y


def setup_panel(outer_gs_cell, fig, DARK, GRID_C,
                title, xlabel, ylabel, xlim, ylim):
    """
    2×2 inner GridSpec: top-hist | (empty) / main_ax | right_hist.
    Returns (main_ax, top_ax, right_ax).
    """
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 2,
        subplot_spec=outer_gs_cell,
        width_ratios=[5, 1],
        height_ratios=[1, 5],
        hspace=0.04,
        wspace=0.04,
    )
    ax_top   = fig.add_subplot(inner[0, 0])
    ax_main  = fig.add_subplot(inner[1, 0])
    ax_right = fig.add_subplot(inner[1, 1])

    for ax in (ax_top, ax_right):
        ax.set_facecolor(DARK)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    ax_main.set_facecolor(DARK)
    ax_main.grid(True, color=GRID_C, linestyle=':', alpha=0.4)
    ax_main.set_xlabel(xlabel, fontsize=9)
    ax_main.set_ylabel(ylabel, fontsize=9)
    ax_main.set_aspect('equal')
    ax_main.set_xlim(*xlim)
    ax_main.set_ylim(*ylim)

    ax_top.set_title(title, fontsize=11, fontweight='bold', color='white', pad=4)
    ax_top.set_facecolor(DARK)
    ax_top.set_xlim(*xlim)
    ax_top.set_ylim(0, 1.15)

    ax_right.set_ylim(*ylim)
    ax_right.set_xlim(0, 1.15)

    return ax_main, ax_top, ax_right


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
    total_density = get_synthpop_total_density("Huston2025_C20Bulge")

    print("Generating Agama potential...")
    full_pot = agama.Potential(type='Multipole', density=total_density, lmax=16, symmetry='triaxial',
                               gridsizeR=100, rmin=0.01, rmax=50.0)

    # Pattern speed from NP model
    omega = 37.67  # km/s/kpc

    is_fast = os.environ.get("FAST_TEST") == "1"
    n_sample = 100 if is_fast else 30000
    p_weights = weights / np.sum(weights)
    sampled_idx = np.random.choice(len(ics), size=n_sample, p=p_weights, replace=False)
    sampled_ics = ics[sampled_idx]

    print(f"Integrating {n_sample} M2M-weighted orbits...")
    n_orbit_steps = 10 if is_fast else 400
    total_time = 0.4  # ~ 400 Myr
    trajs = agama.orbit(potential=full_pot, ic=sampled_ics, time=total_time, trajsize=n_orbit_steps, Omega=omega)

    # Coordinates: shape (n_sample, n_orbit_steps, 6)
    coords = np.stack([trajs[p][1] for p in range(n_sample)])

    print("Classifying orbits to identify X-shape supporting peanut orbits...")
    is_gold = np.zeros(n_sample, dtype=bool)
    for p in range(n_sample):
        c_p = coords[p]
        x_max = np.max(np.abs(c_p[:, 0]))
        z_max = np.max(np.abs(c_p[:, 2]))
        if (z_max >= 0.38) and (z_max <= 1.2) and (x_max >= 0.8) and (x_max <= 2.4):
            ratio = z_max / x_max
            if ratio >= 0.22 and ratio <= 0.65:
                is_gold[p] = True

    n_gold = int(is_gold.sum())
    n_cyan_base = n_sample - n_gold
    print(f"  Classified: {n_gold} gold X-shape orbits, {n_cyan_base} cyan standard orbits "
          f"(Fraction = {n_gold / n_sample * 100:.1f}%)")

    initial_phases = np.random.randint(0, n_orbit_steps, size=n_sample)

    # ─────────────────────────────────────────────────────────
    # Figure layout with marginal histograms
    # ─────────────────────────────────────────────────────────
    DARK   = '#111116'
    GRID_C = '#444455'
    plt.rcParams['text.color']       = 'white'
    plt.rcParams['axes.labelcolor']  = 'white'
    plt.rcParams['xtick.color']      = 'white'
    plt.rcParams['ytick.color']      = 'white'

    fig = plt.figure(figsize=(21, 8), facecolor=DARK)
    outer_gs = gridspec.GridSpec(1, 3, figure=fig, hspace=0.05, wspace=0.12,
                                 left=0.04, right=0.98, top=0.88, bottom=0.08)

    LIM  = 3.8
    ZLIM = 1.8
    panels = [
        ("Top-Down View (XY)",      "x_bar (kpc)", "y_bar (kpc)", (-LIM, LIM),  (-LIM,  LIM)),
        ("Side-On Peanut (XZ)",     "x_bar (kpc)", "z_bar (kpc)", (-LIM, LIM),  (-ZLIM, ZLIM)),
        ("Front-On Profile (YZ)",   "y_bar (kpc)", "z_bar (kpc)", (-LIM, LIM),  (-ZLIM, ZLIM)),
    ]

    axs_main, axs_top, axs_right = [], [], []
    for i, (title, xlabel, ylabel, xlim, ylim) in enumerate(panels):
        ax_m, ax_t, ax_r = setup_panel(outer_gs[i], fig, DARK, GRID_C,
                                        title, xlabel, ylabel, xlim, ylim)
        axs_main.append(ax_m)
        axs_top.append(ax_t)
        axs_right.append(ax_r)

    hist_bins = [
        (np.linspace(-LIM,  LIM,  N_HIST_BINS + 1), np.linspace(-LIM,  LIM,  N_HIST_BINS + 1)),
        (np.linspace(-LIM,  LIM,  N_HIST_BINS + 1), np.linspace(-ZLIM, ZLIM, N_HIST_BINS + 1)),
        (np.linspace(-LIM,  LIM,  N_HIST_BINS + 1), np.linspace(-ZLIM, ZLIM, N_HIST_BINS + 1)),
    ]

    pop_styles = [
        ('#00d8ff', 0.65),
        ('#ffd700', 0.80),
        ('#ff33aa', 0.80),
    ]

    top_step_lines, right_step_lines = [], []
    for pi in range(3):
        t_panel, r_panel = [], []
        for color, alpha in pop_styles:
            tl, = axs_top[pi].plot([], [], color=color, alpha=alpha, linewidth=1.0, solid_capstyle='round')
            rl, = axs_right[pi].plot([], [], color=color, alpha=alpha, linewidth=1.0, solid_capstyle='round')
            t_panel.append(tl)
            r_panel.append(rl)
        top_step_lines.append(t_panel)
        right_step_lines.append(r_panel)

    # ── Main scatter trails & dots ───────────────────────────
    trails_cyan_xy,    = axs_main[0].plot([], [], color='#00d8ff', alpha=0.008, linewidth=0.3)
    trails_cyan_xz,    = axs_main[1].plot([], [], color='#00d8ff', alpha=0.008, linewidth=0.3)
    trails_cyan_yz,    = axs_main[2].plot([], [], color='#00d8ff', alpha=0.008, linewidth=0.3)
    trails_gold_xy,    = axs_main[0].plot([], [], color='#ffd700', alpha=0.015, linewidth=0.4)
    trails_gold_xz,    = axs_main[1].plot([], [], color='#ffd700', alpha=0.015, linewidth=0.4)
    trails_gold_yz,    = axs_main[2].plot([], [], color='#ffd700', alpha=0.015, linewidth=0.4)
    trails_magenta_xy, = axs_main[0].plot([], [], color='#ff33aa', alpha=0.015, linewidth=0.4)
    trails_magenta_xz, = axs_main[1].plot([], [], color='#ff33aa', alpha=0.015, linewidth=0.4)
    trails_magenta_yz, = axs_main[2].plot([], [], color='#ff33aa', alpha=0.015, linewidth=0.4)

    dots_cyan_xy,    = axs_main[0].plot([], [], 'o', color='#00d8ff', markersize=0.5, alpha=0.25)
    dots_cyan_xz,    = axs_main[1].plot([], [], 'o', color='#00d8ff', markersize=0.5, alpha=0.25)
    dots_cyan_yz,    = axs_main[2].plot([], [], 'o', color='#00d8ff', markersize=0.5, alpha=0.25)
    dots_gold_xy,    = axs_main[0].plot([], [], 'o', color='#ffd700', markersize=0.6, alpha=0.45)
    dots_gold_xz,    = axs_main[1].plot([], [], 'o', color='#ffd700', markersize=0.6, alpha=0.45)
    dots_gold_yz,    = axs_main[2].plot([], [], 'o', color='#ffd700', markersize=0.6, alpha=0.45)
    dots_magenta_xy, = axs_main[0].plot([], [], 'o', color='#ff33aa', markersize=0.6, alpha=0.45)
    dots_magenta_xz, = axs_main[1].plot([], [], 'o', color='#ff33aa', markersize=0.6, alpha=0.45)
    dots_magenta_yz, = axs_main[2].plot([], [], 'o', color='#ff33aa', markersize=0.6, alpha=0.45)

    # Stationary bar ellipse and X-shape guide lines
    bar_xy = Ellipse((0, 0), width=5.0, height=2.4, color='#ff9944', fill=False,
                     linestyle='--', linewidth=1.2, alpha=0.7, label='Stationary Bar')
    axs_main[0].add_artist(bar_xy)
    axs_main[0].legend(facecolor=DARK, edgecolor=GRID_C, loc='upper right', fontsize=8)

    axs_main[1].plot([-2.0, 2.0], [-1.0,  1.0], color='#ff4444', linestyle=':', alpha=0.3, linewidth=1.0, label='X-shape guide')
    axs_main[1].plot([-2.0, 2.0], [ 1.0, -1.0], color='#ff4444', linestyle=':', alpha=0.3, linewidth=1.0)
    axs_main[1].legend(facecolor=DARK, edgecolor=GRID_C, loc='upper right', fontsize=8)

    plt.suptitle(
        "N-Body Stellar Flow in the Stationary Rotating Bar Frame\n"
        "(Gold/Magenta = prograde/retrograde X-shape peanut orbits; "
        "Marginals show per-population density)",
        fontsize=12, fontweight='bold', y=0.97, color='white'
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

    scatter_artists = [
        trails_cyan_xy, trails_cyan_xz, trails_cyan_yz,
        trails_gold_xy, trails_gold_xz, trails_gold_yz,
        trails_magenta_xy, trails_magenta_xz, trails_magenta_yz,
        dots_cyan_xy, dots_cyan_xz, dots_cyan_yz,
        dots_gold_xy, dots_gold_xz, dots_gold_yz,
        dots_magenta_xy, dots_magenta_xz, dots_magenta_yz,
    ]
    hist_artists = [line for panel in top_step_lines + right_step_lines for line in panel]
    all_artists = scatter_artists + hist_artists

    def init():
        for art in all_artists:
            art.set_data([], [])
        return all_artists

    def update(frame):
        trail_indices = (initial_phases[:, None] + frame - np.arange(n_tail)[None, :]) % n_orbit_steps
        pos_trail_bar = coords[np.arange(n_sample)[:, None], trail_indices, :]
        vx_bar_current = pos_trail_bar[:, 0, 3]

        xb_t = pos_trail_bar[:, :, 0]
        yb_t = pos_trail_bar[:, :, 1]
        zb_t = pos_trail_bar[:, :, 2]

        # In the bar frame no rotation needed
        is_cyan_f    = ~is_gold
        is_gold_f    = is_gold & (vx_bar_current >= 0.0)
        is_magenta_f = is_gold & (vx_bar_current < 0.0)

        c_idx = np.where(is_cyan_f)[0]
        g_idx = np.where(is_gold_f)[0]
        m_idx = np.where(is_magenta_f)[0]
        n_cy, n_go, n_ma = len(c_idx), len(g_idx), len(m_idx)

        # Fill trails
        x_trail_cyan[:n_cy, :n_tail]    = xb_t[c_idx, :]
        y_trail_cyan[:n_cy, :n_tail]    = yb_t[c_idx, :]
        z_trail_cyan[:n_cy, :n_tail]    = zb_t[c_idx, :]
        x_trail_gold[:n_go, :n_tail]    = xb_t[g_idx, :]
        y_trail_gold[:n_go, :n_tail]    = yb_t[g_idx, :]
        z_trail_gold[:n_go, :n_tail]    = zb_t[g_idx, :]
        x_trail_magenta[:n_ma, :n_tail] = xb_t[m_idx, :]
        y_trail_magenta[:n_ma, :n_tail] = yb_t[m_idx, :]
        z_trail_magenta[:n_ma, :n_tail] = zb_t[m_idx, :]

        trails_cyan_xy.set_data(x_trail_cyan[:n_cy, :].flatten(), y_trail_cyan[:n_cy, :].flatten())
        trails_cyan_xz.set_data(x_trail_cyan[:n_cy, :].flatten(), z_trail_cyan[:n_cy, :].flatten())
        trails_cyan_yz.set_data(y_trail_cyan[:n_cy, :].flatten(), z_trail_cyan[:n_cy, :].flatten())
        trails_gold_xy.set_data(x_trail_gold[:n_go, :].flatten(), y_trail_gold[:n_go, :].flatten())
        trails_gold_xz.set_data(x_trail_gold[:n_go, :].flatten(), z_trail_gold[:n_go, :].flatten())
        trails_gold_yz.set_data(y_trail_gold[:n_go, :].flatten(), z_trail_gold[:n_go, :].flatten())
        trails_magenta_xy.set_data(x_trail_magenta[:n_ma, :].flatten(), y_trail_magenta[:n_ma, :].flatten())
        trails_magenta_xz.set_data(x_trail_magenta[:n_ma, :].flatten(), z_trail_magenta[:n_ma, :].flatten())
        trails_magenta_yz.set_data(y_trail_magenta[:n_ma, :].flatten(), z_trail_magenta[:n_ma, :].flatten())

        dots_cyan_xy.set_data(xb_t[c_idx, 0], yb_t[c_idx, 0])
        dots_cyan_xz.set_data(xb_t[c_idx, 0], zb_t[c_idx, 0])
        dots_cyan_yz.set_data(yb_t[c_idx, 0], zb_t[c_idx, 0])
        dots_gold_xy.set_data(xb_t[g_idx, 0], yb_t[g_idx, 0])
        dots_gold_xz.set_data(xb_t[g_idx, 0], zb_t[g_idx, 0])
        dots_gold_yz.set_data(yb_t[g_idx, 0], zb_t[g_idx, 0])
        dots_magenta_xy.set_data(xb_t[m_idx, 0], yb_t[m_idx, 0])
        dots_magenta_xz.set_data(xb_t[m_idx, 0], zb_t[m_idx, 0])
        dots_magenta_yz.set_data(yb_t[m_idx, 0], zb_t[m_idx, 0])

        # ── Marginal histograms ──────────────────────────────
        pop_x_data = [xb_t[c_idx, 0], xb_t[g_idx, 0], xb_t[m_idx, 0]]
        pop_y_data = [yb_t[c_idx, 0], yb_t[g_idx, 0], yb_t[m_idx, 0]]
        pop_z_data = [zb_t[c_idx, 0], zb_t[g_idx, 0], zb_t[m_idx, 0]]

        panel_horiz = [pop_x_data, pop_x_data, pop_y_data]
        panel_vert  = [pop_y_data, pop_z_data, pop_z_data]

        for pi in range(3):
            bins_x, bins_y = hist_bins[pi]
            for pop_i in range(3):
                hx = panel_horiz[pi][pop_i]
                hy = panel_vert[pi][pop_i]

                if len(hx) > 1:
                    sx, sy = make_step_data(hx, bins_x)
                    top_step_lines[pi][pop_i].set_data(sx, sy)
                else:
                    top_step_lines[pi][pop_i].set_data([], [])

                if len(hy) > 1:
                    sy2, sy2_c = make_step_data(hy, bins_y)
                    right_step_lines[pi][pop_i].set_data(sy2_c, sy2)
                else:
                    right_step_lines[pi][pop_i].set_data([], [])

        return all_artists

    print("Generating stationary bar-frame animation frames...")
    n_frames = 2 if is_fast else 200
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
