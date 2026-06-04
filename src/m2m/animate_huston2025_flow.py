"""
Stellar flow animation for the Huston2025 SynthPop model in the Galactic Centre frame.

Particles are sampled directly from the Huston2025 analytic bulge density, integrated
in an Agama multipole potential + NFW DM halo at the model's pattern speed (CW).
Bar coordinates are rotated back into the GC inertial frame each frame.

Output: results/m2m/huston2025_flow_animation.gif
"""
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import agama

# Set Agama units (kpc, Msun, km/s)
agama.setUnits(length=1, mass=1, velocity=1)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# Append paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(project_root, 'src/shared'))
sys.path.append(os.path.abspath(os.path.join(project_root, '../synthpop')))

import synthpop

# ─────────────────────────────────────────────────────────
# 1. Load Huston2025 model
# ─────────────────────────────────────────────────────────

def load_huston2025(model_name="Huston2025"):
    config = {
        "MANDATORY": {"name_for_output": "h25_anim_gc", "model_name": model_name},
        "SIGHTLINES": {
            "l_set": [0.0], "l_set_type": "list",
            "b_set": [0.0], "b_set_type": "list",
            "solid_angle": 0.01, "solid_angle_unit": "deg^2",
        },
    }
    tmp_conf = f"tmp_h25_gc_{model_name}.json"
    with open(tmp_conf, "w") as f:
        json.dump(config, f)
    sp = synthpop.SynthPop(tmp_conf)
    sp.init_populations()
    os.remove(tmp_conf)

    bar_angle_rad = 0.0
    omega_p = 50.357
    for pop in sp.populations:
        if hasattr(pop.population_density, "bar_ang"):
            bar_angle_rad = pop.population_density.bar_ang
        if hasattr(pop.kinematics, "omega_p"):
            omega_p = pop.kinematics.omega_p

    print(f"  Huston2025 bar_angle = {np.degrees(bar_angle_rad):.2f} deg, "
          f"Omega_p = {omega_p:.3f} km/s/kpc")

    def _mass_corr(pop):
        if pop.population_density.density_unit == "number":
            avg_m = pop.imf.average_mass(min_mass=pop.min_mass, max_mass=pop.max_mass)
            corr = getattr(pop.pop_params, "av_mass_corr", 1.0) or 1.0
            return avg_m * corr
        return 1.0

    def total_density(pos):
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        r = np.sqrt(x**2 + y**2)
        phi_sp = np.arctan2(y, x) + bar_angle_rad
        rho = np.zeros(len(pos))
        for pop in sp.populations:
            rho += pop.population_density.density(r, phi_sp, z) * _mass_corr(pop)
        return rho

    def bulge_density(pos):
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        r = np.sqrt(x**2 + y**2)
        phi_sp = np.arctan2(y, x) + bar_angle_rad
        rho = np.zeros(len(pos))
        for pop in sp.populations:
            if "bulge" in pop.name.lower():
                rho += pop.population_density.density(r, phi_sp, z) * _mass_corr(pop)
        return rho

    return total_density, bulge_density, omega_p, bar_angle_rad

# ─────────────────────────────────────────────────────────
# 2. Rejection-sample initial conditions
# ─────────────────────────────────────────────────────────

def sample_ics(n_stars, potential, dens_func, omega):
    print(f"  Sampling {n_stars:,} particles...")
    ics = np.zeros((n_stars, 6), dtype=np.float32)
    n_filled = 0
    DENS_PEAK = 1e9
    VR_DISP, VT_FRAC, VT_DISP, VZ_DISP = 160.0, 0.4, 160.0, 100.0

    while n_filled < n_stars:
        n_batch = min((n_stars - n_filled) * 15, 1_000_000)
        xs = np.random.uniform(-4.5, 4.5, n_batch)
        ys = np.random.uniform(-4.5, 4.5, n_batch)
        zs = np.random.uniform(-1.8, 1.8, n_batch)
        pos = np.column_stack([xs, ys, zs])
        keep = np.random.uniform(0, DENS_PEAK, n_batch) < dens_func(pos)
        xf, yf, zf = xs[keep], ys[keep], zs[keep]
        if len(xf) == 0:
            continue

        rf = np.maximum(np.sqrt(xf**2 + yf**2), 0.01)
        forces = potential.force(np.column_stack([xf, yf, zf]))
        vc = np.sqrt(rf * np.abs((forces[:, 0] * xf + forces[:, 1] * yf) / rf))
        v_esc = np.sqrt(np.maximum(
            -2.0 * potential.potential(np.column_stack([xf, yf, zf])), 0.0
        ))

        vr = np.random.normal(0, VR_DISP, len(xf))
        vt = np.random.normal(VT_FRAC * vc, VT_DISP, len(xf))
        vz = np.random.normal(0, VZ_DISP, len(xf))
        bound = np.sqrt(vr**2 + vt**2 + vz**2) < (v_esc * 0.95)
        n_to_copy = min(int(np.sum(bound)), n_stars - n_filled)
        if n_to_copy == 0:
            continue

        cos_p = xf[bound][:n_to_copy] / rf[bound][:n_to_copy]
        sin_p = yf[bound][:n_to_copy] / rf[bound][:n_to_copy]
        vr_b, vt_b, vz_b = vr[bound][:n_to_copy], vt[bound][:n_to_copy], vz[bound][:n_to_copy]
        
        # Clockwise rotation coordinates:
        vx_rot = vr_b * cos_p + vt_b * sin_p
        vy_rot = vr_b * sin_p - vt_b * cos_p
        
        # Convert to CW inertial frame
        vx_b = vx_rot + omega * yf[bound][:n_to_copy]
        vy_b = vy_rot - omega * xf[bound][:n_to_copy]
        
        ics[n_filled:n_filled + n_to_copy] = np.column_stack([
            xf[bound][:n_to_copy], yf[bound][:n_to_copy], zf[bound][:n_to_copy],
            vx_b, vy_b, vz_b
        ])
        n_filled += n_to_copy

    return ics

# ─────────────────────────────────────────────────────────
# Marginal histogram helpers
# ─────────────────────────────────────────────────────────

N_HIST_BINS = 60

def make_step_data(values, bins, scale=1.0):
    """Return (x, y) for a step histogram line drawable via set_data."""
    counts, _ = np.histogram(values, bins=bins)
    if scale > 0:
        counts = counts / scale
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

# ─────────────────────────────────────────────────────────
# 3. Main
# ─────────────────────────────────────────────────────────

def main():
    fit_path = os.path.join(project_root, "results/m2m/h25_m2m_fit_results.npz")
    if not os.path.exists(fit_path):
        print(f"Error: {fit_path} not found. Please run validate_h25_with_m2m.py first.")
        return

    print("Loading M2M fit results for Huston2025...")
    data = np.load(fit_path)
    weights = data['weights']
    ics = data['ics']

    print("Loading Huston2025 model from SynthPop...")
    total_density, bulge_density, omega, bar_angle_rad = load_huston2025()

    print("Building Agama potential (stellar multipole + NFW DM halo)...")
    stellar_pot = agama.Potential(
        type="Multipole", density=total_density,
        lmax=16, symmetry="triaxial",
        gridsizeR=100, rmin=0.01, rmax=50.0,
    )
    dm_pot = agama.Potential(type='NFW', mass=1.0e12, scaleRadius=16.0)
    full_pot = agama.Potential(stellar_pot, dm_pot)

    is_fast = os.environ.get("FAST_TEST") == "1"
    n_sample = 100 if is_fast else 30_000
    p_weights = weights / np.sum(weights)
    sampled_idx = np.random.choice(len(ics), size=n_sample, p=p_weights, replace=False)
    sampled_ics = ics[sampled_idx]

    print(f"Integrating {n_sample:,} M2M-weighted orbits in potential (Omega = {-omega:.2f})...")
    n_orbit_steps = 10 if is_fast else 400
    n_frames = 2 if is_fast else 200
    total_time = 0.4  # ~400 Myr
    trajs = agama.orbit(
        potential=full_pot, ic=sampled_ics,
        time=total_time, trajsize=n_orbit_steps, Omega=-omega,
    )
    coords = np.stack([trajs[p][1] for p in range(n_sample)])

    # Classify X-shape orbits
    print("Classifying orbits...")
    is_gold = np.zeros(n_sample, dtype=bool)
    for p in range(n_sample):
        cp = coords[p]
        x_max = np.max(np.abs(cp[:, 0]))
        z_max = np.max(np.abs(cp[:, 2]))
        if (0.38 <= z_max <= 1.2) and (0.8 <= x_max <= 2.4):
            if 0.22 <= z_max / x_max <= 0.65:
                is_gold[p] = True

    n_gold = int(is_gold.sum())
    print(f"  {n_gold} gold X-shape orbits, "
          f"{n_sample - n_gold} cyan standard "
          f"({n_gold / n_sample * 100:.1f}%)")

    initial_phases = np.random.randint(0, n_orbit_steps, size=n_sample)
    dt_frame = total_time / n_orbit_steps

    # ── Dark-theme setup ──────────────────────────────────
    DARK, GRID_C = "#111116", "#444455"
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'

    fig = plt.figure(figsize=(21, 8), facecolor=DARK)
    outer_gs = gridspec.GridSpec(1, 3, figure=fig, hspace=0.05, wspace=0.12,
                                 left=0.04, right=0.98, top=0.88, bottom=0.08)

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
        (np.linspace(-LIM,  LIM,  N_HIST_BINS + 1), np.linspace(-LIM,  LIM,  N_HIST_BINS + 1)),
        (np.linspace(-LIM,  LIM,  N_HIST_BINS + 1), np.linspace(-ZLIM, ZLIM, N_HIST_BINS + 1)),
        (np.linspace(-LIM,  LIM,  N_HIST_BINS + 1), np.linspace(-ZLIM, ZLIM, N_HIST_BINS + 1)),
    ]

    pop_styles = [
        ('#00d8ff', 0.65),  # cyan
        ('#ffd700', 0.80),  # gold
        ('#ff33aa', 0.80),  # magenta
    ]

    top_step_lines = []
    right_step_lines = []
    for pi in range(3):
        t_panel, r_panel = [], []
        for color, alpha in pop_styles:
            tl, = axs_top[pi].plot([], [], color=color, alpha=alpha, linewidth=1.0, solid_capstyle='round')
            rl, = axs_right[pi].plot([], [], color=color, alpha=alpha, linewidth=1.0, solid_capstyle='round')
            t_panel.append(tl)
            r_panel.append(rl)
        top_step_lines.append(t_panel)
        right_step_lines.append(r_panel)

    tr_cy_xy, = axs_main[0].plot([], [], color='#00d8ff', alpha=0.008, linewidth=0.3)
    tr_cy_xz, = axs_main[1].plot([], [], color='#00d8ff', alpha=0.008, linewidth=0.3)
    tr_cy_yz, = axs_main[2].plot([], [], color='#00d8ff', alpha=0.008, linewidth=0.3)
    tr_go_xy, = axs_main[0].plot([], [], color='#ffd700', alpha=0.015, linewidth=0.4)
    tr_go_xz, = axs_main[1].plot([], [], color='#ffd700', alpha=0.015, linewidth=0.4)
    tr_go_yz, = axs_main[2].plot([], [], color='#ffd700', alpha=0.015, linewidth=0.4)
    tr_ma_xy, = axs_main[0].plot([], [], color='#ff33aa', alpha=0.015, linewidth=0.4)
    tr_ma_xz, = axs_main[1].plot([], [], color='#ff33aa', alpha=0.015, linewidth=0.4)
    tr_ma_yz, = axs_main[2].plot([], [], color='#ff33aa', alpha=0.015, linewidth=0.4)

    dt_cy_xy, = axs_main[0].plot([], [], 'o', color='#00d8ff', markersize=0.5, alpha=0.25)
    dt_cy_xz, = axs_main[1].plot([], [], 'o', color='#00d8ff', markersize=0.5, alpha=0.25)
    dt_cy_yz, = axs_main[2].plot([], [], 'o', color='#00d8ff', markersize=0.5, alpha=0.25)
    dt_go_xy, = axs_main[0].plot([], [], 'o', color='#ffd700', markersize=0.6, alpha=0.45)
    dt_go_xz, = axs_main[1].plot([], [], 'o', color='#ffd700', markersize=0.6, alpha=0.45)
    dt_go_yz, = axs_main[2].plot([], [], 'o', color='#ffd700', markersize=0.6, alpha=0.45)
    dt_ma_xy, = axs_main[0].plot([], [], 'o', color='#ff33aa', markersize=0.6, alpha=0.45)
    dt_ma_xz, = axs_main[1].plot([], [], 'o', color='#ff33aa', markersize=0.6, alpha=0.45)
    dt_ma_yz, = axs_main[2].plot([], [], 'o', color='#ff33aa', markersize=0.6, alpha=0.45)

    # Reference boundary circles in XZ, YZ
    for ax in axs_main[1:]:
        boundary = plt.Circle((0, 0), 2.5, color='#444455', fill=False, linestyle=':', alpha=0.4)
        ax.add_artist(boundary)

    plt.suptitle(
        f"Huston2025 Stellar Flow in the Galactic Centre Frame ($\\Omega_p = {omega:.2f}$ km/s/kpc)\n"
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

    scatter_artists = [
        tr_cy_xy, tr_cy_xz, tr_cy_yz,
        tr_go_xy, tr_go_xz, tr_go_yz,
        tr_ma_xy, tr_ma_xz, tr_ma_yz,
        dt_cy_xy, dt_cy_xz, dt_cy_yz,
        dt_go_xy, dt_go_xz, dt_go_yz,
        dt_ma_xy, dt_ma_xz, dt_ma_yz,
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

    def update(frame):
        t_Myr = frame * dt_frame * 977.8
        time_text.set_text(f"t = {t_Myr:.1f} Myr")
        trail_indices = (initial_phases[:, None] + frame - np.arange(n_tail)[None, :]) % n_orbit_steps
        pos_trail_bar = coords[np.arange(n_sample)[:, None], trail_indices, :]
        vx_bar_current = pos_trail_bar[:, 0, 3]

        # Rotate into GC frame (clockwise)
        t_trail = (frame - np.arange(n_tail)) * dt_frame
        alpha_t = bar_angle_rad - omega * t_trail
        cos_a = np.cos(alpha_t)[None, :]
        sin_a = np.sin(alpha_t)[None, :]

        xb_t = pos_trail_bar[:, :, 0]
        yb_t = pos_trail_bar[:, :, 1]
        zb_t = pos_trail_bar[:, :, 2]

        x_gc = xb_t * cos_a - yb_t * sin_a
        y_gc = xb_t * sin_a + yb_t * cos_a
        z_gc = zb_t

        is_cyan_f    = ~is_gold
        is_gold_f    = is_gold & (vx_bar_current >= 0.0)
        is_magenta_f = is_gold & (vx_bar_current < 0.0)

        c_idx = np.where(is_cyan_f)[0]
        g_idx = np.where(is_gold_f)[0]
        m_idx = np.where(is_magenta_f)[0]
        n_cy, n_go, n_ma = len(c_idx), len(g_idx), len(m_idx)

        x_trail_cyan[:n_cy, :n_tail]    = x_gc[c_idx, :]
        y_trail_cyan[:n_cy, :n_tail]    = y_gc[c_idx, :]
        z_trail_cyan[:n_cy, :n_tail]    = z_gc[c_idx, :]
        x_trail_gold[:n_go, :n_tail]    = x_gc[g_idx, :]
        y_trail_gold[:n_go, :n_tail]    = y_gc[g_idx, :]
        z_trail_gold[:n_go, :n_tail]    = z_gc[g_idx, :]
        x_trail_magenta[:n_ma, :n_tail] = x_gc[m_idx, :]
        y_trail_magenta[:n_ma, :n_tail] = y_gc[m_idx, :]
        z_trail_magenta[:n_ma, :n_tail] = z_gc[m_idx, :]

        tr_cy_xy.set_data(x_trail_cyan[:n_cy, :].flatten(), y_trail_cyan[:n_cy, :].flatten())
        tr_cy_xz.set_data(x_trail_cyan[:n_cy, :].flatten(), z_trail_cyan[:n_cy, :].flatten())
        tr_cy_yz.set_data(y_trail_cyan[:n_cy, :].flatten(), z_trail_cyan[:n_cy, :].flatten())
        tr_go_xy.set_data(x_trail_gold[:n_go, :].flatten(), y_trail_gold[:n_go, :].flatten())
        tr_go_xz.set_data(x_trail_gold[:n_go, :].flatten(), z_trail_gold[:n_go, :].flatten())
        tr_go_yz.set_data(y_trail_gold[:n_go, :].flatten(), z_trail_gold[:n_go, :].flatten())
        tr_ma_xy.set_data(x_trail_magenta[:n_ma, :].flatten(), y_trail_magenta[:n_ma, :].flatten())
        tr_ma_xz.set_data(x_trail_magenta[:n_ma, :].flatten(), z_trail_magenta[:n_ma, :].flatten())
        tr_ma_yz.set_data(y_trail_magenta[:n_ma, :].flatten(), z_trail_magenta[:n_ma, :].flatten())

        dt_cy_xy.set_data(x_gc[c_idx, 0], y_gc[c_idx, 0])
        dt_cy_xz.set_data(x_gc[c_idx, 0], z_gc[c_idx, 0])
        dt_cy_yz.set_data(y_gc[c_idx, 0], z_gc[c_idx, 0])
        dt_go_xy.set_data(x_gc[g_idx, 0], y_gc[g_idx, 0])
        dt_go_xz.set_data(x_gc[g_idx, 0], z_gc[g_idx, 0])
        dt_go_yz.set_data(y_gc[g_idx, 0], z_gc[g_idx, 0])
        dt_ma_xy.set_data(x_gc[m_idx, 0], y_gc[m_idx, 0])
        dt_ma_xz.set_data(x_gc[m_idx, 0], z_gc[m_idx, 0])
        dt_ma_yz.set_data(y_gc[m_idx, 0], z_gc[m_idx, 0])

        # ── Marginal histograms ──────────────────────────────
        pop_x_data = [x_gc[c_idx, 0], x_gc[g_idx, 0], x_gc[m_idx, 0]]
        pop_y_data = [y_gc[c_idx, 0], y_gc[g_idx, 0], y_gc[m_idx, 0]]
        pop_z_data = [z_gc[c_idx, 0], z_gc[g_idx, 0], z_gc[m_idx, 0]]

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
                    sy2, sy2_c = make_step_data(hy, bins_y, scale=peak_y)
                    right_step_lines[pi][pop_i].set_data(sy2_c, sy2)
                else:
                    right_step_lines[pi][pop_i].set_data([], [])

        return all_artists

    print("Rendering GC-frame animation...")
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, init_func=init, blit=True
    )

    out_dir = os.path.join(project_root, "results/m2m")
    os.makedirs(out_dir, exist_ok=True)
    out_gif = os.path.join(out_dir, "huston2025_flow_animation.gif")
    anim.save(out_gif, writer="pillow", fps=20, dpi=120)
    plt.close()
    print(f"Saved → {out_gif}")

if __name__ == "__main__":
    main()
