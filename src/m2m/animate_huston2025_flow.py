"""
Stellar flow animation for the Huston2025 SynthPop model in the Galactic Centre frame.

Particles are sampled directly from the Huston2025 analytic bulge density, integrated
in an Agama multipole potential at the model's pattern speed. Bar coordinates are
rotated back into the GC inertial frame each frame so the bar outline spins visibly.

Output: results/m2m/huston2025_flow_animation.gif
"""
import json
import os
import sys

import agama
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

agama.setUnits(length=1, mass=1, velocity=1)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(project_root, "src/shared"))
sys.path.append(os.path.abspath(os.path.join(project_root, "../synthpop")))

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

def sample_ics(n_stars, potential, dens_func):
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
        ics[n_filled:n_filled + n_to_copy] = np.column_stack([
            xf[bound][:n_to_copy], yf[bound][:n_to_copy], zf[bound][:n_to_copy],
            vr_b * cos_p - vt_b * sin_p,
            vr_b * sin_p + vt_b * cos_p,
            vz_b,
        ])
        n_filled += n_to_copy

    return ics


# ─────────────────────────────────────────────────────────
# 3. Main
# ─────────────────────────────────────────────────────────

def main():
    is_fast = os.environ.get("FAST_TEST") == "1"

    print("Loading Huston2025 model from SynthPop...")
    total_density, bulge_density, omega, bar_angle_rad = load_huston2025()

    print("Building Agama triaxial multipole potential...")
    full_pot = agama.Potential(
        type="Multipole", density=total_density,
        lmax=16, symmetry="triaxial",
        gridsizeR=100, rmin=0.01, rmax=50.0,
    )

    n_sample = 100 if is_fast else 30_000
    ics = sample_ics(n_sample, full_pot, bulge_density)

    print(f"Integrating {n_sample:,} orbits...")
    n_orbit_steps = 10 if is_fast else 400
    total_time = 0.4  # ~400 Myr
    trajs = agama.orbit(
        potential=full_pot, ic=ics,
        time=total_time, trajsize=n_orbit_steps, Omega=omega,
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

    # ── Dark-theme figure ─────────────────────────────────
    DARK, GRID_C = "#111116", "#444455"
    for k in ("text.color", "axes.labelcolor", "xtick.color", "ytick.color"):
        plt.rcParams[k] = "white"

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), facecolor=DARK)

    def style_ax(ax, title, xlabel, ylabel, xlim, ylim):
        ax.set_facecolor(DARK)
        ax.grid(True, color=GRID_C, linestyle=":", alpha=0.4)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal")
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-ylim, ylim)

    style_ax(axs[0], "GC Plane (XY)", "x (kpc) [Sun at x≈−8.2]", "y (kpc)", 3.8, 3.8)
    style_ax(axs[1], "Side Profile (XZ)", "x (kpc)", "z (kpc)", 3.8, 1.8)
    style_ax(axs[2], "Front Profile (YZ)", "y (kpc)", "z (kpc)", 3.8, 1.8)

    def _lines(ax, color, alpha, lw):
        return ax.plot([], [], color=color, alpha=alpha, linewidth=lw)[0]

    def _dots(ax, color, ms, alpha):
        return ax.plot([], [], "o", color=color, markersize=ms, alpha=alpha)[0]

    CY, GO, MA = "#00d8ff", "#ffd700", "#ff33aa"

    tr_cy_xy = _lines(axs[0], CY, 0.008, 0.3)
    tr_cy_xz = _lines(axs[1], CY, 0.008, 0.3)
    tr_cy_yz = _lines(axs[2], CY, 0.008, 0.3)
    tr_go_xy = _lines(axs[0], GO, 0.015, 0.4)
    tr_go_xz = _lines(axs[1], GO, 0.015, 0.4)
    tr_go_yz = _lines(axs[2], GO, 0.015, 0.4)
    tr_ma_xy = _lines(axs[0], MA, 0.015, 0.4)
    tr_ma_xz = _lines(axs[1], MA, 0.015, 0.4)
    tr_ma_yz = _lines(axs[2], MA, 0.015, 0.4)

    dt_cy_xy = _dots(axs[0], CY, 0.5, 0.25)
    dt_cy_xz = _dots(axs[1], CY, 0.5, 0.25)
    dt_cy_yz = _dots(axs[2], CY, 0.5, 0.25)
    dt_go_xy = _dots(axs[0], GO, 0.6, 0.45)
    dt_go_xz = _dots(axs[1], GO, 0.6, 0.45)
    dt_go_yz = _dots(axs[2], GO, 0.6, 0.45)
    dt_ma_xy = _dots(axs[0], MA, 0.6, 0.45)
    dt_ma_xz = _dots(axs[1], MA, 0.6, 0.45)
    dt_ma_yz = _dots(axs[2], MA, 0.6, 0.45)

    # Rotating bar ellipse in GC frame
    bar_patch = Ellipse(
        (0, 0), width=5.0, height=2.4,
        color="#ff9944", fill=False, linestyle="--",
        linewidth=1.2, alpha=0.7, label="Rotating Bar Outline",
    )
    axs[0].add_artist(bar_patch)
    axs[0].legend(facecolor=DARK, edgecolor=GRID_C, loc="upper right", fontsize=9)

    for ax in [axs[1], axs[2]]:
        ax.add_artist(plt.Circle((0, 0), 2.5, color=GRID_C, fill=False, linestyle=":", alpha=0.4))

    plt.suptitle(
        "Huston2025 Stellar Flow in the Galactic Centre Frame\n"
        "(Gold/Magenta: prograde/retrograde X-shape peanut orbits  "
        f"| Ω_p = {omega:.1f} km/s/kpc, φ_bar = {np.degrees(bar_angle_rad):.1f}°)",
        fontsize=13, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    # ── Trail buffers ─────────────────────────────────────
    n_frames = 2 if is_fast else 200
    n_tail = 6

    def _trail_buf():
        buf = np.empty((n_sample, n_tail + 1))
        buf[:, n_tail] = np.nan
        return buf

    xt_cy, yt_cy, zt_cy = _trail_buf(), _trail_buf(), _trail_buf()
    xt_go, yt_go, zt_go = _trail_buf(), _trail_buf(), _trail_buf()
    xt_ma, yt_ma, zt_ma = _trail_buf(), _trail_buf(), _trail_buf()

    all_artists = (
        tr_cy_xy, tr_cy_xz, tr_cy_yz,
        tr_go_xy, tr_go_xz, tr_go_yz,
        tr_ma_xy, tr_ma_xz, tr_ma_yz,
        dt_cy_xy, dt_cy_xz, dt_cy_yz,
        dt_go_xy, dt_go_xz, dt_go_yz,
        dt_ma_xy, dt_ma_xz, dt_ma_yz,
    )

    def init():
        for a in all_artists:
            a.set_data([], [])
        return all_artists

    def update(frame):
        trail_idx = (
            initial_phases[:, None] + frame - np.arange(n_tail)[None, :]
        ) % n_orbit_steps

        pos_trail = coords[np.arange(n_sample)[:, None], trail_idx, :]
        vx_curr = pos_trail[:, 0, 3]

        # Rotate bar frame → GC frame for each trail point
        t_trail = (frame - np.arange(n_tail)) * dt_frame
        alpha_t = bar_angle_rad + omega * t_trail   # shape (n_tail,)
        cos_a = np.cos(alpha_t)[None, :]             # (1, n_tail)
        sin_a = np.sin(alpha_t)[None, :]

        xb = pos_trail[:, :, 0]
        yb = pos_trail[:, :, 1]
        zb = pos_trail[:, :, 2]

        x_gc = xb * cos_a - yb * sin_a
        y_gc = xb * sin_a + yb * cos_a
        z_gc = zb

        is_cyan_f = ~is_gold
        is_gold_f = is_gold & (vx_curr >= 0.0)
        is_mage_f = is_gold & (vx_curr < 0.0)
        cy_i = np.where(is_cyan_f)[0]
        go_i = np.where(is_gold_f)[0]
        ma_i = np.where(is_mage_f)[0]
        nc, ng, nm = len(cy_i), len(go_i), len(ma_i)

        xt_cy[:nc, :n_tail] = x_gc[cy_i]
        yt_cy[:nc, :n_tail] = y_gc[cy_i]
        zt_cy[:nc, :n_tail] = z_gc[cy_i]
        xt_go[:ng, :n_tail] = x_gc[go_i]
        yt_go[:ng, :n_tail] = y_gc[go_i]
        zt_go[:ng, :n_tail] = z_gc[go_i]
        xt_ma[:nm, :n_tail] = x_gc[ma_i]
        yt_ma[:nm, :n_tail] = y_gc[ma_i]
        zt_ma[:nm, :n_tail] = z_gc[ma_i]

        tr_cy_xy.set_data(xt_cy[:nc].flatten(), yt_cy[:nc].flatten())
        tr_cy_xz.set_data(xt_cy[:nc].flatten(), zt_cy[:nc].flatten())
        tr_cy_yz.set_data(yt_cy[:nc].flatten(), zt_cy[:nc].flatten())
        tr_go_xy.set_data(xt_go[:ng].flatten(), yt_go[:ng].flatten())
        tr_go_xz.set_data(xt_go[:ng].flatten(), zt_go[:ng].flatten())
        tr_go_yz.set_data(yt_go[:ng].flatten(), zt_go[:ng].flatten())
        tr_ma_xy.set_data(xt_ma[:nm].flatten(), yt_ma[:nm].flatten())
        tr_ma_xz.set_data(xt_ma[:nm].flatten(), zt_ma[:nm].flatten())
        tr_ma_yz.set_data(yt_ma[:nm].flatten(), zt_ma[:nm].flatten())

        dt_cy_xy.set_data(x_gc[cy_i, 0], y_gc[cy_i, 0])
        dt_cy_xz.set_data(x_gc[cy_i, 0], z_gc[cy_i, 0])
        dt_cy_yz.set_data(y_gc[cy_i, 0], z_gc[cy_i, 0])
        dt_go_xy.set_data(x_gc[go_i, 0], y_gc[go_i, 0])
        dt_go_xz.set_data(x_gc[go_i, 0], z_gc[go_i, 0])
        dt_go_yz.set_data(y_gc[go_i, 0], z_gc[go_i, 0])
        dt_ma_xy.set_data(x_gc[ma_i, 0], y_gc[ma_i, 0])
        dt_ma_xz.set_data(x_gc[ma_i, 0], z_gc[ma_i, 0])
        dt_ma_yz.set_data(y_gc[ma_i, 0], z_gc[ma_i, 0])

        # Spin the bar outline in the GC frame
        bar_patch.angle = np.degrees(bar_angle_rad + omega * frame * dt_frame)

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
