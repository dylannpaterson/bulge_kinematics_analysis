"""
Analyse SynthPop NP-kinematics output and compare to VIRAC2 observations.

Processing applied to synthetic stars exactly mirrors run_obs_data_analysis.py:
  1. Subtract Surot 2-D A_Ks  →  ks0 = 2MASS_Ks_obs − A_Ks(2D)
  2. Keep  12 < ks0 < 14
  3. Iterative 5-σ clipping on mu_l, mu_b  (max 4 iterations, tol=0.1)
  4. Bin ks0 into 20 uniform bins [12, 14]  (Δks = 0.1)
  5. In each bin compute mean(mu_l), mean(mu_b), std(mu_l), std(mu_b)

Observed data is read from:
  results/reduction/virac2_magnitude_binned_moments.csv
which already contains the intrinsic binned moments after measurement-error
deconvolution (see mixture_models.get_intrinsic_moments_with_errors).

Usage:
  Run SynthPop first:
      cd /path/to/bulge_model_pipeline
      python3 -m synthpop np_kinematics_test.synthpop_conf

  Then run this script:
      python3 src/m2m/analyse_synthpop_np.py
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
shared_dir   = os.path.join(project_root, "src/shared")
if shared_dir not in sys.path:
    sys.path.insert(0, shared_dir)

from surot_extinction import get_surot_a_ks

SYNTHPOP_OUT_DIR = os.path.join(project_root, "results/synthpop_np_test")
OBS_BINNED_CSV   = os.path.join(project_root, "results/reduction/virac2_magnitude_binned_moments.csv")
OUTPUT_FIGURE    = os.path.join(project_root, "results/inversion/synthpop_np_comparison.png")

# Sightlines to compare
SIGHTLINES = [
    {"l": 5.0,  "b": -3.0},
    {"l": 0.0,  "b": -9.0},
]

# Magnitude binning (must match run_obs_data_analysis.py exactly)
KS0_BIN_EDGES   = np.linspace(12.0, 14.0, 21)
KS0_BIN_CENTERS = 0.5 * (KS0_BIN_EDGES[:-1] + KS0_BIN_EDGES[1:])


# ---------------------------------------------------------------------------
# Helpers — mirror run_obs_data_analysis.py exactly
# ---------------------------------------------------------------------------

def sigma_clip(mu_l: np.ndarray, mu_b: np.ndarray,
               sigma: float = 5.0, max_iter: int = 4, tol: float = 0.1
               ) -> np.ndarray:
    """Return boolean keep-mask after iterative sigma clipping on (mu_l, mu_b)."""
    keep = np.ones(len(mu_l), dtype=bool)
    for _ in range(max_iter):
        std_l = mu_l[keep].std()
        std_b = mu_b[keep].std()
        med_l = np.median(mu_l[keep])
        med_b = np.median(mu_b[keep])

        new_keep = keep & (
            (np.abs(mu_l - med_l) < sigma * std_l) &
            (np.abs(mu_b - med_b) < sigma * std_b)
        )
        diff_l = abs(std_l - mu_l[new_keep].std()) if new_keep.sum() > 1 else 0
        diff_b = abs(std_b - mu_b[new_keep].std()) if new_keep.sum() > 1 else 0

        keep = new_keep
        if (diff_l < tol and diff_b < tol) or keep.sum() < 10:
            break
    return keep


def process_synthpop_df(df: pd.DataFrame, l_deg: float, b_deg: float) -> pd.DataFrame | None:
    """
    Apply the same pipeline as run_obs_data_analysis.py to a SynthPop output
    DataFrame and return a per-bin summary of kinematic moments.

    Parameters
    ----------
    df      : SynthPop output DataFrame (must include '2MASS_Ks', 'mu_l', 'mu_b')
    l_deg   : sightline Galactic longitude (for Surot 2-D look-up)
    b_deg   : sightline Galactic latitude

    Returns
    -------
    binned DataFrame with columns:
        ks0_center, n_stars, mu_l_mean, mu_l_std, mu_b_mean, mu_b_std
    """
    # Identify Ks column name (SynthPop may use '2MASS_Ks' or 'Ks')
    ks_col = next((c for c in ("2MASS_Ks", "Ks") if c in df.columns), None)
    if ks_col is None:
        print(f"  WARNING: no Ks column found in DataFrame. Columns: {list(df.columns)}")
        return None

    # Identify mu_l, mu_b columns
    mul_col = next((c for c in ("mu_l", "mul") if c in df.columns), None)
    mub_col = next((c for c in ("mu_b", "mub") if c in df.columns), None)
    if mul_col is None or mub_col is None:
        print(f"  WARNING: mu columns not found. Columns: {list(df.columns)}")
        return None

    # 1. Subtract Surot 2-D A_Ks
    a_ks = float(get_surot_a_ks(l_deg, b_deg)[0])
    df = df.copy()
    df["ks0"] = df[ks_col] - a_ks

    # 2. Magnitude cut
    df = df[(df["ks0"] > 12.0) & (df["ks0"] < 14.0)].copy()
    if len(df) == 0:
        print(f"  WARNING: No stars survive ks0 ∈ (12,14) at (l={l_deg}, b={b_deg})")
        return None

    # 3. Sigma clipping (no measurement-error cut — synthetic stars have none)
    keep = sigma_clip(df[mul_col].values, df[mub_col].values)
    df = df[keep].copy()
    print(f"  Stars after σ-clip: {len(df):,d}  (removed {(~keep).sum():,d})")

    # 4. Bin by ks0 and compute moments
    rows = []
    for i, kc in enumerate(KS0_BIN_CENTERS):
        mask = (df["ks0"] >= KS0_BIN_EDGES[i]) & (df["ks0"] < KS0_BIN_EDGES[i + 1])
        sub = df[mask]
        if len(sub) < 5:
            continue
        rows.append({
            "ks0_center": kc,
            "n_stars":    len(sub),
            "mu_l_mean":  sub[mul_col].mean(),
            "mu_l_std":   sub[mul_col].std(),
            "mu_b_mean":  sub[mub_col].mean(),
            "mu_b_std":   sub[mub_col].std(),
        })

    return pd.DataFrame(rows) if rows else None


def load_obs_binned(l_deg: float, b_deg: float, tol: float = 0.25) -> pd.DataFrame | None:
    """Load VIRAC2 binned moments for the sightline nearest to (l_deg, b_deg)."""
    obs = pd.read_csv(OBS_BINNED_CSV)
    dist = np.sqrt((obs["l"] - l_deg) ** 2 + (obs["b"] - b_deg) ** 2)
    nearest_mask = dist < tol
    if not nearest_mask.any():
        best = obs.loc[dist.idxmin()]
        print(f"  WARNING: No obs pixel within {tol}° of ({l_deg},{b_deg}). "
              f"Nearest: l={best['l']:.2f}, b={best['b']:.2f} (dist={dist.min():.2f}°)")
        # Use nearest anyway
        nearest_mask = dist == dist.min()
    return obs[nearest_mask].copy()


def find_synthpop_csv(l_deg: float, b_deg: float) -> str | None:
    """Locate the SynthPop output CSV for a given sightline."""
    # SynthPop names files like:  <name>_l<L>_b<B>_<pop>.csv
    pattern = os.path.join(
        SYNTHPOP_OUT_DIR,
        f"np_kinematics_test_l{l_deg:.3f}_b{b_deg:.3f}*.csv"
    )
    files = glob.glob(pattern)
    # Also try with negative b formatted differently
    if not files:
        pattern = os.path.join(
            SYNTHPOP_OUT_DIR,
            f"*l{l_deg:.3f}*b{b_deg:.3f}*.csv"
        )
        files = glob.glob(pattern)
    return files[0] if files else None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(results: list[dict]) -> None:
    """
    Plot binned mu_l and mu_b vs K_s,0 for all sightlines side-by-side.

    Each sightline gets one column; rows are mu_l (top) and mu_b (bottom).
    """
    n_sl = len(results)
    fig = plt.figure(figsize=(5 * n_sl + 1, 9))
    gs  = GridSpec(2, n_sl, figure=fig, hspace=0.45, wspace=0.32,
                   top=0.90, bottom=0.10)

    blue  = "#2a7fbf"
    green = "#2ba04b"
    grey  = "#555555"

    for col, res in enumerate(results):
        l_deg, b_deg = res["l"], res["b"]
        sim   = res["sim"]     # binned DataFrame or None
        obs   = res["obs"]     # binned DataFrame or None

        for row, (moment, ylabel) in enumerate([
            ("mu_l",  r"$\langle\mu_{\ell}\cos b\rangle$  (mas yr$^{-1}$)"),
            ("mu_b",  r"$\langle\mu_b\rangle$  (mas yr$^{-1}$)"),
        ]):
            ax = fig.add_subplot(gs[row, col])

            # Observed
            # virac2_magnitude_binned_moments.csv columns: mu_l, mu_b, sig_l, sig_b,
            # err_mu_l_boot, err_mu_b_boot (bootstrap uncertainty on the mean)
            if obs is not None:
                comp = "l" if moment == "mu_l" else "b"
                obs_mu_col  = f"mu_{comp}"    # intrinsic mean
                obs_sig_col = f"sig_{comp}"   # intrinsic dispersion
                obs_err_col = f"err_mu_{comp}_boot"  # uncertainty on the mean

                if obs_mu_col in obs.columns:
                    ax.errorbar(
                        obs["ks0_center"], obs[obs_mu_col],
                        yerr=(obs[obs_err_col] if obs_err_col in obs.columns else None),
                        fmt="o", color=grey, ms=4, lw=1.2, capsize=3,
                        label="VIRAC2 (obs)", zorder=5
                    )

            # Synthetic (NP model)
            if sim is not None and f"{moment}_mean" in sim.columns:
                ax.plot(
                    sim["ks0_center"], sim[f"{moment}_mean"],
                    "s-", color=blue, ms=4, lw=1.5,
                    label="SynthPop NP", zorder=4
                )
                ax.fill_between(
                    sim["ks0_center"],
                    sim[f"{moment}_mean"] - sim[f"{moment}_std"],
                    sim[f"{moment}_mean"] + sim[f"{moment}_std"],
                    alpha=0.18, color=blue
                )

            ax.set_xlabel(r"$K_{s,0}$ (mag)", fontsize=10)
            ax.set_ylabel(ylabel if col == 0 else "", fontsize=10)
            ax.tick_params(labelsize=9)
            ax.axhline(0, color="0.7", lw=0.8, ls="--")

            if row == 0:
                ax.set_title(
                    rf"$(\ell,\,b) = ({l_deg:+.1f}°,\,{b_deg:+.1f}°)$",
                    fontsize=11
                )
            if col == 0 and row == 0:
                ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "NP kinematics (Huston2025_C20Bulge) vs VIRAC2 — binned kinematic moments",
        fontsize=12, y=0.97
    )
    os.makedirs(os.path.dirname(OUTPUT_FIGURE), exist_ok=True)
    fig.savefig(OUTPUT_FIGURE, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {OUTPUT_FIGURE}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 62)
    print("SynthPop NP kinematics — kinematic moment comparison")
    print("=" * 62)

    results = []
    for sl in SIGHTLINES:
        l_deg, b_deg = sl["l"], sl["b"]
        print(f"\n── Sightline (l={l_deg:+.1f}, b={b_deg:+.1f}) ──")

        # Load SynthPop CSV
        csv_path = find_synthpop_csv(l_deg, b_deg)
        if csv_path is None:
            print(f"  SynthPop output not found in {SYNTHPOP_OUT_DIR}")
            print("  → Run `python3 -m synthpop np_kinematics_test.synthpop_conf` first.")
            sim_binned = None
        else:
            print(f"  Loading {os.path.basename(csv_path)} …")
            df_sim = pd.read_csv(csv_path)
            print(f"  Raw star count: {len(df_sim):,d}")
            print(f"  Columns: {list(df_sim.columns)[:12]}…")
            sim_binned = process_synthpop_df(df_sim, l_deg, b_deg)
            if sim_binned is not None:
                print(f"  Binned rows: {len(sim_binned)}")
                print(f"  mu_l range: {sim_binned['mu_l_mean'].min():.2f} to "
                      f"{sim_binned['mu_l_mean'].max():.2f} mas/yr")
                print(f"  mu_b range: {sim_binned['mu_b_mean'].min():.2f} to "
                      f"{sim_binned['mu_b_mean'].max():.2f} mas/yr")

        # Load VIRAC2 observed binned moments
        obs_binned = load_obs_binned(l_deg, b_deg)
        if obs_binned is not None:
            print(f"  Obs bins: {len(obs_binned)}, "
                  f"columns: {list(obs_binned.columns)[:8]}…")

        results.append({"l": l_deg, "b": b_deg, "sim": sim_binned, "obs": obs_binned})

    plot_comparison(results)
    print("\nDone.")


if __name__ == "__main__":
    main()
