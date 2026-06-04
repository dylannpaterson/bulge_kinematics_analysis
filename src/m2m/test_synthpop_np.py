"""
Test script: compare NonParametricKinematics (NP) vs Koshimoto2021Bulge (parametric)
synthetic proper motions against VIRAC2 observations for two off-centre sightlines.

Sightlines chosen to be far from the central mask, well-covered by data:
  - (l, b) = (+5.0, -3.0)
  - (l, b) = (+5.0, -9.0)

Processing is performed identically to the data reduction pipeline:
  - 2D extinction correction using Surot et al. (2020) map
  - Iterative sigma-clipping of proper motions
  - Magnitude selection 12.0 < Ks,0 < 14.0
  - Intrinsic Color Cut: (J-Ks)_0 > 0.4
"""

from __future__ import annotations
import sys, os, json, tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Paths -----------------------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
synthpop_root = os.path.abspath(os.path.join(project_root, "../synthpop"))
sys.path.insert(0, synthpop_root)
sys.path.insert(0, project_root) # Added project root to sys.path
sys.path.insert(0, os.path.join(project_root, "src/shared"))

# Import reduction helpers for "identical" processing
from src.reduction.run_obs_data_analysis import clean_kinematics
from src.shared.surot_extinction import get_surot_a_ks

# Sightlines ------------------------------------------------------------------
SIGHTLINES = [
    {"l": 5.0,  "b": -3.0, "label": r"$(\ell, b) = (5.0^\circ, -3.0^\circ)$"},
    {"l": 5.0,  "b": -9.0, "label": r"$(\ell, b) = (5.0^\circ, -9.0^\circ)$"},
]

# 0.16 deg² matches one VIRAC2 pixel (0.4° × 0.4°)
SOLID_ANGLE = 0.16   # deg²

# VIRAC2 data -----------------------------------------------------------------
VIRAC_NEW_CSV = os.path.join(project_root, "results/reduction/virac2_magnitude_binned_moments.csv")
VIRAC_OLD_CSV = os.path.join(project_root, "results/reduction/virac2_magnitude_binned_moments_no_color_cut.csv")

# Output ----------------------------------------------------------------------
OUT_PATH = os.path.join(project_root, "results/inversion/synthpop_np_comparison.png")

# Data Loading ----------------------------------------------------------------
SYNTHPOP_DATA_DIR = os.path.join(project_root, "results/synthpop_np_test/np_kinematics_test")

def load_synthpop_results(model_name: str) -> dict[tuple[float, float], pd.DataFrame]:
    """
    Load pre-generated SynthPop CSV files and return a dict:
        {(l, b): DataFrame}
    """
    results = {}
    for sl in SIGHTLINES:
        fname = f"{model_name}_l{sl['l']:.3f}_b{sl['b']:.3f}.csv"
        path = os.path.join(SYNTHPOP_DATA_DIR, fname)
        
        if not os.path.exists(path):
            print(f"  [Warning] File not found: {path}")
            continue
            
        print(f"  Loading {fname}…")
        df = pd.read_csv(path)
        # Keep ALL populations (bulge + disk) to match observations
        results[(sl["l"], sl["b"])] = df
    return results


def load_virac_obs() -> tuple[dict, dict]:
    """Load VIRAC2 binned moments at the two sightlines (new and old)."""
    df_new = pd.read_csv(VIRAC_NEW_CSV)
    df_old = pd.read_csv(VIRAC_OLD_CSV)
    
    obs_new = {}
    obs_old = {}
    
    for sl in SIGHTLINES:
        # NEW
        dist_new = np.sqrt((df_new["l"] - sl["l"])**2 + (df_new["b"] - sl["b"])**2)
        idx_new = dist_new.idxmin()
        l_best, b_best = df_new.loc[idx_new, "l"], df_new.loc[idx_new, "b"]
        sub_new = df_new[(df_new["l"] == l_best) & (df_new["b"] == b_best)].sort_values("ks0_center")
        
        obs_new[(sl["l"], sl["b"])] = {
            "ks0": sub_new["ks0_center"].values,
            "mu_l": sub_new["mu_l"].values,
            "mu_b": sub_new["mu_b"].values,
            "sig_l": sub_new["sig_l"].values,
            "sig_b": sub_new["sig_b"].values,
            "corr_lb": (sub_new["cov_lb"] / (sub_new["sig_l"] * sub_new["sig_b"])).values,
            "err_mu_l": sub_new["err_mu_l_boot"].values,
            "err_mu_b": sub_new["err_mu_b_boot"].values,
            "err_sig_l": sub_new["err_sig_l_boot"].values,
            "err_sig_b": sub_new["err_sig_b_boot"].values,
            "actual_l": l_best,
            "actual_b": b_best,
        }
        
        # OLD
        dist_old = np.sqrt((df_old["l"] - sl["l"])**2 + (df_old["b"] - sl["b"])**2)
        idx_old = dist_old.idxmin()
        l_best_old, b_best_old = df_old.loc[idx_old, "l"], df_old.loc[idx_old, "b"]
        sub_old = df_old[(df_old["l"] == l_best_old) & (df_old["b"] == b_best_old)].sort_values("ks0_center")
        
        obs_old[(sl["l"], sl["b"])] = {
            "ks0": sub_old["ks0_center"].values,
            "mu_l": sub_old["mu_l"].values,
            "mu_b": sub_old["mu_b"].values,
            "sig_l": sub_old["sig_l"].values,
            "sig_b": sub_old["sig_b"].values,
            "corr_lb": (sub_old["cov_lb"] / (sub_old["sig_l"] * sub_old["sig_b"])).values,
            "err_mu_l": sub_old["err_mu_l_boot"].values,
            "err_mu_b": sub_old["err_mu_b_boot"].values,
            "err_sig_l": sub_old["err_sig_l_boot"].values,
            "err_sig_b": sub_old["err_sig_b_boot"].values,
            "actual_l": l_best_old,
            "actual_b": b_best_old,
        }
        
    return obs_new, obs_old


def plot_comparison(np_res: dict, par_res: dict, obs_new: dict, obs_old: dict) -> None:
    """
    Two-column panel (one per sightline), five rows:
      Row 0: <mu_l> vs Ks,0
      Row 1: <mu_b> vs Ks,0
      Row 2: sig_l vs Ks,0
      Row 3: sig_b vs Ks,0
      Row 4: corr_lb vs Ks,0
    """
    n_sl = len(SIGHTLINES)
    fig = plt.figure(figsize=(7 * n_sl, 20))
    gs = GridSpec(5, n_sl, figure=fig, hspace=0.3, wspace=0.3)

    colors = {"NP": "#4a90d9", "H25": "#e87040"}
    ks0_edges = np.linspace(12.0, 14.0, 21)
    ks0_centers = 0.5 * (ks0_edges[:-1] + ks0_edges[1:])

    for col, sl in enumerate(SIGHTLINES):
        key = (sl["l"], sl["b"])
        v_new = obs_new.get(key, {})
        v_old = obs_old.get(key, {})

        # Calculate 2D extinction value for this tile (matching data reduction)
        a_ks_2d = get_surot_a_ks(sl["l"], sl["b"])[0]

        # Precompute binned models for this sightline
        model_binned_data = {}
        for label, res_dict, color in [("NP model", np_res, colors["NP"]), ("H25", par_res, colors["H25"])]:
            df = res_dict.get(key)
            if df is not None and len(df) > 0:
                df = df.copy()
                
                df = df.rename(columns={"mul": "pm_l", "mub": "pm_b"})
                df["pm_l_error"] = 0.1
                df["pm_b_error"] = 0.1
                
                df["ks0"] = df["2MASS_Ks"] - a_ks_2d
                a_j_2d = a_ks_2d * 3.524
                df["j_ks_0"] = (df["2MASS_J"] - a_j_2d) - (df["2MASS_Ks"] - a_ks_2d)
                
                df = df[(df["ks0"] > 12.0) & (df["ks0"] < 14.0) & (df["j_ks_0"] > 0.4)].copy()
                df = clean_kinematics(df)
                
                df["bin"] = pd.cut(df["ks0"], bins=ks0_edges, labels=False)
                
                def calc_binned_stats(group):
                    res = {"count": len(group)}
                    if len(group) > 1:
                        res["mu_l"] = group["pm_l"].mean()
                        res["mu_b"] = group["pm_b"].mean()
                        res["sig_l"] = group["pm_l"].std(ddof=1)
                        res["sig_b"] = group["pm_b"].std(ddof=1)
                        cov = np.cov(group["pm_l"], group["pm_b"])
                        res["cov_lb"] = cov[0, 1]
                        if res["sig_l"] * res["sig_b"] > 0:
                            res["corr_lb"] = res["cov_lb"] / (res["sig_l"] * res["sig_b"])
                        else:
                            res["corr_lb"] = np.nan
                    else:
                        for k in ["mu_l", "mu_b", "sig_l", "sig_b", "cov_lb", "corr_lb"]:
                            res[k] = np.nan
                    return pd.Series(res)

                binned = df.groupby("bin").apply(calc_binned_stats).dropna(subset=["mu_l"])
                model_binned_data[label] = {"binned": binned, "color": color}

        for row, (col_name, obs_key, err_key, ylabel) in enumerate([
            ("mu_l", "mu_l", "err_mu_l", r"$\langle\mu_{\ell}\cos b\rangle$ (mas yr$^{-1}$)"),
            ("mu_b", "mu_b", "err_mu_b", r"$\langle\mu_b\rangle$ (mas yr$^{-1}$)"),
            ("sig_l", "sig_l", "err_sig_l", r"$\sigma_{\mu_{\ell}\cos b}$ (mas yr$^{-1}$)"),
            ("sig_b", "sig_b", "err_sig_b", r"$\sigma_{\mu_b}$ (mas yr$^{-1}$)"),
            ("corr_lb", "corr_lb", None, r"$\rho_{lb}$"),
        ]):
            ax = fig.add_subplot(gs[row, col])

            # 1. Plot Observations (Old)
            if v_old:
                good = np.isfinite(v_old[obs_key])
                if np.any(good):
                    yerr = v_old[err_key][good] if err_key else None
                    ax.errorbar(v_old["ks0"][good], v_old[obs_key][good], yerr=yerr, 
                                fmt="rs-", mfc="none", label="VIRAC2 (No Color Cut)", capsize=2, alpha=0.3)

            # 1b. Plot Observations (New)
            if v_new:
                good = np.isfinite(v_new[obs_key])
                if np.any(good):
                    yerr = v_new[err_key][good] if err_key else None
                    ax.errorbar(v_new["ks0"][good], v_new[obs_key][good], yerr=yerr, 
                                fmt="ko-", mfc="none", label="VIRAC2 (>0.4 Color Cut)", capsize=2, alpha=0.8)

            # 2. Plot Models
            for label, model_data in model_binned_data.items():
                binned = model_data["binned"]
                color = model_data["color"]
                
                if not binned.empty and col_name in binned.columns:
                    valid_centers = ks0_centers[binned.index.astype(int)]
                    y_val = binned[col_name]
                    
                    # Filter nans in y_val (e.g. for corr_lb)
                    valid_mask = np.isfinite(y_val)
                    if not np.any(valid_mask):
                        continue
                    
                    ax.plot(valid_centers[valid_mask], y_val[valid_mask], color=color, lw=2, label=label)
                    
                    if col_name.startswith("mu_"):
                        err_y = binned["sig_" + col_name[-1]] / np.sqrt(binned["count"])
                        ax.fill_between(valid_centers[valid_mask], 
                                       y_val[valid_mask] - err_y[valid_mask], 
                                       y_val[valid_mask] + err_y[valid_mask], 
                                       color=color, alpha=0.4)
                    elif col_name.startswith("sig_"):
                        err_y = y_val / np.sqrt(2 * binned["count"])
                        ax.fill_between(valid_centers[valid_mask], 
                                       y_val[valid_mask] - err_y[valid_mask], 
                                       y_val[valid_mask] + err_y[valid_mask], 
                                       color=color, alpha=0.4)

            if row == 0:
                ax.set_title(sl["label"] + f"\n(data at l={v_new.get('actual_l','?'):.1f}, b={v_new.get('actual_b','?'):.1f})",
                             fontsize=12)
            
            if row == 4:
                ax.set_xlabel(r"$K_{s,0}$ (mag)", fontsize=11)
            
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_xlim(11.9, 14.1)
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=10)
            
            if col == 0 and row == 0:
                ax.legend(fontsize=9, loc="best")

    fig.suptitle(
        "SynthPop NP kinematics vs Parametric vs VIRAC2 Profile\n"
        "Processed identically to data (2D Extinction + Color Cut + Sigma Clipping)",
        fontsize=14, y=0.92
    )
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nSaved binned comparison plot → {OUT_PATH}")
    plt.close(fig)


def main() -> None:
    print("=" * 60)
    print("SynthPop NP kinematics test (Means, Dispersions, Correlations)")
    print("=" * 60)

    # ── Load NP model (Huston2025_C20Bulge with NonParametricKinematics) ──
    print("\n[1/3] Loading NP model results (Huston2025_C20Bulge)…")
    np_results = load_synthpop_results("Huston2025_C20Bulge")

    # ── Load parametric baseline (Huston2025 – uses Koshimoto2021Bulge) ──
    print("\n[2/3] Loading parametric baseline results (Huston2025)…")
    par_results = load_synthpop_results("Huston2025")

    # ── Load VIRAC2 observations ──
    print("\n[3/3] Loading VIRAC2 observations (New and Old)…")
    virac_obs_new, virac_obs_old = load_virac_obs()

    # ── Print quick summary ──
    print("\n── NP model star counts ──")
    for sl in SIGHTLINES:
        key = (sl["l"], sl["b"])
        df = np_results.get(key)
        if df is not None:
            print(f"  {sl['label']}: {len(df)} bulge stars loaded")
        else:
            print(f"  {sl['label']}: No results loaded.")

    # ── Plot ──
    plot_comparison(np_results, par_results, virac_obs_new, virac_obs_old)
    print("\nDone.")


if __name__ == "__main__":
    main()
