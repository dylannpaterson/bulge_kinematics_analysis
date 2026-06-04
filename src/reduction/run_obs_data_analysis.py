"""
Iterative proper-motion cleaning and intrinsic kinematic moment estimation for VVV.

Groups high-resolution VVV Parquet files into spatial tiles, cleans outliers via
iterative sigma-clipping, and calculates proper motion and radial velocity moments
for binned apparent magnitude ranges.
"""
from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
import pandas as pd

# Set up project root imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.reduction.mixture_models import (
    get_intrinsic_moments,
    get_intrinsic_moments_with_errors,
)
from src.shared.surot_extinction import get_surot_a_ks


# ---------------------------------------------------------------------------
# Kinematic Cleaning Helpers
# ---------------------------------------------------------------------------

def clean_kinematics(
    df: pd.DataFrame,
    max_err: float = 5.0,
    sigma: float = 5.0,
    max_iter: int = 4,
    tol: float = 0.1,
) -> pd.DataFrame:
    """
    Apply absolute error cuts and iterative sigma clipping to proper motions.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing proper motions and errors.
    max_err : float, optional
        Maximum allowed proper motion error in mas/yr. Default is 5.0.
    sigma : float, optional
        Sigma multiplier for outlier clipping. Default is 5.0.
    max_iter : int, optional
        Maximum number of clipping iterations. Default is 4.
    tol : float, optional
        Tolerance on standard deviation change for early stopping. Default is 0.1.

    Returns
    -------
    clean_df : pd.DataFrame
        Cleaned proper motion DataFrame.
    """
    if "pm_l_error" not in df.columns or "pm_b_error" not in df.columns:
        return df

    # 1. Hard Error Cut
    clean_df = df[(df["pm_l_error"] < max_err) & (df["pm_b_error"] < max_err)].copy()

    if len(clean_df) == 0:
        return clean_df

    # 2. Iterative Sigma Clipping
    for _ in range(max_iter):
        std_l = clean_df["pm_l"].std()
        std_b = clean_df["pm_b"].std()
        med_l = clean_df["pm_l"].median()
        med_b = clean_df["pm_b"].median()

        # Create boolean mask for stars within bounds
        mask_l = np.abs(clean_df["pm_l"] - med_l) < (sigma * std_l)
        mask_b = np.abs(clean_df["pm_b"] - med_b) < (sigma * std_b)

        new_df = clean_df[mask_l & mask_b]

        # Stopping criteria: if the standard deviation changes by less than tol
        diff_l = np.abs(std_l - (new_df["pm_l"].std() if not new_df.empty else 0))
        diff_b = np.abs(std_b - (new_df["pm_b"].std() if not new_df.empty else 0))

        if diff_l < tol and diff_b < tol:
            return new_df

        clean_df = new_df
        if len(clean_df) < 10:
            break

    return clean_df


# ---------------------------------------------------------------------------
# Main Analysis Pipeline
# ---------------------------------------------------------------------------

def run_obs_analysis() -> None:
    """
    Run spatial tile kinematic analysis on VVV observed data.

    Processes deep VVV Parquet files, performs quality cuts, subtracts extinction,
    and saves binned/integrated intrinsic moments to CSV files.
    """
    data_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../data/virac2_bulge_tiles_deep")
    )
    files = glob.glob(f"{data_dir}/*.parquet")

    # Group 0.2 tiles into 0.4 bins aligned with priors (-9.8, -9.4...)
    groups: dict[tuple[float, float], list[str]] = {}
    for f in files:
        match = re.search(r"l_([-+]?\d*\.\d*)_b_([-+]?\d*\.\d*)", os.path.basename(f))
        if match:
            l, b = float(match.group(1)), float(match.group(2))
            # Align center to nearest 0.4 grid point (-9.8, -9.4...)
            l_bin = np.round((l - 0.2) / 0.4) * 0.4 + 0.2
            b_bin = np.round((b - 0.2) / 0.4) * 0.4 + 0.2
            key = (round(l_bin, 2), round(b_bin, 2))
            if key not in groups:
                groups[key] = []
            groups[key].append(f)

    results = []
    binned_results = []

    # Filter active groups (b < 5.0) beforehand
    active_groups = {k: v for k, v in groups.items() if k[1] < 5.0}
    if os.environ.get("FAST_TEST") == "1":
        active_groups = dict(list(active_groups.items())[:2])
        print(
            f"FAST_TEST mode active: Only analyzing first 2 groups: {list(active_groups.keys())}"
        )

    total_groups = len(active_groups)

    # Define magnitude bins (20 bins between 12 and 14)
    ks0_bin_edges = np.linspace(12.0, 14.0, 21)
    ks0_bin_centers = 0.5 * (ks0_bin_edges[:-1] + ks0_bin_edges[1:])

    for i, ((l_center, b_center), tile_paths) in enumerate(active_groups.items()):
        print(f"[{i+1}/{total_groups}] Analyzing l={l_center}, b={b_center}...")

        try:
            dfs = []
            required_cols = [
                "phot_j_mean_mag",
                "phot_h_mean_mag",
                "phot_ks_mean_mag",
                "pm_l",
                "pm_b",
                "pm_l_error",
                "pm_b_error",
            ]
            for p in tile_paths:
                temp_df = pd.read_parquet(p)
                if all(col in temp_df.columns for col in required_cols):
                    dfs.append(temp_df)

            if not dfs:
                continue
            df = pd.concat(dfs)

            # Pre-process observed data
            df["j_ks"] = df["phot_j_mean_mag"] - df["phot_ks_mean_mag"]
            df["h_ks"] = df["phot_h_mean_mag"] - df["phot_ks_mean_mag"]

            # Extinction and RC Cut
            a_ks_local = get_surot_a_ks(l_center, b_center)[0]
            df["ks0"] = df["phot_ks_mean_mag"] - a_ks_local
            
            # Dereddened color (A_J/A_Ks ~ 3.524 for SODC R_V=2.5 law)
            df["j_ks_0"] = df["j_ks"] - (3.524 - 1.0) * a_ks_local
            
            # Apply magnitude AND color cuts
            data = df[(df["ks0"] > 12.0) & (df["ks0"] < 14.0) & 
                      (df["j_ks_0"] > 0.4)].copy()
            data = data.dropna(subset=["j_ks", "h_ks", "pm_l", "pm_b"])

            # Quality and Sigma Cuts
            data = clean_kinematics(data)

            if len(data) == 0:
                print(f"Skipping l={l_center}, b={b_center}: No stars after cleaning.")
                continue

            lb_corr = data["pm_l_pm_b_corr"] if "pm_l_pm_b_corr" in data.columns else None

            # 1. Magnitude Binned Analysis
            for b_idx in range(len(ks0_bin_centers)):
                bin_mask = (data["ks0"] >= ks0_bin_edges[b_idx]) & (
                    data["ks0"] < ks0_bin_edges[b_idx + 1]
                )
                bin_data = data[bin_mask]
                if len(bin_data) >= 1:
                    w_ones = np.ones(len(bin_data))
                    b_corr = (
                        bin_data["pm_l_pm_b_corr"]
                        if "pm_l_pm_b_corr" in bin_data.columns
                        else None
                    )

                    # Calculate moments with errors (analytic + bootstrap)
                    n_boot_val = 2 if os.environ.get("FAST_TEST") == "1" else 50
                    m_res = get_intrinsic_moments_with_errors(
                        w_ones,
                        bin_data["pm_l"],
                        bin_data["pm_b"],
                        bin_data["pm_l_error"],
                        bin_data["pm_b_error"],
                        pm_lb_corr=b_corr,
                        n_boot=n_boot_val,
                    )

                    if m_res:
                        bin_row = {
                            "l": l_center,
                            "b": b_center,
                            "ks0_center": ks0_bin_centers[b_idx],
                            "n_stars": len(bin_data),
                        }
                        bin_row.update(m_res)
                        binned_results.append(bin_row)

            # 2. Integrated Moments
            mu_l_mix, mu_b_mix, sig_l_mix, sig_b_mix, cov_lb_mix = get_intrinsic_moments(
                np.ones(len(data)),
                data["pm_l"],
                data["pm_b"],
                data["pm_l_error"],
                data["pm_b_error"],
                pm_lb_corr=lb_corr,
            )

            res_row = {
                "l": l_center,
                "b": b_center,
                "n_stars": len(data),
                "mu_l_mix": mu_l_mix,
                "mu_b_mix": mu_b_mix,
                "sig_l_mix": sig_l_mix,
                "sig_b_mix": sig_b_mix,
                "cov_lb_mix": cov_lb_mix,
            }
            results.append(res_row)

        except Exception as e:
            print(f"Error at l={l_center}, b={b_center}: {e}")
            continue

    out_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../results/reduction")
    )
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(
        os.path.join(out_dir, "virac2_kinematics_cross_val_results.csv"), index=False
    )
    pd.DataFrame(binned_results).to_csv(
        os.path.join(out_dir, "virac2_magnitude_binned_moments.csv"), index=False
    )
    print(f"Results saved to {out_dir}. Binned data: {len(binned_results)} rows.")


def main() -> None:
    """Execute the observed data analysis."""
    run_obs_analysis()


if __name__ == "__main__":
    main()
