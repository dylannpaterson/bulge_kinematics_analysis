"""
Align observed kinematic datasets to the SynthPop metadata pixel grid.

Reads the VIRAC2, BRAVA, and GIBS observational results, matches each field to
the nearest pixel in the pre-computed metadata cache, and saves the aligned
arrays as ``results/reduction/*_obs_aligned_h25c20.npz`` ready for inversion.
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd

os.environ["OMP_NUM_THREADS"] = "1"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)


# ---------------------------------------------------------------------------
# GIBS data parser
# ---------------------------------------------------------------------------

def parse_gibs_data(gibs_path: str) -> pd.DataFrame:
    """
    Parse the GIBS full-field text file into per-field kinematic statistics.

    Parameters
    ----------
    gibs_path : str
        Path to ``GIBS_full.txt``.

    Returns
    -------
    grouped : pd.DataFrame
        One row per field with columns ``l_center``, ``b_center``,
        ``rv_mean``, ``rv_disp``, and ``count``.
    """
    rows = []
    with open(gibs_path, "r") as fh:
        for line in fh:
            if not (line.strip() and not line.startswith("#") and "|" in line):
                continue
            parts = line.strip().split("|")
            if len(parts) < 4:
                continue
            try:
                first_col = parts[0].strip()
                if "GIBS" in first_col:
                    field = first_col.split("GIBS")[0].strip()
                else:
                    field_parts = first_col.split()
                    if not field_parts:
                        continue
                    field = field_parts[0]
                coords = parts[2].strip().split()
                glon = float(coords[0])
                if glon > 180:
                    glon -= 360
                glat = float(coords[1])
                rv = float(parts[3].strip())
                rows.append({"field": field, "l": glon, "b": glat, "rv": rv})
            except ValueError:
                continue

    df = pd.DataFrame(rows)
    grouped = df.groupby("field").agg(
        l_center=("l", "mean"),
        b_center=("b", "mean"),
        rv_mean=("rv", "mean"),
        rv_disp=("rv", lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0),
        count=("rv", "count"),
    ).reset_index()
    grouped = grouped[grouped['count'] > 10].dropna(subset=['rv_mean', 'rv_disp'])
    return grouped


# ---------------------------------------------------------------------------
# VIRAC2 alignment
# ---------------------------------------------------------------------------

def align_virac_h25c20() -> None:
    """
    Align VIRAC2 observed proper-motion moments to the H25C20 metadata grid.

    Reads ``results/reduction/virac2_magnitude_binned_moments.csv`` and the
    VIRAC2 metadata cache, matches pixels by (l, b), and writes the aligned
    arrays to ``results/reduction/virac2_obs_aligned_h25c20.npz``.
    """
    virac_csv = os.path.join(
        project_root, "results/reduction/virac2_magnitude_binned_moments.csv"
    )
    meta_cache = os.path.join(
        project_root, "results/inversion/huston25c20_metadata_science_binned.pkl"
    )
    output_npz = os.path.join(
        project_root, "results/reduction/virac2_obs_aligned_h25c20.npz"
    )

    print(f"Aligning VIRAC2 observed data from {virac_csv} with metadata...")
    if not os.path.exists(virac_csv):
        raise FileNotFoundError(
            f"VIRAC CSV not found: {virac_csv}. Run run_obs_data_analysis.py first."
        )
    if not os.path.exists(meta_cache):
        raise FileNotFoundError(
            f"Metadata cache not found: {meta_cache}. "
            "Run generate_metadata_huston25c20.py first."
        )

    df_v = pd.read_csv(virac_csv)
    with open(meta_cache, "rb") as fh:
        meta_v = pickle.load(fh)

    mock_l = np.array(meta_v["l"])
    mock_b = np.array(meta_v["b"])
    n_meta = len(mock_l)

    bins_unique = sorted(df_v["ks0_center"].unique())
    n_bins = len(bins_unique)

    obs_mu_pm = np.zeros((n_meta, n_bins, 2))
    obs_cov_pm = np.zeros((n_meta, n_bins, 2, 2))
    obs_mu_err = np.zeros((n_meta, n_bins, 2))
    obs_mask = np.zeros(n_meta, dtype=bool)

    df_v["coord_key"] = list(zip(df_v["l"].round(4), df_v["b"].round(4)))
    grouped = df_v.groupby("coord_key")

    for i in range(n_meta):
        key = (round(float(mock_l[i]), 4), round(float(mock_b[i]), 4))
        if key not in grouped.groups:
            obs_cov_pm[i] = np.eye(2) * 100.0
            obs_mu_err[i] = 100.0
            continue

        obs_mask[i] = True
        match = grouped.get_group(key)
        for j, b_center in enumerate(bins_unique):
            bin_row = match[np.isclose(match["ks0_center"], b_center)]
            if bin_row.empty:
                obs_cov_pm[i, j] = np.eye(2) * 100.0
                obs_mu_err[i, j] = 100.0
            else:
                obs_mu_pm[i, j, 0] = bin_row["mu_l"].values[0]
                obs_mu_pm[i, j, 1] = bin_row["mu_b"].values[0]
                obs_cov_pm[i, j, 0, 0] = bin_row["sig_l"].values[0] ** 2
                obs_cov_pm[i, j, 1, 1] = bin_row["sig_b"].values[0] ** 2
                obs_cov_pm[i, j, 0, 1] = obs_cov_pm[i, j, 1, 0] = (
                    bin_row["cov_lb"].values[0]
                )
                obs_mu_err[i, j, 0] = bin_row["err_mu_l_ana"].values[0]
                obs_mu_err[i, j, 1] = bin_row["err_mu_b_ana"].values[0]

    np.savez_compressed(
        output_npz,
        l=mock_l, b=mock_b,
        mu_pm=obs_mu_pm, cov_pm=obs_cov_pm,
        mu_err=obs_mu_err, mask=obs_mask,
    )
    print(f"--- VIRAC2: Aligned {np.sum(obs_mask)} pixels → {output_npz} ---")


# ---------------------------------------------------------------------------
# BRAVA alignment
# ---------------------------------------------------------------------------

def align_brava_h25c20() -> None:
    """
    Align BRAVA observed radial-velocity moments to the H25C20 metadata grid.

    Reads ``data/brava_paper_results.csv`` and the BRAVA metadata cache, matches
    fields by (l, b), and writes the aligned arrays to
    ``results/reduction/brava_obs_aligned_h25c20.npz``.
    """
    brava_csv = os.path.join(project_root, "data/brava_paper_results.csv")
    meta_cache = os.path.join(
        project_root, "results/inversion/huston25c20_brava_obs_metadata.pkl"
    )
    output_npz = os.path.join(
        project_root, "results/reduction/brava_obs_aligned_h25c20.npz"
    )

    print(f"Aligning BRAVA observed data from {brava_csv} with metadata...")
    if not os.path.exists(brava_csv):
        raise FileNotFoundError(f"BRAVA CSV not found: {brava_csv}.")
    if not os.path.exists(meta_cache):
        raise FileNotFoundError(
            f"Metadata cache not found: {meta_cache}. "
            "Run generate_metadata_huston25c20.py first."
        )

    df_b = pd.read_csv(brava_csv)
    with open(meta_cache, "rb") as fh:
        meta_b = pickle.load(fh)

    mock_l = np.array(meta_b["l"])
    mock_b = np.array(meta_b["b"])
    n_meta = len(mock_l)

    obs_mu_rv = np.zeros((n_meta, 1))
    obs_var_rv = np.zeros((n_meta, 1))
    obs_mu_err = np.zeros((n_meta, 1))
    obs_mask = np.zeros(n_meta, dtype=bool)

    for i in range(n_meta):
        l, b = mock_l[i], mock_b[i]
        match = df_b[
            np.isclose(df_b["l"], l, atol=1e-3) & np.isclose(df_b["b"], b, atol=1e-3)
        ]
        if match.empty:
            obs_var_rv[i, 0] = 100.0 ** 2
            obs_mu_err[i, 0] = 100.0
        else:
            obs_mask[i] = True
            obs_mu_rv[i, 0] = match["RV_mean"].values[0]
            obs_var_rv[i, 0] = match["RV_disp"].values[0] ** 2
            count = match["N"].values[0]
            obs_mu_err[i, 0] = (
                match["RV_disp"].values[0] / np.sqrt(count) if count > 0 else 100.0
            )

    np.savez_compressed(
        output_npz,
        l=mock_l, b=mock_b,
        mu_rv=obs_mu_rv, var_rv=obs_var_rv,
        mu_err=obs_mu_err, mask=obs_mask,
    )
    print(f"--- BRAVA: Aligned {np.sum(obs_mask)} fields → {output_npz} ---")


# ---------------------------------------------------------------------------
# GIBS alignment
# ---------------------------------------------------------------------------

def align_gibs_h25c20() -> None:
    """
    Align GIBS observed radial-velocity moments to the H25C20 metadata grid.

    Reads ``data/GIBS_full.txt`` and the GIBS metadata cache, matches fields by
    (l, b), and writes the aligned arrays to
    ``results/reduction/gibs_obs_aligned_h25c20.npz``.
    """
    gibs_txt = os.path.join(project_root, "data/GIBS_full.txt")
    meta_cache = os.path.join(
        project_root, "results/inversion/huston25c20_gibs_metadata_science.pkl"
    )
    output_npz = os.path.join(
        project_root, "results/reduction/gibs_obs_aligned_h25c20.npz"
    )

    print(f"Aligning GIBS observed data from {gibs_txt} with metadata...")
    if not os.path.exists(gibs_txt):
        raise FileNotFoundError(f"GIBS file not found: {gibs_txt}.")
    if not os.path.exists(meta_cache):
        raise FileNotFoundError(
            f"Metadata cache not found: {meta_cache}. "
            "Run generate_metadata_huston25c20.py first."
        )

    gibs_stats = parse_gibs_data(gibs_txt)
    with open(meta_cache, "rb") as fh:
        meta_g = pickle.load(fh)

    mock_l = np.array(meta_g["l"])
    mock_b = np.array(meta_g["b"])
    n_meta = len(mock_l)

    obs_mu_rv = np.zeros((n_meta, 1))
    obs_var_rv = np.zeros((n_meta, 1))
    obs_mu_err = np.zeros((n_meta, 1))
    obs_mask = np.zeros(n_meta, dtype=bool)

    for i in range(n_meta):
        l, b = mock_l[i], mock_b[i]
        match = gibs_stats[
            np.isclose(gibs_stats["l_center"], l, atol=1e-3)
            & np.isclose(gibs_stats["b_center"], b, atol=1e-3)
        ]
        if match.empty:
            obs_var_rv[i, 0] = 100.0 ** 2
            obs_mu_err[i, 0] = 100.0
        else:
            obs_mask[i] = True
            obs_mu_rv[i, 0] = match["rv_mean"].values[0]
            obs_var_rv[i, 0] = match["rv_disp"].values[0] ** 2
            count = match["count"].values[0]
            obs_mu_err[i, 0] = (
                match["rv_disp"].values[0] / np.sqrt(count) if count > 0 else 100.0
            )

    np.savez_compressed(
        output_npz,
        l=mock_l, b=mock_b,
        mu_rv=obs_mu_rv, var_rv=obs_var_rv,
        mu_err=obs_mu_err, mask=obs_mask,
    )
    print(f"--- GIBS: Aligned {np.sum(obs_mask)} fields → {output_npz} ---")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    align_virac_h25c20()
    align_brava_h25c20()
    align_gibs_h25c20()
