"""
Surot et al. (2020) 2-D extinction map wrapper.

Loads the high-resolution Surot K_s extinction table, bins it onto a 0.01 deg
grid, and exposes a fast RegularGridInterpolator for arbitrary (l, b) queries.
A module-level singleton avoids repeated file I/O.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import binned_statistic_2d

# Default path — co-located with the synthpop package that ships the table.
_DEFAULT_MAP_PATH = (
    "/Users/paterson.35/Documents/synthpop/synthpop/data/extinction/surot_A_Ks_table.h5"
)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SurotExtinction:
    """
    Interpolator for the Surot et al. (2020) 2-D K_s extinction map.

    The raw map has ~0.0025 deg resolution.  On construction the table is
    binned to 0.01 deg and a ``RegularGridInterpolator`` is built so that
    subsequent queries are O(1).

    Parameters
    ----------
    map_path : str, optional
        Path to the HDF5 extinction table.  Defaults to the synthpop data
        directory set in ``_DEFAULT_MAP_PATH``.
    """

    def __init__(self, map_path: str = _DEFAULT_MAP_PATH) -> None:
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Surot map not found at {map_path}")

        print("Loading Surot map and building interpolator...")
        df = pd.read_hdf(map_path, key="data")

        l_min, l_max = df["l"].min(), df["l"].max()
        b_min, b_max = df["b"].min(), df["b"].max()

        l_grid = np.arange(l_min, l_max + 0.01, 0.01)
        b_grid = np.arange(b_min, b_max + 0.01, 0.01)

        ret = binned_statistic_2d(
            df["l"], df["b"], df["A_Ks"],
            statistic="mean",
            bins=[l_grid, b_grid],
        )

        grid_values = ret.statistic
        grid_values[np.isnan(grid_values)] = 0.0

        l_centers = 0.5 * (l_grid[1:] + l_grid[:-1])
        b_centers = 0.5 * (b_grid[1:] + b_grid[:-1])

        self._interp = RegularGridInterpolator(
            (l_centers, b_centers),
            grid_values,
            bounds_error=False,
            fill_value=None,
        )
        print("Surot interpolator ready.")

    def get_a_ks(self, l: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Return interpolated A_Ks at Galactic coordinates (l, b).

        Longitude values in [180, 360] are wrapped to [-180, 0].

        Parameters
        ----------
        l : array_like
            Galactic longitude in degrees.
        b : array_like
            Galactic latitude in degrees.

        Returns
        -------
        a_ks : np.ndarray
            Interpolated K_s extinction values, same shape as input.
        """
        l = np.atleast_1d(np.asarray(l, dtype=float))
        b = np.atleast_1d(np.asarray(b, dtype=float))
        l_wrapped = np.where(l > 180, l - 360, l)
        return self._interp(np.column_stack([l_wrapped, b]))


# ---------------------------------------------------------------------------
# Module-level singleton helper
# ---------------------------------------------------------------------------

_instance: SurotExtinction | None = None


def get_surot_a_ks(l: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Return A_Ks from the shared SurotExtinction singleton.

    Constructs the singleton on first call; subsequent calls reuse the
    cached interpolator.

    Parameters
    ----------
    l : array_like
        Galactic longitude in degrees.
    b : array_like
        Galactic latitude in degrees.

    Returns
    -------
    a_ks : np.ndarray
        Interpolated K_s extinction.
    """
    global _instance
    if _instance is None:
        _instance = SurotExtinction()
    return _instance.get_a_ks(l, b)


# Keep the old snake_case alias for backwards compatibility with existing callers.
def get_surot_A_Ks(l: np.ndarray, b: np.ndarray) -> np.ndarray:  # noqa: N802
    """Alias for :func:`get_surot_a_ks` retained for backwards compatibility."""
    return get_surot_a_ks(l, b)
