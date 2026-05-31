"""
Shared configuration constants and factory functions for the bulge model pipeline.

Defines the standardised distance grid, kinematic inversion grid, and SynthPop
configuration used consistently across all pipeline phases.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Distance grid (line-of-sight integration)
# ---------------------------------------------------------------------------

N_DIST_BINS = 512
DIST_MIN_KPC = 0.1
DIST_MAX_KPC = 25.0


def get_distance_grid() -> np.ndarray:
    """
    Return the standardised distance grid used for all LOS integrations.

    Returns
    -------
    distances : np.ndarray, shape (N_DIST_BINS,)
        Uniformly spaced distances in kpc from DIST_MIN_KPC to DIST_MAX_KPC.
    """
    return np.linspace(DIST_MIN_KPC, DIST_MAX_KPC, N_DIST_BINS)


# ---------------------------------------------------------------------------
# Kinematic inversion grid (bar-aligned Cartesian frame, kpc)
# ---------------------------------------------------------------------------

GRID_X_BINS = 40
GRID_Y_BINS = 40
GRID_Z_BINS = 15


def get_inverter_grid_axes() -> list[np.ndarray]:
    """
    Return the standardised spatial grid axes for the BulgeKinematicInverter.

    The axes span the first octant of the bar-aligned frame.  The inverter
    mirrors the solution into all eight octants using the assumed symmetries.

    Returns
    -------
    axes : list of np.ndarray
        [axes_x, axes_y, axes_z] each 1-D, in kpc.
    """
    return [
        np.linspace(0.0, 5.0, GRID_X_BINS),
        np.linspace(0.0, 5.0, GRID_Y_BINS),
        np.linspace(0.0, 2.0, GRID_Z_BINS),
    ]


# ---------------------------------------------------------------------------
# Operational constants
# ---------------------------------------------------------------------------

#: Number of pixels processed per JAX vmap batch during metadata generation.
BATCH_SIZE = 200


# ---------------------------------------------------------------------------
# SynthPop base configuration
# ---------------------------------------------------------------------------

#: Extinction settings shared by all SynthPop instantiations in this pipeline.
EXTINCTION_CONFIG: dict = {
    "extinction_map_kwargs": {
        "name": "Surot",
        "project_3d": True,
        "dist_2d": 8.15,
    },
    "extinction_law_kwargs": [{"name": "SODC"}],
}


def get_base_synthpop_config(
    model_name: str,
    output_name: str,
    l_set: list[float] | None = None,
    b_set: list[float] | None = None,
    solid_angle: float = 0.04,
) -> dict:
    """
    Build a base SynthPop configuration dictionary.

    Parameters
    ----------
    model_name : str
        SynthPop model identifier (e.g. ``'Huston2025_C20Bulge'``).
    output_name : str
        Label written into the SynthPop output filename.
    l_set : list of float, optional
        Galactic longitudes in degrees.  Defaults to ``[0.0]``.
    b_set : list of float, optional
        Galactic latitudes in degrees.  Defaults to ``[0.0]``.
    solid_angle : float, optional
        Solid angle in deg².  Default is 0.04.

    Returns
    -------
    config : dict
        Configuration dictionary suitable for ``synthpop.SynthPop()``.
    """
    if l_set is None:
        l_set = [0.0]
    if b_set is None:
        b_set = [0.0]

    return {
        "MANDATORY": {"name_for_output": output_name, "model_name": model_name},
        "SIGHTLINES": {
            "l_set": l_set,
            "l_set_type": "list",
            "b_set": b_set,
            "b_set_type": "list",
            "solid_angle": solid_angle,
            "solid_angle_unit": "deg^2",
        },
        **EXTINCTION_CONFIG,
    }
