import numpy as np

# --- 1. Standardized Distance Grid (LOS Integration) ---
# Used for all LOS integrations across mock generation and metadata extraction
N_DIST_BINS = 512
DIST_MIN = 0.1
DIST_MAX = 25.0

def get_distance_grid():
    """Returns the standardized distance grid array."""
    return np.linspace(DIST_MIN, DIST_MAX, N_DIST_BINS)

# --- 2. Standardized Kinematic Inversion Grid (x, y, z in kpc) ---
# Used by BulgeKinematicInverter to maintain consistency between 
# pre-computed grids and live parametric fits.
GRID_X_BINS = 40
GRID_Y_BINS = 40
GRID_Z_BINS = 15

def get_inverter_grid_axes():
    """Returns the standardized spatial grid axes for the Kinematic Inverter."""
    return [
        np.linspace(0.0, 5.0, GRID_X_BINS),
        np.linspace(0.0, 5.0, GRID_Y_BINS),
        np.linspace(0.0, 2.0, GRID_Z_BINS)
    ]

# --- 3. Operational Constants ---
# Common Batch Size for memory-intensive LOS integrations
BATCH_SIZE = 200

# --- 4. Standardized SynthPop Configuration ---
# Ensures consistent Extinction Map and Extinction Law settings
EXTINCTION_CONFIG = {
    "extinction_map_kwargs": {
        "name": "Surot",
        "project_3d": True,
        "dist_2d": 8.15
    },
    "extinction_law_kwargs": [
        {
            "name": "SODC"
        }
    ]
}

def get_base_synthpop_config(model_name, output_name, l_set=[0.0], b_set=[0.0], solid_angle=0.04):
    """
    Generates a base SynthPop configuration dictionary.
    
    Args:
        model_name (str): The SynthPop model to use (e.g., 'Huston2025').
        output_name (str): The label for internal SynthPop results.
        l_set (list): List of Galactic Longitude values.
        b_set (list): List of Galactic Latitude values.
        solid_angle (float): Solid angle in deg^2.
    """
    return {
        "MANDATORY": {"name_for_output": output_name, "model_name": model_name},
        "SIGHTLINES": {
            "l_set": l_set, "l_set_type": "list", 
            "b_set": b_set, "b_set_type": "list", 
            "solid_angle": solid_angle, "solid_angle_unit": "deg^2"
        },
        **EXTINCTION_CONFIG
    }
