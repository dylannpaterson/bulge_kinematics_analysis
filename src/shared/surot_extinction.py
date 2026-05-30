import pandas as pd
import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator

class SurotExtinction:
    """
    Utility to interpolate the Surot et al. (2020) 2D extinction map.
    Loads the map from the synthpop directory and provides a fast interpolator.
    """
    
    def __init__(self, map_path='/Users/paterson.35/Documents/synthpop/synthpop/data/extinction/surot_A_Ks_table.h5'):
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Surot map not found at {map_path}")
        
        print("Loading Surot map and building interpolator (this may take a moment)...")
        df = pd.read_hdf(map_path, key='data')
        
        # Surot map is very high res (~0.0025 deg). 
        # For general use, binning to 0.01 deg is usually sufficient and much faster/memory efficient.
        # We'll build a regular grid interpolator on a 0.01 deg grid.
        
        l_min, l_max = df['l'].min(), df['l'].max()
        b_min, b_max = df['b'].min(), df['b'].max()
        
        # Define grid
        self.l_grid = np.arange(l_min, l_max + 0.01, 0.01)
        self.b_grid = np.arange(b_min, b_max + 0.01, 0.01)
        
        # Bin data
        from scipy.stats import binned_statistic_2d
        ret = binned_statistic_2d(
            df['l'], df['b'], df['A_Ks'], 
            statistic='mean', 
            bins=[self.l_grid, self.b_grid]
        )
        
        # Fill NaNs (if any) with a small value or nearest neighbor
        grid_values = ret.statistic
        mask = np.isnan(grid_values)
        if np.any(mask):
            grid_values[mask] = 0.0 # Or some sensible default
            
        # Centers for the interpolator
        l_centers = 0.5 * (self.l_grid[1:] + self.l_grid[:-1])
        b_centers = 0.5 * (self.b_grid[1:] + self.b_grid[:-1])
        
        self.interp = RegularGridInterpolator(
            (l_centers, b_centers), 
            grid_values, 
            bounds_error=False, 
            fill_value=None
        )
        print("Interpolator ready.")

    def get_A_Ks(self, l, b):
        """
        Get A_Ks at Galactic longitude l and latitude b.
        
        Args:
            l (float or np.ndarray): Longitude (deg), handled as [0, 360] -> [-180, 180]
            b (float or np.ndarray): Latitude (deg)
            
        Returns:
            np.ndarray: Interpolated A_Ks values
        """
        # Handle coordinate wrapping if l is in [0, 360]
        l = np.atleast_1d(l)
        b = np.atleast_1d(b)
        l_use = np.where(l > 180, l - 360, l)
        
        # Query interpolator
        pts = np.vstack([l_use, b]).T
        return self.interp(pts)

# Global instance for easy access
_instance = None

def get_surot_A_Ks(l, b):
    """Convenience function to get A_Ks using a shared instance."""
    global _instance
    if _instance is None:
        _instance = SurotExtinction()
    return _instance.get_A_Ks(l, b)
