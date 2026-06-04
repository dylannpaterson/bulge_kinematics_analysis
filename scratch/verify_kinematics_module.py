import os
import sys
import numpy as np
import pandas as pd

# Setup paths
project_root = "/Users/paterson.35/Documents/bulge_model_pipeline"
synthpop_root = "/Users/paterson.35/Documents/synthpop"
sys.path.insert(0, synthpop_root)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src/shared"))

import synthpop
from src.reduction.run_obs_data_analysis import clean_kinematics
from src.shared.surot_extinction import get_surot_a_ks

def main():
    print("Initializing SynthPop...")
    sp = synthpop.SynthPop("np_kinematics_test.synthpop_conf")
    sp.init_populations()
    bulge_pop = sp.populations[0]
    coord_trans = bulge_pop.coord_trans
    
    # Load Huston2025_C20Bulge_l5.000_b-3.000.csv
    csv_path = "results/synthpop_np_test/np_kinematics_test/Huston2025_C20Bulge_l5.000_b-3.000.csv"
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter bulge stars
    df = df[df['pop'] == 0].copy()
    print(f"Loaded {len(df)} bulge stars.")
    
    # Apply standard cuts
    a_ks_2d = get_surot_a_ks(5.0, -3.0)[0]
    df["ks0"] = df["2MASS_Ks"] - a_ks_2d
    a_j_2d = a_ks_2d * 3.524
    df["j_ks_0"] = (df["2MASS_J"] - a_j_2d) - (df["2MASS_Ks"] - a_ks_2d)
    
    df = df[(df["ks0"] > 12.0) & (df["ks0"] < 14.0) & (df["j_ks_0"] > 0.4)].copy()
    df = clean_kinematics(df)
    print(f"After cuts, {len(df)} stars remain.")
    
    print("\nCSV Dispersions:")
    print(f"  sig_mul: {df['mul'].std():.4f} mas/yr")
    print(f"  sig_mub: {df['mub'].std():.4f} mas/yr")
    
    # Draw new velocities using current kinematics implementation
    x = df['x'].values
    y = df['y'].values
    z = df['z'].values
    
    print("\nDrawing new velocities from current NonParametricKinematics...")
    vx, vy, vz = bulge_pop.kinematics.draw_random_velocity(x, y, z, density_class=bulge_pop.population_density)
    
    # Convert to proper motions
    l = df['l'].values
    b = df['b'].values
    dist = df['Dist'].values
    vr, mu_l, mu_b = coord_trans.uvw_to_vrmulb(l, b, dist, vx, vy, vz)
    
    print("New Dispersions:")
    print(f"  sig_mul: {mu_l.std():.4f} mas/yr")
    print(f"  sig_mub: {mu_b.std():.4f} mas/yr")

if __name__ == "__main__":
    main()
