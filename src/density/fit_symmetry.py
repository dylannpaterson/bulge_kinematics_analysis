import sys
import os

# Limit CPU threading for emcee Pool
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Set up project root imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import emcee
import corner
from coleman_bulge_density import bulge_density_model
from multiprocessing import Pool
import matplotlib.pyplot as plt

# 1. Define the grid of probe points in the "bar-aligned" frame (first octant)
nx, ny, nz = 6, 4, 3
x_probes = np.linspace(0.2, 3.5, nx)
y_probes = np.linspace(0.2, 1.5, ny)
z_probes = np.linspace(0.1, 1.0, nz)

Xp, Yp, Zp = np.meshgrid(x_probes, y_probes, z_probes, indexing='ij')
probes_first_octant = np.stack([Xp.flatten(), Yp.flatten(), Zp.flatten()], axis=-1)

# Generate the 8 symmetric combinations
combinations = np.array([
    [ 1,  1,  1], [-1,  1,  1], [ 1, -1,  1], [ 1,  1, -1],
    [-1, -1,  1], [-1,  1, -1], [ 1, -1, -1], [-1, -1, -1]
])

# Pre-calculate all 8-fold symmetric points in the bar frame
bar_frame_points = (probes_first_octant[:, np.newaxis, :] * combinations[np.newaxis, :, :]).reshape(-1, 3)

def get_density_at_params(params):
    x0, y0, z0, alpha = params
    xp, yp, zp = bar_frame_points[:, 0], bar_frame_points[:, 1], bar_frame_points[:, 2]
    
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    x = x0 + xp * cos_a - yp * sin_a
    y = y0 + xp * sin_a + yp * cos_a
    z = z0 + zp
    
    dens, _ = bulge_density_model(x=x, y=y, z=z)
    return dens.reshape(-1, 8)

def log_likelihood(params):
    x0, y0, z0, alpha = params
    # Expand alpha bounds to allow negative angles
    if not (-2.0 < x0 < 2.0 and -2.0 < y0 < 2.0 and -1.0 < z0 < 1.0 and -np.pi < alpha < np.pi):
        return -np.inf
    
    try:
        dens = get_density_at_params(params)
        mean_dens = np.mean(dens, axis=1)
        
        if np.any(mean_dens <= 0):
            mean_dens[mean_dens <= 0] = 1e-10
            
        mad = np.mean(np.abs(dens - mean_dens[:, np.newaxis]), axis=1)
        score = -np.sum(mad / mean_dens)
        
        if np.isnan(score):
            return -np.inf
        return score
    except Exception as e:
        return -np.inf

if __name__ == '__main__':
    # 2. Set up emcee
    n_dim = 4
    is_fast = os.environ.get("FAST_TEST") == "1"
    n_walkers = 8 if is_fast else 48
    n_steps = 5 if is_fast else 5000 # Reduced for test-warmup safety, can scale up by user
    burn_in = 2 if is_fast else 1000

    # Initial guess around -20 degrees (-0.349 rad)
    initial_guess = np.array([0.0, 0.0, 0.0, np.radians(-20.0)])
    pos = [initial_guess + 1e-2 * np.random.randn(n_dim) for i in range(n_walkers)]

    print(f"Starting MCMC with {n_steps} steps and {burn_in} burn-in...")
    # Use 2 workers if FAST_TEST is set to avoid overhead
    n_workers = 2 if is_fast else None
    with Pool(processes=n_workers) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_likelihood, pool=pool)
        sampler.run_mcmc(pos, n_steps, progress=True)

    print("\nMCMC finished.")

    # 3. Analyze results
    thin_val = 1 if is_fast else 50
    flat_samples = sampler.get_chain(discard=burn_in, thin=thin_val, flat=True)
    
    # Convert alpha (column 3) to degrees for analysis and plotting
    samples_plot = flat_samples.copy()
    samples_plot[:, 3] = np.degrees(samples_plot[:, 3])
    
    labels_plot = ["x0", "y0", "z0", "alpha (deg)"]
    
    # Calculate medians for the lines on the plot
    medians = np.median(samples_plot, axis=0)

    print("\nBest fit (median values):")
    for i in range(n_dim):
        mcmc = np.percentile(samples_plot[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        unit = " deg" if i == 3 else ""
        print(f"{labels_plot[i]}: {mcmc[1]:.4f}{unit} +{q[1]:.4f}/-{q[0]:.4f}")

    # 4. Visualization
    plots_dir = os.path.join(project_root, 'results/density')
    os.makedirs(plots_dir, exist_ok=True)
    data_dir = os.path.join(project_root, 'results/density')
    os.makedirs(data_dir, exist_ok=True)

    fig = corner.corner(samples_plot, labels=labels_plot, truths=medians, 
                        show_titles=True, title_fmt=".3f")
    fig.savefig(os.path.join(plots_dir, "corner_plot_symmetry.png"))
    print(f"\nCorner plot saved to {os.path.join(plots_dir, 'corner_plot_symmetry.png')}")
    
    np.save(os.path.join(data_dir, "symmetry_mcmc_samples.npy"), flat_samples)
    print(f"MCMC chains saved to {os.path.join(data_dir, 'symmetry_mcmc_samples.npy')}")
