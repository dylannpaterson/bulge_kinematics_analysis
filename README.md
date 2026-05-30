# Galactic Bulge Modelling & Dynamical Unification Pipeline

A unified, self-contained pipeline for high-fidelity Milky Way Galactic Bulge structural and dynamical modeling. This repository integrates four sequential scientific phases: proper motion reduction, triaxial density modeling, non-parametric kinematic inversion (with parametric prior tuning), and self-consistent Made-to-Measure (M2M) orbital superposition.

---

## 1. Directory Structure

This pipeline cleanly separates raw input datasets (`data/`) from synthesized output files (`results/` subfolders organized by step), avoiding folder clutter:

```
bulge_model_pipeline/
├── data/               # Unified raw external datasets (symbolic links)
│   ├── virac2_bulge_tiles_deep/
│   ├── vvvdaophot_completed_masked_mlferr_newzp.fits
│   ├── vvvdaophot_kserr.fits
│   ├── brava_paper_results.csv
│   └── GIBS_full.txt
├── results/            # Pipeline synthesized outputs
│   ├── reduction/      # Step 1: Intrinsic moments CSV & aligned Bravas/GIBS/VIRAC2 .npz files
│   ├── density/        # Step 2: emcee VVV MCMC chains, corners, symmetry optimization
│   ├── inversion/      # Step 3: Metadata pkl, parametric/NP fits, vector comparison plots, continuity check
│   └── m2m/            # Step 4: Agama orbit superposition weights, validation diagnostics, flow animations
└── src/
    ├── reduction/      # Proper motion moment calculation and coordinate alignments
    ├── density/        # emcee VVV density fitting & octant symmetry optimizations
    ├── inversion/      # JAX kinematic parametric prior + non-parametric inverter
    ├── m2m/            # Agama orbit library integration & weighting solver
    └── shared/         # Common configurations, providers, and extinction models
```

---

## 2. Execution Workflow

To run the unified Bulge model machinery, execute the phases in chronological order:

### **Phase 1: VIRAC Proper Motion Moment Reduction & Coordinate Alignment**
Reduces raw proper motion data from deep parquet tiles, cleans proper motion errors via iterative sigma clipping, applies extinction corrections, and calculates binned proper motion moments. Afterwards, coordinate alignment is run to map Bravas, GIBS, and VIRAC2 datasets to the coordinate system.
```bash
# 1. Reduce proper motion moments
python src/reduction/run_real_data_analysis.py

# 2. Align coordinates and moments
python src/reduction/align_all_h25c20.py
```
* **Primary Outputs**: 
  * `results/reduction/virac2_magnitude_binned_moments.csv`
  * `results/reduction/virac2_real_aligned_h25c20.npz`
  * `results/reduction/brava_real_aligned_h25c20.npz`
  * `results/reduction/gibs_real_aligned_h25c20.npz`

---

### **Phase 2: Triaxial VVV Density MCMC Fitting**
Fits a 3D triaxial density model to the original VVV star counts using `emcee` parallel MCMC walkers, calling the 8-way symmetric JAX-compiled density core.
```bash
# 1. Precompute selection grids
python src/density/fit_vvv_density.py

# 2. Run MCMC Ensemble sampler
python src/density/fit_vvv_density.py --mcmc
```
* **Primary Outputs**: 
  * `results/density/vvv_fit_results.json`
  * `results/density/corner_plot.png`
  * `results/density/best_fit_residuals_2d.png`

*(Optional Calibration)*: Optimize offset ($x_0, y_0, z_0$) and bar tilt ($\alpha$) parameters by minimizing relative Mean Absolute Deviation across all 8 octants:
```bash
python src/density/fit_symmetry.py
```
* **Primary Outputs**: `results/density/corner_plot_symmetry.png`, `results/density/symmetry_mcmc_samples.npy`

---

### **Phase 3: Kinematic Inversion & Parametric Prior Sequence**
Runs the free-form non-parametric kinematic inversion to solve for JAX bulk streaming velocities ($u_x, u_y$) and the 3D velocity dispersion tensor on the coordinate grid. This phase requires a strict sequence to construct the parametric anisotropic prior:

```bash
# 1. Generate fit metadata caches
python src/inversion/generate_metadata_huston25c20.py

# 2. Fit the Parametric Kinematic Model (Prerequisite Prior)
python src/inversion/fit_real_parametric_h25c20.py

# 3. Compute Pattern Speed Ωp Uncertainty bounds
python src/inversion/compute_omega_uncertainty.py

# 4. Run the Combined Non-Parametric Solver (Uses the parametric fit as anisotropic prior)
python src/inversion/fit_real_combined_h25c20.py
```
* **Primary Outputs**: 
  * `results/inversion/huston25c20_metadata_science_binned.pkl`
  * `results/inversion/parametric_fit_results_real_h25c20.npz`
  * `results/inversion/fit_results_real_combined_h25c20.npz`

*(Inversion Diagnostics & Comparison plots)*:
```bash
# A. Visualize parametric fit profiles
python src/inversion/visualise_parametric_fit_h25c20.py

# B. Visualize non-parametric fit profiles
python src/inversion/visualise_non_parametric_fit_h25c20.py

# C. Plot bulk-motion comparison dashboard in the Bar frame
python src/inversion/plot_bulk_motion_bar_comparison.py

# D. Plot bulk-motion comparison dashboard in the inertial GC frame
python src/inversion/plot_bulk_motion_gc_comparison.py

# E. Evaluate mass conservation check (continuity equation residuals)
python src/inversion/check_continuity_equation.py
```
* **Primary Plots**: 
  * `results/inversion/parametric_fit_real_h25c20_sigl.png`
  * `results/inversion/non_parametric_fit_mu_l_h25c20.png`
  * `results/inversion/bulk_motion_bar_comparison.png`
  * `results/inversion/bulk_motion_gc_comparison.png`
  * `results/inversion/np_mass_conservation_check.png`

---

### **Phase 4: Agama Made-to-Measure (M2M) Validation**
Integrates 40,000 stellar orbits for 3 Gyr in the Agama triaxial multipole potential, optimizing particle weights to fit the non-parametric density and kinematics, and compiles dynamic animations.
```bash
# 1. Integrate orbit library and fit weights
python src/m2m/validate_np_with_m2m.py

# 2. Plot diagnostic comparisons (kinematics, dispersions, weights)
python src/m2m/plot_m2m_validation.py

# 3. Generate 30,000-particle N-body flow animations (GC & Stationary Bar frames)
python src/m2m/animate_m2m_flow.py
python src/m2m/animate_m2m_flow_bar.py
```
* **Primary Outputs**: 
  * `results/m2m/np_m2m_fit_results.npz`
  * `results/m2m/m2m_validation_kinematics.png`
  * `results/m2m/m2m_validation_dispersions.png`
  * `results/m2m/m2m_weight_entropy.png`
  * `results/m2m/m2m_flow_animation.gif`
  * `results/m2m/m2m_flow_animation_bar.gif`

---

## 3. Style & Performance Guidelines
1. **CPU Threading Safety**: Thread limits are propagated automatically inside each fit script (`OMP_NUM_THREADS=1`) to prevent CPU thrashing in parallel workers.
2. **Visual Standard**: All generated plots are strictly formatted in the high-contrast `#111116` dark-theme visual style.
