#!/bin/bash
set -e # exit immediately if a command exits with a non-zero status

echo "=== STARTING FAST TEST END-TO-END PIPELINE ==="
export FAST_TEST=1

echo "--- Phase 1: Reduction & Coordinate Alignment ---"
/Users/paterson.35/anaconda3/bin/conda run -n bulge_env python src/reduction/run_obs_data_analysis.py

echo "--- Phase 2: Density Precomputation & MCMC Fitting ---"
/Users/paterson.35/anaconda3/bin/conda run -n bulge_env python src/density/fit_vvv_density.py
/Users/paterson.35/anaconda3/bin/conda run -n bulge_env python src/density/fit_vvv_density.py --mcmc
/Users/paterson.35/anaconda3/bin/conda run -n bulge_env python src/density/fit_symmetry.py

echo "--- Phase 3: Kinematic Inversion Metadata ---"
/Users/paterson.35/anaconda3/bin/conda run -n bulge_env python src/inversion/generate_metadata_huston25c20.py

echo "--- Phase 1: Aligning coordinates ---"
/Users/paterson.35/anaconda3/bin/conda run -n bulge_env python src/reduction/align_all_h25c20.py

echo "--- Phase 3: Fit Parametric & Combined Non-Parametric ---"
/Users/paterson.35/anaconda3/bin/conda run -n bulge_env python src/inversion/fit_obs_parametric_h25c20.py
/Users/paterson.35/anaconda3/bin/conda run -n bulge_env python src/inversion/compute_omega_uncertainty.py
/Users/paterson.35/anaconda3/bin/conda run -n bulge_env python src/inversion/fit_obs_combined_h25c20.py

echo "--- Phase 3: Diagnostics ---"
/Users/paterson.35/anaconda3/bin/conda run -n bulge_env python src/inversion/visualise_parametric_fit_h25c20.py
/Users/paterson.35/anaconda3/bin/conda run -n bulge_env python src/inversion/visualise_non_parametric_fit_h25c20.py
/Users/paterson.35/anaconda3/bin/conda run -n bulge_env python src/inversion/plot_bulk_motion_bar_comparison.py
/Users/paterson.35/anaconda3/bin/conda run -n bulge_env python src/inversion/plot_bulk_motion_gc_comparison.py
/Users/paterson.35/anaconda3/bin/conda run -n bulge_env python src/inversion/check_continuity_equation.py

echo "--- Phase 4: Made-to-Measure (M2M) validation & Animations ---"
/Users/paterson.35/anaconda3/bin/conda run -n bulge_env python src/m2m/validate_np_with_m2m.py
/Users/paterson.35/anaconda3/bin/conda run -n bulge_env python src/m2m/plot_m2m_validation.py
/Users/paterson.35/anaconda3/bin/conda run -n bulge_env python src/m2m/animate_m2m_flow.py
/Users/paterson.35/anaconda3/bin/conda run -n bulge_env python src/m2m/animate_m2m_flow_bar.py

echo "=== PIPELINE EXECUTION SUCCESSFUL ==="
