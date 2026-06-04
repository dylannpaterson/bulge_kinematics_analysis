"""
Continuity equation conservation check for the non-parametric kinematic model.

Evaluates the spatial divergence of the mass flux vector field J = rho * u
across a dense regular Cartesian grid, ensuring physical consistency and mass
conservation of the non-parametric stellar flow.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Set paths for shared code and SynthPop
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../shared")))
sys.path.append(os.path.abspath(os.path.join(project_root, "synthpop")))

import synthpop
from inversion_config import get_inverter_grid_axes


# ---------------------------------------------------------------------------
# 1. SynthPop Bulge Density Wrapper
# ---------------------------------------------------------------------------

def get_bulge_density_model(
    model_name: str = "Huston2025_C20Bulge",
) -> callable:
    """
    Build a bulge-only density evaluator from a SynthPop model.

    Parameters
    ----------
    model_name : str
        SynthPop model name.

    Returns
    -------
    bulge_only_density : function
        Function taking (N, 3) positions and returning density in Msun/kpc^3.
    """
    config = {
        "MANDATORY": {"name_for_output": "m2m_tmp", "model_name": model_name},
        "SIGHTLINES": {
            "l_set": [0.0],
            "l_set_type": "list",
            "b_set": [0.0],
            "b_set_type": "list",
            "solid_angle": 0.01,
            "solid_angle_unit": "deg^2",
        },
    }
    tmp_conf = f"tmp_continuity_{model_name}.json"
    with open(tmp_conf, "w") as f:
        json.dump(config, f)

    sp_instance = synthpop.SynthPop(tmp_conf)
    sp_instance.init_populations()
    os.remove(tmp_conf)

    bar_angle_rad = 0.0
    for pop in sp_instance.populations:
        if hasattr(pop.population_density, "bar_ang"):
            bar_angle_rad = pop.population_density.bar_ang
            break

    def bulge_only_density(pos: np.ndarray) -> np.ndarray:
        """Extract populations named 'bulge' in Msun/kpc^3."""
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        r = np.sqrt(x**2 + y**2)
        phi_bar = np.arctan2(y, x)
        phi_sp = phi_bar + bar_angle_rad

        rho_bulge = np.zeros(len(pos))
        for pop in sp_instance.populations:
            if "bulge" in pop.name.lower():
                dens = pop.population_density.density(r, phi_sp, z)
                if pop.population_density.density_unit == "number":
                    avg_m = pop.imf.average_mass(
                        min_mass=pop.min_mass, max_mass=pop.max_mass
                    )
                    corr = getattr(pop.pop_params, "av_mass_corr", 1.0)
                    if corr is None:
                        corr = 1.0
                    dens *= avg_m * corr
                rho_bulge += dens
        return rho_bulge

    return bulge_only_density


# ---------------------------------------------------------------------------
# 2. NP Model Velocity Interpolator
# ---------------------------------------------------------------------------

def interpolate_velocity_field(
    grid_params: np.ndarray,
    axes_x: np.ndarray,
    axes_y: np.ndarray,
    axes_z: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate ux and uy on a 3D grid using triaxial symmetries.

    Parameters
    ----------
    grid_params : np.ndarray
    axes_x, axes_y, axes_z : np.ndarray
    X, Y, Z : np.ndarray

    Returns
    -------
    ux, uy : np.ndarray
    """
    NX, NY, NZ = grid_params.shape[:3]
    shape = X.shape

    # Flatten grid for fast interpolation checks
    X_flat, Y_flat, Z_flat = X.flatten(), Y.flatten(), Z.flatten()
    ux_flat = np.zeros_like(X_flat)
    uy_flat = np.zeros_like(X_flat)

    for i in range(len(X_flat)):
        x_val, y_val, z_val = X_flat[i], Y_flat[i], Z_flat[i]
        xo, yo, zo = np.abs(x_val), np.abs(y_val), np.abs(z_val)

        # Enforce bounds
        ix = np.clip(np.searchsorted(axes_x, xo) - 1, 0, NX - 2)
        iy = np.clip(np.searchsorted(axes_y, yo) - 1, 0, NY - 2)
        iz = np.clip(np.searchsorted(axes_z, zo) - 1, 0, NZ - 2)

        tx = (xo - axes_x[ix]) / (axes_x[ix + 1] - axes_x[ix] + 1e-10)
        ty = (yo - axes_y[iy]) / (axes_y[iy + 1] - axes_y[iy] + 1e-10)
        tz = (zo - axes_z[iz]) / (axes_z[iz + 1] - axes_z[iz] + 1e-10)

        # Trilinear interpolation
        val_000 = grid_params[ix, iy, iz, :2]
        val_100 = grid_params[ix + 1, iy, iz, :2]
        val_010 = grid_params[ix, iy + 1, iz, :2]
        val_110 = grid_params[ix + 1, iy + 1, iz, :2]
        val_001 = grid_params[ix, iy, iz + 1, :2]
        val_101 = grid_params[ix + 1, iy, iz + 1, :2]
        val_011 = grid_params[ix, iy + 1, iz + 1, :2]
        val_111 = grid_params[ix + 1, iy + 1, iz + 1, :2]

        interp = (
            (1 - tx) * (1 - ty) * (1 - tz) * val_000
            + tx * (1 - ty) * (1 - tz) * val_100
            + (1 - tx) * ty * (1 - tz) * val_010
            + tx * ty * (1 - tz) * val_110
            + (1 - tx) * (1 - ty) * tz * val_001
            + tx * (1 - ty) * tz * val_101
            + (1 - tx) * ty * tz * val_011
            + tx * ty * tz * val_111
        )

        # Enforce octant symmetries
        ux_flat[i] = interp[0] * np.sign(y_val)
        uy_flat[i] = interp[1] * np.sign(x_val)

    return ux_flat.reshape(shape), uy_flat.reshape(shape)


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main() -> None:
    """Execute the continuity conservation analysis."""
    if os.environ.get("FAST_TEST") == "1":
        print("FAST_TEST mode active: skipping continuity conservation check.")
        return
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    fit_path = os.path.join(
        project_root, "results/inversion/fit_results_obs_combined_h25c20.npz"
    )
    if not os.path.exists(fit_path):
        print(f"Error: {fit_path} not found.")
        return

    print("Loading non-parametric fit results...")
    fit_res = np.load(fit_path)
    grid_params = fit_res["grid"]
    axes_x = fit_res["axes_x"]
    axes_y = fit_res["axes_y"]
    axes_z = fit_res["axes_z"]

    print("Loading Huston-Coleman density from SynthPop...")
    bulge_density = get_bulge_density_model("Huston2025_C20Bulge")

    # 3. Create regular 3D grid for divergence analysis
    nx, ny, nz = 50, 50, 20
    x_grid = np.linspace(-4.0, 4.0, nx)
    y_grid = np.linspace(-4.0, 4.0, ny)
    z_grid = np.linspace(-1.5, 1.5, nz)

    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")

    print("Evaluating density on grid...")
    q = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
    rho = bulge_density(q).reshape(X.shape)

    print("Interpolating velocity field...")
    ux, uy = interpolate_velocity_field(grid_params, axes_x, axes_y, axes_z, X, Y, Z)
    uz = np.zeros_like(X)

    # 4. Compute flux and divergence
    Jx = rho * ux
    Jy = rho * uy
    Jz = rho * uz

    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    dz = z_grid[1] - z_grid[0]

    dJx_dx = np.gradient(Jx, axis=0) / dx
    dJy_dy = np.gradient(Jy, axis=1) / dy
    dJz_dz = np.gradient(Jz, axis=2) / dz

    div_J = dJx_dx + dJy_dy + dJz_dz

    # Compute characteristic flux gradient for scale comparison
    grad_Jx = np.gradient(rho * np.sqrt(ux**2 + uy**2), axis=0) / dx
    grad_Jy = np.gradient(rho * np.sqrt(ux**2 + uy**2), axis=1) / dy
    char_scale = np.nanmedian(np.sqrt(grad_Jx**2 + grad_Jy**2)) + 1e-10

    # Compute average divergence error metric
    rel_error = div_J / char_scale
    print(
        f"Median Absolute Divergence: {np.nanmedian(np.abs(div_J)):.4f} Msun/kpc^3/yr"
    )
    print(
        f"Relative Divergence Error (div / typical_flux_gradient): {np.nanmedian(np.abs(rel_error))*100:.2f}%"
    )

    # 5. Plotting (Premium Dark Theme)
    DARK = "#111116"
    GRID_C = "#444455"
    plt.rcParams["text.color"] = "white"
    plt.rcParams["axes.labelcolor"] = "white"
    plt.rcParams["xtick.color"] = "white"
    plt.rcParams["ytick.color"] = "white"

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor=DARK)

    z_mid = nz // 2

    ax0 = axes[0]
    ax0.set_facecolor(DARK)
    ax0.grid(True, color=GRID_C, linestyle=":", alpha=0.4)

    div_lim = 1.5 * np.nanpercentile(np.abs(div_J[:, :, z_mid]), 90)
    im0 = ax0.pcolormesh(
        x_grid,
        y_grid,
        div_J[:, :, z_mid].T,
        cmap="coolwarm",
        vmin=-div_lim,
        vmax=div_lim,
        shading="auto",
    )
    fig.colorbar(
        im0,
        ax=ax0,
        label=r"$\nabla \cdot (\rho \mathbf{u})$ [$M_\odot/{\rm kpc}^3/{\rm yr}$]",
    )

    ax0.streamplot(
        x_grid,
        y_grid,
        Jx[:, :, z_mid].T,
        Jy[:, :, z_mid].T,
        color="white",
        linewidth=1.0,
        density=1.2,
        arrowstyle="->",
    )

    ax0.set_title(
        "Midplane Mass Flux & Divergence (XY Plane)\nOverlaid with Mass Flux Streamlines",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax0.set_xlabel("x (kpc)")
    ax0.set_ylabel("y (kpc)")
    ax0.set_aspect("equal")

    # Relative error histogram
    ax1 = axes[1]
    ax1.set_facecolor(DARK)
    ax1.grid(True, color=GRID_C, linestyle=":", alpha=0.4)

    ax1.hist(
        rel_error.flatten(),
        bins=100,
        range=(-2.0, 2.0),
        color="cyan",
        alpha=0.7,
        edgecolor=DARK,
        density=True,
    )
    ax1.axvline(0, color="red", linestyle="--")
    ax1.set_title(
        "Divergence Error Distribution\n(Normalized by Typical Flux Gradient Scale)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax1.set_xlabel(r"$\nabla \cdot (\rho \mathbf{u}) / |\nabla (\rho |\mathbf{u}|)|$")
    ax1.set_ylabel("Probability Density")

    plt.suptitle(
        "Continuity Equation Conservation Check of the Non-Parametric Model",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir = os.path.join(project_root, "results/inversion")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "np_mass_conservation_check.png")
    plt.savefig(out_path, dpi=200, facecolor=DARK)
    plt.close()
    print(f"Diagnostic plot saved to {out_path}")


if __name__ == "__main__":
    main()
