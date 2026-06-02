# Bulge Kinematic Inversion & Density Modeling

This document describes the mathematical framework and implementation details for the 3D kinematic reconstruction and structural fitting of the Galactic bulge.

## 1. Bulge Density Model

The underlying stellar density field $\rho(\vec{x})$ is based on the **Coleman2020BulgeSymmetric** model, an 8-way symmetrized representation of the VVV bulge. The model is extended with independent scaling along its principal axes.

### Parameters
The density model is controlled by 5 free parameters:
- $n_0$: Overall density normalization factor.
- $\alpha$: Bar angle (degrees), defined as the orientation of the major axis relative to the Sun-GC line.
- $x_{scale}, y_{scale}, z_{scale}$: Spatial scale factors applied along the bar's intrinsic **minor**, **major**, and **vertical** axes respectively.

### Coordinate Transformation
Scaling is applied in the bar's rest frame. For a point $\vec{x}_{GC} = (x, y, z)^T$ in Galactocentric Cartesian coordinates, the transformation to the scaled bar frame $\vec{x}'_{bar}$ is:

$$ \begin{pmatrix} x' \\ y' \end{pmatrix}_{bar} = \begin{pmatrix} \cos\alpha & \sin\alpha \\ -\sin\alpha & \cos\alpha \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix}_{GC} \circ \begin{pmatrix} 1/x_{scale} \\ 1/y_{scale} \end{pmatrix} $$
$$ z'_{bar} = z_{GC} / z_{scale} $$

This mapping assumes the bar nearside is oriented towards $y < 0$ (negative Galactic longitude).

---

## 2. Intrinsic 3D Kinematics

The model reconstructs the intrinsic 3D kinematic fields—the mean velocity vector $\vec{\mu}_{3D}(\vec{x})$ and the velocity dispersion tensor $\Sigma_{int}(\vec{x})$—across a discrete spatial grid in the bar frame.

### Non-Parametric Grid (Nodes)
$\Sigma_{int}$ is parameterized via a lower-triangular Cholesky decomposition, $\mathbf{L}$, to ensure it remains symmetric and positive semi-definite:
$$\Sigma_{int}(\vec{x}) = \mathbf{L}(\vec{x})\mathbf{L}(\vec{x})^T = \begin{pmatrix} L_{11} & 0 & 0 \\ L_{21} & L_{22} & 0 \\ 0 & 0 & L_{33} \end{pmatrix} \begin{pmatrix} L_{11} & L_{21} & 0 \\ 0 & L_{22} & 0 \\ 0 & 0 & L_{33} \end{pmatrix}$$

The optimized parameters at each node are $\theta = \{u_x, u_y, \log L_{11}, L_{21}, \log L_{22}, \log L_{33}\}$. We assume triaxial symmetries:
$$\vec{\mu}_{3D}(\vec{x}) = \begin{pmatrix} u_x(|x|, |y|, |z|) \cdot \text{sgn}(y) \\ u_y(|x|, |y|, |z|) \cdot \text{sgn}(x) \\ 0 \end{pmatrix}$$

### Parametric Reference Model
For initialization and priors, a Koshimoto-style parametric form is used. Streaming velocity follows a sigmoid profile along the major axis, and velocity dispersion follows a core-disk profile.

---

## 3. Geometry and Observational Projection

Intrinsic kinematics are projected into the heliocentric frame $(l, b, D)$ using a projection matrix $\mathbf{P}$:

$$\vec{\mu}_{loc} = \frac{\kappa}{D} \left( \mathbf{P}\vec{\mu}_{3D} + \mathbf{R}_{helio \to lb} (\vec{v}_{rot} - \vec{v}_\odot) \right)$$
$$\Sigma_{loc} = \left(\frac{\kappa}{D}\right)^2 \mathbf{P} \Sigma_{int} \mathbf{P}^T$$

where:
- $\vec{v}_{rot} = (\Omega_p y_{bar}, -\Omega_p x_{bar}, 0)$ accounts for the pattern speed rotation.
- $\kappa = 0.2108$ is the conversion factor for mas/yr.
- $\mathbf{P} = \mathbf{R}_{helio \to lb} \mathbf{H} \mathbf{R}_{bar \to GC}$ incorporates the heliocentric basis, the Galactic plane tilt ($\mathbf{H}$), and the bar rotation.

---

## 4. Selection-Weighted Line-of-Sight Integration

To compare the model to binned data, we integrate along the line-of-sight distance $s$, weighting by the effective density $\rho_{eff}(s, k) = \rho(s) \cdot S_k(s)$, where $S_k(s)$ is the selection function for bin $k$.

### Selection Functions ($S_k(s)$)
$S_k(s)$ is the fraction of the Initial Mass Function (IMF) that falls within the bin's magnitude limits $[m_{min}, m_{max}]$ at distance $s$:
$$S_k(s) = \frac{1}{\langle M \rangle} \int_{M(m_{min}, s)}^{M(m_{max}, s)} M \cdot \text{IMF}(M) dM$$

### Integrated Observables
The predicted mean velocity $\vec{\mu}_{pred, k}$ and variance $\Sigma_{pred, k}$ are:
$$W_k = \sum_s \rho_{eff}(s, k) \cdot s^2$$
$$\vec{\mu}_{pred, k} = \frac{1}{W_k} \sum_s \rho_{eff}(s, k) \vec{\mu}_{loc}(s) \cdot s^2$$
$$\Sigma_{pred, k} = \frac{1}{W_k} \sum_s \rho_{eff}(s, k) \left( \Sigma_{loc}(s) + \Delta\vec{\mu}_s \Delta\vec{\mu}_s^T \right) s^2$$
where $\Delta\vec{\mu}_s = \vec{\mu}_{loc}(s) - \vec{\mu}_{pred, k}$.

---

## 5. Optimization Pipeline

The pipeline follows a two-stage process:

### Stage 1: Structural Fit (Density)
The 5 density parameters are optimized against VVV star counts using a Poisson likelihood:
1. **L-BFGS Pre-optimization**: Quickly finds the local minimum.
2. **MCMC Sampling**: Explores the parameter space (including degeneracies between $\alpha$ and scales) to provide full posterior uncertainties.

### Stage 2: Kinematic Inversion
The kinematic grid parameters are optimized by minimizing the negative log-likelihood of the proper motion and radial velocity distributions:
$$\mathcal{L} = \sum_{pixels} \sum_{bins} \left[ \ln |\Sigma_{pred}| + \text{Tr}(\Sigma_{pred}^{-1} \Sigma_{obs}) + (\Delta\vec{\mu})^T \Sigma_{pred}^{-1} \Delta\vec{\mu} \right] + \lambda_{KL} S_{KL}$$
A Kullback-Leibler (KL) divergence penalty $S_{KL}$ is applied against the parametric model to regularize poorly constrained regions.
