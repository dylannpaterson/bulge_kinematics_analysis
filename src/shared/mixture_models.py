"""
Stellar kinematic mixture models and intrinsic moment estimators.

Provides:

* ``BGMMColorSplitter`` — Bayesian GMM colour-based bulge/disk separator.
* ``NPMaxEntMixture`` — Non-parametric maximum-entropy template mixture model.
* ``get_intrinsic_moments`` — Weighted intrinsic PM moments with noise subtraction.
* ``get_intrinsic_moments_with_errors`` — As above, plus analytic and bootstrap errors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter  # noqa: F401 — re-exported for callers
from scipy.stats import binned_statistic_2d
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture


# ---------------------------------------------------------------------------
# Colour-based bulge/disk separator
# ---------------------------------------------------------------------------

class BGMMColorSplitter:
    """
    Bayesian GMM wrapper for separating bulge and disk stellar populations.

    Fits a multi-component BGMM to photometric colour data, then clusters
    the component means into two groups (bulge / disk) based on J-K_s colour.

    Parameters
    ----------
    n_components : int, optional
        Maximum number of BGMM components.  Default is 10.
    weight_concentration_prior : float, optional
        Dirichlet concentration prior; smaller values encourage sparsity.
        Default is 1e-2.
    max_iter : int, optional
        EM iteration limit.  Default is 500.
    random_state : int, optional
        Random seed for reproducibility.  Default is 42.
    """

    def __init__(
        self,
        n_components: int = 10,
        weight_concentration_prior: float = 1e-2,
        max_iter: int = 500,
        random_state: int = 42,
    ) -> None:
        self.bgmm = BayesianGaussianMixture(
            n_components=n_components,
            weight_concentration_prior=weight_concentration_prior,
            max_iter=max_iter,
            random_state=random_state,
        )
        self.comp_is_bulge: np.ndarray | None = None

    def fit(self, colors: np.ndarray | pd.DataFrame) -> BGMMColorSplitter:
        """
        Fit the BGMM and identify which components belong to the bulge.

        The bulge group is the cluster of component means with the higher
        mean J-K_s colour.

        Parameters
        ----------
        colors : array_like, shape (N, 2)
            Colour columns ``[j_ks, h_ks]`` for each star.

        Returns
        -------
        self : BGMMColorSplitter
        """
        self.bgmm.fit(colors)

        j_ks_means = self.bgmm.means_[:, 0].reshape(-1, 1)
        splitter = GaussianMixture(n_components=2, random_state=42)
        splitter.fit(j_ks_means)

        bulge_group = np.argmax(splitter.means_)
        self.comp_is_bulge = splitter.predict(j_ks_means) == bulge_group
        return self

    def predict_proba(self, colors: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Return per-star probability of belonging to the bulge population.

        Parameters
        ----------
        colors : array_like, shape (N, 2)
            Colour data ``[j_ks, h_ks]``.

        Returns
        -------
        p_bulge : np.ndarray, shape (N,)
            Bulge membership probability in [0, 1].
        """
        probs = self.bgmm.predict_proba(colors)
        return probs[:, self.comp_is_bulge].sum(axis=1)


# ---------------------------------------------------------------------------
# Non-parametric maximum-entropy mixture model
# ---------------------------------------------------------------------------

class NPMaxEntMixture:
    """
    Non-parametric maximum-entropy bulge/disk mixture model.

    Uses simulation-derived template colour distributions as priors and
    iteratively updates both the templates and the bulge fraction using
    an EM-like maximum-entropy scheme.

    Parameters
    ----------
    h_bins : np.ndarray
        Bin edges for the H-K_s colour axis.
    j_bins : np.ndarray
        Bin edges for the J-K_s colour axis.
    alpha : float, optional
        Template prior strength, scaled by the number of stars at fit time.
        Default is 0.05.
    beta : float, optional
        Fraction prior strength, scaled by the number of stars at fit time.
        Default is 0.001.
    """

    def __init__(
        self,
        h_bins: np.ndarray,
        j_bins: np.ndarray,
        alpha: float = 0.05,
        beta: float = 0.001,
    ) -> None:
        self.h_bins = h_bins
        self.j_bins = j_bins
        self._alpha = alpha
        self._beta = beta

        self.w_b: float | None = None
        self.p_b: np.ndarray | None = None
        self.p_d: np.ndarray | None = None

    def fit(
        self,
        data_colors: pd.DataFrame,
        q_b_prior: np.ndarray,
        q_d_prior: np.ndarray,
        w_sim_prior: float,
        iterations: int = 50,
    ) -> NPMaxEntMixture:
        """
        Solve for population templates and bulge mixing fraction.

        Parameters
        ----------
        data_colors : pd.DataFrame
            Observed colour columns ``[h_ks, j_ks]`` for each star.
        q_b_prior : np.ndarray
            Initial bulge colour template (normalised 2-D histogram).
        q_d_prior : np.ndarray
            Initial disk colour template (normalised 2-D histogram).
        w_sim_prior : float
            Prior estimate of the bulge fraction.
        iterations : int, optional
            Maximum EM iterations.  Default is 50.

        Returns
        -------
        self : NPMaxEntMixture
        """
        n_stars = len(data_colors)
        alpha = self._alpha * n_stars
        beta = self._beta * n_stars

        d_counts, _, _, _ = binned_statistic_2d(
            data_colors.iloc[:, 0],
            data_colors.iloc[:, 1],
            None,
            "count",
            bins=[self.h_bins, self.j_bins],
        )

        self.w_b = w_sim_prior
        self.p_b = q_b_prior.copy()
        self.p_d = q_d_prior.copy()

        for _ in range(iterations):
            w_d = 1.0 - self.w_b
            model = self.w_b * self.p_b + w_d * self.p_d + 1e-12

            gamma_b = (self.w_b * self.p_b) / model
            gamma_d = (w_d * self.p_d) / model
            n_b = d_counts * gamma_b
            n_d = d_counts * gamma_d

            p_b_new = (n_b + alpha * q_b_prior) / (np.sum(n_b) + alpha)
            p_d_new = (n_d + alpha * q_d_prior) / (np.sum(n_d) + alpha)
            w_b_new = (np.sum(n_b) + beta * w_sim_prior) / (
                np.sum(n_b) + np.sum(n_d) + beta
            )

            if abs(w_b_new - self.w_b) < 1e-6:
                break

            self.w_b, self.p_b, self.p_d = w_b_new, p_b_new, p_d_new

        return self

    def predict_proba(self, data_colors: pd.DataFrame) -> np.ndarray:
        """
        Return per-star bulge membership probability from colour-bin lookup.

        Parameters
        ----------
        data_colors : pd.DataFrame
            Colour data with columns ``[h_ks, j_ks]``.

        Returns
        -------
        p_bulge : np.ndarray, shape (N,)
        """
        h_idx = np.clip(
            np.digitize(data_colors.iloc[:, 0], self.h_bins) - 1,
            0, len(self.h_bins) - 2,
        )
        j_idx = np.clip(
            np.digitize(data_colors.iloc[:, 1], self.j_bins) - 1,
            0, len(self.j_bins) - 2,
        )

        prob_b = self.w_b * self.p_b[h_idx, j_idx]
        prob_d = (1.0 - self.w_b) * self.p_d[h_idx, j_idx]
        return prob_b / (prob_b + prob_d + 1e-12)


# ---------------------------------------------------------------------------
# Intrinsic proper-motion moment estimators
# ---------------------------------------------------------------------------

def get_intrinsic_moments(
    wb: np.ndarray,
    pm_l: np.ndarray,
    pm_b: np.ndarray,
    pm_l_error: np.ndarray,
    pm_b_error: np.ndarray,
    pm_lb_corr: np.ndarray | None = None,
) -> tuple[float, float, float, float, float]:
    """
    Compute weighted intrinsic proper-motion moments with noise subtraction.

    The intrinsic dispersions are obtained by quadrature subtraction of the
    mean squared measurement error from the observed weighted variance.

    Parameters
    ----------
    wb : np.ndarray, shape (N,)
        Per-star weights (e.g. unity or selection weights).
    pm_l : np.ndarray, shape (N,)
        Proper motion in Galactic longitude (mas/yr).
    pm_b : np.ndarray, shape (N,)
        Proper motion in Galactic latitude (mas/yr).
    pm_l_error : np.ndarray, shape (N,)
        Measurement uncertainty on ``pm_l`` (mas/yr).
    pm_b_error : np.ndarray, shape (N,)
        Measurement uncertainty on ``pm_b`` (mas/yr).
    pm_lb_corr : np.ndarray, shape (N,), optional
        Correlation coefficient between ``pm_l`` and ``pm_b`` errors.

    Returns
    -------
    mu_l : float
        Weighted mean proper motion in l.
    mu_b : float
        Weighted mean proper motion in b.
    sig_l : float
        Intrinsic dispersion in l (mas/yr).
    sig_b : float
        Intrinsic dispersion in b (mas/yr).
    cov_lb : float
        Intrinsic covariance between l and b (mas²/yr²).
    """
    sum_wb = np.sum(wb)
    if sum_wb <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    mu_l = np.sum(wb * pm_l) / sum_wb
    mu_b = np.sum(wb * pm_b) / sum_wb

    err_l2 = np.sum(wb * pm_l_error**2) / sum_wb
    err_b2 = np.sum(wb * pm_b_error**2) / sum_wb

    sig_l = np.sqrt(max(0.0, np.sum(wb * (pm_l - mu_l)**2) / sum_wb - err_l2))
    sig_b = np.sqrt(max(0.0, np.sum(wb * (pm_b - mu_b)**2) / sum_wb - err_b2))

    obs_cov_lb = np.sum(wb * (pm_l - mu_l) * (pm_b - mu_b)) / sum_wb
    if pm_lb_corr is not None:
        err_cov_lb = np.sum(wb * pm_lb_corr * pm_l_error * pm_b_error) / sum_wb
        cov_lb = obs_cov_lb - err_cov_lb
    else:
        cov_lb = obs_cov_lb

    return mu_l, mu_b, sig_l, sig_b, cov_lb


def get_intrinsic_moments_with_errors(
    wb: np.ndarray,
    pm_l: np.ndarray,
    pm_b: np.ndarray,
    pm_l_error: np.ndarray,
    pm_b_error: np.ndarray,
    pm_lb_corr: np.ndarray | None = None,
    n_boot: int = 50,
) -> dict | None:
    """
    Compute intrinsic moments with analytic and bootstrap uncertainty estimates.

    Parameters
    ----------
    wb : np.ndarray, shape (N,)
        Per-star weights.
    pm_l : np.ndarray, shape (N,)
        Proper motion in l (mas/yr).
    pm_b : np.ndarray, shape (N,)
        Proper motion in b (mas/yr).
    pm_l_error : np.ndarray, shape (N,)
        Uncertainty on ``pm_l`` (mas/yr).
    pm_b_error : np.ndarray, shape (N,)
        Uncertainty on ``pm_b`` (mas/yr).
    pm_lb_corr : np.ndarray, shape (N,), optional
        Correlation coefficient between ``pm_l`` and ``pm_b`` errors.
    n_boot : int, optional
        Number of bootstrap resamples.  Default is 50.

    Returns
    -------
    result : dict or None
        Dictionary with keys ``mu_l``, ``mu_b``, ``sig_l``, ``sig_b``,
        ``cov_lb``, plus ``err_*_ana`` and ``err_*_boot`` suffixed variants
        for analytic and bootstrap errors respectively.  Returns ``None``
        if the moment calculation fails (e.g. empty sample).
    """
    wb = np.asarray(wb)
    pm_l = np.asarray(pm_l)
    pm_b = np.asarray(pm_b)
    pm_l_error = np.asarray(pm_l_error)
    pm_b_error = np.asarray(pm_b_error)
    if pm_lb_corr is not None:
        pm_lb_corr = np.asarray(pm_lb_corr)

    mu_l, mu_b, sig_l, sig_b, cov_lb = get_intrinsic_moments(
        wb, pm_l, pm_b, pm_l_error, pm_b_error, pm_lb_corr
    )
    if np.isnan(mu_l):
        return None

    # Analytic errors via effective sample size
    sum_w = np.sum(wb)
    sum_w2 = np.sum(wb**2)
    if sum_w2 <= 0:
        return None
    n_eff = sum_w**2 / sum_w2

    err_mu_l_ana = sig_l / np.sqrt(n_eff)
    err_mu_b_ana = sig_b / np.sqrt(n_eff)
    err_sig_l_ana = sig_l / np.sqrt(2 * n_eff)
    err_sig_b_ana = sig_b / np.sqrt(2 * n_eff)
    err_cov_lb_ana = np.sqrt((sig_l**2 * sig_b**2 + cov_lb**2) / n_eff)

    # Bootstrap errors
    n_stars = len(wb)
    indices = np.arange(n_stars)
    boot_results = []
    for _ in range(n_boot):
        idx = np.random.choice(indices, size=n_stars, replace=True)
        corr_boot = pm_lb_corr[idx] if pm_lb_corr is not None else None
        res = get_intrinsic_moments(
            wb[idx], pm_l[idx], pm_b[idx],
            pm_l_error[idx], pm_b_error[idx], corr_boot,
        )
        if not np.isnan(res[0]):
            boot_results.append(res)

    if len(boot_results) > 2:
        errs_boot = np.nanstd(np.array(boot_results), axis=0)
    else:
        errs_boot = [np.nan] * 5

    return {
        "mu_l": mu_l, "mu_b": mu_b,
        "sig_l": sig_l, "sig_b": sig_b,
        "cov_lb": cov_lb,
        "err_mu_l_ana": err_mu_l_ana,
        "err_mu_b_ana": err_mu_b_ana,
        "err_sig_l_ana": err_sig_l_ana,
        "err_sig_b_ana": err_sig_b_ana,
        "err_cov_lb_ana": err_cov_lb_ana,
        "err_mu_l_boot": errs_boot[0],
        "err_mu_b_boot": errs_boot[1],
        "err_sig_l_boot": errs_boot[2],
        "err_sig_b_boot": errs_boot[3],
        "err_cov_lb_boot": errs_boot[4],
    }
