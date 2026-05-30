import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter

class BGMMColorSplitter:
    """
    A Bayesian Gaussian Mixture Model (BGMM) wrapper designed to separate 
    stellar populations based on color distributions.
    
    It fits a multi-component GMM and then groups those components into 
    two populations (typically Bulge and Disk) based on their mean colors.
    """
    
    def __init__(self, n_components=10, weight_concentration_prior=1e-2, max_iter=500, random_state=42):
        self.bgmm = BayesianGaussianMixture(
            n_components=n_components,
            weight_concentration_prior=weight_concentration_prior,
            max_iter=max_iter,
            random_state=random_state
        )
        self.comp_is_bulge = None

    def fit(self, colors):
        """
        Fit the BGMM to color data and identify bulge components.
        
        Args:
            colors (pd.DataFrame or np.ndarray): Color data, e.g., [['j_ks', 'h_ks']]
        """
        self.bgmm.fit(colors)
        
        # Identify bulge components by splitting component means into two groups
        # The group with the redder (higher) J-Ks color is considered the bulge.
        comp_means = self.bgmm.means_
        j_ks_means = comp_means[:, 0].reshape(-1, 1)
        
        color_splitter = GaussianMixture(n_components=2, random_state=42)
        color_splitter.fit(j_ks_means)
        
        # Bulge is the group with the higher mean J-Ks
        bulge_group_idx = np.argmax(color_splitter.means_)
        self.comp_is_bulge = (color_splitter.predict(j_ks_means) == bulge_group_idx)
        return self

    def predict_proba(self, colors):
        """
        Predict the probability of a star belonging to the bulge.
        """
        probs = self.bgmm.predict_proba(colors)
        return probs[:, self.comp_is_bulge].sum(axis=1)


class NPMaxEntMixture:
    """
    A Non-Parametric Maximum Entropy Mixture Model.
    
    This model uses template color distributions (priors) from simulations
    and iteratively fits the population fraction and templates to the observed data
    using a maximum entropy approach.
    """
    
    def __init__(self, h_bins, j_bins, alpha=0.05, beta=0.001):
        """
        Args:
            h_bins (np.ndarray): Edges for the H-Ks color bins.
            j_bins (np.ndarray): Edges for the J-Ks color bins.
            alpha (float): Strength of the template prior (smoothing/regularization).
                           Scaled by total number of stars in fit().
            beta (float): Strength of the population fraction prior.
                          Scaled by total number of stars in fit().
        """
        self.h_bins = h_bins
        self.j_bins = j_bins
        self.alpha_param = alpha
        self.beta_param = beta
        
        self.w_b = None # Population fraction (bulge)
        self.p_b = None # Fitted bulge template
        self.p_d = None # Fitted disk template

    def fit(self, data_colors, q_b_prior, q_d_prior, w_sim_prior, iterations=50):
        """
        Iteratively solve for the population templates and mixing fraction.
        
        Args:
            data_colors (pd.DataFrame): Observed color data.
            q_b_prior (np.ndarray): Initial bulge template (prior).
            q_d_prior (np.ndarray): Initial disk template (prior).
            w_sim_prior (float): Initial bulge fraction (prior).
        """
        n_stars = len(data_colors)
        alpha = self.alpha_param * n_stars
        beta = self.beta_param * n_stars
        
        # Observed counts
        d_counts, _, _, _ = binned_statistic_2d(
            data_colors.iloc[:, 0], data_colors.iloc[:, 1], 
            None, 'count', bins=[self.h_bins, self.j_bins]
        )
        
        # Initialize
        self.w_b = w_sim_prior
        self.p_b = q_b_prior.copy()
        self.p_d = q_d_prior.copy()
        
        for _ in range(iterations):
            w_d = 1.0 - self.w_b
            
            # M is the model distribution
            M = self.w_b * self.p_b + w_d * self.p_d + 1e-12
            
            # Responsibility (probability of being bulge/disk in each bin)
            gamma_b = (self.w_b * self.p_b) / M
            gamma_d = (w_d * self.p_d) / M
            
            # Fractional counts in each bin
            n_b, n_d = d_counts * gamma_b, d_counts * gamma_d
            
            # Update templates
            p_b_new = (n_b + alpha * q_b_prior) / (np.sum(n_b) + alpha)
            p_d_new = (n_d + alpha * q_d_prior) / (np.sum(n_d) + alpha)
            
            # Update fraction
            w_b_new = (np.sum(n_b) + beta * w_sim_prior) / (np.sum(n_b) + np.sum(n_d) + beta)
            
            if abs(w_b_new - self.w_b) < 1e-6:
                break
                
            self.w_b, self.p_b, self.p_d = w_b_new, p_b_new, p_d_new
            
        return self

    def predict_proba(self, data_colors):
        """
        Calculate the bulge probability for individual stars based on their bin.
        """
        h_idx = np.clip(np.digitize(data_colors.iloc[:, 0], self.h_bins) - 1, 0, len(self.h_bins)-2)
        j_idx = np.clip(np.digitize(data_colors.iloc[:, 1], self.j_bins) - 1, 0, len(self.j_bins)-2)
        
        prob_b = self.w_b * self.p_b[h_idx, j_idx]
        prob_d = (1.0 - self.w_b) * self.p_d[h_idx, j_idx]
        
        return prob_b / (prob_b + prob_d + 1e-12)


def get_intrinsic_moments(wb, pm_l, pm_b, pm_l_error, pm_b_error, pm_lb_corr=None):
    """
    Calculates weighted intrinsic moments with quadrature noise subtraction.
    
    Args:
        wb (np.ndarray): Weights for each star.
        pm_l, pm_b (np.ndarray): Proper motions.
        pm_l_error, pm_b_error (np.ndarray): Measurement errors.
        pm_lb_corr (np.ndarray, optional): Correlation between l and b errors.
    
    Returns:
        mu_l, mu_b, sig_l, sig_b, cov_lb
    """
    sum_wb = np.sum(wb)
    if sum_wb <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Means
    mu_l = np.sum(wb * pm_l) / sum_wb
    mu_b = np.sum(wb * pm_b) / sum_wb
    
    # Weighted squared errors for quadrature subtraction
    err_l2 = np.sum(wb * pm_l_error**2) / sum_wb
    err_b2 = np.sum(wb * pm_b_error**2) / sum_wb

    # Intrinsic dispersion (Quadrature Subtraction APPLIED)
    sig_l = np.sqrt(max(0, np.sum(wb * (pm_l - mu_l)**2) / sum_wb - err_l2))
    sig_b = np.sqrt(max(0, np.sum(wb * (pm_b - mu_b)**2) / sum_wb - err_b2))
    
    # Intrinsic Covariance
    obs_cov_lb = np.sum(wb * (pm_l - mu_l) * (pm_b - mu_b)) / sum_wb
    
    if pm_lb_corr is not None:
        err_cov_lb = np.sum(wb * pm_lb_corr * pm_l_error * pm_b_error) / sum_wb
        cov_lb = obs_cov_lb - err_cov_lb
    else:
        cov_lb = obs_cov_lb

    return mu_l, mu_b, sig_l, sig_b, cov_lb


def get_intrinsic_moments_with_errors(wb, pm_l, pm_b, pm_l_error, pm_b_error, pm_lb_corr=None, n_boot=50):
    """
    Calculates weighted intrinsic moments and estimates errors using two methods:
    1. Analytic (using effective N)
    2. Bootstrap
    
    Returns:
        Dictionary containing moments and both sets of errors.
    """
    # Force everything to numpy arrays to prevent indexing errors during bootstrap
    wb = np.asarray(wb)
    pm_l = np.asarray(pm_l)
    pm_b = np.asarray(pm_b)
    pm_l_error = np.asarray(pm_l_error)
    pm_b_error = np.asarray(pm_b_error)
    if pm_lb_corr is not None:
        pm_lb_corr = np.asarray(pm_lb_corr)

    # 1. Base Moments
    mu_l, mu_b, sig_l, sig_b, cov_lb = get_intrinsic_moments(wb, pm_l, pm_b, pm_l_error, pm_b_error, pm_lb_corr)
    
    if np.isnan(mu_l):
        return None

    # 2. Analytic Errors (using effective N to account for weights)
    sum_w = np.sum(wb)
    sum_w2 = np.sum(wb**2)
    if sum_w2 <= 0: return None
    neff = sum_w**2 / sum_w2
    
    err_mu_l_ana = sig_l / np.sqrt(neff)
    err_mu_b_ana = sig_b / np.sqrt(neff)
    err_sig_l_ana = sig_l / np.sqrt(2 * neff)
    err_sig_b_ana = sig_b / np.sqrt(2 * neff)
    # Analytic error on covariance: sqrt((var_l*var_b + cov^2)/N)
    err_cov_lb_ana = np.sqrt((sig_l**2 * sig_b**2 + cov_lb**2) / neff)
    
    # 3. Bootstrap Errors
    boot_res = []
    n_stars = len(wb)
    indices = np.arange(n_stars)
    for _ in range(n_boot):
        idx = np.random.choice(indices, size=n_stars, replace=True)
        c_boot = pm_lb_corr[idx] if pm_lb_corr is not None else None
        
        res = get_intrinsic_moments(wb[idx], pm_l[idx], pm_b[idx], pm_l_error[idx], pm_b_error[idx], c_boot)
        if not np.isnan(res[0]):
            boot_res.append(res)
            
    if len(boot_res) > 2:
        boot_res = np.array(boot_res)
        errs_boot = np.nanstd(boot_res, axis=0)
    else:
        errs_boot = [np.nan] * 5

    return {
        'mu_l': mu_l, 'mu_b': mu_b, 'sig_l': sig_l, 'sig_b': sig_b, 'cov_lb': cov_lb,
        'err_mu_l_ana': err_mu_l_ana, 'err_mu_b_ana': err_mu_b_ana,
        'err_sig_l_ana': err_sig_l_ana, 'err_sig_b_ana': err_sig_b_ana,
        'err_cov_lb_ana': err_cov_lb_ana,
        'err_mu_l_boot': errs_boot[0], 'err_mu_b_boot': errs_boot[1],
        'err_sig_l_boot': errs_boot[2], 'err_sig_b_boot': errs_boot[3],
        'err_cov_lb_boot': errs_boot[4]
    }
