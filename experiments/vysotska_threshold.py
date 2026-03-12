"""
Implementation of Vysotska et al. (ICRA 2025) adaptive thresholding.

Reimplemented from the paper description:
  1. Extract a p×p patch from the similarity matrix (p=20)
  2. KS test: check if patch values are bimodal (reject unimodal null hypothesis)
  3. If bimodal: fit 2-component GMM, find decision boundary between components
  4. Smooth threshold with 1D Kalman filter

This is just the thresholding method — we use our own matching pipeline
(not their graph-based sequence matcher) so we can compare threshold
methods apples-to-apples.

Usage:
    from experiments.vysotska_threshold import VysotskaDaptiveThreshold
"""

import numpy as np
from scipy.stats import kstest, norm
from sklearn.mixture import GaussianMixture


class KalmanFilter1D:
    """Simple 1D Kalman filter for smoothing threshold estimates."""

    def __init__(self, process_noise=1e-3, measurement_noise=1e-2):
        self.Q = process_noise      # process noise
        self.R = measurement_noise  # measurement noise
        self.x = None               # state estimate
        self.P = 1.0                # error covariance

    def update(self, measurement):
        if self.x is None:
            self.x = measurement
            return self.x

        # Predict (state doesn't change, just uncertainty grows)
        P_pred = self.P + self.Q

        # Update
        K = P_pred / (P_pred + self.R)  # Kalman gain
        self.x = self.x + K * (measurement - self.x)
        self.P = (1 - K) * P_pred

        return self.x


def gmm_decision_boundary(means, stds, weights):
    """
    Find the decision boundary between two Gaussian components.

    Solves: π₁ g(θ, μ₁, σ₁) = π₂ g(θ, μ₂, σ₂)

    Which becomes the quadratic (from the paper):
    log(π₁σ₂/π₂σ₁) = (θ-μ₁)²/(2σ₁²) - (θ-μ₂)²/(2σ₂²)

    Returns the boundary between the two means, or None if no valid solution.
    """
    mu1, mu2 = means
    s1, s2 = stds
    w1, w2 = weights

    # Ensure mu1 < mu2
    if mu1 > mu2:
        mu1, mu2 = mu2, mu1
        s1, s2 = s2, s1
        w1, w2 = w2, w1

    # Avoid degenerate cases
    if s1 < 1e-8 or s2 < 1e-8:
        return (mu1 + mu2) / 2

    # Quadratic equation: a*θ² + b*θ + c = 0
    # From expanding the log equation
    a = 1 / (2 * s1**2) - 1 / (2 * s2**2)
    b = mu2 / (s2**2) - mu1 / (s1**2)
    c = mu1**2 / (2 * s1**2) - mu2**2 / (2 * s2**2) - np.log((s2 * w1) / (s1 * w2 + 1e-12) + 1e-12)

    if abs(a) < 1e-12:
        # Linear case
        if abs(b) < 1e-12:
            return (mu1 + mu2) / 2
        return -c / b

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return (mu1 + mu2) / 2

    sqrt_disc = np.sqrt(discriminant)
    theta1 = (-b + sqrt_disc) / (2 * a)
    theta2 = (-b - sqrt_disc) / (2 * a)

    # Pick the solution between the two means
    candidates = []
    for theta in [theta1, theta2]:
        if mu1 <= theta <= mu2:
            candidates.append(theta)

    if candidates:
        # If multiple, pick the one closest to midpoint
        midpoint = (mu1 + mu2) / 2
        return min(candidates, key=lambda t: abs(t - midpoint))

    # No solution between means — return midpoint
    return (mu1 + mu2) / 2


class VysotskaDaptiveThreshold:
    """
    Vysotska et al. adaptive thresholding for sequence-based place recognition.

    For each query image, extracts a patch from the similarity matrix,
    tests for bimodality (KS test), fits GMM if bimodal, and computes
    adaptive threshold smoothed by Kalman filter.
    """

    def __init__(self, patch_size=20, ks_significance=0.05,
                 kf_process_noise=1e-3, kf_measurement_noise=1e-2):
        self.patch_size = patch_size
        self.ks_significance = ks_significance
        self.kf = KalmanFilter1D(kf_process_noise, kf_measurement_noise)

    def _extract_patch(self, S, query_idx, ref_idx):
        """
        Extract a p×p patch from similarity matrix S with bottom-right
        corner at (query_idx, ref_idx).
        """
        p = self.patch_size
        q_start = max(0, query_idx - p + 1)
        r_start = max(0, ref_idx - p + 1)
        q_end = query_idx + 1
        r_end = ref_idx + 1

        patch = S[q_start:q_end, r_start:r_end]
        return patch.flatten()

    def _ks_test_bimodal(self, values):
        """
        KS test: is the distribution unimodal (Gaussian)?
        H0: values come from a single Gaussian.
        If we reject H0 (p < significance), the distribution is likely bimodal.

        Returns True if bimodal (reject H0), False if unimodal (accept H0).
        """
        if len(values) < 5:
            return False

        # Fit a single Gaussian to the values
        mu = np.mean(values)
        sigma = np.std(values)
        if sigma < 1e-8:
            return False

        # KS test against fitted Gaussian
        stat, p_value = kstest(values, 'norm', args=(mu, sigma))
        return p_value < self.ks_significance

    def _fit_gmm_threshold(self, values):
        """
        Fit a 2-component GMM and find the decision boundary.
        """
        values = values.reshape(-1, 1)

        gmm = GaussianMixture(n_components=2, random_state=42,
                               max_iter=100, n_init=3)
        gmm.fit(values)

        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())
        weights = gmm.weights_.flatten()

        theta = gmm_decision_boundary(means, stds, weights)
        return theta, means, stds, weights

    def compute_thresholds(self, S, best_matches=None):
        """
        Compute adaptive thresholds for all query images.

        Args:
            S: (n_query, n_ref) similarity matrix
            best_matches: optional list of best reference match per query.
                         If None, uses argmax of each row.

        Returns:
            thresholds: array of per-query thresholds
            history: list of dicts with debug info
        """
        n_query, n_ref = S.shape

        if best_matches is None:
            best_matches = np.argmax(S, axis=1)

        thresholds = np.zeros(n_query)
        history = []

        for q_idx in range(n_query):
            ref_idx = best_matches[q_idx]

            # Extract patch
            patch_values = self._extract_patch(S, q_idx, ref_idx)

            record = {
                "query": q_idx,
                "ref_match": ref_idx,
                "patch_size": len(patch_values),
            }

            # KS test
            is_bimodal = self._ks_test_bimodal(patch_values)
            record["is_bimodal"] = is_bimodal

            if is_bimodal and len(patch_values) >= 10:
                # Fit GMM and get threshold
                try:
                    theta_raw, means, stds, weights = self._fit_gmm_threshold(
                        patch_values)
                    record["gmm_means"] = means.tolist()
                    record["gmm_stds"] = stds.tolist()
                    record["gmm_weights"] = weights.tolist()
                    record["theta_raw"] = theta_raw
                except Exception:
                    # GMM failed — use patch mean as fallback
                    theta_raw = np.mean(patch_values)
                    record["theta_raw"] = theta_raw
                    record["gmm_failed"] = True

                # Kalman filter smoothing
                theta_smooth = self.kf.update(theta_raw)
            else:
                # No path detected — don't update threshold
                theta_smooth = self.kf.x if self.kf.x is not None else np.mean(patch_values)
                record["skipped"] = True

            thresholds[q_idx] = theta_smooth
            record["theta"] = theta_smooth
            history.append(record)

        return thresholds, history
