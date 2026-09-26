"""Pairwise Baseline Matcher Module.

Member 3 (Features + Matcher) ownership.
Implements a transparent weighted baseline matcher and lightweight Logistic Regression.
"""

from typing import Dict, List, Optional
import numpy as np

from src.features import FEATURE_NAMES
from src.utils import get_logger

logger = get_logger("Matcher")


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class WeightedScoreMatcher:
    """Transparent, interpretable weighted baseline matcher.

    Combines similarity features using domain-driven weights and penalizes mismatches.
    Outputs calibrated matching scores in [0.0, 1.0].
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or FEATURE_NAMES

        # Domain-tuned interpretable weights prioritizing name, address numbers, and country agreement
        self.weights: Dict[str, float] = {
            "feat_name_exact": 0.40,
            "feat_name_token_jaccard": 0.20,
            "feat_name_token_overlap": 0.10,
            "feat_name_char_sim": 0.15,
            "feat_name_len_diff": -0.10,
            "feat_name_token_diff": -0.02,
            "feat_addr_exact": 0.25,
            "feat_addr_token_jaccard": 0.15,
            "feat_addr_num_overlap": 0.15,
            "feat_addr_char_sim": 0.10,
            "feat_addr_len_diff": -0.05,
            "feat_country_match": 0.10,
            "feat_country_missing": -0.05,
            "feat_name_missing": -0.30,
            "feat_addr_missing": -0.10,
            "feat_source_is_s3": 0.0,
        }

        # Weight vector
        self.w_vec = np.array([self.weights.get(f, 0.0) for f in self.feature_names], dtype=np.float32)
        # Normalization factor
        pos_weights = np.sum(np.maximum(0.0, self.w_vec))
        self.norm_factor = pos_weights if pos_weights > 0 else 1.0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute matching probability scores for feature matrix X."""
        if X.shape[0] == 0:
            return np.empty((0,), dtype=np.float32)

        # Linear combination
        raw_scores = X @ self.w_vec

        # Country mismatch gate: if country_match is 0 and country_missing is 0 (i.e. both have different countries),
        # apply hard penalty
        ctry_match_idx = self.feature_names.index("feat_country_match")
        ctry_miss_idx = self.feature_names.index("feat_country_missing")
        mismatch_mask = (X[:, ctry_match_idx] == 0.0) & (X[:, ctry_miss_idx] == 0.0)
        raw_scores[mismatch_mask] -= 0.50

        # Scale into [0, 1] range
        scores = np.clip(raw_scores / self.norm_factor, 0.0, 1.0)
        return scores


class LogisticRegressionMatcher:
    """Lightweight Logistic Regression matcher implemented in pure NumPy.

    Optimized via L-BFGS or gradient descent with L2 regularization.
    """

    def __init__(self, l2_reg: float = 1.0, max_iter: int = 200, lr: float = 0.05):
        self.l2_reg = l2_reg
        self.max_iter = max_iter
        self.lr = lr
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionMatcher":
        """Fit logistic regression weights on candidate pairs with binary labels."""
        n_samples, n_features = X.shape
        logger.info(f"Fitting Logistic Regression on {n_samples} samples ({np.sum(y == 1)} positive)...")

        # Initialize weights
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0

        if n_samples == 0 or np.all(y == y[0]):
            return self

        # Standard gradient descent
        for it in range(self.max_iter):
            linear_pred = X @ self.weights + self.bias
            preds = sigmoid(linear_pred)

            error = preds - y
            grad_w = (X.T @ error) / n_samples + (self.l2_reg / n_samples) * self.weights
            grad_b = np.mean(error)

            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

        logger.info("Logistic regression training completed.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute matching probabilities."""
        if self.weights is None:
            raise RuntimeError("Model has not been fitted yet.")
        linear_pred = X @ self.weights + self.bias
        return sigmoid(linear_pred)
