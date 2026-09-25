"""
Pairwise baseline matcher, threshold optimization, and prediction postprocessing.

Member 3 Component: Features + Baseline Matcher.
Provides WeightedScoreMatcher and LogisticRegressionMatcher.
"""

from typing import Dict, List, Set, Tuple, Union, Optional, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_NAMES
from src.metrics import evaluate_predictions


class BaseMatcher:
    """Abstract base matcher for pairwise candidate scoring."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return match probability array of shape (N,) with values in [0, 1]."""
        raise NotImplementedError


class WeightedScoreMatcher(BaseMatcher):
    """
    Transparent, deterministic rule-based linear weighted baseline matcher.
    Combines name, address, numeric, and country similarity features into a score in [0, 1].
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        # Default domain-informed weights for ER
        self.weights = weights or {
            "name_exact_match": 0.35,
            "name_token_jaccard": 0.25,
            "name_token_overlap": 0.10,
            "name_char_jaccard": 0.10,
            "addr_exact_match": 0.25,
            "addr_token_jaccard": 0.15,
            "addr_numeric_overlap": 0.15,
            "addr_char_jaccard": 0.05,
            "country_exact_match": 0.10,
        }
        self.feature_names = FEATURE_NAMES
        self._feat_to_idx = {name: i for i, name in enumerate(self.feature_names)}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute weighted similarity score for each candidate pair."""
        if X is None or X.shape[0] == 0:
            return np.empty((0,), dtype=np.float32)

        scores = np.zeros(X.shape[0], dtype=np.float32)
        total_weight = sum(self.weights.values())

        for feat_name, weight in self.weights.items():
            if feat_name in self._feat_to_idx:
                col_idx = self._feat_to_idx[feat_name]
                scores += (X[:, col_idx] * weight).astype(np.float32)

        # Normalize to [0, 1]
        if total_weight > 0:
            scores /= total_weight

        # Heavy penalty if countries are present and explicitly mismatch
        if "country_exact_match" in self._feat_to_idx and "country_missing" in self._feat_to_idx:
            c_exact_idx = self._feat_to_idx["country_exact_match"]
            c_miss_idx = self._feat_to_idx["country_missing"]
            country_mismatch = (X[:, c_exact_idx] == 0.0) & (X[:, c_miss_idx] == 0.0)
            scores[country_mismatch] *= 0.2  # 80% penalty for mismatched country

        return np.clip(scores, 0.0, 1.0)


class LogisticRegressionMatcher(BaseMatcher):
    """
    Supervised Logistic Regression pairwise matcher with feature standardization.
    Trained on positive ground truth matches and negative candidate non-matches.
    """

    def __init__(
        self,
        C: float = 1.0,
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
        max_iter: int = 1000,
    ):
        self.clf = LogisticRegression(
            C=C,
            class_weight=class_weight,
            random_state=random_state,
            max_iter=max_iter,
            solver="lbfgs",
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self._single_class_fallback: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionMatcher":
        """Fit model on feature matrix X and binary labels y (1=match, 0=non-match)."""
        if len(y) == 0:
            raise ValueError("Cannot fit LogisticRegressionMatcher on empty data.")
        
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            # Handle edge case where training set has only one class
            self._single_class_fallback = 1.0 if unique_classes[0] == 1 else 0.0
            self.is_fitted = True
            return self

        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        self._single_class_fallback = None
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of positive match (class 1)."""
        if not self.is_fitted:
            raise ValueError("Matcher is not fitted yet. Call fit() first.")
        if X is None or X.shape[0] == 0:
            return np.empty((0,), dtype=np.float32)

        if self._single_class_fallback is not None:
            return np.full(X.shape[0], self._single_class_fallback, dtype=np.float32)

        X_scaled = self.scaler.transform(X)
        probs = self.clf.predict_proba(X_scaled)
        # Class 1 probability
        return probs[:, 1].astype(np.float32)


def aggregate_predictions(
    candidate_pairs: Union[List[Tuple[str, str]], pd.DataFrame],
    scores: Union[np.ndarray, List[float]],
    threshold: float,
    all_s1_ids: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """
    Aggregate scored candidate pairs into predicted match sets per Source 1 entity.
    
    Args:
        candidate_pairs: List of (source1_entity_id, candidate_entity_id) or DataFrame
        scores: 1D array of match scores / probabilities matching candidate_pairs
        threshold: Decision threshold for positive link
        all_s1_ids: Complete list of all S1 IDs to guarantee presence (even singletons)
        
    Returns:
        Dict mapping source1_entity_id -> List of matched candidate IDs
    """
    if isinstance(candidate_pairs, pd.DataFrame):
        s1_col = candidate_pairs["source1_entity_id"].astype(str).values
        cand_col = candidate_pairs["candidate_entity_id"].astype(str).values
    else:
        s1_col = [str(pair[0]) for pair in candidate_pairs]
        cand_col = [str(pair[1]) for pair in candidate_pairs]

    scores_arr = np.asarray(scores, dtype=np.float32)

    predictions: Dict[str, List[Tuple[str, float]]] = {}
    if all_s1_ids is not None:
        for s1_id in all_s1_ids:
            predictions[str(s1_id)] = []

    for idx, (s1_id, cand_id) in enumerate(zip(s1_col, cand_col)):
        score = float(scores_arr[idx])
        if score >= threshold:
            # Avoid self-matches if any S1 ID accidentally appears as candidate
            if not cand_id.startswith("S1-") and cand_id != s1_id:
                if s1_id not in predictions:
                    predictions[s1_id] = []
                predictions[s1_id].append((cand_id, score))

    # Sort each entity's predictions by score descending and deduplicate IDs
    final_predictions: Dict[str, List[str]] = {}
    for s1_id, candidate_list in predictions.items():
        # Sort by score desc
        sorted_cands = sorted(candidate_list, key=lambda x: x[1], reverse=True)
        # Deduplicate preserving order
        seen = set()
        deduped = []
        for cid, _ in sorted_cands:
            if cid not in seen:
                seen.add(cid)
                deduped.append(cid)
        final_predictions[s1_id] = deduped

    return final_predictions


def tune_threshold(
    matcher: BaseMatcher,
    X_val: np.ndarray,
    candidate_pairs_val: Union[List[Tuple[str, str]], pd.DataFrame],
    ground_truth_val: Dict[str, Set[str]],
    val_s1_ids: List[str],
    candidate_thresholds: Optional[List[float]] = None,
) -> Tuple[float, pd.DataFrame]:
    """
    Evaluate validation performance across multiple candidate thresholds.
    Selects the threshold maximizing macro F0.5.
    
    Args:
        matcher: Fitted BaseMatcher or WeightedScoreMatcher
        X_val: Validation pairwise feature matrix
        candidate_pairs_val: Candidate pairs on validation set
        ground_truth_val: Ground truth match sets for validation S1 entities
        val_s1_ids: List of Source 1 entity IDs in the validation split
        candidate_thresholds: List of thresholds to evaluate
        
    Returns:
        (best_threshold, evaluation_report_df)
    """
    if candidate_thresholds is None:
        candidate_thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    scores = matcher.predict_proba(X_val)
    
    records = []
    best_threshold = candidate_thresholds[0]
    best_f05 = -1.0

    for thresh in candidate_thresholds:
        preds = aggregate_predictions(
            candidate_pairs=candidate_pairs_val,
            scores=scores,
            threshold=thresh,
            all_s1_ids=val_s1_ids,
        )
        metrics = evaluate_predictions(
            ground_truth=ground_truth_val,
            predictions=preds,
            s1_ids=val_s1_ids,
        )
        rec = {
            "threshold": thresh,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f05": metrics["f05"],
            "total_entities": metrics["total_entities"],
            "singleton_count": metrics["singleton_count"],
        }
        records.append(rec)

        if metrics["f05"] > best_f05:
            best_f05 = metrics["f05"]
            best_threshold = thresh

    df_report = pd.DataFrame(records)
    return best_threshold, df_report


def format_matching_results_tsv(
    predictions: Dict[str, List[str]],
    all_s1_ids: List[str],
) -> pd.DataFrame:
    """
    Format predictions into official matching_results.tsv DataFrame schema.
    Columns: ["source1_entity_id", "matched_entity_ids"]
    """
    rows = []
    for s1_id in all_s1_ids:
        cands = predictions.get(str(s1_id), [])
        matched_str = ",".join(cands) if cands else ""
        rows.append({
            "source1_entity_id": str(s1_id),
            "matched_entity_ids": matched_str,
        })
    return pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])
