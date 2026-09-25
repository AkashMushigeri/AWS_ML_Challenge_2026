"""Official Metric and Threshold Evaluation Module.

Member 1 (Team Lead) ownership.
Implements entity-level Precision, Recall, and macro F0.5 with explicit singleton handling.
"""

from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger("Evaluation")


def compute_entity_metrics(
    true_matches: Set[str],
    pred_matches: Set[str],
) -> Tuple[float, float, float]:
    """Compute (precision, recall, f0.5) for a single Source 1 entity.

    Follows the official challenge scoring rules:
    - If true matches is empty (singleton):
        - If pred matches is empty: Precision=1.0, Recall=1.0, F0.5=1.0 (correct singleton).
        - If pred matches is non-empty: Precision=0.0, Recall=0.0, F0.5=0.0 (false merge penalty).
    - If true matches is non-empty:
        - If pred matches is empty: Precision=0.0, Recall=0.0, F0.5=0.0.
        - If pred matches is non-empty:
            P = |true n pred| / |pred|
            R = |true n pred| / |true|
            F0.5 = (1.25 * P * R) / (0.25 * P + R) if (0.25 * P + R) > 0 else 0.0
    """
    is_true_singleton = len(true_matches) == 0
    is_pred_singleton = len(pred_matches) == 0

    if is_true_singleton:
        if is_pred_singleton:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0

    if is_pred_singleton:
        return 0.0, 0.0, 0.0

    overlap = len(true_matches & pred_matches)
    if overlap == 0:
        return 0.0, 0.0, 0.0

    prec = overlap / len(pred_matches)
    rec = overlap / len(true_matches)

    denom = 0.25 * prec + rec
    f05 = (1.25 * prec * rec) / denom if denom > 0 else 0.0
    return prec, rec, f05


def evaluate_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
) -> Dict[str, float]:
    """Compute macro-averaged Precision, Recall, and F0.5 across all evaluated S1 entities."""
    s1_entities = list(ground_truth.keys())
    if not s1_entities:
        return {"precision": 0.0, "recall": 0.0, "f05": 0.0, "count": 0}

    precisions: List[float] = []
    recalls: List[float] = []
    f05_scores: List[float] = []

    singletons = 0
    correct_singletons = 0

    for s1 in s1_entities:
        t_matches = ground_truth.get(s1, set())
        p_matches = predictions.get(s1, set())

        if len(t_matches) == 0:
            singletons += 1
            if len(p_matches) == 0:
                correct_singletons += 1

        p, r, f = compute_entity_metrics(t_matches, p_matches)
        precisions.append(p)
        recalls.append(r)
        f05_scores.append(f)

    macro_p = float(np.mean(precisions))
    macro_r = float(np.mean(recalls))
    macro_f05 = float(np.mean(f05_scores))

    return {
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f05": macro_f05,
        "total_evaluated_s1": float(len(s1_entities)),
        "true_singletons": float(singletons),
        "correct_singletons": float(correct_singletons),
    }


def find_optimal_threshold(
    candidate_pairs_df: pd.DataFrame,
    scores: np.ndarray,
    ground_truth: Dict[str, Set[str]],
    threshold_grid: List[float],
) -> Tuple[float, pd.DataFrame]:
    """Evaluate performance across a grid of thresholds and select the best F0.5 threshold.

    Returns:
        optimal_threshold: float
        summary_df: DataFrame containing metrics for each evaluated threshold
    """
    logger.info(f"Evaluating candidate thresholds on validation set: {threshold_grid}")
    s1_ids = list(ground_truth.keys())
    results: List[Dict[str, float]] = []

    # Map candidate pairs to scores
    pair_scores: Dict[Tuple[str, str], float] = {}
    for i, (_, row) in enumerate(candidate_pairs_df.iterrows()):
        s1 = row["source1_entity_id"]
        c = row["candidate_entity_id"]
        pair_scores[(s1, c)] = float(scores[i])

    best_thresh = threshold_grid[0]
    best_f05 = -1.0

    for thresh in threshold_grid:
        preds: Dict[str, Set[str]] = {s1: set() for s1 in s1_ids}
        for (s1, c), sc in pair_scores.items():
            if s1 in preds and sc >= thresh:
                preds[s1].add(c)

        metrics = evaluate_predictions(ground_truth, preds)
        metrics["threshold"] = thresh
        results.append(metrics)

        logger.info(
            f"Threshold {thresh:.2f} -> "
            f"F0.5: {metrics['macro_f05']:.4f} "
            f"(P: {metrics['macro_precision']:.4f}, R: {metrics['macro_recall']:.4f})"
        )

        if metrics["macro_f05"] > best_f05:
            best_f05 = metrics["macro_f05"]
            best_thresh = thresh

    summary_df = pd.DataFrame(results)[
        [
            "threshold",
            "macro_f05",
            "macro_precision",
            "macro_recall",
            "true_singletons",
            "correct_singletons",
        ]
    ]

    logger.info(f"Optimal validation threshold: {best_thresh:.2f} with Macro F0.5 = {best_f05:.4f}")
    return best_thresh, summary_df
