"""
Evaluation metrics for Entity Resolution adhering to Section 11 & Section 4 specifications.

Calculates entity-level Precision, Recall, and Macro F0.5 with official singleton handling.
"""

from typing import Dict, List, Set, Tuple, Union, Optional
import numpy as np
import pandas as pd


def compute_entity_f05(
    true_matches: Set[str],
    pred_matches: Set[str],
    beta: float = 0.5,
) -> Tuple[float, float, float]:
    """
    Compute Precision, Recall, and F_beta (default F0.5) for a single Source 1 entity.
    
    Handles singletons (empty true matches) according to official challenge rules:
    - If True = empty and Pred = empty: Precision = 1.0, Recall = 1.0, F0.5 = 1.0
    - If True = empty and Pred != empty: Precision = 0.0, Recall = 1.0, F0.5 = 0.0
    - If True != empty and Pred = empty: Precision = 1.0, Recall = 0.0, F0.5 = 0.0
    - If True != empty and Pred != empty:
        Precision = |P ∩ T| / |P|
        Recall = |P ∩ T| / |T|
        F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)
    """
    beta_sq = beta ** 2  # 0.25 for beta=0.5
    weight = 1.0 + beta_sq  # 1.25 for beta=0.5

    # Case 1: Ground truth is singleton (no true matches)
    if not true_matches:
        if not pred_matches:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 1.0, 0.0

    # Case 2: Ground truth has matches, but prediction is empty
    if not pred_matches:
        return 1.0, 0.0, 0.0

    # Case 3: Both have matches
    intersection = len(true_matches.intersection(pred_matches))
    prec = float(intersection) / float(len(pred_matches))
    rec = float(intersection) / float(len(true_matches))

    if prec + rec == 0.0 or intersection == 0:
        f_score = 0.0
    else:
        denom = (beta_sq * prec) + rec
        f_score = (weight * prec * rec) / denom if denom > 0 else 0.0

    return prec, rec, f_score


def evaluate_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Union[Set[str], List[str]]],
    s1_ids: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Evaluate all Source 1 entities and return macro-averaged metrics.
    
    Args:
        ground_truth: Dict mapping source1_entity_id to set of true matching candidate IDs
        predictions: Dict mapping source1_entity_id to set/list of predicted candidate IDs
        s1_ids: Optional list of S1 IDs to evaluate (defaults to union of keys)
        
    Returns:
        Dict with keys: "precision", "recall", "f05", "total_entities", "singleton_count"
    """
    if s1_ids is None:
        all_ids = set(ground_truth.keys()).union(set(predictions.keys()))
        eval_ids = sorted(list(all_ids))
    else:
        eval_ids = list(s1_ids)

    n_entities = len(eval_ids)
    if n_entities == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f05": 0.0,
            "total_entities": 0,
            "singleton_count": 0,
        }

    precisions: List[float] = []
    recalls: List[float] = []
    f_scores: List[float] = []
    singleton_count = 0

    for s1_id in eval_ids:
        true_set = ground_truth.get(str(s1_id), set())
        if not true_set:
            singleton_count += 1
            
        pred_val = predictions.get(str(s1_id), set())
        pred_set = set(pred_val) if not isinstance(pred_val, set) else pred_val

        p, r, f05 = compute_entity_f05(true_set, pred_set, beta=0.5)
        precisions.append(p)
        recalls.append(r)
        f_scores.append(f05)

    return {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f05": float(np.mean(f_scores)),
        "total_entities": n_entities,
        "singleton_count": singleton_count,
    }


def parse_ground_truth_df(df_gt: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    Parse a ground truth DataFrame (source1_entity_id, matched_entity_ids)
    into a dictionary of sets.
    """
    gt_dict: Dict[str, Set[str]] = {}
    for row in df_gt.itertuples(index=False):
        s1_id = str(getattr(row, "source1_entity_id", "")).strip()
        matched_val = getattr(row, "matched_entity_ids", "")
        if pd.isna(matched_val) or matched_val is None:
            gt_dict[s1_id] = set()
            continue
            
        matched_str = str(matched_val).strip()
        if matched_str.lower() in {"", "nan", "none", "null", "[]", "set()"}:
            gt_dict[s1_id] = set()
        else:
            # Comma-separated list
            ids = {x.strip() for x in matched_str.split(",") if x.strip()}
            gt_dict[s1_id] = ids
    return gt_dict
