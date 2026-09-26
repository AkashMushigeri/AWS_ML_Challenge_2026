"""
Unit tests for Matcher module (Member 3).
"""

import pytest
import numpy as np
import pandas as pd
from src.features import EntityRecord, FeatureExtractor, build_entity_record_cache
from src.matcher import (
    WeightedScoreMatcher,
    LogisticRegressionMatcher,
    aggregate_predictions,
    tune_threshold,
    format_matching_results_tsv,
)


def test_weighted_score_matcher_logic():
    matcher = WeightedScoreMatcher()

    # Exact match record pair
    r1 = EntityRecord("S1-1", "Google LLC", "1600 Amphitheatre Pkwy Mountain View CA 94043", "US")
    r2 = EntityRecord("S2-1", "Google LLC", "1600 Amphitheatre Pkwy Mountain View CA 94043", "US")
    r3 = EntityRecord("S2-2", "Completely Different Store", "500 Elm Street Dallas TX 75001", "US")
    r4 = EntityRecord("S2-3", "Google LLC", "1600 Amphitheatre Pkwy Mountain View CA 94043", "France")

    s1_cache = {"S1-1": r1}
    cand_cache = {"S2-1": r2, "S2-2": r3, "S2-3": r4}

    extractor = FeatureExtractor()
    pairs = [("S1-1", "S2-1"), ("S1-1", "S2-2"), ("S1-1", "S2-3")]
    X = extractor.extract_features(pairs, s1_cache, cand_cache)

    scores = matcher.predict_proba(X)
    assert len(scores) == 3
    # High score for exact match
    assert scores[0] > 0.90
    # Low score for completely different
    assert scores[1] < 0.20
    # Heavily penalized for country mismatch
    assert scores[2] < scores[0] * 0.5


def test_logistic_regression_matcher():
    # Synthetic dataset with large unscaled length difference
    X_train = np.array([
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 100.0, 20.0, 0.0, 0.0, 0.0, 0.0, 80.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.9, 0.8, 0.9, 0.8, 2.0, 1.0, 0.8, 0.7, 1.0, 0.8, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.1, 0.1, 0.1, 0.1, 95.0, 18.0, 0.0, 0.0, 0.0, 0.0, 90.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)
    y_train = np.array([1, 0, 1, 0], dtype=np.int32)

    clf_matcher = LogisticRegressionMatcher(random_state=42)
    clf_matcher.fit(X_train, y_train)

    probs = clf_matcher.predict_proba(X_train)
    assert probs[0] > probs[1]
    assert probs[2] > probs[3]


def test_logistic_regression_single_class_edge_case():
    X_train = np.ones((5, 17), dtype=np.float32)
    y_train = np.zeros(5, dtype=np.int32)  # All 0s

    clf = LogisticRegressionMatcher(random_state=42)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_train)
    assert len(probs) == 5
    assert np.all(probs == 0.0)


def test_aggregate_predictions_and_singletons():
    pairs = [
        ("S1-1", "S2-1"),
        ("S1-1", "S3-1"),
        ("S1-2", "S2-2"),  # Below threshold -> singleton
        ("S1-3", "S1-3"),  # Self-match (should be filtered out)
    ]
    scores = np.array([0.95, 0.85, 0.30, 0.99], dtype=np.float32)
    all_s1 = ["S1-1", "S1-2", "S1-3", "S1-4"]

    preds = aggregate_predictions(pairs, scores, threshold=0.70, all_s1_ids=all_s1)

    # S1-1 has two valid matches ordered by score descending
    assert preds["S1-1"] == ["S2-1", "S3-1"]
    # S1-2 had score 0.30 < 0.70 -> singleton
    assert preds["S1-2"] == []
    # S1-3 had self-match -> filtered -> singleton
    assert preds["S1-3"] == []
    # S1-4 had no candidate pairs -> singleton
    assert preds["S1-4"] == []


def test_threshold_tuning():
    val_s1 = ["S1-1", "S1-2"]
    pairs = [
        ("S1-1", "S2-1"),
        ("S1-1", "S2-2"),
        ("S1-2", "S3-1"),
    ]
    gt = {
        "S1-1": {"S2-1"},
        "S1-2": set(),
    }
    X_val = np.zeros((3, 17), dtype=np.float32)
    X_val[0, 0] = 1.0
    X_val[0, 1] = 1.0
    X_val[0, 6] = 1.0
    X_val[0, 11] = 1.0

    X_val[1, 1] = 0.5
    X_val[1, 11] = 1.0

    X_val[2, 1] = 0.3
    X_val[2, 11] = 1.0

    matcher = WeightedScoreMatcher()
    best_thresh, df_report = tune_threshold(
        matcher=matcher,
        X_val=X_val,
        candidate_pairs_val=pairs,
        ground_truth_val=gt,
        val_s1_ids=val_s1,
        candidate_thresholds=[0.50, 0.70, 0.85],
    )
    assert best_thresh >= 0.50
    assert "f05" in df_report.columns


def test_format_matching_results_tsv():
    preds = {
        "S1-1": ["S2-10", "S3-20"],
        "S1-2": [],
    }
    all_s1 = ["S1-1", "S1-2"]
    df = format_matching_results_tsv(preds, all_s1)
    assert list(df.columns) == ["source1_entity_id", "matched_entity_ids"]
    assert len(df) == 2
    assert df.loc[df["source1_entity_id"] == "S1-1", "matched_entity_ids"].values[0] == "S2-10,S3-20"
    assert df.loc[df["source1_entity_id"] == "S1-2", "matched_entity_ids"].values[0] == ""
