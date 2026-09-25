"""
Unit tests for Metrics module (Member 3).
"""

import pytest
from src.metrics import compute_entity_f05, evaluate_predictions


def test_singleton_perfect_credit():
    # True = empty, Pred = empty -> F0.5 = 1.0
    p, r, f05 = compute_entity_f05(true_matches=set(), pred_matches=set())
    assert p == 1.0
    assert r == 1.0
    assert f05 == 1.0


def test_singleton_false_positive_penalty():
    # True = empty, Pred = {'S2-1'} -> F0.5 = 0.0
    p, r, f05 = compute_entity_f05(true_matches=set(), pred_matches={"S2-1"})
    assert p == 0.0
    assert r == 1.0
    assert f05 == 0.0


def test_non_singleton_missed_match():
    # True = {'S2-1'}, Pred = empty -> F0.5 = 0.0
    p, r, f05 = compute_entity_f05(true_matches={"S2-1"}, pred_matches=set())
    assert p == 1.0
    assert r == 0.0
    assert f05 == 0.0


def test_exact_match_f05():
    # True = {'S2-1', 'S3-2'}, Pred = {'S2-1', 'S3-2'} -> F0.5 = 1.0
    p, r, f05 = compute_entity_f05(true_matches={"S2-1", "S3-2"}, pred_matches={"S2-1", "S3-2"})
    assert p == 1.0
    assert r == 1.0
    assert f05 == 1.0


def test_partial_match_f05_formula():
    # True = {'S2-1'}, Pred = {'S2-1', 'S2-2'}
    # Precision = 1/2 = 0.5, Recall = 1/1 = 1.0
    # F0.5 = 1.25 * 0.5 * 1.0 / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 5/9 ≈ 0.5555...
    p, r, f05 = compute_entity_f05(true_matches={"S2-1"}, pred_matches={"S2-1", "S2-2"})
    assert p == 0.5
    assert r == 1.0
    assert abs(f05 - (5.0 / 9.0)) < 1e-5


def test_macro_evaluation():
    gt = {
        "S1-1": {"S2-1"},
        "S1-2": set(),  # Singleton
    }
    preds = {
        "S1-1": {"S2-1"},
        "S1-2": set(),  # Correct singleton
    }
    res = evaluate_predictions(gt, preds)
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f05"] == 1.0
    assert res["total_entities"] == 2
    assert res["singleton_count"] == 1
