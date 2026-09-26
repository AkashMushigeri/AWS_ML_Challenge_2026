"""Comprehensive Unit & Regression Tests for V1 Pipeline.

Covers all 14 required challenge test cases:
1. exact business-name match
2. punctuation variation
3. capitalization variation
4. Unicode/diacritics
5. address variations
6. numeric address matching
7. country matching
8. France records
9. missing address
10. missing country
11. duplicate candidates
12. generic business names
13. unmatched entities
14. singleton entities scoring
"""

import unittest
import numpy as np
import pandas as pd

from src.blocking import extract_blocking_keys, generate_candidate_pairs
from src.evaluation import compute_entity_metrics, evaluate_predictions
from src.features import extract_features, get_char_ngrams, jaccard_similarity
from src.matcher import WeightedScoreMatcher
from src.normalizer import (
    extract_core_name_tokens,
    extract_numeric_tokens,
    normalize_address,
    normalize_business_name,
    normalize_country,
    remove_diacritics,
)
from src.postprocess import build_candidate_and_matching_outputs, format_id_list


class TestV1Pipeline(unittest.TestCase):
    """Test suite covering the 14 V1 pipeline requirements."""

    # 1. Exact business-name match
    def test_01_exact_business_name_match(self):
        n1 = normalize_business_name("Amazon Web Services Inc")
        n2 = normalize_business_name("Amazon Web Services Inc")
        self.assertEqual(n1, n2)
        self.assertEqual(n1, "amazon web services")

    # 2. Punctuation variation
    def test_02_punctuation_variation(self):
        n1 = normalize_business_name("B & M Retail, Corp.")
        n2 = normalize_business_name("B and M Retail Corp")
        self.assertEqual(n1, n2)

    # 3. Capitalization variation
    def test_03_capitalization_variation(self):
        n1 = normalize_business_name("GLOBAL LOGISTICS SOLUTIONS LLC")
        n2 = normalize_business_name("global logistics solutions llc")
        self.assertEqual(n1, n2)

    # 4. Unicode / diacritics
    def test_04_unicode_diacritics(self):
        n1 = normalize_business_name("Café Étoile SARL")
        n2 = normalize_business_name("Cafe Etoile")
        self.assertEqual(n1, n2)
        self.assertNotIn("é", n1)

    # 5. Address variations (abbreviations)
    def test_05_address_variations(self):
        a1 = normalize_address("123 Main Rd., Apt 4B")
        a2 = normalize_address("123 Main Road, Apartment 4B")
        self.assertEqual(a1, a2)

    # 6. Numeric address matching (PIN codes / street numbers)
    def test_06_numeric_address_matching(self):
        nums1 = extract_numeric_tokens("560001 MG Road Bengaluru")
        nums2 = extract_numeric_tokens("MG Road 560001 Karnataka")
        self.assertIn("560001", nums1)
        self.assertIn("560001", nums2)
        self.assertEqual(set(nums1), set(nums2))

    # 7. Country matching
    def test_07_country_matching(self):
        c1 = normalize_country("US")
        c2 = normalize_country("  us  ")
        self.assertEqual(c1, c2)
        self.assertEqual(c1, "US")

    # 8. France records (open-set country compatibility)
    def test_08_france_records(self):
        c_france = normalize_country("France")
        self.assertEqual(c_france, "FRANCE")
        # Ensure blocking produces keys for France
        keys = extract_blocking_keys("Renault SAS", "15 Avenue des Champs, Paris", "France")
        self.assertTrue(any("FRANCE" in k for k_list in keys.values() for k in k_list))

    # 9. Missing address
    def test_09_missing_address(self):
        norm_empty = normalize_address("")
        self.assertEqual(norm_empty, "")
        nums = extract_numeric_tokens("")
        self.assertEqual(nums, [])

    # 10. Missing country
    def test_10_missing_country(self):
        c_empty = normalize_country("")
        c_none = normalize_country(None)
        self.assertEqual(c_empty, "UNKNOWN")
        self.assertEqual(c_none, "UNKNOWN")

    # 11. Duplicate candidates deduplication
    def test_11_duplicate_candidates_deduplication(self):
        id_set = {"S2-00001", "S2-00001", "S3-00002"}
        formatted = format_id_list(id_set)
        # S2-00001 must appear only once
        self.assertEqual(formatted, "S2-00001,S3-00002")

    # 12. Generic business names & suffix handling
    def test_12_generic_business_names(self):
        core = extract_core_name_tokens("Pinnacle Technologies Private Limited")
        self.assertNotIn("private", core)
        self.assertNotIn("limited", core)
        self.assertIn("pinnacle", core)
        self.assertIn("technologies", core)

    # 13. Unmatched entities (singletons stay empty)
    def test_13_unmatched_entities(self):
        all_s1 = ["S1-001", "S1-002"]
        cand_df = pd.DataFrame([{"source1_entity_id": "S1-001", "candidate_entity_id": "S2-999"}])
        scores = np.array([0.40])  # Below threshold 0.75
        _, matches = build_candidate_and_matching_outputs(all_s1, cand_df, scores, threshold=0.75)
        # S1-001 has no match passing threshold -> must be empty
        self.assertEqual(len(matches["S1-001"]), 0)
        # S1-002 had no candidates -> must be empty
        self.assertEqual(len(matches["S1-002"]), 0)

    # 14. Singleton entities scoring (evaluator logic)
    def test_14_singleton_entities_scoring(self):
        # Case A: True singleton correctly predicted as empty -> F0.5 = 1.0
        p, r, f = compute_entity_metrics(true_matches=set(), pred_matches=set())
        self.assertEqual(f, 1.0)
        self.assertEqual(p, 1.0)
        self.assertEqual(r, 1.0)

        # Case B: True singleton incorrectly predicted with a match -> F0.5 = 0.0
        p_err, r_err, f_err = compute_entity_metrics(true_matches=set(), pred_matches={"S2-123"})
        self.assertEqual(f_err, 0.0)
        self.assertEqual(p_err, 0.0)
        self.assertEqual(r_err, 0.0)

        # Case C: Non-singleton match
        true_m = {"S2-001", "S3-002"}
        pred_m = {"S2-001", "S3-002", "S2-003"}  # 2 true, 1 false
        p_c, r_c, f_c = compute_entity_metrics(true_m, pred_m)
        self.assertAlmostEqual(p_c, 2 / 3, places=3)
        self.assertAlmostEqual(r_c, 1.0, places=3)
        expected_f05 = (1.25 * (2 / 3) * 1.0) / (0.25 * (2 / 3) + 1.0)
        self.assertAlmostEqual(f_c, expected_f05, places=3)


if __name__ == "__main__":
    unittest.main()
