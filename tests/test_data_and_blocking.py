"""Dedicated Unit & Integration Tests for Member 2: Data + Blocking.

Covers:
1. Exact business name matching
2. Punctuation variations & conjunctions (&, +)
3. Capitalization variations
4. Unicode and diacritics (French accents, ligatures œ, æ)
5. Address abbreviation expansion (street, road, ave, blvd, etc.)
6. Numeric address and postal/PIN code extraction
7. Open-set country normalization (never hard-coded)
8. France record handling
9. Missing address handling
10. Missing country handling
11. Candidate deduplication
12. Generic business names and legal suffixes (LLC at start vs end)
13. Unmatched entities / singletons
14. Non-Latin scripts (Devanagari / Hindi preservation)
15. TSV loader schema validation and streaming
16. Candidate pairs TSV exporter adherence to submission contract
"""

import io
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

from src.blocking import (
    BlockingIndex,
    export_candidate_pairs_tsv,
    extract_blocking_keys,
    generate_candidate_pairs,
)
from src.data_loader import (
    load_ground_truth,
    load_source_tsv,
    validate_source_columns,
)
from src.normalizer import (
    extract_core_name,
    extract_core_name_tokens,
    extract_numeric_tokens,
    extract_postal_code,
    normalize_address,
    normalize_business_name,
    normalize_country,
    normalize_dataframe,
    remove_diacritics,
)


class TestMember2DataAndBlocking(unittest.TestCase):
    """Exhaustive test suite for Member 2 (Data + Normalization + Blocking)."""

    def test_01_exact_business_name_match(self):
        n1 = normalize_business_name("Amazon Web Services Inc")
        n2 = normalize_business_name("Amazon Web Services Inc")
        self.assertEqual(n1, n2)
        self.assertEqual(n1, "amazon web services")

    def test_02_punctuation_and_conjunction_variation(self):
        n1 = normalize_business_name("B & M Retail, Corp.")
        n2 = normalize_business_name("B and M Retail Corp")
        n3 = normalize_business_name("B + M Retail Corp")
        self.assertEqual(n1, n2)
        self.assertEqual(n2, n3)

    def test_03_capitalization_variation(self):
        n1 = normalize_business_name("GLOBAL LOGISTICS SOLUTIONS LLC")
        n2 = normalize_business_name("global logistics solutions llc")
        self.assertEqual(n1, n2)

    def test_04_unicode_diacritics_and_ligatures(self):
        n1 = normalize_business_name("Café Étoile SARL")
        n2 = normalize_business_name("Cafe Etoile")
        self.assertEqual(n1, n2)
        self.assertEqual(n1, "cafe etoile")

        # Test French ligature
        n_lig = normalize_business_name("Sœur & Frères SARL")
        self.assertEqual(n_lig, "soeur and freres")

    def test_05_address_variations(self):
        a1 = normalize_address("123 Main Rd., Apt 4B, 1st Fl.")
        a2 = normalize_address("123 Main Road, Apartment 4B, 1st Floor")
        self.assertEqual(a1, a2)

    def test_06_numeric_address_and_postal_extraction(self):
        addr_in = "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi 110041"
        nums = extract_numeric_tokens(addr_in)
        pin = extract_postal_code(addr_in)
        self.assertIn("570", nums)
        self.assertIn("13", nums)
        self.assertIn("110041", nums)
        self.assertEqual(pin, "110041")

        # US ZIP
        addr_us = "1795 Westchester Drive, High Point, NC 27262"
        self.assertEqual(extract_postal_code(addr_us), "27262")

        # France Postal
        addr_fr = "175 Boulevard du Président Franklin Roosevelt, Bordeaux 33000"
        self.assertEqual(extract_postal_code(addr_fr), "33000")

    def test_07_country_normalization(self):
        self.assertEqual(normalize_country("US"), "US")
        self.assertEqual(normalize_country("  india  "), "INDIA")
        self.assertEqual(normalize_country("France"), "FRANCE")
        # Unseen country
        self.assertEqual(normalize_country("Brazil"), "BRAZIL")

    def test_08_france_records(self):
        keys = extract_blocking_keys(
            "<< Team Ecole",
            "175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine 33000",
            "France"
        )
        all_keys = [k for k_list in keys.values() for k in k_list]
        self.assertTrue(any("FRANCE" in k for k in all_keys))
        self.assertTrue(any("33000" in k for k in all_keys))

    def test_09_missing_address(self):
        self.assertEqual(normalize_address(""), "")
        self.assertEqual(normalize_address(None), "")
        self.assertEqual(extract_numeric_tokens(""), [])
        self.assertEqual(extract_postal_code(""), "")

    def test_10_missing_country(self):
        self.assertEqual(normalize_country(""), "UNKNOWN")
        self.assertEqual(normalize_country(None), "UNKNOWN")
        keys = extract_blocking_keys("Test Entity", "123 Main St", None)
        all_keys = [k for k_list in keys.values() for k in k_list]
        self.assertTrue(any("UNKNOWN" in k for k in all_keys))

    def test_11_candidate_deduplication(self):
        df_s1 = pd.DataFrame([
            {"entity_id": "S1-1", "business_name": "Acme Corp", "business_address": "100 Broadway, NY", "country": "US"}
        ])
        # Two identical candidate records
        df_s2 = pd.DataFrame([
            {"entity_id": "S2-1", "business_name": "Acme Inc", "business_address": "100 Broadway, New York", "country": "US"},
            {"entity_id": "S2-1", "business_name": "Acme Inc", "business_address": "100 Broadway, New York", "country": "US"}
        ])
        df_s3 = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])

        pairs_df, metrics = generate_candidate_pairs(df_s1, df_s2, df_s3)
        # S2-1 must appear only once for S1-1
        self.assertEqual(len(pairs_df), 1)
        self.assertEqual(pairs_df.iloc[0]["source1_entity_id"], "S1-1")
        self.assertEqual(pairs_df.iloc[0]["candidate_entity_id"], "S2-1")

    def test_12_generic_business_names_and_legal_positions(self):
        # LLC at beginning vs LLC at end
        n_lead = normalize_business_name("LLC Moncada Learning Center")
        n_trail = normalize_business_name("Moncada Learning Center LLC")
        self.assertEqual(n_lead, n_trail)
        self.assertEqual(n_lead, "moncada learning center")

        # Core name tokens exclude generic stopwords
        tokens = extract_core_name_tokens("Pinnacle Technologies Private Limited")
        self.assertNotIn("private", tokens)
        self.assertNotIn("limited", tokens)
        self.assertIn("pinnacle", tokens)

    def test_13_unmatched_entities(self):
        df_s1 = pd.DataFrame([
            {"entity_id": "S1-Isolated", "business_name": "Xylophone Quantum Robotics", "business_address": "888 Zzz Way, Nowhere", "country": "US"}
        ])
        df_s2 = pd.DataFrame([
            {"entity_id": "S2-Unrelated", "business_name": "Alpha Bakery", "business_address": "12 First St, City", "country": "US"}
        ])
        df_s3 = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])

        pairs_df, metrics = generate_candidate_pairs(df_s1, df_s2, df_s3)
        # No candidate should match
        self.assertEqual(len(pairs_df), 0)
        self.assertEqual(metrics["dedup_candidates"], 0.0)

    def test_14_non_latin_indic_preservation(self):
        hindi_name = "राम मार्केटिंग प्राइवेट लिमिटेड"
        norm = normalize_business_name(hindi_name)
        # Ensure Devanagari script is completely preserved and not erased
        self.assertIn("राम", norm)
        self.assertIn("मार्केटिंग", norm)

        # Test blocking key extraction for Hindi
        keys = extract_blocking_keys(hindi_name, "123 Main Road", "India")
        all_keys = [k for k_list in keys.values() for k in k_list]
        self.assertTrue(any("राम" in k for k in all_keys))

    def test_15_tsv_loader_schema_and_streaming(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tsv_path = Path(tmpdir) / "test_source.tsv"
            tsv_path.write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                "S2-001\tBeta Corp\t456 Elm St\tUS\n"
                "S2-002\tGamma Ltd\t789 Oak Ave\tUS\n",
                encoding="utf-8"
            )

            df = load_source_tsv(tsv_path, nrows=1)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]["entity_id"], "S2-001")

            # Missing columns should raise ValueError
            bad_tsv = Path(tmpdir) / "bad.tsv"
            bad_tsv.write_text("id\tname\n1\tA\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_source_tsv(bad_tsv)

    def test_16_export_candidate_pairs_tsv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "candidate_pairs.tsv"
            pairs_df = pd.DataFrame([
                {"source1_entity_id": "S1-A", "candidate_entity_id": "S2-1"},
                {"source1_entity_id": "S1-A", "candidate_entity_id": "S3-2"},
            ])
            all_s1 = ["S1-A", "S1-B"]  # S1-B is a singleton
            export_candidate_pairs_tsv(pairs_df, out_file, all_s1_ids=all_s1)

            lines = out_file.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "source1_entity_id\tcandidate_entity_ids")
            self.assertEqual(lines[1], "S1-A\tS2-1,S3-2")
            self.assertEqual(lines[2], "S1-B\t")


if __name__ == "__main__":
    unittest.main()
