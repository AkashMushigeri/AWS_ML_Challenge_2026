"""
Unit tests for Feature Extraction module (Member 3).
"""

import pytest
import numpy as np
import pandas as pd
from src.features import (
    EntityRecord,
    FeatureExtractor,
    build_entity_record_cache,
    compute_pair_features,
    FEATURE_NAMES,
)


def test_feature_vector_shape_and_names():
    rec1 = EntityRecord("S1-1", "Google LLC", "1600 Amphitheatre Pkwy Mountain View CA", "US")
    rec2 = EntityRecord("S2-1", "Google Inc", "1600 Amphitheatre Parkway Mountain View", "US")
    feats = compute_pair_features(rec1, rec2)
    assert len(feats) == len(FEATURE_NAMES)
    assert len(feats) == 17


def test_exact_business_name_match():
    rec1 = EntityRecord("S1-1", "Amazon Web Services", "410 Terry Ave N Seattle WA", "US")
    rec2 = EntityRecord("S2-1", "Amazon Web Services", "410 Terry Ave N Seattle WA", "US")
    feats = compute_pair_features(rec1, rec2)
    feat_dict = dict(zip(FEATURE_NAMES, feats))
    assert feat_dict["name_exact_match"] == 1.0
    assert feat_dict["name_token_jaccard"] == 1.0
    assert feat_dict["name_length_diff"] == 0.0


def test_capitalization_and_punctuation_variation():
    rec1 = EntityRecord("S1-1", "AT&T INC.", "208 S. Akard St., Dallas, TX", "US")
    rec2 = EntityRecord("S2-1", "at&t inc", "208 s akard st dallas tx", "US")
    feats = compute_pair_features(rec1, rec2)
    feat_dict = dict(zip(FEATURE_NAMES, feats))
    # Tokens match regardless of case/punctuation
    assert feat_dict["name_token_jaccard"] == 1.0
    assert feat_dict["addr_token_jaccard"] == 1.0


def test_unicode_and_diacritics_french_records():
    rec1 = EntityRecord("S1-FR1", "Société Générale", "29 boulevard Haussmann Paris", "France")
    rec2 = EntityRecord("S2-FR1", "societe generale", "29 blvd haussmann paris", "France")
    feats = compute_pair_features(rec1, rec2)
    feat_dict = dict(zip(FEATURE_NAMES, feats))
    assert feat_dict["name_exact_match"] == 1.0  # NFKD strips combining accents to match
    assert feat_dict["country_exact_match"] == 1.0
    assert feat_dict["country_missing"] == 0.0
    assert feat_dict["addr_numeric_overlap"] == 1.0  # both have '29'


def test_short_string_padded_ngrams():
    rec1 = EntityRecord("S1-1", "IBM", "Armonk NY", "US")
    rec2 = EntityRecord("S2-1", "IBM Corp", "Armonk NY", "US")
    feats = compute_pair_features(rec1, rec2)
    feat_dict = dict(zip(FEATURE_NAMES, feats))
    assert feat_dict["name_char_jaccard"] > 0.25  # Combined 2/3-grams find common root
    assert feat_dict["name_token_overlap"] == 1.0  # IBM is full subset of IBM Corp


def test_numeric_address_matching():
    rec1 = EntityRecord("S1-IN1", "Tata Consultancy Services", "Bandra Kurla Complex Mumbai 400051", "India")
    rec2 = EntityRecord("S2-IN1", "TCS Ltd", "BKC Mumbai 400051", "India")
    rec3 = EntityRecord("S3-IN2", "TCS Ltd", "Electronic City Bangalore 560100", "India")
    
    feats_match = compute_pair_features(rec1, rec2)
    dict_match = dict(zip(FEATURE_NAMES, feats_match))
    assert dict_match["addr_numeric_overlap"] == 1.0  # both share 400051
    
    feats_mismatch = compute_pair_features(rec1, rec3)
    dict_mismatch = dict(zip(FEATURE_NAMES, feats_mismatch))
    assert dict_mismatch["addr_numeric_overlap"] == 0.0  # 400051 != 560100


def test_country_matching_and_missing_country():
    # Matching countries
    r1 = EntityRecord("S1-1", "Acme", "Road 1", "France")
    r2 = EntityRecord("S2-1", "Acme", "Road 1", "France")
    f = dict(zip(FEATURE_NAMES, compute_pair_features(r1, r2)))
    assert f["country_exact_match"] == 1.0
    assert f["country_missing"] == 0.0

    # Mismatching countries
    r3 = EntityRecord("S2-2", "Acme", "Road 1", "US")
    f_diff = dict(zip(FEATURE_NAMES, compute_pair_features(r1, r3)))
    assert f_diff["country_exact_match"] == 0.0
    assert f_diff["country_missing"] == 0.0

    # Missing country
    r4 = EntityRecord("S2-3", "Acme", "Road 1", "")
    f_miss = dict(zip(FEATURE_NAMES, compute_pair_features(r1, r4)))
    assert f_miss["country_exact_match"] == 0.0
    assert f_miss["country_missing"] == 1.0


def test_missing_name_and_missing_address():
    r1 = EntityRecord("S1-1", "", "123 Main St", "US")
    r2 = EntityRecord("S2-1", "Acme Corp", "", "US")
    f = dict(zip(FEATURE_NAMES, compute_pair_features(r1, r2)))
    assert f["missing_name"] == 1.0
    assert f["missing_addr"] == 1.0
    assert f["name_exact_match"] == 0.0
    assert f["addr_exact_match"] == 0.0


def test_source_type_indicators():
    r1 = EntityRecord("S1-1", "Acme", "Addr", "US")
    r2 = EntityRecord("S2-100", "Acme", "Addr", "US")
    r3 = EntityRecord("S3-200", "Acme", "Addr", "US")
    
    f2 = dict(zip(FEATURE_NAMES, compute_pair_features(r1, r2)))
    assert f2["is_source_2"] == 1.0
    assert f2["is_source_3"] == 0.0

    f3 = dict(zip(FEATURE_NAMES, compute_pair_features(r1, r3)))
    assert f3["is_source_2"] == 0.0
    assert f3["is_source_3"] == 1.0


def test_feature_extractor_batch_and_empty():
    s1_df = pd.DataFrame([
        {"entity_id": "S1-1", "business_name": "Apple Inc", "business_address": "1 Infinite Loop", "country": "US"},
        {"entity_id": "S1-2", "business_name": "Microsoft", "business_address": "One Microsoft Way", "country": "US"},
    ])
    cand_df = pd.DataFrame([
        {"entity_id": "S2-1", "business_name": "Apple Computer", "business_address": "1 Infinite Loop", "country": "US"},
        {"entity_id": "S3-2", "business_name": "Microsoft Corp", "business_address": "1 Microsoft Way", "country": "US"},
    ])
    s1_cache = build_entity_record_cache(s1_df)
    cand_cache = build_entity_record_cache(cand_df)

    pairs = [("S1-1", "S2-1"), ("S1-2", "S3-2")]
    extractor = FeatureExtractor()
    X = extractor.extract_features(pairs, s1_cache, cand_cache)

    assert X.shape == (2, len(FEATURE_NAMES))
    assert X.dtype == np.float32

    # Empty pairs check
    X_empty = extractor.extract_features([], s1_cache, cand_cache)
    assert X_empty.shape == (0, len(FEATURE_NAMES))
