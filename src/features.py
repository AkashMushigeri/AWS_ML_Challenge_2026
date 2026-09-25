"""Pairwise Similarity Feature Extraction Module.

Member 3 (Features + Matcher) ownership.
Extracts compact, lightweight numeric similarity features for candidate pairs.
"""

from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd

from src.normalizer import (
    extract_numeric_tokens,
    normalize_address,
    normalize_business_name,
    normalize_country,
)
from src.utils import get_logger

logger = get_logger("Features")

FEATURE_NAMES = [
    "feat_name_exact",
    "feat_name_token_jaccard",
    "feat_name_token_overlap",
    "feat_name_char_sim",
    "feat_name_len_diff",
    "feat_name_token_diff",
    "feat_addr_exact",
    "feat_addr_token_jaccard",
    "feat_addr_num_overlap",
    "feat_addr_char_sim",
    "feat_addr_len_diff",
    "feat_country_match",
    "feat_country_missing",
    "feat_name_missing",
    "feat_addr_missing",
    "feat_source_is_s3",
]


def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Return set of character n-grams."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard_similarity(set_a: Set, set_b: Set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def overlap_coefficient(set_a: Set, set_b: Set) -> float:
    """Compute Overlap coefficient: |A n B| / min(|A|, |B|)."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    min_len = min(len(set_a), len(set_b))
    return intersection / min_len if min_len > 0 else 0.0


class ProcessedRecord:
    """Cached preprocessed representation of a record for fast feature extraction."""

    __slots__ = (
        "entity_id",
        "norm_name",
        "name_tokens",
        "name_token_set",
        "name_char_ngrams",
        "norm_addr",
        "addr_tokens",
        "addr_token_set",
        "addr_num_set",
        "addr_char_ngrams",
        "norm_ctry",
        "is_s3",
    )

    def __init__(self, entity_id: str, name: str, addr: str, ctry: str):
        self.entity_id = entity_id
        self.norm_name = normalize_business_name(name)
        self.name_tokens = self.norm_name.split()
        self.name_token_set = set(self.name_tokens)
        self.name_char_ngrams = get_char_ngrams(self.norm_name, 3)

        self.norm_addr = normalize_address(addr)
        self.addr_tokens = self.norm_addr.split()
        self.addr_token_set = set(self.addr_tokens)
        self.addr_num_set = set(extract_numeric_tokens(self.norm_addr))
        self.addr_char_ngrams = get_char_ngrams(self.norm_addr, 3)

        self.norm_ctry = normalize_country(ctry)
        self.is_s3 = 1.0 if entity_id.startswith("S3-") else 0.0


def compute_pair_features(r1: ProcessedRecord, r2: ProcessedRecord) -> List[float]:
    """Compute the 16 numeric features for a single candidate pair."""
    # Name features
    feat_name_exact = 1.0 if r1.norm_name and r1.norm_name == r2.norm_name else 0.0
    feat_name_token_jaccard = jaccard_similarity(r1.name_token_set, r2.name_token_set)
    feat_name_token_overlap = overlap_coefficient(r1.name_token_set, r2.name_token_set)
    feat_name_char_sim = jaccard_similarity(r1.name_char_ngrams, r2.name_char_ngrams)

    max_name_len = max(len(r1.norm_name), len(r2.norm_name), 1)
    feat_name_len_diff = abs(len(r1.norm_name) - len(r2.norm_name)) / max_name_len
    feat_name_token_diff = float(abs(len(r1.name_tokens) - len(r2.name_tokens)))

    # Address features
    feat_addr_exact = 1.0 if r1.norm_addr and r1.norm_addr == r2.norm_addr else 0.0
    feat_addr_token_jaccard = jaccard_similarity(r1.addr_token_set, r2.addr_token_set)
    feat_addr_num_overlap = jaccard_similarity(r1.addr_num_set, r2.addr_num_set)
    feat_addr_char_sim = jaccard_similarity(r1.addr_char_ngrams, r2.addr_char_ngrams)

    max_addr_len = max(len(r1.norm_addr), len(r2.norm_addr), 1)
    feat_addr_len_diff = abs(len(r1.norm_addr) - len(r2.norm_addr)) / max_addr_len

    # Country features
    feat_country_match = 1.0 if r1.norm_ctry != "UNKNOWN" and r1.norm_ctry == r2.norm_ctry else 0.0
    feat_country_missing = 1.0 if r1.norm_ctry == "UNKNOWN" or r2.norm_ctry == "UNKNOWN" else 0.0

    # Meta features
    feat_name_missing = 1.0 if not r1.norm_name or not r2.norm_name else 0.0
    feat_addr_missing = 1.0 if not r1.norm_addr or not r2.norm_addr else 0.0
    feat_source_is_s3 = r2.is_s3

    return [
        feat_name_exact,
        feat_name_token_jaccard,
        feat_name_token_overlap,
        feat_name_char_sim,
        feat_name_len_diff,
        feat_name_token_diff,
        feat_addr_exact,
        feat_addr_token_jaccard,
        feat_addr_num_overlap,
        feat_addr_char_sim,
        feat_addr_len_diff,
        feat_country_match,
        feat_country_missing,
        feat_name_missing,
        feat_addr_missing,
        feat_source_is_s3,
    ]


def extract_features(
    candidate_pairs_df: pd.DataFrame,
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
) -> np.ndarray:
    """Extract similarity feature matrix for all candidate pairs in candidate_pairs_df.

    Returns:
        np.ndarray of shape (N_pairs, 16) with float32 values.
    """
    total_pairs = len(candidate_pairs_df)
    logger.info(f"Extracting features for {total_pairs:,} candidate pairs...")
    if total_pairs == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    # Pre-index all records by entity_id
    records: Dict[str, ProcessedRecord] = {}

    for _, row in df_s1.iterrows():
        eid = row["entity_id"]
        records[eid] = ProcessedRecord(eid, row["business_name"], row["business_address"], row["country"])

    for df in (df_s2, df_s3):
        for _, row in df.iterrows():
            eid = row["entity_id"]
            if eid not in records:
                records[eid] = ProcessedRecord(eid, row["business_name"], row["business_address"], row["country"])

    feature_rows: List[List[float]] = []
    default_rec = ProcessedRecord("MISSING", "", "", "UNKNOWN")

    for _, row in candidate_pairs_df.iterrows():
        s1_id = row["source1_entity_id"]
        cand_id = row["candidate_entity_id"]

        r1 = records.get(s1_id, default_rec)
        r2 = records.get(cand_id, default_rec)

        feature_rows.append(compute_pair_features(r1, r2))

    X = np.array(feature_rows, dtype=np.float32)
    logger.info(f"Feature matrix created with shape: {X.shape}")
    return X
