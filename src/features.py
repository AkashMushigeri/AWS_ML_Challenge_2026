"""
Pairwise similarity feature engineering for Entity Resolution.

Member 3 Component: Features + Baseline Matcher.
Generates lightweight, compact numeric features for candidate pairs.
"""

from typing import Dict, List, Tuple, Union, Optional, Any, Set
import re
import numpy as np
import pandas as pd


FEATURE_NAMES = [
    # Business Name Features
    "name_exact_match",
    "name_token_jaccard",
    "name_token_overlap",
    "name_char_jaccard",
    "name_length_diff",
    "name_token_count_diff",
    # Address Features
    "addr_exact_match",
    "addr_token_jaccard",
    "addr_numeric_overlap",
    "addr_char_jaccard",
    "addr_length_diff",
    # Country Features
    "country_exact_match",
    "country_missing",
    # Meta Features
    "missing_name",
    "missing_addr",
    "is_source_2",
    "is_source_3",
]


def _get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from text."""
    if not text:
        return set()
    if len(text) < n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def _extract_numeric_tokens(text: str) -> Set[str]:
    """Extract numeric/digit tokens (PIN, ZIP, building numbers)."""
    if not text:
        return set()
    return set(re.findall(r"\b\d+\b", text))


def _tokenize(text: str) -> List[str]:
    """Tokenize string into lowercase alphanumeric words."""
    if not text:
        return []
    return re.findall(r"\b\w+\b", text.lower())


def jaccard_similarity(set1: Set[Any], set2: Set[Any]) -> float:
    """Compute Jaccard similarity between two sets: |A ∩ B| / |A ∪ B|."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return float(intersection) / float(union) if union > 0 else 0.0


def overlap_coefficient(set1: Set[Any], set2: Set[Any]) -> float:
    """Compute Overlap coefficient: |A ∩ B| / min(|A|, |B|)."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    min_len = min(len(set1), len(set2))
    if min_len == 0:
        return 0.0
    return float(len(set1.intersection(set2))) / float(min_len)


class EntityRecord:
    """
    Compact preprocessed entity representation to cache tokenization
    and avoid repeated string computations during pairwise comparison.
    """
    __slots__ = (
        "entity_id",
        "name",
        "name_len",
        "name_tokens",
        "name_token_set",
        "name_token_count",
        "name_char_ngrams",
        "addr",
        "addr_len",
        "addr_tokens",
        "addr_token_set",
        "addr_char_ngrams",
        "addr_numeric_tokens",
        "country",
        "has_name",
        "has_addr",
        "has_country",
        "source_type",
    )

    def __init__(
        self,
        entity_id: str,
        business_name: Optional[str] = "",
        business_address: Optional[str] = "",
        country: Optional[str] = "",
    ):
        self.entity_id = str(entity_id)
        
        # Name preprocessing
        name_clean = str(business_name).strip().lower() if business_name is not None and not pd.isna(business_name) else ""
        self.name = name_clean
        self.has_name = bool(name_clean)
        self.name_len = len(name_clean)
        self.name_tokens = _tokenize(name_clean)
        self.name_token_set = set(self.name_tokens)
        self.name_token_count = len(self.name_tokens)
        self.name_char_ngrams = _get_char_ngrams(name_clean, n=3)

        # Address preprocessing
        addr_clean = str(business_address).strip().lower() if business_address is not None and not pd.isna(business_address) else ""
        self.addr = addr_clean
        self.has_addr = bool(addr_clean)
        self.addr_len = len(addr_clean)
        self.addr_tokens = _tokenize(addr_clean)
        self.addr_token_set = set(self.addr_tokens)
        self.addr_char_ngrams = _get_char_ngrams(addr_clean, n=3)
        self.addr_numeric_tokens = _extract_numeric_tokens(addr_clean)

        # Country preprocessing
        country_clean = str(country).strip().upper() if country is not None and not pd.isna(country) else ""
        self.country = country_clean
        self.has_country = bool(country_clean)

        # Source identification based on entity_id prefix
        if self.entity_id.startswith("S2-"):
            self.source_type = 2
        elif self.entity_id.startswith("S3-"):
            self.source_type = 3
        elif self.entity_id.startswith("S1-"):
            self.source_type = 1
        else:
            self.source_type = 0


def build_entity_record_cache(
    df_or_records: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Dict[str, Any]]]
) -> Dict[str, EntityRecord]:
    """
    Builds a dictionary cache of EntityRecord objects from a DataFrame or record collection.
    """
    cache: Dict[str, EntityRecord] = {}

    if isinstance(df_or_records, pd.DataFrame):
        for row in df_or_records.itertuples(index=False):
            # Assumes columns entity_id, business_name, business_address, country
            eid = getattr(row, "entity_id", "")
            bname = getattr(row, "business_name", "")
            baddr = getattr(row, "business_address", "")
            ctry = getattr(row, "country", "")
            cache[str(eid)] = EntityRecord(
                entity_id=eid,
                business_name=bname,
                business_address=baddr,
                country=ctry,
            )
    elif isinstance(df_or_records, dict):
        for eid, data in df_or_records.items():
            cache[str(eid)] = EntityRecord(
                entity_id=eid,
                business_name=data.get("business_name", ""),
                business_address=data.get("business_address", ""),
                country=data.get("country", ""),
            )
    elif isinstance(df_or_records, list):
        for item in df_or_records:
            eid = item.get("entity_id", "")
            cache[str(eid)] = EntityRecord(
                entity_id=eid,
                business_name=item.get("business_name", ""),
                business_address=item.get("business_address", ""),
                country=item.get("country", ""),
            )
    return cache


def compute_pair_features(rec1: EntityRecord, rec2: EntityRecord) -> List[float]:
    """
    Compute similarity feature vector for a single pair of entities.
    Returns a list of 17 float values matching FEATURE_NAMES.
    """
    # 1. Name features
    if rec1.has_name and rec2.has_name:
        name_exact = 1.0 if rec1.name == rec2.name else 0.0
        name_jaccard = jaccard_similarity(rec1.name_token_set, rec2.name_token_set)
        name_overlap = overlap_coefficient(rec1.name_token_set, rec2.name_token_set)
        name_char_jaccard = jaccard_similarity(rec1.name_char_ngrams, rec2.name_char_ngrams)
        name_len_diff = float(abs(rec1.name_len - rec2.name_len))
        name_token_count_diff = float(abs(rec1.name_token_count - rec2.name_token_count))
    else:
        name_exact = 0.0
        name_jaccard = 0.0
        name_overlap = 0.0
        name_char_jaccard = 0.0
        name_len_diff = float(abs(rec1.name_len - rec2.name_len))
        name_token_count_diff = float(abs(rec1.name_token_count - rec2.name_token_count))

    # 2. Address features
    if rec1.has_addr and rec2.has_addr:
        addr_exact = 1.0 if rec1.addr == rec2.addr else 0.0
        addr_jaccard = jaccard_similarity(rec1.addr_token_set, rec2.addr_token_set)
        addr_num_overlap = overlap_coefficient(rec1.addr_numeric_tokens, rec2.addr_numeric_tokens)
        addr_char_jaccard = jaccard_similarity(rec1.addr_char_ngrams, rec2.addr_char_ngrams)
        addr_len_diff = float(abs(rec1.addr_len - rec2.addr_len))
    else:
        addr_exact = 0.0
        addr_jaccard = 0.0
        addr_num_overlap = 0.0
        addr_char_jaccard = 0.0
        addr_len_diff = float(abs(rec1.addr_len - rec2.addr_len))

    # 3. Country features
    if rec1.has_country and rec2.has_country:
        country_exact = 1.0 if rec1.country == rec2.country else 0.0
        country_missing = 0.0
    else:
        country_exact = 0.0
        country_missing = 1.0

    # 4. Meta features
    missing_name = 1.0 if (not rec1.has_name or not rec2.has_name) else 0.0
    missing_addr = 1.0 if (not rec1.has_addr or not rec2.has_addr) else 0.0
    is_s2 = 1.0 if rec2.source_type == 2 else 0.0
    is_s3 = 1.0 if rec2.source_type == 3 else 0.0

    return [
        name_exact,
        name_jaccard,
        name_overlap,
        name_char_jaccard,
        name_len_diff,
        name_token_count_diff,
        addr_exact,
        addr_jaccard,
        addr_num_overlap,
        addr_char_jaccard,
        addr_len_diff,
        country_exact,
        country_missing,
        missing_name,
        missing_addr,
        is_s2,
        is_s3,
    ]


class FeatureExtractor:
    """
    Feature extraction engine that processes candidate pairs and produces
    compact numeric feature matrices.
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or FEATURE_NAMES

    def extract_features(
        self,
        candidate_pairs: Union[List[Tuple[str, str]], pd.DataFrame],
        s1_records: Dict[str, EntityRecord],
        cand_records: Dict[str, EntityRecord],
        batch_size: int = 50000,
    ) -> np.ndarray:
        """
        Extract numeric feature matrix for candidate pairs.
        
        Args:
            candidate_pairs: List of (source1_entity_id, candidate_entity_id) or DataFrame
            s1_records: Dict mapping source1_entity_id to EntityRecord
            cand_records: Dict mapping candidate_entity_id to EntityRecord
            batch_size: Batch size for feature array allocation
            
        Returns:
            np.ndarray of shape (N, num_features) with dtype=np.float32
        """
        if isinstance(candidate_pairs, pd.DataFrame):
            pairs_list = list(
                zip(
                    candidate_pairs["source1_entity_id"].astype(str),
                    candidate_pairs["candidate_entity_id"].astype(str),
                )
            )
        else:
            pairs_list = candidate_pairs

        num_pairs = len(pairs_list)
        if num_pairs == 0:
            return np.empty((0, len(self.feature_names)), dtype=np.float32)

        # Allocate contiguous float32 matrix
        X = np.zeros((num_pairs, len(self.feature_names)), dtype=np.float32)

        empty_rec = EntityRecord(entity_id="UNKNOWN")

        for idx, (s1_id, cand_id) in enumerate(pairs_list):
            rec1 = s1_records.get(str(s1_id), empty_rec)
            rec2 = cand_records.get(str(cand_id), empty_rec)
            X[idx] = compute_pair_features(rec1, rec2)

        return X
