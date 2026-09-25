"""Candidate Generation and Blocking Module.

Member 2 (Data + Blocking) ownership.
Implements multi-pass inverted-index candidate generation with deduplication and metrics.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.normalizer import (
    extract_core_name_tokens,
    extract_numeric_tokens,
    normalize_address,
    normalize_business_name,
    normalize_country,
)
from src.utils import get_logger

logger = get_logger("Blocking")


def extract_blocking_keys(
    business_name: str,
    business_address: str,
    country: str,
) -> Dict[str, Set[str]]:
    """Extract multi-pass blocking keys for a record.

    Returns dict of pass_name -> set of keys:
      - block1_name_core: exact first 2 core name tokens
      - block2_name_prefix_country: first 4 chars of primary name token + country
      - block3_numeric_addr_country: postal/street numbers + country
      - block4_token_sig_country: sorted initial letters of name tokens + country
    """
    keys: Dict[str, Set[str]] = {
        "block1_name_core": set(),
        "block2_name_prefix_country": set(),
        "block3_numeric_addr_country": set(),
        "block4_token_sig_country": set(),
    }

    norm_name = normalize_business_name(business_name)
    norm_addr = normalize_address(business_address)
    norm_ctry = normalize_country(country)

    core_tokens = extract_core_name_tokens(norm_name)
    num_tokens = extract_numeric_tokens(norm_addr)

    # BLOCK 1: Normalized business-name key (first 2 core tokens or full name if short)
    if core_tokens:
        k1 = " ".join(core_tokens[:2])
        if len(k1) >= 3:
            keys["block1_name_core"].add(f"b1:{k1}")

    # BLOCK 2: Business-name prefix (first 4 chars of primary token) + country
    if core_tokens and len(core_tokens[0]) >= 3:
        pfx = core_tokens[0][:4]
        keys["block2_name_prefix_country"].add(f"b2:{pfx}#{norm_ctry}")

    # BLOCK 3: Postal/numeric address token + country where available
    for num in num_tokens:
        # Meaningful numbers: street numbers (1-5 digits) or postal codes (5-6 digits)
        if 2 <= len(num) <= 6:
            keys["block3_numeric_addr_country"].add(f"b3:{num}#{norm_ctry}")

    # BLOCK 4: Simple token/character signature + country
    if core_tokens:
        # Signature: initials of first 3 tokens or first 3 chars
        sig = "".join(sorted([t[0] for t in core_tokens[:3]]))
        if sig:
            keys["block4_token_sig_country"].add(f"b4:{sig}#{norm_ctry}")

    return keys


class BlockingIndex:
    """Inverted index for multi-pass blocking across candidate sources."""

    def __init__(self, max_block_size: int = 1000):
        self.max_block_size = max_block_size
        self.index: Dict[str, List[str]] = defaultdict(list)
        self.total_candidate_records = 0

    def index_candidates(self, df_candidates: pd.DataFrame) -> None:
        """Build the inverted index from combined candidate records (S2 + S3)."""
        logger.info(f"Indexing {len(df_candidates)} candidate records...")
        self.total_candidate_records = len(df_candidates)
        for _, row in df_candidates.iterrows():
            cid = row["entity_id"]
            b_name = row["business_name"]
            b_addr = row["business_address"]
            b_ctry = row["country"]

            key_dict = extract_blocking_keys(b_name, b_addr, b_ctry)
            for pass_name, key_set in key_dict.items():
                for k in key_set:
                    self.index[k].append(cid)

        # Filter out overly generic / massive blocks (stop-blocks)
        overlarge = 0
        for k, cids in list(self.index.items()):
            if len(cids) > self.max_block_size:
                # Truncate or drop over-dense blocks to protect memory and runtime
                self.index[k] = cids[: self.max_block_size]
                overlarge += 1
        if overlarge:
            logger.info(f"Truncated {overlarge} excessively large blocks exceeding {self.max_block_size} candidates.")

    def query(
        self,
        business_name: str,
        business_address: str,
        country: str,
        max_candidates: int = 100,
    ) -> Tuple[List[str], int]:
        """Query index for a single Source 1 record.

        Returns (deduplicated_candidates_list, raw_candidate_count_before_dedup).
        """
        key_dict = extract_blocking_keys(business_name, business_address, country)
        raw_count = 0
        seen_candidates: Set[str] = set()

        for pass_name, key_set in key_dict.items():
            for k in key_set:
                matched_cids = self.index.get(k, [])
                raw_count += len(matched_cids)
                for cid in matched_cids:
                    seen_candidates.add(cid)
                    if len(seen_candidates) >= max_candidates * 2:
                        break

        # Deduplicate and cap
        dedup_list = list(seen_candidates)[:max_candidates]
        return dedup_list, raw_count


def generate_candidate_pairs(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    max_candidates_per_s1: int = 100,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Generate candidate pairs using multi-pass blocking.

    Returns:
        candidate_pairs_df: DataFrame with ['source1_entity_id', 'candidate_entity_id']
        metrics: Dictionary of blocking performance metrics
    """
    logger.info("Combining Source 2 and Source 3 for candidate indexing...")
    df_candidates = pd.concat([df_s2, df_s3], ignore_index=True).drop_duplicates(subset=["entity_id"])

    indexer = BlockingIndex(max_block_size=500)
    indexer.index_candidates(df_candidates)

    total_s1 = len(df_s1)
    cartesian_space = total_s1 * len(df_candidates)
    logger.info(
        f"Querying blocking index for {total_s1} S1 entities against {len(df_candidates)} candidates "
        f"(Cartesian space: {cartesian_space:,} pairs)..."
    )

    pairs: List[Tuple[str, str]] = []
    candidates_per_s1_counts: List[int] = []
    total_raw_pairs = 0

    for _, row in df_s1.iterrows():
        s1_id = row["entity_id"]
        cands, raw_c = indexer.query(
            row["business_name"],
            row["business_address"],
            row["country"],
            max_candidates=max_candidates_per_s1,
        )
        total_raw_pairs += raw_c
        candidates_per_s1_counts.append(len(cands))
        for cid in cands:
            pairs.append((s1_id, cid))

    candidate_df = pd.DataFrame(pairs, columns=["source1_entity_id", "candidate_entity_id"])

    # Compute blocking metrics
    counts_arr = np.array(candidates_per_s1_counts) if candidates_per_s1_counts else np.array([0])
    total_dedup_pairs = len(candidate_df)
    reduction_ratio = 1.0 - (total_dedup_pairs / cartesian_space) if cartesian_space > 0 else 1.0

    metrics: Dict[str, float] = {
        "num_s1": float(total_s1),
        "total_candidates_pool": float(len(df_candidates)),
        "raw_candidates": float(total_raw_pairs),
        "dedup_candidates": float(total_dedup_pairs),
        "avg_candidates_per_s1": float(np.mean(counts_arr)),
        "median_candidates_per_s1": float(np.median(counts_arr)),
        "p95_candidates_per_s1": float(np.percentile(counts_arr, 95)),
        "max_candidates_per_s1": float(np.max(counts_arr)),
        "min_candidates_per_s1": float(np.min(counts_arr)),
        "reduction_ratio": float(reduction_ratio),
    }

    logger.info(
        f"Blocking completed: Generated {total_dedup_pairs:,} candidate pairs. "
        f"Reduction ratio: {reduction_ratio:.6f} "
        f"(Avg: {metrics['avg_candidates_per_s1']:.1f}, Max: {metrics['max_candidates_per_s1']:.0f})"
    )

    return candidate_df, metrics
