"""Candidate Generation and Blocking Module.

Member 2 (Data + Blocking) ownership.
Implements multi-pass inverted-index candidate generation with deduplication,
protection against overlarge blocks, and detailed candidate reduction metrics.

Four Required Blocking Passes:
- BLOCK 1: Normalized business-name key (exact core tokens & normalized name)
- BLOCK 2: Business-name prefix + country (4-char prefix & token bigram)
- BLOCK 3: Postal/numeric address token + country (postal PIN/ZIP & street number + word)
- BLOCK 4: Simple token/character signature (distinctive name tokens & initials signature)
"""

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from src.normalizer import (
    extract_core_name_tokens,
    extract_numeric_tokens,
    extract_postal_code,
    normalize_address,
    normalize_business_name,
    normalize_country,
)
from src.utils import get_logger

logger = get_logger("Blocking")

# Generic business words that are excluded from single-token blocking keys (Block 4)
# to avoid runaway candidate explosion
GENERIC_STOPWORDS: Set[str] = {
    "and", "the", "of", "in", "for", "at", "on", "a", "an", "to", "by", "with",
    "retail", "trading", "services", "service", "enterprises", "enterprise",
    "solutions", "group", "holdings", "holding", "international", "center",
    "centre", "store", "shop", "market", "management", "consultants",
    "ventures", "associates", "industries", "industry", "agency"
}


def extract_blocking_keys(
    business_name: str,
    business_address: str,
    country: str,
) -> Dict[str, Set[str]]:
    """Extract multi-pass blocking keys for a record.

    Returns dict of pass_name -> set of keys:
      - block1_name_core: exact first 2 core name tokens and exact normalized name
      - block2_name_prefix_country: first 4 chars of primary name token + country, plus token bigram
      - block3_numeric_addr_country: postal codes, street numbers, and number+street combos + country
      - block4_token_sig_country: distinctive name tokens and sorted initials signature + country
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
    postal_code = extract_postal_code(norm_addr)

    # -------------------------------------------------------------
    # BLOCK 1: Normalized business-name key
    # -------------------------------------------------------------
    if core_tokens:
        k1 = " ".join(core_tokens[:2])
        if len(k1) >= 3:
            keys["block1_name_core"].add(f"b1:{k1}")
        if len(norm_name) >= 3:
            keys["block1_name_core"].add(f"b1:name#{norm_name}#{norm_ctry}")

    # -------------------------------------------------------------
    # BLOCK 2: Business-name prefix + country
    # -------------------------------------------------------------
    if core_tokens and len(core_tokens[0]) >= 3:
        pfx = core_tokens[0][:4]
        keys["block2_name_prefix_country"].add(f"b2:{pfx}#{norm_ctry}")

    if len(core_tokens) >= 2:
        bi = f"{core_tokens[0]}_{core_tokens[1]}"
        keys["block2_name_prefix_country"].add(f"b2:bi:{bi}#{norm_ctry}")

    # -------------------------------------------------------------
    # BLOCK 3: Postal/numeric address token + country where available
    # -------------------------------------------------------------
    if postal_code:
        keys["block3_numeric_addr_country"].add(f"b3:postal:{postal_code}#{norm_ctry}")

    # Street numbers (2-6 digits)
    addr_words = [
        w for w in norm_addr.split()
        if len(w) >= 4 and not w.isdigit() and w not in GENERIC_STOPWORDS
    ]
    for num in num_tokens:
        if 2 <= len(num) <= 6:
            keys["block3_numeric_addr_country"].add(f"b3:{num}#{norm_ctry}")
            # Combine house/building number with first significant street word
            if addr_words:
                keys["block3_numeric_addr_country"].add(f"b3:num_street:{num}_{addr_words[0]}#{norm_ctry}")

    # -------------------------------------------------------------
    # BLOCK 4: Simple token/character signature + country
    # -------------------------------------------------------------
    if core_tokens:
        # Sorted initials of first 3 tokens
        sig = "".join(sorted([t[0] for t in core_tokens[:3]]))
        if sig:
            keys["block4_token_sig_country"].add(f"b4:{sig}#{norm_ctry}")

        # Distinctive name tokens (length >= 4 or non-ASCII or containing digits)
        for tok in core_tokens:
            if tok not in GENERIC_STOPWORDS and (
                len(tok) >= 4
                or any(ord(ch) > 127 for ch in tok)
                or any(ch.isdigit() for ch in tok)
            ):
                keys["block4_token_sig_country"].add(f"b4:tok:{tok}#{norm_ctry}")

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

        # Fast tuple iteration
        for row in df_candidates.itertuples(index=False):
            cid = str(row.entity_id)
            b_name = str(row.business_name)
            b_addr = str(row.business_address)
            b_ctry = str(row.country)

            key_dict = extract_blocking_keys(b_name, b_addr, b_ctry)
            for pass_name, key_set in key_dict.items():
                for k in key_set:
                    self.index[k].append(cid)

        # Cap overlarge blocks to protect memory and runtime
        overlarge = 0
        for k, cids in list(self.index.items()):
            if len(cids) > self.max_block_size:
                self.index[k] = cids[: self.max_block_size]
                overlarge += 1
        if overlarge:
            logger.info(f"Capped {overlarge} excessively large blocks exceeding {self.max_block_size} candidates.")

    def query(
        self,
        business_name: str,
        business_address: str,
        country: str,
        max_candidates: int = 100,
    ) -> Tuple[List[str], int]:
        """Query index for a single Source 1 record.

        Ranks candidates by the number of shared blocking keys across passes.
        Returns (deduplicated_candidates_list, raw_candidate_count_before_dedup).
        """
        key_dict = extract_blocking_keys(business_name, business_address, country)
        raw_count = 0
        candidate_scores: Counter = Counter()

        for pass_name, key_set in key_dict.items():
            for k in key_set:
                matched_cids = self.index.get(k, [])
                raw_count += len(matched_cids)
                for cid in matched_cids:
                    candidate_scores[cid] += 1

        if not candidate_scores:
            return [], raw_count

        # Prioritize candidates by number of shared blocking keys (multi-key agreement)
        # Deterministic tie-breaking on candidate entity_id
        ranked_candidates = sorted(
            candidate_scores.keys(),
            key=lambda cid: (-candidate_scores[cid], cid)
        )

        dedup_list = ranked_candidates[:max_candidates]
        return dedup_list, raw_count


def generate_candidate_pairs(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    max_candidates_per_s1: int = 100,
    max_block_size: int = 1000,
    ground_truth: Optional[Dict[str, Set[str]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Generate candidate pairs using multi-pass blocking.

    Returns:
        candidate_pairs_df: DataFrame with ['source1_entity_id', 'candidate_entity_id']
        metrics: Dictionary of blocking performance metrics
    """
    logger.info("Combining Source 2 and Source 3 for candidate indexing...")
    df_candidates = pd.concat([df_s2, df_s3], ignore_index=True).drop_duplicates(subset=["entity_id"])

    indexer = BlockingIndex(max_block_size=max_block_size)
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

    for row in df_s1.itertuples(index=False):
        s1_id = str(row.entity_id)
        cands, raw_c = indexer.query(
            str(row.business_name),
            str(row.business_address),
            str(row.country),
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

    # If ground truth is provided, evaluate candidate recall
    if ground_truth:
        cand_map: Dict[str, Set[str]] = defaultdict(set)
        for s1, cid in pairs:
            cand_map[s1].add(cid)

        total_true = 0
        captured_true = 0
        for s1_id, true_set in ground_truth.items():
            if true_set:
                total_true += len(true_set)
                found = cand_map.get(s1_id, set())
                captured_true += len(true_set & found)

        recall = (captured_true / total_true) if total_true > 0 else 1.0
        metrics["blocking_recall"] = float(recall)
        metrics["total_ground_truth_matches"] = float(total_true)
        metrics["captured_ground_truth_matches"] = float(captured_true)
        logger.info(f"Blocking recall on ground truth: {recall:.4f} ({captured_true}/{total_true} matches)")

    logger.info(
        f"Blocking completed: Generated {total_dedup_pairs:,} candidate pairs. "
        f"Reduction ratio: {reduction_ratio:.6f} "
        f"(Avg: {metrics['avg_candidates_per_s1']:.1f}, Max: {metrics['max_candidates_per_s1']:.0f})"
    )

    return candidate_df, metrics


def export_candidate_pairs_tsv(
    candidate_df: pd.DataFrame,
    output_path: Union[str, Path],
    all_s1_ids: Optional[List[str]] = None,
) -> None:
    """Exports candidate pairs to official candidate_pairs.tsv format.

    Header:
    source1_entity_id\\tcandidate_entity_ids

    - Strictly TAB-delimited
    - Comma-separated candidate IDs
    - Every S1 entity has exactly one row
    - Unmatched entities have empty string
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cand_map: Dict[str, List[str]] = defaultdict(list)
    for row in candidate_df.itertuples(index=False):
        cand_map[str(row.source1_entity_id)].append(str(row.candidate_entity_id))

    # Determine full list of S1 IDs
    if all_s1_ids is None:
        all_ids = sorted(list(cand_map.keys()))
    else:
        all_ids = all_s1_ids

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_ids:
            cands = cand_map.get(s1_id, [])
            cands_str = ",".join(cands)
            f.write(f"{s1_id}\t{cands_str}\n")

    logger.info(f"Candidate pairs exported to {output_path} ({len(all_ids)} rows).")
