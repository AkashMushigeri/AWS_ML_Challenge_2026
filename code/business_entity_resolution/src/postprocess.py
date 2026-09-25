"""Post-Processing and Output Generation Module.

Member 1 (Team Lead) ownership.
Generates candidate_pairs.tsv and matching_results.tsv adhering strictly to official TSV formatting rules.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger("Postprocess")


def format_id_list(id_iterable: Set[str]) -> str:
    """Format a set of IDs into a sorted comma-separated string without duplicates."""
    # Only keep valid S2- and S3- IDs, exclude any S1- self-matches
    valid = sorted([i for i in id_iterable if i.startswith(("S2-", "S3-"))])
    return ",".join(valid)


def build_candidate_and_matching_outputs(
    all_s1_ids: List[str],
    candidate_pairs_df: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Aggregate candidate pairs and threshold-filtered matches per Source 1 entity.

    Returns:
        candidates_per_s1: Dict[s1_id, Set[candidate_id]]
        matches_per_s1: Dict[s1_id, Set[matched_id]]
    """
    candidates_per_s1: Dict[str, Set[str]] = {s1: set() for s1 in all_s1_ids}
    matches_per_s1: Dict[str, Set[str]] = {s1: set() for s1 in all_s1_ids}

    for i, (_, row) in enumerate(candidate_pairs_df.iterrows()):
        s1 = row["source1_entity_id"]
        c = row["candidate_entity_id"]

        # Ensure entity is in our universe
        if s1 not in candidates_per_s1:
            candidates_per_s1[s1] = set()
            matches_per_s1[s1] = set()

        if c.startswith(("S2-", "S3-")):
            candidates_per_s1[s1].add(c)
            if float(scores[i]) >= threshold:
                matches_per_s1[s1].add(c)

    return candidates_per_s1, matches_per_s1


def write_submission_tsv(
    data_dict: Dict[str, Set[str]],
    file_path: Path,
    header_col2: str,
) -> None:
    """Write an entity ID list TSV file with exact formatting.

    - Tab-separated
    - Header: source1_entity_id \t {header_col2}
    - One line per S1 entity
    - No quoting
    - UTF-8 encoding
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing {len(data_dict):,} rows to {file_path}...")

    with open(file_path, "w", encoding="utf-8", newline="\n") as f:
        # Write header
        f.write(f"source1_entity_id\t{header_col2}\n")
        # Write rows
        for s1_id in sorted(data_dict.keys()):
            id_list_str = format_id_list(data_dict[s1_id])
            f.write(f"{s1_id}\t{id_list_str}\n")

    logger.info(f"Successfully generated {file_path}")


def export_pipeline_outputs(
    all_s1_ids: List[str],
    candidate_pairs_df: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    output_dir: Path,
) -> Tuple[Path, Path, Dict[str, int]]:
    """Export both candidate_pairs.tsv and matching_results.tsv.

    Returns:
        candidate_file_path, matching_file_path, summary_counts
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cand_file = output_dir / "candidate_pairs.tsv"
    match_file = output_dir / "matching_results.tsv"

    candidates_map, matches_map = build_candidate_and_matching_outputs(
        all_s1_ids=all_s1_ids,
        candidate_pairs_df=candidate_pairs_df,
        scores=scores,
        threshold=threshold,
    )

    write_submission_tsv(candidates_map, cand_file, "candidate_entity_ids")
    write_submission_tsv(matches_map, match_file, "matched_entity_ids")

    matched_s1_count = sum(1 for m in matches_map.values() if len(m) > 0)
    unmatched_s1_count = len(matches_map) - matched_s1_count
    total_matches_count = sum(len(m) for m in matches_map.values())

    stats = {
        "total_s1_entities": len(all_s1_ids),
        "matched_s1_entities": matched_s1_count,
        "unmatched_s1_entities": unmatched_s1_count,
        "total_matches_linked": total_matches_count,
    }

    logger.info(
        f"Output generation complete: {matched_s1_count} matched S1 entities, "
        f"{unmatched_s1_count} unmatched (singletons)."
    )

    return cand_file, match_file, stats
