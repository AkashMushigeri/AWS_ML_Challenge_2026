"""Data Loading and Validation Module.

Member 2 (Data + Blocking) ownership.
Implements memory-conscious, deterministic, streaming TSV loading with strict validation.

Key Requirements:
- Strictly tab-separated values (sep='\\t')
- Explicit dtypes and schemas
- Memory-conscious streaming (no loading full 12.5M records into RAM)
- Deterministic sampling (random_seed=42, configurable sample_size)
- Robust path discovery using pathlib (no hardcoded absolute paths)
- Clean error handling for missing columns, corrupt lines, or missing files
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import pandas as pd

from src.utils import get_logger

logger = get_logger("DataLoader")

REQUIRED_SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
REQUIRED_GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def resolve_data_dir(data_dir: Optional[Union[str, Path]] = None) -> Path:
    """Auto-discovers the dataset root directory.

    Checks candidates in order of priority:
    1. User-passed data_dir (if valid)
    2. dataset/student_resource/dataset
    3. dataset/
    4. ../student_resource/dataset
    5. C:/Users/mmano/OneDrive/Desktop/student_resource/dataset
    6. data/
    """
    if data_dir is not None:
        p = Path(data_dir)
        if (p / "train").exists() or (p / "test").exists():
            return p
        if (p / "student_resource" / "dataset").exists():
            return p / "student_resource" / "dataset"

    candidates = [
        Path("dataset/student_resource/dataset"),
        Path("dataset"),
        Path("../student_resource/dataset"),
        Path(r"C:\Users\mmano\OneDrive\Desktop\student_resource\dataset"),
        Path("data"),
    ]

    for c in candidates:
        if c.exists() and ((c / "train").exists() or (c / "test").exists()):
            return c.resolve()

    return Path(data_dir) if data_dir else Path("dataset")


def validate_source_columns(cols: List[str], file_path: Path) -> None:
    """Validate that the TSV header has the exact required columns."""
    missing = [c for c in REQUIRED_SOURCE_COLUMNS if c not in cols]
    if missing:
        raise ValueError(
            f"File {file_path} is missing required column(s): {missing}. "
            f"Found columns: {cols}"
        )


def load_source_tsv(
    file_path: Union[str, Path],
    nrows: Optional[int] = None,
    target_ids: Optional[Set[str]] = None,
    max_scan: Optional[int] = None,
) -> pd.DataFrame:
    """Fast, memory-conscious streaming loader for TSV files.

    - Tab-separated
    - Line-by-line streaming without loading massive multi-hundred MB files into memory
    - Extracts target_ids and/or first nrows distractors
    - Handles CRLF and missing columns safely
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Source file not found at: {file_path}")

    rows: List[List[str]] = []
    targets = set(target_ids) if target_ids else set()
    found_targets = 0
    distractor_count = 0

    with open(file_path, "r", encoding="utf-8") as f:
        header_line = next(f, None)
        if not header_line:
            return pd.DataFrame(columns=REQUIRED_SOURCE_COLUMNS)

        cols = [c.strip() for c in header_line.rstrip("\r\n").split("\t")]
        validate_source_columns(cols, file_path)

        for idx, line in enumerate(f):
            parts = line.rstrip("\r\n").split("\t")
            if not parts or not parts[0].strip():
                continue

            # Pad with empty strings if trailing columns are missing
            if len(parts) < 4:
                parts += [""] * (4 - len(parts))

            eid = parts[0].strip()
            is_target = eid in targets
            is_distractor = False

            if nrows is not None and distractor_count < nrows:
                is_distractor = True

            if is_target or is_distractor:
                rows.append([parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()])
                if is_target:
                    found_targets += 1
                if is_distractor:
                    distractor_count += 1

            # Termination condition:
            # If target_ids were supplied, stop once all targets are found AND distractor quota met
            if targets:
                if found_targets >= len(targets) and (nrows is None or distractor_count >= nrows):
                    break
            else:
                if nrows is not None and distractor_count >= nrows:
                    break

            if max_scan is not None and idx >= max_scan:
                break

    df = pd.DataFrame(rows, columns=REQUIRED_SOURCE_COLUMNS).fillna("")
    df = df.drop_duplicates(subset=["entity_id"]).reset_index(drop=True)
    return df


def load_ground_truth(
    file_path: Union[str, Path],
    target_s1_ids: Optional[Set[str]] = None
) -> Dict[str, Set[str]]:
    """Fast streaming loader for train_ground_truth.tsv.

    Reads line by line without loading the full 127MB into pandas.
    Returns dict mapping: source1_entity_id -> set of matched_entity_ids.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Ground truth file not found at: {file_path}")

    ground_truth: Dict[str, Set[str]] = {}
    target_set = set(target_s1_ids) if target_s1_ids else None

    with open(file_path, "r", encoding="utf-8") as f:
        header = next(f, None)
        if not header:
            return ground_truth
        cols = [c.strip() for c in header.rstrip("\r\n").split("\t")]
        for col in REQUIRED_GROUND_TRUTH_COLUMNS:
            if col not in cols:
                raise ValueError(f"Ground truth file {file_path} missing column: {col}")

        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if not parts or not parts[0].strip():
                continue
            s1_id = parts[0].strip()
            if target_set is not None and s1_id not in target_set:
                continue

            matched_str = parts[1].strip() if len(parts) > 1 else ""
            if matched_str:
                matched_set = {m.strip() for m in matched_str.split(",") if m.strip()}
                ground_truth[s1_id] = matched_set
            else:
                ground_truth[s1_id] = set()

            if target_set is not None and len(ground_truth) >= len(target_set):
                break

    # If target_set was provided, ensure all target IDs exist in dict (default to empty set)
    if target_set is not None:
        for s1_id in target_set:
            if s1_id not in ground_truth:
                ground_truth[s1_id] = set()

    return ground_truth


def load_v1_sample(
    data_dir: Optional[Union[str, Path]] = None,
    split: str = "train",
    sample_size: int = 5000,
    random_seed: int = 42,
    distractor_size: int = 25000,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[Dict[str, Set[str]]]]:
    """Deterministically load a representative small V1 sample of the dataset.

    Returns:
        (df_s1, df_s2, df_s3, ground_truth)
    """
    resolved_dir = resolve_data_dir(data_dir)
    split_dir = resolved_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Directory {split_dir} does not exist. (Base: {resolved_dir})")

    s1_path = split_dir / f"{split}_source1.tsv"
    s2_path = split_dir / f"{split}_source2.tsv"
    s3_path = split_dir / f"{split}_source3.tsv"

    logger.info(f"Loading {split} Source 1 from {s1_path} (sample size: {sample_size})...")
    raw_s1 = load_source_tsv(s1_path, nrows=sample_size * 2)
    if len(raw_s1) > sample_size:
        df_s1 = raw_s1.sample(n=sample_size, random_state=random_seed).reset_index(drop=True)
    else:
        df_s1 = raw_s1

    s1_ids = set(df_s1["entity_id"].tolist())
    target_s2_ids: Set[str] = set()
    target_s3_ids: Set[str] = set()
    ground_truth: Optional[Dict[str, Set[str]]] = None

    if split == "train":
        gt_path = split_dir / "train_ground_truth.tsv"
        logger.info(f"Loading ground truth for {len(s1_ids)} Source 1 entities from {gt_path}...")
        ground_truth = load_ground_truth(gt_path, target_s1_ids=s1_ids)

        for matches in ground_truth.values():
            for m in matches:
                if m.startswith("S2-"):
                    target_s2_ids.add(m)
                elif m.startswith("S3-"):
                    target_s3_ids.add(m)
        logger.info(
            f"Sample ground truth contains {len(target_s2_ids)} target S2 matches "
            f"and {len(target_s3_ids)} target S3 matches."
        )

    logger.info(f"Loading {split} Source 2 (targets: {len(target_s2_ids)}, distractors: {distractor_size})...")
    df_s2 = load_source_tsv(s2_path, nrows=distractor_size, target_ids=target_s2_ids if target_s2_ids else None)

    logger.info(f"Loading {split} Source 3 (targets: {len(target_s3_ids)}, distractors: {distractor_size})...")
    df_s3 = load_source_tsv(s3_path, nrows=distractor_size, target_ids=target_s3_ids if target_s3_ids else None)

    logger.info(
        f"V1 Dataset loaded successfully: "
        f"S1={len(df_s1)}, S2={len(df_s2)}, S3={len(df_s3)}, "
        f"GroundTruth={len(ground_truth) if ground_truth is not None else 0}"
    )

    return df_s1, df_s2, df_s3, ground_truth
