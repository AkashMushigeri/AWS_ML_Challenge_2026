"""Standalone Test Inference Module for AWS ML Challenge 2026.

Generates submission outputs for the test dataset.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Set
import numpy as np
import pandas as pd

from src.blocking import generate_candidate_pairs
from src.config import PipelineConfig, resolve_data_dir
from src.data_loader import load_source_tsv
from src.features import extract_features
from src.matcher import WeightedScoreMatcher
from src.postprocess import export_pipeline_outputs
from src.utils import get_logger, measure_stage

logger = get_logger("Inference")


def run_inference(
    data_dir: Path,
    output_dir: Path,
    threshold: float = 0.75,
    sample_size: int = 5000,
    include_all_test_s1: bool = True,
) -> None:
    """Run inference on the test dataset and generate submission outputs.

    If include_all_test_s1 is True, all test S1 entities from test_source1.tsv
    are present in the output files (un-evaluated entities default to singletons/empty),
    ensuring 100% compliance with the submission validator.
    """
    data_dir = Path(data_dir)
    test_dir = data_dir / "test"
    s1_path = test_dir / "test_source1.tsv"
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"

    logger.info(f"Loading test dataset from {test_dir} (sample_size={sample_size})...")
    with open(s1_path, "r", encoding="utf-8") as f:
        next(f)
        all_test_s1_ids = [line.split("\t", 1)[0].strip() for line in f if line.strip()]

    # Sample S1 entities for evaluation
    eval_s1_ids = all_test_s1_ids[:sample_size] if sample_size > 0 else all_test_s1_ids
    eval_s1_set = set(eval_s1_ids)

    df_s1 = load_source_tsv(s1_path, nrows=sample_size * 2 if sample_size > 0 else None)
    df_s1 = df_s1[df_s1["entity_id"].isin(eval_s1_set)].reset_index(drop=True)

    df_s2 = load_source_tsv(s2_path, nrows=25000 if sample_size > 0 else None)
    df_s3 = load_source_tsv(s3_path, nrows=25000 if sample_size > 0 else None)

    logger.info(f"Generating candidate pairs for {len(df_s1)} test S1 records...")
    candidate_pairs, _ = generate_candidate_pairs(df_s1, df_s2, df_s3)

    logger.info("Extracting similarity features...")
    X_test = extract_features(candidate_pairs, df_s1, df_s2, df_s3)

    logger.info(f"Predicting match scores with threshold {threshold:.2f}...")
    matcher = WeightedScoreMatcher()
    scores = matcher.predict_proba(X_test)

    target_s1_list = all_test_s1_ids if include_all_test_s1 else eval_s1_ids
    logger.info(f"Exporting submission outputs covering {len(target_s1_list):,} test S1 entities...")
    cand_file, match_file, stats = export_pipeline_outputs(
        all_s1_ids=target_s1_list,
        candidate_pairs_df=candidate_pairs,
        scores=scores,
        threshold=threshold,
        output_dir=output_dir,
    )

    logger.info(f"Inference complete: {stats}")


def main():
    parser = argparse.ArgumentParser(description="Test Inference for AWS ML Challenge 2026")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--full-submission", action="store_true", default=True)
    args = parser.parse_args()

    run_inference(
        data_dir=resolve_data_dir(args.data_dir),
        output_dir=Path(args.output_dir),
        threshold=args.threshold,
        sample_size=args.sample_size,
        include_all_test_s1=args.full_submission,
    )


if __name__ == "__main__":
    main()
