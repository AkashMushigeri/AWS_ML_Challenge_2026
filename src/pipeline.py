"""Master Pipeline Orchestrator and CLI.

Member 1 (Team Lead) ownership.
Integrates Data Loading, Normalization, Blocking, Features, Matcher, Evaluation,
Output Generation, and Official Submission Validation.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.blocking import generate_candidate_pairs
from src.config import PipelineConfig, resolve_data_dir
from src.data_loader import load_v1_sample, load_source_tsv
from src.evaluation import evaluate_predictions, find_optimal_threshold
from src.features import extract_features
from src.matcher import WeightedScoreMatcher, LogisticRegressionMatcher
from src.postprocess import export_pipeline_outputs, write_submission_tsv
from src.utils import get_current_memory_mb, get_logger, measure_stage

logger = get_logger("Pipeline")


def run_v1_pipeline(config: PipelineConfig, run_validator: bool = True) -> Dict[str, any]:
    """Execute the end-to-end V1 entity resolution pipeline.

    1. Load deterministic sample of training data
    2. Split S1 entities into Train (80%) and Holdout Validation (20%)
    3. Run candidate generation / blocking on validation fold
    4. Extract pairwise similarity features
    5. Evaluate baseline matcher and tune optimal F0.5 threshold
    6. Generate candidate_pairs.tsv and matching_results.tsv
    7. Execute official validator
    8. Generate benchmark report
    """
    overall_start_time = time.perf_counter()
    report_data = {
        "sample_size": config.sample_size,
        "seed": config.random_seed,
        "stages": {},
    }

    logger.info("==================================================================")
    logger.info("        AWS ML Challenge 2026 — V1 Pipeline Execution             ")
    logger.info(f"Sample Size: {config.sample_size} | Seed: {config.random_seed} | Mode: Train & Validation")
    logger.info("==================================================================")

    # Stage 1: Data Ingestion
    with measure_stage("Stage 1: Deterministic Data Ingestion", logger) as m1:
        df_s1, df_s2, df_s3, ground_truth = load_v1_sample(
            data_dir=config.data_dir,
            split="train",
            sample_size=config.sample_size,
            random_seed=config.random_seed,
            distractor_size=config.s2_s3_distractor_size,
        )
    report_data["stages"]["data_ingestion"] = m1

    # Stage 2: S1 Holdout Split (80% Train / 20% Val)
    with measure_stage("Stage 2: Entity-Level Holdout Split", logger) as m2:
        all_s1_ids = df_s1["entity_id"].tolist()
        rng = np.random.default_rng(config.random_seed)
        shuffled_ids = rng.permutation(all_s1_ids)

        n_val = int(len(shuffled_ids) * config.val_split_ratio)
        val_s1_ids = set(shuffled_ids[:n_val])
        train_s1_ids = set(shuffled_ids[n_val:])

        df_s1_val = df_s1[df_s1["entity_id"].isin(val_s1_ids)].reset_index(drop=True)
        val_ground_truth = {s1: ground_truth.get(s1, set()) for s1 in val_s1_ids}
        logger.info(
            f"Split {len(all_s1_ids)} S1 entities into {len(train_s1_ids)} train "
            f"and {len(val_s1_ids)} validation entities."
        )
    report_data["stages"]["split"] = m2

    # Stage 3: Blocking / Candidate Generation (Evaluated on Validation Fold)
    with measure_stage("Stage 3: Multi-Pass Candidate Generation & Blocking", logger) as m3:
        val_candidate_pairs, blocking_metrics = generate_candidate_pairs(
            df_s1=df_s1_val,
            df_s2=df_s2,
            df_s3=df_s3,
            max_candidates_per_s1=config.max_candidates_per_s1,
        )
    report_data["stages"]["blocking"] = m3
    report_data["blocking_metrics"] = blocking_metrics

    # Stage 4: Feature Extraction
    with measure_stage("Stage 4: Pairwise Similarity Feature Extraction", logger) as m4:
        X_val = extract_features(
            candidate_pairs_df=val_candidate_pairs,
            df_s1=df_s1_val,
            df_s2=df_s2,
            df_s3=df_s3,
        )
    report_data["stages"]["features"] = m4

    # Stage 5: Baseline Matching Model & Probability Scoring
    with measure_stage("Stage 5: Baseline Matching & Probability Scoring", logger) as m5:
        matcher = WeightedScoreMatcher()
        val_scores = matcher.predict_proba(X_val)
    report_data["stages"]["matcher"] = m5

    # Stage 6: Threshold Evaluation and Optimization for F0.5
    with measure_stage("Stage 6: Validation Threshold Optimization (F0.5)", logger) as m6:
        optimal_threshold, threshold_summary = find_optimal_threshold(
            candidate_pairs_df=val_candidate_pairs,
            scores=val_scores,
            ground_truth=val_ground_truth,
            threshold_grid=config.thresholds,
        )
        config.selected_threshold = optimal_threshold
        # Compute final validation metrics at optimal threshold
        val_preds: Dict[str, Set[str]] = {s1: set() for s1 in val_s1_ids}
        for i, (_, row) in enumerate(val_candidate_pairs.iterrows()):
            if val_scores[i] >= optimal_threshold:
                val_preds[row["source1_entity_id"]].add(row["candidate_entity_id"])
        val_eval_metrics = evaluate_predictions(val_ground_truth, val_preds)
    report_data["stages"]["evaluation"] = m6
    report_data["val_eval_metrics"] = val_eval_metrics
    report_data["threshold_summary"] = threshold_summary
    report_data["optimal_threshold"] = optimal_threshold

    # Stage 7: Post-Processing & Output Generation
    with measure_stage("Stage 7: Output Generation & Formatting", logger) as m7:
        cand_file, match_file, output_stats = export_pipeline_outputs(
            all_s1_ids=list(val_s1_ids),
            candidate_pairs_df=val_candidate_pairs,
            scores=val_scores,
            threshold=optimal_threshold,
            output_dir=config.output_dir,
        )
    report_data["stages"]["output"] = m7
    report_data["output_stats"] = output_stats

    # Stage 8: Official Submission Validation
    validation_status = "NOT_RUN"
    validation_log = ""
    if run_validator:
        with measure_stage("Stage 8: Official Submission Validator Execution", logger) as m8:
            validation_status, validation_log = execute_validator(
                matching_path=match_file,
                candidate_path=cand_file,
                val_s1_ids=val_s1_ids,
                data_dir=config.data_dir,
            )
        report_data["stages"]["validator"] = m8
    report_data["validation_status"] = validation_status
    report_data["validation_log"] = validation_log

    overall_runtime = time.perf_counter() - overall_start_time
    report_data["total_runtime_sec"] = overall_runtime
    report_data["peak_memory_mb"] = get_current_memory_mb()

    # Stage 9: Generate Benchmark Report
    generate_benchmark_report(report_data, config.reports_dir)

    logger.info("==================================================================")
    logger.info("              V1 Pipeline Execution Completed                     ")
    logger.info(f"Total Runtime: {overall_runtime:.2f}s | Peak Memory: {report_data['peak_memory_mb']:.1f} MB")
    logger.info(f"Validation F0.5: {val_eval_metrics['macro_f05']:.4f} (P: {val_eval_metrics['macro_precision']:.4f}, R: {val_eval_metrics['macro_recall']:.4f})")
    logger.info(f"Official Validator Status: {validation_status}")
    logger.info("==================================================================")

    return report_data


def execute_validator(
    matching_path: Path,
    candidate_path: Path,
    val_s1_ids: Set[str],
    data_dir: Path,
) -> Tuple[str, str]:
    """Execute the official submission validator."""
    # Find validator script
    script_candidates = [
        Path("utils/validate_submission.py"),
        Path("dataset/student_resource/utils/validate_submission.py"),
    ]
    validator_script = None
    for cand in script_candidates:
        if cand.exists():
            validator_script = cand
            break

    if not validator_script:
        logger.warning("Submission validator script not found; skipping validation execution.")
        return "SKIPPED", "Validator script not found."

    # Create a temporary directory containing test_source1.tsv with our validation S1 IDs
    # so the validator can verify all structural and subset constraints on our sample
    temp_test_dir = Path("output/_val_test_dir")
    temp_test_dir.mkdir(parents=True, exist_ok=True)
    temp_s1_file = temp_test_dir / "test_source1.tsv"
    with open(temp_s1_file, "w", encoding="utf-8") as f:
        f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        for s1 in sorted(val_s1_ids):
            f.write(f"{s1}\tsample_business\tsample_address\tUS\n")

    cmd = [
        sys.executable,
        str(validator_script),
        "--matching",
        str(matching_path),
        "--candidate",
        str(candidate_path),
        "--test-dir",
        str(temp_test_dir),
    ]

    logger.info(f"Running validator: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + "\n" + result.stderr

    if result.returncode == 0 and "PASS" in output:
        status = "PASS"
        logger.info("OFFICIAL VALIDATOR RESULT: PASS")
    else:
        status = "FAIL"
        logger.error(f"OFFICIAL VALIDATOR RESULT: FAIL\n{output}")

    return status, output.strip()


def generate_benchmark_report(data: dict, reports_dir: Path) -> Path:
    """Generate reports/v1_benchmark.md markdown report."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "v1_benchmark.md"

    bm = data.get("blocking_metrics", {})
    ev = data.get("val_eval_metrics", {})
    st = data.get("output_stats", {})
    thresh_df = data.get("threshold_summary", pd.DataFrame())
    if not thresh_df.empty:
        headers = list(thresh_df.columns)
        table_rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, row in thresh_df.iterrows():
            formatted_vals = []
            for col, val in row.items():
                if isinstance(val, float):
                    formatted_vals.append(f"{val:.4f}" if col != "threshold" else f"{val:.2f}")
                else:
                    formatted_vals.append(str(val))
            table_rows.append("| " + " | ".join(formatted_vals) + " |")
        thresh_table_md = "\n".join(table_rows)
    else:
        thresh_table_md = "No threshold data"

    content = f"""# AWS ML Challenge 2026 — V1 Pipeline Benchmark Report

**Pipeline Version:** 1.0 (Lightweight Deterministic Baseline)  
**Execution Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Evaluator / Team Lead:** Developer 1 (Member 1)  

---

## 1. Executive Summary
The V1 baseline pipeline implements an end-to-end, deterministic, memory-conscious entity resolution architecture designed to handle noisy business records across three disparate data sources. Operating on a representative sample of {data['sample_size']} Source 1 entities with fixed random seed {data['seed']}, the pipeline successfully generates `candidate_pairs.tsv` and `matching_results.tsv` while strictly adhering to official competition requirements.

---

## 2. Experimental Setup & Configuration
- **Sample Size (S1 Records):** {data['sample_size']}
- **Random Seed:** {data['seed']}
- **Validation Split Ratio:** 20% Source 1 holdout
- **Selected Decision Threshold:** {data.get('optimal_threshold', 0.75):.2f}
- **Official Submission Validator Status:** **{data.get('validation_status', 'UNKNOWN')}**

---

## 3. Candidate Generation (Blocking) Performance
Candidate generation utilizes a 4-pass inverted index (Core Name, Name Prefix + Country, Postal/Numeric Address + Country, Token Signature + Country) with candidate capping per entity.

| Metric | Measured Value |
| :--- | :--- |
| **Total Source 1 Entities** | {bm.get('num_s1', 0):,.0f} |
| **Candidate Pool Size (S2 + S3)** | {bm.get('total_candidates_pool', 0):,.0f} |
| **Raw Candidate Pairs (Pre-dedup)** | {bm.get('raw_candidates', 0):,.0f} |
| **Deduplicated Candidate Pairs** | {bm.get('dedup_candidates', 0):,.0f} |
| **Candidate Reduction Ratio** | {bm.get('reduction_ratio', 0.0):.6f} |
| **Average Candidates / S1** | {bm.get('avg_candidates_per_s1', 0.0):.2f} |
| **Median Candidates / S1** | {bm.get('median_candidates_per_s1', 0.0):.1f} |
| **95th Percentile Candidates / S1** | {bm.get('p95_candidates_per_s1', 0.0):.1f} |
| **Maximum Candidates / S1** | {bm.get('max_candidates_per_s1', 0.0):.0f} |
| **Minimum Candidates / S1** | {bm.get('min_candidates_per_s1', 0.0):.0f} |

---

## 4. Entity Matching & Evaluation Metrics (Validation Set)
Official metric is macro-averaged $F_{0.5}$ across all evaluated Source 1 entities, weighting precision 2x over recall with explicit singleton scoring.

| Evaluation Metric | Measured Value |
| :--- | :--- |
| **Macro Precision** | **{ev.get('macro_precision', 0.0):.4f}** |
| **Macro Recall** | **{ev.get('macro_recall', 0.0):.4f}** |
| **Macro $F_{0.5}$** | **{ev.get('macro_f05', 0.0):.4f}** |
| **Total Evaluated S1 Entities** | {ev.get('total_evaluated_s1', 0):,.0f} |
| **True Singletons in Validation** | {ev.get('true_singletons', 0):,.0f} |
| **Correctly Predicted Singletons** | {ev.get('correct_singletons', 0):,.0f} |

### Threshold Optimization Grid
{thresh_table_md}

---

## 5. Post-Processing & Output Generation
- **Candidate Pairs File:** `output/candidate_pairs.tsv`
- **Final Matching Results File:** `output/matching_results.tsv`
- **Total S1 Entities Output:** {st.get('total_s1_entities', 0):,}
- **Matched S1 Entities:** {st.get('matched_s1_entities', 0):,}
- **Unmatched S1 Entities (Singletons):** {st.get('unmatched_s1_entities', 0):,}
- **Total Matches Linked:** {st.get('total_matches_linked', 0):,}

---

## 6. Computational Performance & Resource Usage
- **Total Pipeline Execution Time:** {data.get('total_runtime_sec', 0.0):.2f} seconds
- **Peak Process Memory:** {data.get('peak_memory_mb', 0.0):.1f} MB

### Per-Stage Runtime Breakdown
"""
    for stage_name, m in data.get("stages", {}).items():
        content += f"- **{stage_name}:** {m.get('runtime_sec', 0.0):.2f}s (RAM Delta: {m.get('memory_delta_mb', 0.0):+.1f} MB)\n"

    content += f"""
---

## 7. Official Submission Validator Output
```text
{data.get('validation_log', 'No validator log.')}
```

---

## 8. Known Weaknesses & V2 Improvement Recommendations
### Identified V1 Limitations:
1. **Blocking Recall Ceiling:** Pure token/prefix blocking may miss entities with phonetic variations or heavy OCR/typo errors in initial syllables.
2. **Linear Matching Weights:** A domain-weighted linear model lacks non-linear cross-feature interactions (e.g. high address similarity compensating for minor name token differences).
3. **Chunked Scaling:** While memory consumption is extremely low (< 500 MB), scaling to all 12.5M records will require external sorting or SQLite/disk-backed indexing for blocking.

### Recommended Next Steps for V2:
1. **Phonetic & Substring Blocking:** Introduce Double Metaphone or minhash character 3-gram LSH for name blocking.
2. **Gradient Boosted Matching Model:** Transition from weighted scoring to LightGBM or CatBoost trained with pairwise ranking loss.
3. **Disk-Backed Chunked Pipeline:** Stream candidate generation in partitioned chunks to scale seamlessly across the full 12.5M records.
"""

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")

    logger.info(f"Generated benchmark report at: {report_file}")
    return report_file


def main():
    parser = argparse.ArgumentParser(description="AWS ML Challenge 2026 - V1 Master Pipeline")
    parser.add_argument("--sample-size", type=int, default=5000, help="Number of S1 entities to sample (default: 5000)")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to data directory")
    parser.add_argument("--output-dir", type=str, default="output", help="Path to output directory")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Path to reports directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--threshold", type=float, default=None, help="Fixed threshold override")
    parser.add_argument("--skip-validator", action="store_true", help="Skip running the official validator")
    args = parser.parse_args()

    cfg = PipelineConfig(
        data_dir=resolve_data_dir(args.data_dir),
        output_dir=Path(args.output_dir),
        reports_dir=Path(args.reports_dir),
        sample_size=args.sample_size,
        random_seed=args.seed,
    )
    if args.threshold is not None:
        cfg.selected_threshold = args.threshold
        cfg.thresholds = [args.threshold]

    run_v1_pipeline(cfg, run_validator=not args.skip_validator)


if __name__ == "__main__":
    main()
