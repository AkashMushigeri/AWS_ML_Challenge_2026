# AWS ML Challenge 2026 — V1 Pipeline Benchmark Report

**Pipeline Version:** 1.0 (Lightweight Deterministic Baseline)  
**Execution Timestamp:** 2026-09-26 10:28:01  
**Evaluator / Team Lead:** Developer 1 (Member 1)  

---

## 1. Executive Summary
The V1 baseline pipeline implements an end-to-end, deterministic, memory-conscious entity resolution architecture designed to handle noisy business records across three disparate data sources. Operating on a representative sample of 5000 Source 1 entities with fixed random seed 42, the pipeline successfully generates `candidate_pairs.tsv` and `matching_results.tsv` while strictly adhering to official competition requirements.

---

## 2. Experimental Setup & Configuration
- **Sample Size (S1 Records):** 5000
- **Random Seed:** 42
- **Validation Split Ratio:** 20% Source 1 holdout
- **Selected Decision Threshold:** 0.50
- **Official Submission Validator Status:** **PASS**

---

## 3. Candidate Generation (Blocking) Performance
Candidate generation utilizes a 4-pass inverted index (Core Name, Name Prefix + Country, Postal/Numeric Address + Country, Token Signature + Country) with candidate capping per entity.

| Metric | Measured Value |
| :--- | :--- |
| **Total Source 1 Entities** | 1,000 |
| **Candidate Pool Size (S2 + S3)** | 67,235 |
| **Raw Candidate Pairs (Pre-dedup)** | 458,433 |
| **Deduplicated Candidate Pairs** | 93,723 |
| **Candidate Reduction Ratio** | 0.998606 |
| **Average Candidates / S1** | 93.72 |
| **Median Candidates / S1** | 100.0 |
| **95th Percentile Candidates / S1** | 100.0 |
| **Maximum Candidates / S1** | 100 |
| **Minimum Candidates / S1** | 13 |

---

## 4. Entity Matching & Evaluation Metrics (Validation Set)
Official metric is macro-averaged $F_0.5$ across all evaluated Source 1 entities, weighting precision 2x over recall with explicit singleton scoring.

| Evaluation Metric | Measured Value |
| :--- | :--- |
| **Macro Precision** | **0.7363** |
| **Macro Recall** | **0.3845** |
| **Macro $F_0.5$** | **0.5892** |
| **Total Evaluated S1 Entities** | 1,000 |
| **True Singletons in Validation** | 47 |
| **Correctly Predicted Singletons** | 47 |

### Threshold Optimization Grid
| threshold | macro_f05 | macro_precision | macro_recall | true_singletons | correct_singletons |
| --- | --- | --- | --- | --- | --- |
| 0.50 | 0.5892 | 0.7363 | 0.3845 | 47.0000 | 47.0000 |
| 0.60 | 0.3707 | 0.4964 | 0.2221 | 47.0000 | 47.0000 |
| 0.70 | 0.2790 | 0.3850 | 0.1621 | 47.0000 | 47.0000 |
| 0.75 | 0.2472 | 0.3420 | 0.1447 | 47.0000 | 47.0000 |
| 0.80 | 0.1276 | 0.1720 | 0.0838 | 47.0000 | 47.0000 |
| 0.85 | 0.0676 | 0.0780 | 0.0569 | 47.0000 | 47.0000 |
| 0.90 | 0.0676 | 0.0780 | 0.0569 | 47.0000 | 47.0000 |
| 0.95 | 0.0676 | 0.0780 | 0.0569 | 47.0000 | 47.0000 |

---

## 5. Post-Processing & Output Generation
- **Candidate Pairs File:** `output/candidate_pairs.tsv`
- **Final Matching Results File:** `output/matching_results.tsv`
- **Total S1 Entities Output:** 1,000
- **Matched S1 Entities:** 705
- **Unmatched S1 Entities (Singletons):** 295
- **Total Matches Linked:** 1,291

---

## 6. Computational Performance & Resource Usage
- **Total Pipeline Execution Time:** 42.54 seconds
- **Peak Process Memory:** 212.3 MB

### Per-Stage Runtime Breakdown
- **data_ingestion:** 15.63s (RAM Delta: +31.2 MB)
- **split:** 0.01s (RAM Delta: +0.8 MB)
- **blocking:** 5.20s (RAM Delta: +11.2 MB)
- **features:** 8.00s (RAM Delta: +13.1 MB)
- **matcher:** 0.01s (RAM Delta: +1.3 MB)
- **evaluation:** 8.76s (RAM Delta: -0.2 MB)
- **output:** 4.72s (RAM Delta: -1.2 MB)
- **validator:** 0.20s (RAM Delta: +0.2 MB)

---

## 7. Official Submission Validator Output
```text
ML Challenge 2026 — submission validator
  test dir: output\_val_test_dir
  required S1 entities: 1000
  matching_results.tsv: 1000 rows (295 empty, 705 non-empty).
  candidate_pairs.tsv: 1000 rows (0 empty, 1000 non-empty).

WARNING: ID-existence check is OFF (the default) — not checking that matched/candidate IDs exist in the test set. Every other rule is still checked. Re-run with --check-ids to enable it (needs test_source2/3.tsv; uses more memory). A nonexistent ID only lowers your score, never rejects your submission.
PASS — no blocking issues found. Safe to submit.
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
