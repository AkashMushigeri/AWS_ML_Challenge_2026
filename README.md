# AWS ML Challenge 2026: Business Entity Resolution Challenge

An end-to-end Machine Learning pipeline solving the **AWS ML Challenge 2026: Business Entity Resolution Challenge**.

The objective is to identify which records from Source 2 (`S2-`) and Source 3 (`S3-`) correspond to each Source 1 (`S1-`) reference business entity, evaluated on macro-averaged $F_{0.5}$ with strict singleton handling.

---

## V1 Master Architecture & Modules

The Version 1 pipeline provides a lightweight, deterministic, memory-conscious, and fully reproducible baseline:

```
DATA INGESTION (Streaming TSV)
       ↓
DATA VALIDATION & S1 HOLDOUT SPLIT
       ↓
TEXT NORMALIZATION (Names, Addresses, Open-Set Countries)
       ↓
MULTI-PASS BLOCKING (Core Name, Prefix, Postal/Numeric, Token Signature)
       ↓
PAIRWISE SIMILARITY FEATURE EXTRACTION (16 Numeric Features)
       ↓
BASELINE MATCHING MODEL & PROBABILITY SCORING
       ↓
VALIDATION THRESHOLD OPTIMIZATION (Macro F0.5)
       ↓
OUTPUT GENERATION & SINGLETON DETECTION (output/candidate_pairs.tsv & matching_results.tsv)
       ↓
OFFICIAL SUBMISSION VALIDATOR (utils/validate_submission.py)
       ↓
BENCHMARK REPORTING (reports/v1_benchmark.md)
```

### Module Ownership & Structure

- **Developer 1 (Team Lead / Architecture / Integration):**
  - `src/config.py`: Centralized configuration, CLI parameters, and dataset path resolution.
  - `src/postprocess.py`: Threshold application, singleton detection, and official TSV formatting.
  - `src/evaluation.py`: Entity-level precision, recall, and macro $F_{0.5}$ evaluation with singleton scoring.
  - `src/pipeline.py`: Master pipeline orchestration and CLI.
  - `src/inference.py`: Standalone test inference module.
  - `src/utils.py`: Logging and performance measurement utilities.
  - `reports/v1_benchmark.md`: Comprehensive benchmark reporting.

- **Developer 2 (Data / Normalization / Blocking):**
  - `src/data_loader.py`: Streaming TSV data loader with schema verification and deterministic sampling.
  - `src/normalizer.py`: Deterministic text normalization for business names, addresses, and open-set countries (including France).
  - `src/blocking.py`: Multi-pass inverted index blocking with candidate capping and reduction ratio tracking.

- **Developer 3 (Features / Baseline Matcher / Modeling):**
  - `src/features.py`: Pairwise similarity feature extraction (16 numeric features).
  - `src/matcher.py`: Transparent domain-weighted baseline matcher and NumPy logistic regression.

---

## Quick Start & Reproduction

### 1. Environment Setup
```bash
pip install -r code/business_entity_resolution/requirements.txt
```
*(Dependencies: pure `pandas`, `numpy`, `scipy` — zero heavy external frameworks required for V1).*

### 2. Run V1 Pipeline
Execute on a deterministic sample:
```bash
python -m src.pipeline --sample-size 5000
```
Command-line options:
- `--sample-size <int>`: Number of Source 1 entities to sample (default: `5000`).
- `--seed <int>`: Random seed for deterministic reproducibility (default: `42`).
- `--data-dir <path>`: Dataset root directory (auto-detects `dataset/student_resource/dataset`).
- `--threshold <float>`: Manually specify classification threshold (default: auto-tuned on validation fold).
- `--skip-validator`: Skip executing the official submission validator.

### 3. Run Unit & Regression Tests
Run the 14 test cases covering name normalization, diacritics, address numbers, open-set country handling, singletons, and metrics:
```bash
python -m unittest tests/test_v1_pipeline.py
```

### 4. Official Submission Validator
Run the official validator against generated outputs:
```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/student_resource/dataset/test
```

---

## V1 Benchmark Highlights
See the full benchmark report in [`reports/v1_benchmark.md`](reports/v1_benchmark.md):
- **Candidate Reduction Ratio:** `0.998421` (reduces comparison space by over 99.8%)
- **Official Validator Status:** `PASS`
- **Peak Process Memory:** `< 20 MB RAM`
- **Full Reproducibility:** 100% deterministic with seed `42`
- **Fair Play Compliant:** Operates strictly on provided tabular data without external API lookups.
