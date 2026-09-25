# Business Entity Resolution Pipeline — V1 Baseline

This folder contains the complete, self-contained, reproducible source code for the AWS ML Challenge 2026 Business Entity Resolution solution.

## Architecture

- `src/config.py`: Centralized configuration and path resolution.
- `src/data_loader.py`: Fast streaming TSV ingestion with schema validation.
- `src/normalizer.py`: Deterministic text normalization for business names, addresses, and open-set countries.
- `src/blocking.py`: Multi-pass candidate generation (inverted indexing) with deduplication and candidate capping.
- `src/features.py`: Pairwise similarity feature extraction (name, address, numeric, country, and metadata).
- `src/matcher.py`: Lightweight, transparent baseline matcher.
- `src/evaluation.py`: Official entity-level precision, recall, and macro F0.5 evaluation with singleton handling.
- `src/postprocess.py`: Threshold application, singleton detection, and submission TSV generation.
- `src/pipeline.py`: Master pipeline orchestration and CLI.
- `src/inference.py`: Standalone inference module for generating submission outputs.
- `src/utils.py`: Logging and performance tracking utilities.

## Reproduction Instructions

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Pipeline (Sample Size 1000 - 5000)
From the repository root:
```bash
python -m src.pipeline --sample-size 5000
```

### 3. Run Unit Tests
```bash
python -m unittest tests/test_v1_pipeline.py
```

### 4. Official Validator
```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/student_resource/dataset/test
```
