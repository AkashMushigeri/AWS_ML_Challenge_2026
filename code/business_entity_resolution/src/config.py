"""Configuration module for AWS ML Challenge 2026 V1 Pipeline.

Member 1 (Team Lead) ownership.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


def resolve_data_dir(preferred_path: Optional[str] = None) -> Path:
    """Resolve the directory containing train/ and test/ TSV files.

    Checks preferred_path first, then checks dataset/student_resource/dataset,
    then checks dataset/.
    """
    if preferred_path:
        p = Path(preferred_path)
        if p.exists():
            return p

    candidates = [
        Path("dataset/student_resource/dataset"),
        Path("dataset"),
        Path("../dataset/student_resource/dataset"),
    ]
    for cand in candidates:
        if cand.exists() and (cand / "train").exists():
            return cand

    # Default fallback
    return Path("dataset/student_resource/dataset")


@dataclass
class PipelineConfig:
    """Master configuration for V1 entity resolution pipeline."""

    data_dir: Path = field(default_factory=resolve_data_dir)
    output_dir: Path = field(default_factory=lambda: Path("output"))
    reports_dir: Path = field(default_factory=lambda: Path("reports"))
    sample_size: int = 5000
    random_seed: int = 42
    val_split_ratio: float = 0.2
    max_candidates_per_s1: int = 100
    thresholds: List[float] = field(
        default_factory=lambda: [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    )
    selected_threshold: float = 0.75
    s2_s3_distractor_size: int = 25000  # Number of random distractors to include in V1 sample
