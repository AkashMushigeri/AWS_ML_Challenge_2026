"""Utility functions and logging helpers for AWS ML Challenge 2026.

Member 1 (Team Lead) ownership.
"""

import logging
import sys
import time
import tracemalloc
from contextlib import contextmanager
from typing import Generator, Optional


def get_logger(name: str = "AWS_ML_ER_V1") -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def get_current_memory_mb() -> float:
    """Estimate current process memory usage in Megabytes."""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        # Fallback to tracemalloc if psutil is unavailable
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        current, _ = tracemalloc.get_traced_memory()
        return current / (1024 * 1024)


@contextmanager
def measure_stage(stage_name: str, logger: Optional[logging.Logger] = None) -> Generator[dict, None, None]:
    """Context manager to measure runtime and memory for a pipeline stage."""
    log = logger or get_logger("Timer")
    mem_before = get_current_memory_mb()
    t_start = time.perf_counter()
    metrics = {"runtime_sec": 0.0, "memory_delta_mb": 0.0, "mem_peak_mb": 0.0}
    log.info(f"--- Starting: {stage_name} (Initial Memory: {mem_before:.1f} MB) ---")
    try:
        yield metrics
    finally:
        t_end = time.perf_counter()
        mem_after = get_current_memory_mb()
        runtime = t_end - t_start
        delta_mem = mem_after - mem_before
        metrics["runtime_sec"] = runtime
        metrics["memory_delta_mb"] = delta_mem
        metrics["mem_peak_mb"] = mem_after
        log.info(
            f"--- Completed: {stage_name} in {runtime:.2f}s "
            f"(Mem: {mem_after:.1f} MB, Delta: {delta_mem:+.1f} MB) ---"
        )
