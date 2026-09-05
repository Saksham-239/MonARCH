"""Shared, versioned evaluation protocol for benchmark and ablation runs."""

from __future__ import annotations

from collections.abc import Sequence

from src.data_loader import build_lobo_folds


# These defaults are part of the reported protocol. Do not silently change one
# experiment without changing the other.
BENCHMARK_SEEDS: tuple[int, ...] = (42, 43, 44)
BENCHMARK_EPOCHS: int = 100


def benchmark_folds(cell_ids: Sequence[str]) -> list[tuple[list[str], str]]:
    """Return canonical leave-one-battery-out folds for a cell collection."""
    return build_lobo_folds(list(cell_ids))
