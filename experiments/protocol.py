"""
Centralized Experimental Protocol & Evaluation Registry
=========================================================
Strict, centralized protocol for publication-grade reproducibility:
- Deterministic 5-seed list: [42, 100, 2024, 2025, 2026]
- End-of-Life (EOL) thresholds: 70% and 80% nominal capacity
- Leave-One-Cell-Out (LOCO) cross-validation folds across NASA, CALCE, and Oxford
- Canonical metric calculation (RMSE, MAE, MAPE, R2, RUL_AE, Alpha-Lambda Accuracy)
- Causal cycle normalization scale (1000.0) preventing lifespan leakage
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import r2_score


# ═══════════════════════════════════════════════════════════════════════════════
# DETERMINISTIC PROTOCOL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

BENCHMARK_SEEDS: list[int] = [42, 100, 2024, 2025, 2026]
SEEDS: list[int] = BENCHMARK_SEEDS

BENCHMARK_EPOCHS: int = 100
QUICK_EPOCHS: int = 10

# End-of-life conventions: 30% fade (1.4 Ah for 2.0 Ah rated) or 20% fade (1.6 Ah)
EOL_FADE_FRACTION_70: float = 0.30
EOL_FADE_FRACTION_80: float = 0.20
EOL_FADE_FRACTION: float = EOL_FADE_FRACTION_80
DEFAULT_EOL_FADE_FRACTION: float = EOL_FADE_FRACTION_80

# Fixed reference cycle scale for causal normalization across all cells.
# At cycle t, cycle_idx = t / CYCLE_NORMALIZATION_SCALE.
# This prevents leaking the unknown total cell lifespan to test inference.
CYCLE_NORMALIZATION_SCALE: float = 1000.0

# RUL rolling damage window k
RUL_WINDOW_K: int = 10


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICAL & PROGNOSTIC METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def calc_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def calc_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def calc_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (in %). Avoids division by zero."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > 1e-6)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def calc_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination (R^2). Returns NaN if variance is zero."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return np.nan
    yt, yp = y_true[mask], y_pred[mask]
    if np.ptp(yt) < 1e-9:
        return np.nan
    return float(r2_score(yt, yp))


def calc_alpha_lambda_accuracy(
    rul_true: np.ndarray,
    rul_pred: np.ndarray,
    alpha: float = 0.20,
    lambda_horizon: float = 0.50,
) -> float:
    """Compute Alpha-Lambda Prognostic Accuracy.

    Verifies whether RUL predictions fall within an alpha error bound
    (e.g., +/- 20% of true RUL or total lifespan) at prediction time
    horizon lambda (e.g., from 50% of life onwards).

    Parameters
    ----------
    rul_true : np.ndarray
        True RUL in cycles.
    rul_pred : np.ndarray
        Predicted RUL in cycles.
    alpha : float
        Acceptable error fraction (default 0.20 for 20%).
    lambda_horizon : float
        Fraction of trajectory elapsed (default 0.50 for second half of life).

    Returns
    -------
    float
        Fraction of predictions within the [1 - alpha, 1 + alpha] bound.
    """
    mask = np.isfinite(rul_true) & np.isfinite(rul_pred)
    if mask.sum() == 0:
        return np.nan

    yt = rul_true[mask]
    yp = rul_pred[mask]
    n = len(yt)

    # Consider points where true RUL is in the evaluation window
    start_idx = int(n * (1.0 - lambda_horizon))
    if start_idx >= n:
        start_idx = 0

    sub_true = yt[start_idx:]
    sub_pred = yp[start_idx:]

    if len(sub_true) == 0:
        return np.nan

    # Error bound: +/- max(alpha * sub_true, 5 cycles)
    bound = np.maximum(alpha * sub_true, 5.0)
    in_bound = np.abs(sub_pred - sub_true) <= bound
    return float(np.mean(in_bound))


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    """Compute all standard evaluation metrics for an estimation array."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {
            f"{prefix}rmse": np.nan,
            f"{prefix}mae": np.nan,
            f"{prefix}mape": np.nan,
            f"{prefix}r2": np.nan,
        }

    return {
        f"{prefix}rmse": calc_rmse(y_true, y_pred),
        f"{prefix}mae": calc_mae(y_true, y_pred),
        f"{prefix}mape": calc_mape(y_true, y_pred),
        f"{prefix}r2": calc_r2(y_true, y_pred),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION FOLD REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_folds(cell_ids: Sequence[str]) -> list[tuple[list[str], str]]:
    """Return canonical Leave-One-Cell-Out (LOCO) folds for a cell collection."""
    c_list = list(cell_ids)
    folds = []
    for test_cell in c_list:
        train_cells = [c for c in c_list if c != test_cell]
        folds.append((train_cells, test_cell))
    return folds


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT PROTOCOL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentProtocol:
    """Centralized configuration container for reproducible benchmarks."""
    name: str = "PI-MDCNet-Publication-Protocol"
    seeds: list[int] = field(default_factory=lambda: list(BENCHMARK_SEEDS))
    epochs: int = BENCHMARK_EPOCHS
    eol_fade_fraction: float = DEFAULT_EOL_FADE_FRACTION
    cycle_norm_scale: float = CYCLE_NORMALIZATION_SCALE
    rul_window_k: int = RUL_WINDOW_K
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 20

    def get_folds(self, cell_ids: Sequence[str]) -> list[tuple[list[str], str]]:
        return benchmark_folds(cell_ids)


def extrapolate_damage_latent(
    D_history: np.ndarray,
    soh_0: float = 1.0,
    target_soh: float = 0.80,
    window_k: int = 15,
    alpha: float = 0.15,
) -> float:
    """Non-learned causal linear extrapolation of monotonic damage latent D_t to D_fail.

    Uses exponentially-weighted causal projection of recent damage increments:
    D_fail = soh_0 - target_soh
    Since D_t is strictly monotonic non-decreasing, dot{D}_t > 0.
    RUL = max(0, (D_fail - D_t) / dot{D}_t).
    """
    n = len(D_history)
    D_curr = D_history[-1]
    D_fail = max(0.05, soh_0 - target_soh)

    if D_curr >= D_fail:
        return 0.0

    diffs = np.diff(D_history)
    if len(diffs) == 0:
        return max(0.0, float((D_fail - D_curr) / 0.002))

    # Exponentially weighted causal rate over past window
    k = min(len(diffs), window_k)
    recent_diffs = diffs[-k:]
    weights = (1.0 - alpha) ** np.arange(k)[::-1]
    weights /= weights.sum()
    dot_D = float(np.sum(recent_diffs * weights))

    if dot_D <= 1e-7:
        dot_D = max(D_curr / max(n, 1), 1e-5)

    rul = max(0.0, float((D_fail - D_curr) / dot_D))
    return rul

