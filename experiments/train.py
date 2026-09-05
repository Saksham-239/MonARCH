"""
PI-MDCNet v4 -- Training Loop & LOBO Cross-Validation
=====================================================

Handles:
- Dataset/DataLoader construction from data_loader DataFrames
- Feature normalization (StandardScaler, fit on train only)
- PI-MDCNet SLAC training loop with combined loss
- LOBO (Leave-One-Battery-Out) cross-validation
- Baseline model training & evaluation
- Metric computation (RMSE, MAE, MAPE for SoH/SoC/RUL)
- Per-model wall-clock time logging (spec correction #6)
- R_t correlation diagnostic (spec correction #4)

Usage:
    python -m experiments.train --mode benchmark
    python -m experiments.train --mode quick
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.data_loader import load_all_cells, build_lobo_folds
from src.models.pi_mdcnet import PIMDCNet, SLACConfig, SLACLoss
from src.models.baselines import (
    BaselineModel, PredictionResult, BASELINE_REGISTRY,
    get_all_baselines, FEATURE_COLS,
)
from experiments.protocol import BENCHMARK_EPOCHS, BENCHMARK_SEEDS, benchmark_folds

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

SEED = BENCHMARK_SEEDS[0]
N_SEEDS = len(BENCHMARK_SEEDS)

# Feature column groups for PI-MDCNet input tensors
ELEC_COLS = [
    "v_mean", "v_min", "v_max", "v_std", "v_slope",
    "i_mean", "i_std",
    "dq_dv_peak_v", "dq_dv_peak_mag",
    "v_ocv", "r0", "r1", "c1",
]

THERM_COLS = ["t_mean", "t_max", "t_std", "t_rise"]

# ``throughput_cycle_ah`` is a numerical re-integration of capacity on these
# full-discharge records, so it is near-identical to the SoH label and unsafe.
STRESSOR_COLS = ["mean_temp_cycle", "c_rate"]

REST_FEATURE_COLS = ["rest_duration_hours", "soc_at_rest_onset", "ambient_temp_rest"]

ALL_INPUT_COLS = ELEC_COLS + THERM_COLS + STRESSOR_COLS + REST_FEATURE_COLS


# =============================================================================
# FEATURE NORMALIZATION
# =============================================================================

class FeatureScaler:
    """StandardScaler for PI-MDCNet feature groups.

    Computes mean/std from training data and applies z-normalization
    separately to each feature group. Must be fit on training data only,
    then applied to both train and test.
    """

    def __init__(self) -> None:
        self.means: dict[str, np.ndarray] = {}
        self.stds: dict[str, np.ndarray] = {}

    def fit(self, train_dfs: list[pd.DataFrame]) -> "FeatureScaler":
        """Fit scaler on concatenated training data."""
        combined = pd.concat(train_dfs, ignore_index=True).fillna(0.0)
        for group_name, cols in [
            ("elec", ELEC_COLS),
            ("therm", THERM_COLS),
            ("stressor", STRESSOR_COLS),
            ("rest", REST_FEATURE_COLS),
        ]:
            vals = combined[cols].values.astype(np.float64)
            self.means[group_name] = vals.mean(axis=0).astype(np.float32)
            std = vals.std(axis=0).astype(np.float32)
            std[std < 1e-8] = 1.0  # prevent div-by-zero
            self.stds[group_name] = std
        return self

    def transform_tensor(self, t: torch.Tensor, group: str) -> torch.Tensor:
        """Apply z-normalization to a tensor."""
        mean = torch.tensor(self.means[group], dtype=t.dtype, device=t.device)
        std = torch.tensor(self.stds[group], dtype=t.dtype, device=t.device)
        return (t - mean) / std


# =============================================================================
# METRICS
# =============================================================================

@dataclass
class EvalMetrics:
    """Evaluation metrics for a single model on a single fold."""
    model_name: str
    fold_id: str  # test cell ID
    seed: int
    # SoH metrics
    soh_rmse: float = np.nan
    soh_mae: float = np.nan
    soh_mape: float = np.nan
    # SoC metrics
    soc_rmse: float = np.nan
    soc_mae: float = np.nan
    # RUL metrics
    rul_mae: float = np.nan
    rul_rmse: float = np.nan
    # Timing
    train_wall_clock_s: float = 0.0
    # Diagnostic: R_t correlation with delta_SoH_post_rest (correction #4)
    rt_corr_post_rest: float = np.nan


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    """Compute metrics for valid, non-degenerate targets.

    A constant target carries no estimation signal. Returning NaN avoids
    presenting a score on the all-zero end-of-discharge SoC label as model
    quality.
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return {
            f"{prefix}rmse": np.nan,
            f"{prefix}mae": np.nan,
            f"{prefix}mape": np.nan,
        }
    yt = y_true[mask]
    yp = y_pred[mask]
    if np.ptp(yt) <= 1e-12:
        return {
            f"{prefix}rmse": np.nan,
            f"{prefix}mae": np.nan,
            f"{prefix}mape": np.nan,
        }
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae = float(np.mean(np.abs(yt - yp)))
    # MAPE: compute only over non-trivial denominators (|yt| > 0.05) to avoid explosion
    # when target reaches 0 (e.g. SoC at end of discharge, RUL at EOL)
    nz_mask = np.abs(yt) > 0.05
    if nz_mask.sum() > 0:
        mape = float(np.mean(np.abs(yt[nz_mask] - yp[nz_mask]) / np.abs(yt[nz_mask])) * 100)
    else:
        mape = np.nan
    return {
        f"{prefix}rmse": rmse,
        f"{prefix}mae": mae,
        f"{prefix}mape": mape,
    }


# =============================================================================
# PYTORCH DATASET FOR PI-MDCNet
# =============================================================================

class CycleSequenceDataset(Dataset):
    """PyTorch Dataset wrapping a cell's per-cycle DataFrame into tensors.

    Each sample is the ENTIRE sequence for one cell (all cycles).
    For LOBO, each fold trains on multiple cells = multiple sequences.

    If a FeatureScaler is provided, all features are z-normalized.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        scaler: FeatureScaler | None = None,
    ) -> None:
        self.n = len(df)

        # Fill NaN for tensor conversion
        df_filled = df.fillna(0.0)

        self.x_elec = torch.tensor(
            df_filled[ELEC_COLS].values, dtype=torch.float32
        )
        self.x_therm = torch.tensor(
            df_filled[THERM_COLS].values, dtype=torch.float32
        )
        self.s_t = torch.tensor(
            df_filled[STRESSOR_COLS].values, dtype=torch.float32
        )
        self.rest_features = torch.tensor(
            df_filled[REST_FEATURE_COLS].values, dtype=torch.float32
        )

        # Apply normalization if scaler is provided
        if scaler is not None:
            self.x_elec = scaler.transform_tensor(self.x_elec, "elec")
            self.x_therm = scaler.transform_tensor(self.x_therm, "therm")
            self.s_t = scaler.transform_tensor(self.s_t, "stressor")
            self.rest_features = scaler.transform_tensor(self.rest_features, "rest")

        self.rest_flag = torch.tensor(
            df_filled["rest_flag"].values, dtype=torch.float32
        ).unsqueeze(-1)

        # Normalized cycle index
        max_idx = max(self.n - 1, 1)
        self.cycle_idx = torch.arange(self.n, dtype=torch.float32).unsqueeze(-1) / max_idx

        # Targets (NOT normalized -- these are in original SoH/SoC/RUL units)
        self.soh = torch.tensor(df["soh"].values, dtype=torch.float32).unsqueeze(-1)
        self.soc = torch.tensor(
            df_filled["soc_eod"].values, dtype=torch.float32
        ).unsqueeze(-1)
        soc_raw = df["soc_eod"].values.astype(np.float64)
        soc_is_informative = (
            np.isfinite(soc_raw).all() and np.ptp(soc_raw) > 1e-12
        )
        self.soc_mask = torch.full(
            (self.n, 1), float(soc_is_informative), dtype=torch.float32
        )

        rul_raw = df["rul"].values.copy()
        self.rul_mask = torch.tensor(
            ~np.isnan(rul_raw), dtype=torch.float32
        ).unsqueeze(-1)
        rul_raw[np.isnan(rul_raw)] = 0.0
        self.rul = torch.tensor(rul_raw, dtype=torch.float32).unsqueeze(-1)

    def __len__(self) -> int:
        return 1  # one sequence per cell

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x_elec": self.x_elec,         # (T, elec_dim)
            "x_therm": self.x_therm,        # (T, therm_dim)
            "s_t": self.s_t,                # (T, stressor_dim)
            "rest_features": self.rest_features,  # (T, 3)
            "rest_flag": self.rest_flag,     # (T, 1)
            "cycle_idx": self.cycle_idx,    # (T, 1)
            "soh": self.soh,                # (T, 1)
            "soc": self.soc,                # (T, 1)
            "soc_mask": self.soc_mask,      # (T, 1)
            "rul": self.rul,                # (T, 1)
            "rul_mask": self.rul_mask,      # (T, 1)
        }


def collate_sequences(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Custom collate: pad sequences to max length in batch."""
    keys = batch[0].keys()
    max_T = max(b["x_elec"].shape[0] for b in batch)

    result = {}
    for key in keys:
        tensors = []
        for b in batch:
            t = b[key]
            T = t.shape[0]
            if T < max_T:
                pad_shape = list(t.shape)
                pad_shape[0] = max_T - T
                t = torch.cat([t, torch.zeros(*pad_shape)], dim=0)
            tensors.append(t.unsqueeze(0))
        result[key] = torch.cat(tensors, dim=0)  # (B, T, *)

    return result


# =============================================================================
# PI-MDCNet TRAINING
# =============================================================================

def train_pimdcnet(
    train_dfs: list[pd.DataFrame],
    cfg: SLACConfig | None = None,
    n_epochs: int = BENCHMARK_EPOCHS,
    device: str = "auto",
    seed: int = 42,
    model_cls: type[PIMDCNet] = PIMDCNet,
) -> tuple[PIMDCNet, FeatureScaler, float]:
    """Train PI-MDCNet on a list of cell DataFrames.

    Parameters
    ----------
    train_dfs : list of DataFrames
        Per-cell training data.
    cfg : SLACConfig, optional
    n_epochs : int
    device : str
        "auto", "cuda", or "cpu".
    seed : int
    model_cls : type[PIMDCNet]
        Model class to instantiate (defaults to PIMDCNet).

    Returns
    -------
    model : PIMDCNet
        Trained model.
    scaler : FeatureScaler
        Fitted feature scaler (needed at inference).
    wall_clock : float
        Training wall-clock time in seconds.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = cfg or SLACConfig()
    model = model_cls(cfg).to(device)
    loss_fn = SLACLoss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Fit scaler on training data
    scaler = FeatureScaler().fit(train_dfs)

    # Build datasets with normalization
    datasets = [CycleSequenceDataset(df, scaler=scaler) for df in train_dfs]
    # Combine into a DataLoader with batch_size=1 (no zero padding, exact cell length)
    combined = torch.utils.data.ConcatDataset(datasets)
    loader = DataLoader(
        combined,
        batch_size=1,
        shuffle=True,
    )

    t0 = time.monotonic()
    model.train()

    best_loss = float("inf")
    patience_counter = 0
    patience = 20  # early stopping patience

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            # Move to device
            batch_dev = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            outputs = model(
                x_elec=batch_dev["x_elec"],
                x_therm=batch_dev["x_therm"],
                s_t=batch_dev["s_t"],
                rest_features=batch_dev["rest_features"],
                rest_flag=batch_dev["rest_flag"],
                cycle_idx=batch_dev["cycle_idx"],
            )

            targets = {
                "soh": batch_dev["soh"],
                "soc": batch_dev["soc"],
                "rul": batch_dev["rul"],
            }

            losses = loss_fn(
                outputs,
                targets,
                rul_mask=batch_dev["rul_mask"],
                soc_mask=batch_dev["soc_mask"],
            )
            losses["total"].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += losses["total"].item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        # Early stopping
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch > 30:
                logger.info("Early stopping at epoch %d (no improvement for %d epochs)",
                            epoch + 1, patience)
                break

        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info(
                "Epoch %d/%d: loss=%.6f, lr=%.6f",
                epoch + 1, n_epochs, avg_loss, scheduler.get_last_lr()[0],
            )

    wall_clock = time.monotonic() - t0
    logger.info("PI-MDCNet training complete: %.1fs", wall_clock)
    return model, scaler, wall_clock


def predict_pimdcnet(
    model: PIMDCNet,
    test_df: pd.DataFrame,
    scaler: FeatureScaler,
    device: str = "auto",
) -> PredictionResult:
    """Run inference on test cell data."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = CycleSequenceDataset(test_df, scaler=scaler)
    batch = ds[0]
    # Add batch dim
    batch_dev = {k: v.unsqueeze(0).to(device) for k, v in batch.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(
            x_elec=batch_dev["x_elec"],
            x_therm=batch_dev["x_therm"],
            s_t=batch_dev["s_t"],
            rest_features=batch_dev["rest_features"],
            rest_flag=batch_dev["rest_flag"],
            cycle_idx=batch_dev["cycle_idx"],
        )

    # Extract predictions
    soh = outputs["soh"][0, :, 0].cpu().numpy()
    soc = outputs["soc"][0, :, 0].cpu().numpy()
    rul = outputs["rul"][0, :, 0].cpu().numpy()

    return PredictionResult(soh=soh, soc=soc, rul=rul)


# =============================================================================
# LOBO CROSS-VALIDATION BENCHMARK
# =============================================================================

def run_benchmark(
    data_dir: Path | str = "data/raw",
    output_dir: Path | str = "artifacts",
    n_epochs: int = 100,
    n_seeds: int = N_SEEDS,
    device: str = "auto",
) -> pd.DataFrame:
    """Run full LOBO cross-validation benchmark across all models.

    For each fold (leave one cell out):
    1. Train all baselines on remaining cells
    2. Train PI-MDCNet on remaining cells (with feature normalization)
    3. Evaluate on held-out cell
    4. Compute metrics

    Results are saved to CSV and returned.

    Parameters
    ----------
    data_dir : Path
        Directory containing .mat files.
    output_dir : Path
        Directory for output artifacts.
    n_epochs : int
        Training epochs for PI-MDCNet.
    n_seeds : int
        Number of random seeds per fold.
    device : str
        PyTorch device.

    Returns
    -------
    pd.DataFrame
        All metrics across models, folds, and seeds.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all cells
    logger.info("Loading all cells from %s...", data_dir)
    cell_data = load_all_cells(data_dir)
    cell_ids = list(cell_data.keys())
    logger.info("Loaded %d cells: %s", len(cell_ids), cell_ids)

    # Build LOBO folds
    folds = benchmark_folds(cell_ids)
    logger.info("Built %d LOBO folds", len(folds))

    all_metrics: list[dict[str, Any]] = []

    for fold_idx, (train_ids, test_id) in enumerate(folds):
        logger.info("=" * 60)
        logger.info("FOLD %d: test=%s, train=%s", fold_idx, test_id, train_ids)
        logger.info("=" * 60)

        train_dfs = [cell_data[cid] for cid in train_ids]
        train_combined = pd.concat(train_dfs, ignore_index=True)
        test_df = cell_data[test_id]

        # --- Baselines ---
        for model_name, model_cls in BASELINE_REGISTRY.items():
            for seed_idx in range(n_seeds):
                seed = BENCHMARK_SEEDS[seed_idx]
                np.random.seed(seed)

                model = model_cls()
                t0 = time.monotonic()
                model.fit(train_combined, random_state=seed)
                wall_clock = time.monotonic() - t0

                result = model.predict(test_df)
                result.wall_clock_seconds = wall_clock

                # Compute metrics
                soh_m = compute_metrics(test_df["soh"].values, result.soh, "soh_")
                soc_m = compute_metrics(test_df["soc_eod"].values, result.soc, "soc_")
                rul_m = compute_metrics(test_df["rul"].values, result.rul, "rul_")

                metrics_row = {
                    "model": model_name,
                    "fold": test_id,
                    "seed": seed,
                    "train_wall_clock_s": wall_clock,
                    **soh_m, **soc_m, **rul_m,
                }
                all_metrics.append(metrics_row)

                logger.info(
                    "  %s (seed=%d): SoH RMSE=%.4f, MAE=%.4f",
                    model_name, seed,
                    soh_m.get("soh_rmse", np.nan),
                    soh_m.get("soh_mae", np.nan),
                )

        # --- PI-MDCNet ---
        for seed_idx in range(n_seeds):
            seed = BENCHMARK_SEEDS[seed_idx]
            cfg = SLACConfig()

            model, scaler, wall_clock = train_pimdcnet(
                train_dfs, cfg=cfg, n_epochs=n_epochs,
                device=device, seed=seed,
            )

            result = predict_pimdcnet(model, test_df, scaler=scaler, device=device)

            soh_m = compute_metrics(test_df["soh"].values, result.soh, "soh_")
            soc_m = compute_metrics(test_df["soc_eod"].values, result.soc, "soc_")
            rul_m = compute_metrics(test_df["rul"].values, result.rul, "rul_")

            # Diagnostic: R_t correlation with post-rest SoH delta (correction #4)
            rt_corr = _compute_rt_correlation(model, test_df, scaler, device)

            metrics_row = {
                "model": "PI-MDCNet",
                "fold": test_id,
                "seed": seed,
                "train_wall_clock_s": wall_clock,
                "rt_corr_post_rest": rt_corr,
                **soh_m, **soc_m, **rul_m,
            }
            all_metrics.append(metrics_row)

            logger.info(
                "  PI-MDCNet (seed=%d): SoH RMSE=%.4f, MAE=%.4f, R_t corr=%.4f",
                seed,
                soh_m.get("soh_rmse", np.nan),
                soh_m.get("soh_mae", np.nan),
                rt_corr,
            )

    # Save results
    results_df = pd.DataFrame(all_metrics)
    results_path = output_dir / "benchmark_results.csv"
    results_df.to_csv(results_path, index=False)
    results_df.to_csv(output_dir / "comparison_table.csv", index=False)
    logger.info("Results saved to %s", results_path)

    # Print summary table
    _print_summary_table(results_df)

    return results_df


def _compute_rt_correlation(
    model: PIMDCNet,
    test_df: pd.DataFrame,
    scaler: FeatureScaler,
    device: str,
) -> float:
    """Compute correlation between R_t and delta_SoH_post_rest.

    Diagnostic metric (spec correction #4): checks whether the model's
    predicted R_t correlates with actual SoH recovery after rest events.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = CycleSequenceDataset(test_df, scaler=scaler)
    batch = ds[0]
    batch_dev = {k: v.unsqueeze(0).to(device) for k, v in batch.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(
            x_elec=batch_dev["x_elec"],
            x_therm=batch_dev["x_therm"],
            s_t=batch_dev["s_t"],
            rest_features=batch_dev["rest_features"],
            rest_flag=batch_dev["rest_flag"],
            cycle_idx=batch_dev["cycle_idx"],
        )

    R_t = outputs["R_t"][0, :, 0].cpu().numpy()
    rest_flags = test_df["rest_flag"].values

    # Delta SoH after rest: SoH[t] - SoH[t-1] on rest-flagged cycles
    soh = test_df["soh"].values
    delta_soh = np.diff(soh, prepend=soh[0])

    # Only look at rest cycles
    rest_mask = rest_flags == 1
    if rest_mask.sum() < 2:
        return np.nan

    r_rest = R_t[rest_mask]
    d_rest = delta_soh[rest_mask]

    # Pearson correlation
    if np.std(r_rest) < 1e-8 or np.std(d_rest) < 1e-8:
        return 0.0

    corr = float(np.corrcoef(r_rest, d_rest)[0, 1])
    return corr


def _print_summary_table(results_df: pd.DataFrame) -> None:
    """Print aggregated results table (spec correction #5).

    Aggregation order: mean across seeds per fold first, then across N=4 cells.
    """
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS (mean +/- std across seeds, aggregated per fold)")
    print("=" * 80)

    # Aggregate: mean across seeds per (model, fold), then mean across folds
    agg = results_df.groupby(["model", "fold"]).agg({
        "soh_rmse": "mean",
        "soh_mae": "mean",
        "rul_mae": "mean",
        "train_wall_clock_s": "mean",
    }).reset_index()

    # Then aggregate across folds
    summary = agg.groupby("model").agg({
        "soh_rmse": ["mean", "std"],
        "soh_mae": ["mean", "std"],
        "rul_mae": ["mean", "std"],
        "train_wall_clock_s": ["mean"],
    })

    print(f"\n{'Model':<20} {'SoH RMSE':>14} {'SoH MAE':>14} {'RUL MAE':>14} {'Time (s)':>10}")
    print("-" * 72)

    for model_name in summary.index:
        row = summary.loc[model_name]
        soh_rmse_m = row[("soh_rmse", "mean")]
        soh_rmse_s = row[("soh_rmse", "std")]
        soh_mae_m = row[("soh_mae", "mean")]
        soh_mae_s = row[("soh_mae", "std")]
        rul_mae_m = row[("rul_mae", "mean")]
        rul_mae_s = row[("rul_mae", "std")]
        time_m = row[("train_wall_clock_s", "mean")]

        soh_rmse_str = f"{soh_rmse_m:.4f}+/-{soh_rmse_s:.4f}" if not np.isnan(soh_rmse_m) else "N/A"
        soh_mae_str = f"{soh_mae_m:.4f}+/-{soh_mae_s:.4f}" if not np.isnan(soh_mae_m) else "N/A"
        rul_mae_str = f"{rul_mae_m:.1f}+/-{rul_mae_s:.1f}" if not np.isnan(rul_mae_m) else "N/A"

        print(f"{model_name:<20} {soh_rmse_str:>14} {soh_mae_str:>14} {rul_mae_str:>14} {time_m:>10.1f}")

    print("=" * 72)
    print("Note: n=4 cells (Wilcoxon signed-rank has limited power at n=4)")
    print("=" * 72)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main() -> None:
    """CLI entry point for training and evaluation."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="PI-MDCNet v4 Training & Evaluation")
    parser.add_argument(
        "--mode", choices=["benchmark", "quick"], default="quick",
        help="benchmark: full LOBO benchmark; quick: fast smoke test",
    )
    parser.add_argument("--data-dir", default="data/raw", help="Path to .mat files")
    parser.add_argument("--output-dir", default="artifacts", help="Output directory")
    parser.add_argument("--epochs", type=int, default=BENCHMARK_EPOCHS, help="PI-MDCNet training epochs")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--device", default="auto", help="PyTorch device")
    args = parser.parse_args()

    if args.mode == "quick":
        logger.info("Quick mode: 10 epochs, 1 seed")
        run_benchmark(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            n_epochs=10,
            n_seeds=1,
            device=args.device,
        )
    else:
        run_benchmark(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            n_epochs=args.epochs,
            n_seeds=args.seeds,
            device=args.device,
        )


if __name__ == "__main__":
    main()
