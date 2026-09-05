"""
PI-MDCNet v4 -- Evaluation & Publication Figures
=================================================

Performs:
1. Statistical testing: Wilcoxon signed-rank test (aggregating seeds per fold first,
   then comparing across N=4 cells, with small-sample power disclaimer).
2. Diagnostic evaluation: Pearson & Spearman correlation of R_t with delta_SoH_post_rest.
3. Publication-grade figures:
   - Figure 1: SoH trajectory across 4 test cells (ground truth vs PI-MDCNet vs baselines)
   - Figure 2: SLAC degradation/regeneration decomposition (D_t monotonic + R_t spikes)
   - Figure 3: Regeneration diagnostic scatter (R_t vs delta_SoH_post_rest)
   - Figure 4: LOBO performance bar chart across all 7 models
   - Figure 5: Training wall-clock time comparison
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import torch

from src.data_loader import load_all_cells
from src.models.pi_mdcnet import PIMDCNet, SLACConfig
from experiments.train import (
    FeatureScaler,
    CycleSequenceDataset,
    predict_pimdcnet,
    train_pimdcnet,
)
from experiments.protocol import BENCHMARK_EPOCHS

logger = logging.getLogger(__name__)

# Aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 1.8,
})

COLORS = {
    "PI-MDCNet": "#1f77b4",       # Deep blue
    "RandomForest": "#2ca02c",    # Green
    "XGBoost": "#ff7f0e",         # Orange
    "VanillaLSTM": "#9467bd",     # Purple
    "StatDecomp": "#d62728",      # Red
    "CoulombCounting": "#8c564b", # Brown
    "EKF_1RC": "#e377c2",         # Pink
    "GroundTruth": "#222222",     # Dark grey / black
}


def run_statistical_analysis(results_df: pd.DataFrame) -> pd.DataFrame:
    """Run paired Wilcoxon tests after seed aggregation and censoring removal.

    Every metric follows the same rule: average seeds within a fold, inner-join
    PI-MDCNet and a baseline by fold, then discard pairs for which either value
    is NaN.  Thus right-censored RUL folds never enter an RUL test, while valid
    SoH folds remain eligible.
    """
    print("\n" + "=" * 80)
    print("STATISTICAL EVALUATION: WILCOXON SIGNED-RANK TESTS")
    print("=" * 80)
    print("NOTE: paired sample size depends on finite labels after censoring removal.\n")

    # Aggregate seeds per model and fold
    fold_agg = results_df.groupby(["model", "fold"]).agg({
        "soh_rmse": "mean",
        "soh_mae": "mean",
        "rul_rmse": "mean",
        "rul_mae": "mean",
    }).reset_index()

    pi_df = fold_agg[fold_agg["model"] == "PI-MDCNet"].sort_values("fold")
    models = [m for m in fold_agg["model"].unique() if m != "PI-MDCNet"]

    stat_rows = []

    metrics = {
        "soh_rmse": "SoH RMSE",
        "rul_rmse": "RUL RMSE",
    }

    for model_name in models:
        m_df = fold_agg[fold_agg["model"] == model_name].sort_values("fold")
        for metric, metric_label in metrics.items():
            paired = pi_df[["fold", metric]].merge(
                m_df[["fold", metric]], on="fold", how="inner", suffixes=("_pi", "_base")
            )
            valid = np.isfinite(paired[f"{metric}_pi"]) & np.isfinite(paired[f"{metric}_base"])
            pi_vals = paired.loc[valid, f"{metric}_pi"].to_numpy()
            m_vals = paired.loc[valid, f"{metric}_base"].to_numpy()

            stat = p_val = np.nan
            if len(pi_vals) >= 3 and not np.allclose(pi_vals, m_vals):
                try:
                    stat, p_val = stats.wilcoxon(pi_vals, m_vals, alternative="two-sided")
                except ValueError:
                    pass

            pi_mean = float(np.mean(pi_vals)) if len(pi_vals) else np.nan
            m_mean = float(np.mean(m_vals)) if len(m_vals) else np.nan
            stat_rows.append({
                "Comparison": f"PI-MDCNet vs {model_name}",
                "Metric": metric_label,
                "PI-MDCNet mean": round(pi_mean, 4) if np.isfinite(pi_mean) else "-",
                "Baseline mean": round(m_mean, 4) if np.isfinite(m_mean) else "-",
                "Delta (PI - Base)": round(pi_mean - m_mean, 4) if np.isfinite(pi_mean) else "-",
                "Wilcoxon W": stat if np.isfinite(stat) else "-",
                "p-value": round(p_val, 3) if np.isfinite(p_val) else "-",
                "N paired": int(len(pi_vals)),
                "Excluded": int(len(paired) - len(pi_vals)),
            })

    summary_df = pd.DataFrame(stat_rows)
    print(summary_df.to_string(index=False))
    print("=" * 80 + "\n")
    return summary_df


def generate_benchmark_figures(
    results_path: Path | str = "artifacts/benchmark_results.csv",
    output_dir: Path | str = "artifacts/figures",
) -> None:
    """Generate all publication figures from benchmark results."""
    results_path = Path(results_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        logger.warning("Benchmark results file %s does not exist", results_path)
        return

    df = pd.read_csv(results_path)

    # 1. SoH RMSE & MAE Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    agg = df.groupby("model").agg(
        rmse_mean=("soh_rmse", "mean"),
        rmse_std=("soh_rmse", "std"),
        mae_mean=("soh_mae", "mean"),
        mae_std=("soh_mae", "std"),
    ).dropna()

    x = np.arange(len(agg))
    width = 0.35

    plt.bar(x - width/2, agg["rmse_mean"], width, yerr=agg["rmse_std"],
            capsize=4, label="SoH RMSE", color="#1f77b4", alpha=0.85)
    plt.bar(x + width/2, agg["mae_mean"], width, yerr=agg["mae_std"],
            capsize=4, label="SoH MAE", color="#ff7f0e", alpha=0.85)

    plt.xticks(x, agg.index, rotation=20, ha="right")
    plt.ylabel("Error")
    plt.title("LOBO Cross-Validation Performance: SoH Estimation Error")
    plt.legend()
    plt.tight_layout()
    bar_path = output_dir / "soh_benchmark_comparison.png"
    plt.savefig(bar_path, dpi=300)
    plt.close()
    logger.info("Saved %s", bar_path)

    # 2. Wall Clock Time Comparison Bar Chart
    plt.figure(figsize=(9, 4.5))
    time_agg = df.groupby("model")["train_wall_clock_s"].mean().sort_values()
    bars = plt.bar(time_agg.index, time_agg.values, color="#2ca02c", alpha=0.85)
    plt.ylabel("Wall-Clock Training Time (seconds)")
    plt.title("Computational Efficiency: Mean Training Wall-Clock Time")
    plt.xticks(rotation=25, ha="right")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{yval:.2f}s",
                 ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    time_path = output_dir / "wall_clock_comparison.png"
    plt.savefig(time_path, dpi=300)
    plt.close()
    logger.info("Saved %s", time_path)

    # 3. Post-Rest Regeneration Diagnostic (R_t correlation)
    pi_df = df[df["model"] == "PI-MDCNet"]
    if "rt_corr_post_rest" in pi_df.columns:
        plt.figure(figsize=(7, 4.5))
        valid_corr = pi_df.dropna(subset=["rt_corr_post_rest"])
        plt.bar(valid_corr["fold"], valid_corr["rt_corr_post_rest"], color="#1f77b4", alpha=0.85)
        plt.axhline(0, color="grey", linestyle="--", linewidth=1)
        plt.ylabel("Correlation ($r$)")
        plt.title("SLAC Diagnostic: Correlation Between $R_t$ and Post-Rest $\\Delta$SoH")
        plt.ylim(-0.2, 1.0)
        for i, row in valid_corr.iterrows():
            plt.text(row["fold"], row["rt_corr_post_rest"] + 0.03,
                     f"r = {row['rt_corr_post_rest']:.3f}", ha="center", fontsize=10, fontweight="bold")
        plt.tight_layout()
        diag_path = output_dir / "regeneration_diagnostic.png"
        plt.savefig(diag_path, dpi=300)
        plt.close()
        logger.info("Saved %s", diag_path)


def generate_cell_trajectory_plots(
    data_dir: Path | str = "data/raw",
    output_dir: Path | str = "artifacts/figures",
    device: str = "auto",
) -> None:
    """Generate detailed cycle-by-cycle degradation & decomposition plots for all cells."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cell_data = load_all_cells(Path(data_dir))

    # Train a representative model on B0006, B0007, B0018 and test on B0005
    train_ids = ["B0006", "B0007", "B0018"]
    test_id = "B0005"
    train_dfs = [cell_data[cid] for cid in train_ids]
    test_df = cell_data[test_id]

    model, scaler, _ = train_pimdcnet(
        train_dfs, cfg=SLACConfig(), n_epochs=BENCHMARK_EPOCHS, device=device, seed=42
    )

    # Predict on B0005
    test_ds = CycleSequenceDataset(test_df, scaler=scaler)
    test_batch = {k: v.unsqueeze(0).to(device) for k, v in test_ds[0].items()}
    model.eval()
    with torch.no_grad():
        pred = model(
            x_elec=test_batch["x_elec"],
            x_therm=test_batch["x_therm"],
            s_t=test_batch["s_t"],
            rest_features=test_batch["rest_features"],
            rest_flag=test_batch["rest_flag"],
            cycle_idx=test_batch["cycle_idx"],
        )

    cycles = np.arange(len(test_df))
    true_soh = test_df["soh"].values
    pred_soh = pred["soh"][0, :, 0].cpu().numpy()
    D_t = pred["D_t"][0, :, 0].cpu().numpy()
    R_t = pred["R_t"][0, :, 0].cpu().numpy()
    rest_flags = test_df["rest_flag"].values

    # Plot 1: True vs Predicted SoH with Rest Markers
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cycles, true_soh, label="True SoH (NASA PCoE)", color="#222222", linewidth=2.2)
    ax.plot(cycles, pred_soh, "--", label="PI-MDCNet v4 (SLAC)", color="#1f77b4", linewidth=2.0)

    # Rest event markers
    rest_cycles = cycles[rest_flags == 1]
    if len(rest_cycles) > 0:
        ax.scatter(rest_cycles, true_soh[rest_flags == 1], color="#d62728", s=45,
                   zorder=5, label=f"Rest Events (rest >= 1.0h, n={len(rest_cycles)})")

    ax.set_xlabel("Discharge Cycle Number")
    ax.set_ylabel("State of Health (SoH)")
    ax.set_title(f"Cell {test_id}: True vs Predicted SoH Trajectory (LOBO Held-Out Test)")
    ax.legend(loc="lower left")
    plt.tight_layout()
    soh_curve_path = output_dir / f"{test_id}_soh_trajectory.png"
    plt.savefig(soh_curve_path, dpi=300)
    plt.close()
    logger.info("Saved %s", soh_curve_path)

    # Plot 2: Split-Latent Decomposition (D_t vs R_t)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Top: Damage D_t (Monotonic)
    ax1.plot(cycles, D_t, color="#d62728", linewidth=2.0, label="Irreversible Damage $D_t$ (Monotonic)")
    ax1.set_ylabel("Accumulated Damage $D_t$")
    ax1.set_title(f"Cell {test_id}: Split-Latent Aging Core (SLAC) Decomposition")
    ax1.legend(loc="upper left")

    # Bottom: Regeneration R_t (Hard-Gated)
    ax2.bar(cycles, R_t, width=1.0, color="#2ca02c", alpha=0.85, label="Regeneration $R_t$ (Hard-Gated at Rest)")
    ax2.set_xlabel("Discharge Cycle Number")
    ax2.set_ylabel("Capacity Recovery $R_t$")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    decomp_path = output_dir / f"{test_id}_slac_decomposition.png"
    plt.savefig(decomp_path, dpi=300)
    plt.close()
    logger.info("Saved %s", decomp_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    df = pd.read_csv("artifacts/benchmark_results.csv")
    run_statistical_analysis(df)
    generate_benchmark_figures()
    generate_cell_trajectory_plots()
