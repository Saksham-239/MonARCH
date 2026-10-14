from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
from experiments.protocol import BENCHMARK_EPOCHS, BENCHMARK_SEEDS

logger = logging.getLogger(__name__)

# Publication aesthetic styling (IEEE Transactions / Nature Energy grade)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 1.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLORS = {
    "PI-MDCNet": "#1f77b4",       # Deep blue
    "RandomForest": "#2ca02c",    # Green
    "XGBoost": "#ff7f0e",         # Orange
    "VanillaLSTM": "#9467bd",     # Purple
    "StatDecomp": "#d62728",      # Red
    "CoulombCounting": "#8c564b", # Brown
    "EKF_1RC": "#e377c2",         # Pink
    "MLP": "#7f7f7f",             # Gray
    "TCN": "#17becf",             # Cyan
    "Transformer": "#bcbd22",     # Olive
    "PINN": "#e377c2",            # Magenta
    "GroundTruth": "#222222",     # Dark grey / black
}


def run_statistical_analysis(results_df: pd.DataFrame, output_dir: Path | str = "artifacts") -> pd.DataFrame:
    """Run paired Wilcoxon and Student's t-tests between PI-MDCNet and all baselines."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print("STATISTICAL EVALUATION: PAIRED WILCOXON SIGNED-RANK & STUDENT'S T-TESTS")
    print("=" * 90)

    # Aggregate seeds per model and fold
    agg_candidates = ["soh_rmse", "soh_mae", "rul_mae", "rul_rmse", "train_wall_clock_s"]
    agg_dict = {c: "mean" for c in agg_candidates if c in results_df.columns}
    fold_agg = results_df.groupby(["model", "fold"]).agg(agg_dict).reset_index()

    pi_df = fold_agg[fold_agg["model"] == "PI-MDCNet"].sort_values("fold").set_index("fold")
    models = [m for m in fold_agg["model"].unique() if m != "PI-MDCNet"]

    stat_rows = []
    comp_table_rows = []

    # Add PI-MDCNet row to comparison table first
    pi_sub = fold_agg[fold_agg["model"] == "PI-MDCNet"]
    comp_table_rows.append({
        "model": "PI-MDCNet",
        "soh_rmse_mean": float(pi_sub["soh_rmse"].mean()) if "soh_rmse" in pi_sub.columns else np.nan,
        "soh_rmse_std": float(pi_sub["soh_rmse"].std()) if "soh_rmse" in pi_sub.columns else np.nan,
        "soh_mae_mean": float(pi_sub["soh_mae"].mean()) if "soh_mae" in pi_sub.columns else np.nan,
        "soh_mae_std": float(pi_sub["soh_mae"].std()) if "soh_mae" in pi_sub.columns else np.nan,
        "rul_mae_mean": float(pi_sub["rul_mae"].mean()) if "rul_mae" in pi_sub.columns else np.nan,
        "rul_mae_std": float(pi_sub["rul_mae"].std()) if "rul_mae" in pi_sub.columns else np.nan,
        "rul_rmse_mean": float(pi_sub["rul_rmse"].mean()) if "rul_rmse" in pi_sub.columns else np.nan,
        "rul_rmse_std": float(pi_sub["rul_rmse"].std()) if "rul_rmse" in pi_sub.columns else np.nan,
        "train_wall_clock_s_mean": float(pi_sub["train_wall_clock_s"].mean()) if "train_wall_clock_s" in pi_sub.columns else 0.0,
        "p_value_wilcoxon": 1.0,
        "p_value_ttest": 1.0,
    })

    metrics = {
        "soh_rmse": "SoH RMSE",
        "soh_mae": "SoH MAE",
        "rul_mae": "RUL MAE",
        "rul_rmse": "RUL RMSE",
    }

    for model_name in sorted(models):
        m_df = fold_agg[fold_agg["model"] == model_name].sort_values("fold").set_index("fold")
        common_folds = pi_df.index.intersection(m_df.index)

        p_wilc_soh = 1.0
        p_ttest_soh = 1.0

        for metric, metric_label in metrics.items():
            if metric not in pi_df.columns or metric not in m_df.columns:
                continue
            pi_vals = pi_df.loc[common_folds, metric].dropna().to_numpy()
            m_vals = m_df.loc[common_folds, metric].dropna().to_numpy()

            stat_w = p_w = np.nan
            stat_t = p_t = np.nan

            min_len = min(len(pi_vals), len(m_vals))
            excluded = int(len(common_folds) - min_len)
            if min_len >= 3 and not np.allclose(pi_vals[:min_len], m_vals[:min_len]):
                try:
                    stat_w, p_w = stats.wilcoxon(m_vals[:min_len], pi_vals[:min_len], alternative="two-sided")
                except ValueError:
                    pass
                try:
                    stat_t, p_t = stats.ttest_rel(m_vals[:min_len], pi_vals[:min_len])
                except ValueError:
                    pass

            if metric == "soh_rmse":
                p_wilc_soh = p_w if not np.isnan(p_w) else 1.0
                p_ttest_soh = p_t if not np.isnan(p_t) else 1.0

            pi_mean = float(np.mean(pi_vals)) if len(pi_vals) else np.nan
            m_mean = float(np.mean(m_vals)) if len(m_vals) else np.nan
            stat_rows.append({
                "Comparison": f"PI-MDCNet vs {model_name}",
                "Metric": metric_label,
                "PI-MDCNet mean": round(pi_mean, 4) if np.isfinite(pi_mean) else "-",
                "Baseline mean": round(m_mean, 4) if np.isfinite(m_mean) else "-",
                "Delta (PI - Base)": round(pi_mean - m_mean, 4) if np.isfinite(pi_mean) else "-",
                "Wilcoxon p-val": round(p_w, 4) if np.isfinite(p_w) else "-",
                "t-test p-val": round(p_t, 4) if np.isfinite(p_t) else "-",
                "N paired": int(min_len),
                "Excluded": excluded,
            })

        sub = fold_agg[fold_agg["model"] == model_name]
        comp_table_rows.append({
            "model": model_name,
            "soh_rmse_mean": float(sub["soh_rmse"].mean()) if "soh_rmse" in sub.columns else np.nan,
            "soh_rmse_std": float(sub["soh_rmse"].std()) if "soh_rmse" in sub.columns else np.nan,
            "soh_mae_mean": float(sub["soh_mae"].mean()) if "soh_mae" in sub.columns else np.nan,
            "soh_mae_std": float(sub["soh_mae"].std()) if "soh_mae" in sub.columns else np.nan,
            "rul_mae_mean": float(sub["rul_mae"].mean()) if "rul_mae" in sub.columns else np.nan,
            "rul_mae_std": float(sub["rul_mae"].std()) if "rul_mae" in sub.columns else np.nan,
            "rul_rmse_mean": float(sub["rul_rmse"].mean()) if "rul_rmse" in sub.columns else np.nan,
            "rul_rmse_std": float(sub["rul_rmse"].std()) if "rul_rmse" in sub.columns else np.nan,
            "train_wall_clock_s_mean": float(sub["train_wall_clock_s"].mean()) if "train_wall_clock_s" in sub.columns else 0.0,
            "p_value_wilcoxon": float(p_wilc_soh),
            "p_value_ttest": float(p_ttest_soh),
        })

    summary_df = pd.DataFrame(stat_rows)
    print(summary_df.to_string(index=False))
    print("=" * 90 + "\n")

    comp_table = pd.DataFrame(comp_table_rows)
    comp_path = output_dir / "comparison_table.csv"
    comp_table.to_csv(comp_path, index=False)
    logger.info("Saved aggregated comparison table with p-values to %s", comp_path)

    return summary_df


def generate_benchmark_figures(
    results_path: Path | str = "artifacts/benchmark_results.csv",
    output_dir: Path | str = "artifacts/figures",
) -> None:
    """Generate publication figures in both vector .pdf and 300-DPI .png formats."""
    results_path = Path(results_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        logger.warning("Benchmark results file %s does not exist", results_path)
        return

    df = pd.read_csv(results_path)

    # 1. SoH RMSE & MAE Comparison Bar Chart
    plt.figure(figsize=(11, 5))
    agg = df.groupby("model").agg(
        rmse_mean=("soh_rmse", "mean"),
        rmse_std=("soh_rmse", "std"),
        mae_mean=("soh_mae", "mean"),
        mae_std=("soh_mae", "std"),
    ).dropna()

    x = np.arange(len(agg))
    width = 0.35

    plt.bar(x - width/2, agg["rmse_mean"], width, yerr=agg["rmse_std"],
            capsize=4, label="SoH RMSE", color="#1f77b4", alpha=0.85, edgecolor="black", linewidth=0.8)
    plt.bar(x + width/2, agg["mae_mean"], width, yerr=agg["mae_std"],
            capsize=4, label="SoH MAE", color="#ff7f0e", alpha=0.85, edgecolor="black", linewidth=0.8)

    plt.xticks(x, agg.index, rotation=25, ha="right")
    plt.ylabel("Error (Normalized SoH)")
    plt.title("LOBO Cross-Validation Performance: SoH Estimation Error")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_dir / "soh_benchmark_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "soh_benchmark_comparison.pdf", bbox_inches="tight")
    plt.close()

    # 2. Wall Clock Time Comparison Bar Chart
    plt.figure(figsize=(10, 4.5))
    time_agg = df.groupby("model")["train_wall_clock_s"].mean().sort_values()
    bars = plt.bar(time_agg.index, time_agg.values, color="#2ca02c", alpha=0.85, edgecolor="black", linewidth=0.8)
    plt.ylabel("Wall-Clock Training Time (seconds)")
    plt.title("Computational Efficiency: Mean Training Wall-Clock Time")
    plt.xticks(rotation=25, ha="right")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{yval:.2f}s",
                 ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "wall_clock_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "wall_clock_comparison.pdf", bbox_inches="tight")
    plt.close()

    # 3. Post-Rest Regeneration Diagnostic (R_t correlation)
    pi_df = df[df["model"] == "PI-MDCNet"]
    if "rt_corr_post_rest" in pi_df.columns:
        plt.figure(figsize=(7, 4.5))
        valid_corr = pi_df.dropna(subset=["rt_corr_post_rest"]).groupby("fold")["rt_corr_post_rest"].mean().reset_index()
        plt.bar(valid_corr["fold"], valid_corr["rt_corr_post_rest"], color="#1f77b4", alpha=0.85, edgecolor="black", linewidth=0.8)
        plt.axhline(0, color="grey", linestyle="--", linewidth=1)
        plt.ylabel("Correlation ($r$)")
        plt.title("SLAC Diagnostic: Correlation Between $R_t$ and Post-Rest $\\Delta$SoH")
        plt.ylim(-0.2, 1.0)
        for _, row in valid_corr.iterrows():
            plt.text(row["fold"], row["rt_corr_post_rest"] + 0.03,
                     f"r = {row['rt_corr_post_rest']:.3f}", ha="center", fontsize=10, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_dir / "regeneration_diagnostic.png", dpi=300, bbox_inches="tight")
        plt.savefig(output_dir / "regeneration_diagnostic.pdf", bbox_inches="tight")
        plt.close()


def generate_cell_trajectory_plots(
    data_dir: Path | str = "data/raw",
    output_dir: Path | str = "artifacts/figures",
    device: str = "auto",
) -> None:
    """Generate detailed cycle-by-cycle degradation & decomposition plots with ±1 std shaded error bands."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cell_data = load_all_cells(Path(data_dir))

    train_ids = ["B0006", "B0007", "B0018"]
    test_id = "B0005"
    train_dfs = [cell_data[cid] for cid in train_ids]
    test_df = cell_data[test_id]

    # Predict across seeds to compute shaded error band
    all_pred_soh = []
    all_D_t = []
    all_R_t = []

    for seed in BENCHMARK_SEEDS:
        model, scaler, _ = train_pimdcnet(
            train_dfs, cfg=SLACConfig(), n_epochs=BENCHMARK_EPOCHS, device=device, seed=seed
        )
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
                soh_lag=test_batch.get("soh_lag", None),
            )
        all_pred_soh.append(pred["soh"][0, :, 0].cpu().numpy())
        all_D_t.append(pred["D_t"][0, :, 0].cpu().numpy())
        all_R_t.append(pred["R_t"][0, :, 0].cpu().numpy())

    cycles = np.arange(len(test_df))
    true_soh = test_df["soh"].values
    rest_flags = test_df["rest_flag"].values

    pred_soh_mean = np.mean(all_pred_soh, axis=0)
    pred_soh_std = np.std(all_pred_soh, axis=0)
    D_t_mean = np.mean(all_D_t, axis=0)
    D_t_std = np.std(all_D_t, axis=0)
    R_t_mean = np.mean(all_R_t, axis=0)

    # Plot 1: True vs Predicted SoH with Shaded ±1 std Band
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cycles, true_soh, label="True SoH (NASA PCoE)", color="#222222", linewidth=2.2)
    ax.plot(cycles, pred_soh_mean, "--", label="PI-MDCNet v4 Mean (SLAC)", color="#1f77b4", linewidth=2.0)
    ax.fill_between(cycles, pred_soh_mean - pred_soh_std, pred_soh_mean + pred_soh_std,
                    color="#1f77b4", alpha=0.25, label="±1 Std Dev (5 Seeds)")

    rest_cycles = cycles[rest_flags == 1]
    if len(rest_cycles) > 0:
        ax.scatter(rest_cycles, true_soh[rest_flags == 1], color="#d62728", s=45,
                   zorder=5, label=f"Rest Events (rest >= 1.0h, n={len(rest_cycles)})")

    ax.set_xlabel("Discharge Cycle Number")
    ax.set_ylabel("State of Health (SoH)")
    ax.set_title(f"Cell {test_id}: True vs Predicted SoH Trajectory (LOBO Held-Out Test)")
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_dir / f"{test_id}_soh_trajectory.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / f"{test_id}_soh_trajectory.pdf", bbox_inches="tight")
    plt.close()

    # Plot 2: Split-Latent Decomposition (D_t vs R_t)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Top: Damage D_t (Monotonic)
    ax1.plot(cycles, D_t_mean, color="#d62728", linewidth=2.0, label="Irreversible Damage $D_t$ (Monotonic)")
    ax1.fill_between(cycles, D_t_mean - D_t_std, D_t_mean + D_t_std, color="#d62728", alpha=0.25, label="±1 Std Dev")
    ax1.set_ylabel("Accumulated Damage $D_t$")
    ax1.set_title(f"Cell {test_id}: Split-Latent Aging Core (SLAC) Decomposition")
    ax1.legend(loc="upper left")

    # Bottom: Regeneration R_t (Hard-Gated)
    ax2.bar(cycles, R_t_mean, width=1.0, color="#2ca02c", alpha=0.85, label="Regeneration $R_t$ (Hard-Gated at Rest)")
    ax2.set_xlabel("Discharge Cycle Number")
    ax2.set_ylabel("Capacity Recovery $R_t$")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_dir / f"{test_id}_slac_decomposition.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / f"{test_id}_slac_decomposition.pdf", bbox_inches="tight")
    plt.close()


def _fmt_stat(mean_val: float, std_val: float | None = None, precision: int = 4) -> str:
    """Format mean +/- std cleanly, eliminating any '\\pm nan' in LaTeX output."""
    if pd.isna(mean_val) or np.isnan(mean_val):
        return "--"
    if std_val is not None and pd.notna(std_val) and not np.isnan(std_val):
        return f"{mean_val:.{precision}f} $\\pm$ {std_val:.{precision}f}"
    return f"{mean_val:.{precision}f}"


def generate_latex_tables(output_path: Path | str = "artifacts/paper_tables.tex") -> None:
    """Generate publication-ready LaTeX tables using booktabs package."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    latex_content = []
    latex_content.append("% Auto-generated publication tables for IEEE Transactions on Industrial Informatics / Applied Energy\n")

    # Table 1: Model Comparison Table
    comp_file = Path("artifacts/comparison_table.csv")
    if comp_file.exists():
        df_comp = pd.read_csv(comp_file)
        latex_content.append("% Table I: Multi-Model Benchmark Comparison")
        latex_content.append("\\begin{table*}[t]")
        latex_content.append("\\centering")
        latex_content.append("\\caption{Cross-Dataset LOCO Benchmark Performance across NASA, CALCE, and Oxford (Mean $\\pm$ Std across 5 seeds: [42, 100, 2024, 2025, 2026])}")
        latex_content.append("\\label{tab:benchmark_comparison}")
        latex_content.append("\\begin{tabular}{lcccccc}")
        latex_content.append("\\toprule")
        latex_content.append("Model & SoH RMSE & SoH MAE & RUL MAE (cyc) & Time (s) & $p$-val (Wilcoxon) & $p$-val ($t$-test) \\\\")
        latex_content.append("\\midrule")
        for _, row in df_comp.iterrows():
            m = row["model"]
            s_rmse = _fmt_stat(row.get("soh_rmse_mean"), row.get("soh_rmse_std"), 4)
            s_mae = _fmt_stat(row.get("soh_mae_mean"), row.get("soh_mae_std"), 4)
            if pd.notna(row.get("rul_mae_mean")) and row["rul_mae_mean"] > 0:
                r_mae = _fmt_stat(row.get("rul_mae_mean"), row.get("rul_mae_std"), 1)
            else:
                r_mae = "--"
            t_s = f"{row['train_wall_clock_s_mean']:.1f}" if pd.notna(row.get("train_wall_clock_s_mean")) else "--"
            pw = f"{row['p_value_wilcoxon']:.3f}" if pd.notna(row.get("p_value_wilcoxon")) and m != "PI-MDCNet" else "--"
            pt = f"{row['p_value_ttest']:.3e}" if pd.notna(row.get("p_value_ttest")) and m != "PI-MDCNet" else "--"

            bold = "\\textbf{" if m == "PI-MDCNet" else ""
            bold_end = "}" if m == "PI-MDCNet" else ""
            latex_content.append(f"{bold}{m}{bold_end} & {s_rmse} & {s_mae} & {r_mae} & {t_s} & {pw} & {pt} \\\\")
        latex_content.append("\\bottomrule")
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table*}\n\n")

    # Table 2: Ablation Study Table
    abl_file = Path("artifacts/ablation_table.csv")
    if abl_file.exists():
        df_abl = pd.read_csv(abl_file)
        latex_content.append("% Table II: Architectural Ablation Study")
        latex_content.append("\\begin{table}[t]")
        latex_content.append("\\centering")
        latex_content.append("\\caption{SLAC Architectural Ablation Study Across 5 Seeds}")
        latex_content.append("\\label{tab:ablation_study}")
        latex_content.append("\\begin{tabular}{lcccc}")
        latex_content.append("\\toprule")
        latex_content.append("Ablation Variant & SoH RMSE & SoH MAE & Monotonic $D_t$ & $p$-val vs Full \\\\")
        latex_content.append("\\midrule")
        for _, row in df_abl.iterrows():
            v = row["variant"]
            s_rmse = _fmt_stat(row.get("soh_rmse_mean"), row.get("soh_rmse_std"), 4)
            s_mae = _fmt_stat(row.get("soh_mae_mean"), row.get("soh_mae_std"), 4)
            mono = "Yes" if row.get("monotonic_damage", False) else "No"
            pv = f"{row['p_value_vs_full']:.3f}" if v != "Full_SLAC" and pd.notna(row.get("p_value_vs_full")) else "--"
            latex_content.append(f"{v} & {s_rmse} & {s_mae} & {mono} & {pv} \\\\")
        latex_content.append("\\bottomrule")
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}\n\n")

    # Table 3: RUL Validation Summary
    rul_file = Path("artifacts/rul_validation_summary.csv")
    if rul_file.exists():
        df_rul = pd.read_csv(rul_file)
        latex_content.append("% Table III: Four-Way RUL Validation Benchmark")
        latex_content.append("\\begin{table}[t]")
        latex_content.append("\\centering")
        latex_content.append("\\caption{Four-Way RUL Estimation Benchmark (LOBO Valid EOL Folds)}")
        latex_content.append("\\label{tab:rul_validation}")
        latex_content.append("\\begin{tabular}{lcc}")
        latex_content.append("\\toprule")
        latex_content.append("RUL Method & Mean MAE $\\pm$ Std (cycles) & Mean RMSE $\\pm$ Std (cycles) \\\\")
        latex_content.append("\\midrule")
        for _, row in df_rul.iterrows():
            mae_str = _fmt_stat(row.get("mean_mae"), row.get("std_mae"), 2)
            rmse_str = _fmt_stat(row.get("mean_rmse"), row.get("std_rmse"), 2)
            latex_content.append(f"{row['method']} & {mae_str} & {rmse_str} \\\\")
        latex_content.append("\\bottomrule")
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}\n\n")

    # Table 4: Coupled Intracycle SOC Estimation
    soc_file = Path("artifacts/intracycle_soc_results.csv")
    if soc_file.exists():
        df_soc = pd.read_csv(soc_file)
        soc_agg = df_soc.groupby(["dataset", "aging_tier", "method"])[["rmse", "mae"]].mean().reset_index()
        latex_content.append("% Table IV: Aging-Tiered Intra-Cycle SOC Estimation")
        latex_content.append("\\begin{table*}[t]")
        latex_content.append("\\centering")
        latex_content.append("\\caption{Coupled Intra-Cycle Continuous SoC Estimation Across Aging Stages}")
        latex_content.append("\\label{tab:intracycle_soc}")
        latex_content.append("\\begin{tabular}{llccc}")
        latex_content.append("\\toprule")
        latex_content.append("Dataset & Aging Tier (SoH) & Method & SoC RMSE & SoC MAE \\\\")
        latex_content.append("\\midrule")
        for _, row in soc_agg.iterrows():
            latex_content.append(f"{row['dataset']} & {row['aging_tier']} & {row['method']} & {row['rmse']:.4f} & {row['mae']:.4f} \\\\")
        latex_content.append("\\bottomrule")
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_content))
    logger.info("Saved LaTeX publication tables to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    bench_file = Path("artifacts/benchmark_results.csv")
    if bench_file.exists():
        df = pd.read_csv(bench_file)
        run_statistical_analysis(df)
        generate_benchmark_figures()
        generate_cell_trajectory_plots()
        generate_latex_tables()
    else:
        logger.warning("artifacts/benchmark_results.csv does not exist. Run train.py first.")

