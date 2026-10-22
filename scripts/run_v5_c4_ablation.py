"""V5 Verification Script: Part C4 (Ablation Study under C2 Open-Loop).

Re-runs the four ablation variants on NASA across 4 folds and 5 seeds:
- Full_SLAC
- Ablation_NoCrossAttn
- Ablation_NoHardGating
- Ablation_NoMonotonicity
Under C2: Open-loop training (loss on cumulative trajectory) and open-loop evaluation (learned soh_0, no test labels).
"""

import logging
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import torch
from scipy import stats

from src.data_loader import load_all_cells
from experiments.protocol import BENCHMARK_SEEDS, BENCHMARK_EPOCHS, CYCLE_NORMALIZATION_SCALE
from experiments.train import train_pimdcnet, predict_pimdcnet, compute_metrics
from experiments.ablation import ABLATION_VARIANTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("V5_Ablation_C2")

NASA_CELLS = ["B0005", "B0006", "B0007", "B0018"]

def run_c4_ablation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Running C4 Open-Loop Ablation Study on device: %s", device)

    data_dir = Path("data/raw") if Path("data/raw").exists() else Path("data")
    output_dir = Path("artifacts/v5")
    output_dir.mkdir(parents=True, exist_ok=True)

    cell_data = load_all_cells(data_dir, dataset="nasa")
    raw_results = []

    for variant_name, model_cls in ABLATION_VARIANTS.items():
        logger.info("==================================================")
        logger.info("RUNNING C2 OPEN-LOOP ABLATION: %s", variant_name)
        logger.info("==================================================")

        for fold_idx, held_out in enumerate(NASA_CELLS):
            train_cells = [cid for cid in NASA_CELLS if cid != held_out]
            train_dfs = [cell_data[cid] for cid in train_cells]
            test_df = cell_data[held_out]

            for seed in BENCHMARK_SEEDS:
                # Train open-loop (C2)
                model, scaler, wall_clock = train_pimdcnet(
                    train_dfs,
                    n_epochs=BENCHMARK_EPOCHS,
                    device=device,
                    seed=seed,
                    model_cls=model_cls,
                    open_loop=True,
                    cycle_norm_scale=CYCLE_NORMALIZATION_SCALE,
                )

                # Predict open-loop (C2)
                res = predict_pimdcnet(
                    model, test_df, scaler=scaler, device=device,
                    eval_mode="open_loop", cycle_norm_scale=CYCLE_NORMALIZATION_SCALE,
                )

                soh_m = compute_metrics(test_df["soh"].values, res.soh, "soh_")
                rul_m = compute_metrics(test_df["rul"].values, res.rul, "rul_")

                # Monotonicity check on D_t
                D_t = res.D_t
                d_diff = np.diff(D_t)
                is_monotonic = bool(np.all(d_diff >= -1e-6))
                max_violation = float(max(0.0, -np.min(d_diff))) if len(d_diff) > 0 else 0.0

                record = {
                    "variant": variant_name,
                    "fold": held_out,
                    "seed": seed,
                    "train_wall_clock_s": wall_clock,
                    "is_damage_monotonic": is_monotonic,
                    "max_monotonicity_violation": max_violation,
                    **soh_m,
                    **rul_m,
                }
                raw_results.append(record)
                logger.info(
                    "  [%s] %s seed=%d: SoH RMSE=%.4f, MAE=%.4f | Monotonic=%s",
                    variant_name, held_out, seed, soh_m["soh_rmse"], soh_m["soh_mae"], is_monotonic
                )

    raw_df = pd.DataFrame(raw_results)
    raw_path = output_dir / "ablation_openloop_raw.csv"
    raw_df.to_csv(raw_path, index=False)
    logger.info("Saved raw open-loop ablation results to %s", raw_path)

    # Paired statistical analysis against Full_SLAC
    full_scores = raw_df[raw_df["variant"] == "Full_SLAC"].groupby(["fold", "seed"])["soh_rmse"].mean()

    summary_rows = []
    for variant in ABLATION_VARIANTS.keys():
        sub = raw_df[raw_df["variant"] == variant]
        v_scores = sub.groupby(["fold", "seed"])["soh_rmse"].mean()

        p_val = np.nan
        if variant != "Full_SLAC":
            common_idx = full_scores.index.intersection(v_scores.index)
            if len(common_idx) >= 3:
                a_vals = full_scores.loc[common_idx].values
                b_vals = v_scores.loc[common_idx].values
                if not np.allclose(a_vals, b_vals):
                    try:
                        _, p_val = stats.ttest_rel(b_vals, a_vals)
                    except Exception:
                        pass

        # Also aggregate per fold first, then mean/std
        fold_agg = sub.groupby("fold")[["soh_rmse", "soh_mae", "rul_mae"]].mean()

        summary_rows.append({
            "variant": variant,
            "soh_rmse_mean": float(fold_agg["soh_rmse"].mean()),
            "soh_rmse_std": float(fold_agg["soh_rmse"].std()),
            "soh_mae_mean": float(fold_agg["soh_mae"].mean()),
            "soh_mae_std": float(fold_agg["soh_mae"].std()),
            "rul_mae_mean": float(fold_agg["rul_mae"].mean()),
            "rul_mae_std": float(fold_agg["rul_mae"].std()),
            "monotonic_damage": bool(sub["is_damage_monotonic"].all()),
            "p_value_vs_full": float(p_val) if not np.isnan(p_val) else 1.0,
        })

    summary_df = pd.DataFrame(summary_rows)
    sum_path = output_dir / "ablation_openloop_summary.csv"
    summary_df.to_csv(sum_path, index=False)
    logger.info("Saved open-loop ablation summary table to %s", sum_path)

    print("\n" + "=" * 95)
    print("OPEN-LOOP (C2) ABLATION STUDY SUMMARY (NASA, 5 seeds)")
    print("=" * 95)
    print(summary_df.to_string(index=False))
    print("=" * 95 + "\n")

if __name__ == "__main__":
    run_c4_ablation()
