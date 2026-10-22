"""V5 Verification Script: Part B (Baselines) and Part C (Open-Loop PI-MDCNet).

Evaluates across all 12 folds (NASA 4, CALCE 4, Oxford 4) and 5 seeds:
- B1: Persistence
- B2: Causal Persistence + Drift
- B3: RandomForest + lag
- B3: XGBoost + lag
- B4: PI-MDCNet (lag-conditioned)
- C1: PI-MDCNet (lag-trained, open-loop evaluated)
- C2: PI-MDCNet (open-loop trained, open-loop evaluated)
- C3: PI-MDCNet (lag-trained, free-running recursive evaluated)
"""

import time
import logging
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from src.data_loader import load_all_cells
from src.models.baselines import FEATURE_COLS
from src.models.pi_mdcnet import PIMDCNet, SLACConfig
from experiments.protocol import BENCHMARK_SEEDS, BENCHMARK_EPOCHS, CYCLE_NORMALIZATION_SCALE
from experiments.train import train_pimdcnet, predict_pimdcnet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("V5_Benchmark")

DATASETS = {
    "nasa": ["B0005", "B0006", "B0007", "B0018"],
    "calce": ["CS2_35", "CS2_36", "CS2_37", "CS2_38"],
    "oxford": ["Cell1", "Cell2", "Cell3", "Cell4"],
}

def add_lag1(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    soh = df_copy["soh"].values
    lag = np.concatenate([[soh[0]], soh[:-1]])
    df_copy["soh_lag1"] = lag
    return df_copy

def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan, np.nan, np.nan
    t = y_true[mask]
    p = y_pred[mask]
    rmse = float(np.sqrt(np.mean((t - p) ** 2)))
    mae = float(np.mean(np.abs(t - p)))
    max_ae = float(np.max(np.abs(t - p)))
    return rmse, mae, max_ae

def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Running V5 benchmark on device: %s", device)
    
    data_dir = Path("data/raw") if Path("data/raw").exists() else Path("data")
    output_dir = Path("artifacts/v5")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    raw_records = []

    for ds_name, cell_ids in DATASETS.items():
        logger.info("=== DATASET: %s ===", ds_name.upper())
        cells = load_all_cells(data_dir, dataset=ds_name)
        
        for fold_idx, held_out in enumerate(cell_ids):
            logger.info("--- Fold %d/%d: Test Cell %s ---", fold_idx + 1, len(cell_ids), held_out)
            train_cells = [cid for cid in cell_ids if cid != held_out]
            train_dfs = [cells[cid] for cid in train_cells]
            test_df = cells[held_out]
            
            y_test = test_df["soh"].values
            n_cycles = len(test_df)
            
            # --- B1: Persistence ---
            pred_b1 = np.concatenate([[y_test[0]], y_test[:-1]])
            b1_rmse, b1_mae, b1_max = calc_metrics(y_test, pred_b1)
            for seed in BENCHMARK_SEEDS:
                raw_records.append({
                    "dataset": ds_name, "fold": held_out, "seed": seed, "model": "Persistence",
                    "soh_rmse": b1_rmse, "soh_mae": b1_mae, "soh_max_ae": b1_max,
                })
            
            # --- B2: Causal Persistence + Drift ---
            pred_b2 = np.zeros(n_cycles, dtype=np.float64)
            pred_b2[0] = y_test[0]
            if n_cycles > 1:
                pred_b2[1] = y_test[0]
            for t in range(2, n_cycles):
                running_drift = (y_test[t - 1] - y_test[0]) / float(t - 1)
                pred_b2[t] = y_test[t - 1] + running_drift
            b2_rmse, b2_mae, b2_max = calc_metrics(y_test, pred_b2)
            for seed in BENCHMARK_SEEDS:
                raw_records.append({
                    "dataset": ds_name, "fold": held_out, "seed": seed, "model": "Persistence+drift",
                    "soh_rmse": b2_rmse, "soh_mae": b2_mae, "soh_max_ae": b2_max,
                })
            
            # Prepare tabular data with lag for RF / XGB
            train_dfs_lag = [add_lag1(df) for df in train_dfs]
            train_combined_lag = pd.concat(train_dfs_lag, ignore_index=True)
            test_df_lag = add_lag1(test_df)
            
            feature_cols_with_lag = FEATURE_COLS + ["soh_lag1"]
            X_train = np.nan_to_num(train_combined_lag[feature_cols_with_lag].values.astype(np.float64), nan=0.0)
            y_train = train_combined_lag["soh"].values.astype(np.float64)
            X_test = np.nan_to_num(test_df_lag[feature_cols_with_lag].values.astype(np.float64), nan=0.0)

            # Loop over seeds for learned models
            for seed in BENCHMARK_SEEDS:
                # --- B3: RandomForest + lag ---
                rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=seed, n_jobs=-1)
                rf.fit(X_train, y_train)
                rf_pred = rf.predict(X_test)
                rf_rmse, rf_mae, rf_max = calc_metrics(y_test, rf_pred)
                raw_records.append({
                    "dataset": ds_name, "fold": held_out, "seed": seed, "model": "RF+lag",
                    "soh_rmse": rf_rmse, "soh_mae": rf_mae, "soh_max_ae": rf_max,
                })

                # --- B3: XGBoost + lag ---
                xgbr = xgb.XGBRegressor(
                    n_estimators=200, max_depth=6, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8, random_state=seed, verbosity=0,
                )
                xgbr.fit(X_train, y_train)
                xgb_pred = xgbr.predict(X_test)
                xgb_rmse, xgb_mae, xgb_max = calc_metrics(y_test, xgb_pred)
                raw_records.append({
                    "dataset": ds_name, "fold": held_out, "seed": seed, "model": "XGB+lag",
                    "soh_rmse": xgb_rmse, "soh_mae": xgb_mae, "soh_max_ae": xgb_max,
                })

                # --- Train Model 1: Lag-Conditioned (for B4, C1, C3) ---
                model_lag, scaler_lag, _ = train_pimdcnet(
                    train_dfs, cfg=SLACConfig(), n_epochs=BENCHMARK_EPOCHS,
                    device=device, seed=seed, open_loop=False, cycle_norm_scale=CYCLE_NORMALIZATION_SCALE,
                )

                # B4: PI-MDCNet lag-conditioned
                res_b4 = predict_pimdcnet(model_lag, test_df, scaler=scaler_lag, device=device, eval_mode="lag")
                b4_rmse, b4_mae, b4_max = calc_metrics(y_test, res_b4.soh)
                raw_records.append({
                    "dataset": ds_name, "fold": held_out, "seed": seed, "model": "PI-MDCNet lag-conditioned",
                    "soh_rmse": b4_rmse, "soh_mae": b4_mae, "soh_max_ae": b4_max,
                })

                # C1: Lag-trained, open-loop evaluated
                res_c1 = predict_pimdcnet(model_lag, test_df, scaler=scaler_lag, device=device, eval_mode="open_loop")
                c1_rmse, c1_mae, c1_max = calc_metrics(y_test, res_c1.soh)
                raw_records.append({
                    "dataset": ds_name, "fold": held_out, "seed": seed, "model": "C1",
                    "soh_rmse": c1_rmse, "soh_mae": c1_mae, "soh_max_ae": c1_max,
                })

                # C3: Free-running recursive (lag-trained, recursive evaluated)
                res_c3 = predict_pimdcnet(model_lag, test_df, scaler=scaler_lag, device=device, eval_mode="recursive")
                c3_rmse, c3_mae, c3_max = calc_metrics(y_test, res_c3.soh)
                raw_records.append({
                    "dataset": ds_name, "fold": held_out, "seed": seed, "model": "C3",
                    "soh_rmse": c3_rmse, "soh_mae": c3_mae, "soh_max_ae": c3_max,
                })

                # --- Train Model 2: Open-Loop (for C2) ---
                model_open, scaler_open, _ = train_pimdcnet(
                    train_dfs, cfg=SLACConfig(), n_epochs=BENCHMARK_EPOCHS,
                    device=device, seed=seed, open_loop=True, cycle_norm_scale=CYCLE_NORMALIZATION_SCALE,
                )

                # C2: Open-loop trained, open-loop evaluated
                res_c2 = predict_pimdcnet(model_open, test_df, scaler=scaler_open, device=device, eval_mode="open_loop")
                c2_rmse, c2_mae, c2_max = calc_metrics(y_test, res_c2.soh)
                raw_records.append({
                    "dataset": ds_name, "fold": held_out, "seed": seed, "model": "C2",
                    "soh_rmse": c2_rmse, "soh_mae": c2_mae, "soh_max_ae": c2_max,
                })

                logger.info(
                    "Fold %s [seed %d] | Lag: RMSE=%.4f | C1: RMSE=%.4f | C2: RMSE=%.4f | C3: RMSE=%.4f",
                    held_out, seed, b4_rmse, c1_rmse, c2_rmse, c3_rmse
                )

    raw_df = pd.DataFrame(raw_records)
    raw_path = output_dir / "soh_benchmark_raw.csv"
    raw_df.to_csv(raw_path, index=False)
    logger.info("Saved raw per-run results to %s", raw_path)

    # Aggregation:
    # 1. Mean over seeds per (dataset, model, fold)
    fold_df = raw_df.groupby(["dataset", "model", "fold"])[["soh_rmse", "soh_mae", "soh_max_ae"]].mean().reset_index()

    # 2. Per dataset mean and std over fold values
    model_order = [
        "Persistence", "Persistence+drift", "RF+lag", "XGB+lag",
        "PI-MDCNet lag-conditioned", "C1", "C2", "C3"
    ]
    
    summary_rows = []
    for model in model_order:
        row = {"model": model}
        for ds in ["nasa", "calce", "oxford"]:
            sub = fold_df[(fold_df["model"] == model) & (fold_df["dataset"] == ds)]
            rmse_vals = sub["soh_rmse"].values
            mae_vals = sub["soh_mae"].values
            max_ae_vals = sub["soh_max_ae"].values
            row[f"{ds}_rmse_mean"] = float(np.mean(rmse_vals))
            row[f"{ds}_rmse_std"] = float(np.std(rmse_vals, ddof=1)) if len(rmse_vals) > 1 else 0.0
            row[f"{ds}_mae_mean"] = float(np.mean(mae_vals))
            row[f"{ds}_mae_std"] = float(np.std(mae_vals, ddof=1)) if len(mae_vals) > 1 else 0.0
            row[f"{ds}_max_ae_mean"] = float(np.mean(max_ae_vals))
        
        # Overall across all 12 folds
        all_sub = fold_df[fold_df["model"] == model]
        all_rmse = all_sub["soh_rmse"].values
        all_mae = all_sub["soh_mae"].values
        row["overall_rmse_mean"] = float(np.mean(all_rmse))
        row["overall_rmse_std"] = float(np.std(all_rmse, ddof=1)) if len(all_rmse) > 1 else 0.0
        row["overall_mae_mean"] = float(np.mean(all_mae))
        row["overall_mae_std"] = float(np.std(all_mae, ddof=1)) if len(all_mae) > 1 else 0.0
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "soh_benchmark_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved summary results to %s", summary_path)

    print("\n" + "=" * 90)
    print("SOH BENCHMARK SUMMARY (SoH RMSE: Mean ± Std over folds)")
    print("=" * 90)
    fmt_rows = []
    for _, r in summary_df.iterrows():
        fmt_rows.append({
            "Model": r["model"],
            "NASA SoH RMSE": f"{r['nasa_rmse_mean']:.4f} ± {r['nasa_rmse_std']:.4f}",
            "CALCE SoH RMSE": f"{r['calce_rmse_mean']:.4f} ± {r['calce_rmse_std']:.4f}",
            "Oxford SoH RMSE": f"{r['oxford_rmse_mean']:.4f} ± {r['oxford_rmse_std']:.4f}",
            "Overall 12-Fold": f"{r['overall_rmse_mean']:.4f} ± {r['overall_rmse_std']:.4f}",
        })
    print(pd.DataFrame(fmt_rows).to_string(index=False))
    print("=" * 90 + "\n")

if __name__ == "__main__":
    run_benchmark()
