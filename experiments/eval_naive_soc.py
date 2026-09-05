"""
Zero-Learning Naive Coulomb-Counting Baseline for SoC
=====================================================

Calculates SoC via open-loop Coulomb integration normalized by RATED capacity (2.0 Ah):
    SoC_naive(t) = 1.0 - (∫₀ᵗ |I(τ)| dτ) / 2.0 Ah

Evaluated across all 4 NASA cells (B0005, B0006, B0007, B0018) on the exact same
intra-cycle discharge profiles as the GRU module.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from src.models.intracycle_soc import extract_intracycle_profiles

CELL_IDS = ["B0005", "B0006", "B0007", "B0018"]
data_dir = Path("data/raw")

print("=" * 85, flush=True)
print("NAIVE COULOMB-COUNTING ZERO-LEARNING BASELINE (Q_rated = 2.0 Ah)", flush=True)
print("=" * 85, flush=True)
print(f"{'Cell':<8} | {'Cycles':<8} | {'Timesteps':<10} | {'RMSE':<10} | {'MAE':<10} | {'Max Err':<10}", flush=True)
print("-" * 85, flush=True)

all_preds = []
all_trues = []
cell_metrics = []

for cell in CELL_IDS:
    profs = extract_intracycle_profiles(data_dir / f"{cell}.mat")
    cell_preds = []
    cell_trues = []
    for p in profs:
        dt = np.diff(p.time_s)
        curr_mid = np.abs(p.current[1:] + p.current[:-1]) / 2.0
        dq = curr_mid * dt / 3600.0
        q = np.concatenate([[0.0], np.cumsum(dq)]).astype(np.float32)
        soc_naive = np.clip(1.0 - (q / 2.0), 0.0, 1.0)
        cell_preds.append(soc_naive)
        cell_trues.append(p.soc_true)

    cp = np.concatenate(cell_preds)
    ct = np.concatenate(cell_trues)
    rmse = float(np.sqrt(np.mean((cp - ct) ** 2)))
    mae = float(np.mean(np.abs(cp - ct)))
    max_e = float(np.max(np.abs(cp - ct)))

    cell_metrics.append({"cell": cell, "rmse": rmse, "mae": mae, "max_err": max_e, "timesteps": len(ct), "cycles": len(profs)})
    print(f"{cell:<8} | {len(profs):<8} | {len(ct):<10} | {rmse:<10.4f} | {mae:<10.4f} | {max_e:<10.4f}", flush=True)

    all_preds.append(cp)
    all_trues.append(ct)

concat_all_p = np.concatenate(all_preds)
concat_all_t = np.concatenate(all_trues)
overall_rmse = float(np.sqrt(np.mean((concat_all_p - concat_all_t) ** 2)))
overall_mae = float(np.mean(np.abs(concat_all_p - concat_all_t)))

print("=" * 85, flush=True)
print(f"Overall Naive Coulomb-Counting RMSE : {overall_rmse:.4f} ({overall_rmse*100:.2f}%)", flush=True)
print(f"Overall Naive Coulomb-Counting MAE  : {overall_mae:.4f} ({overall_mae*100:.2f}%)", flush=True)
print("=" * 85, flush=True)

# Comparison with GRU
print("\n" + "=" * 85, flush=True)
print("DIRECT COMPARISON: NAIVE COULOMB COUNTING vs. INTRA-CYCLE GRU", flush=True)
print("=" * 85, flush=True)
df_gru = pd.read_csv("artifacts/intracycle_soc_results.csv")
gru_summary = df_gru.groupby("fold")[["rmse", "mae"]].mean()

print(f"{'Cell':<8} | {'Naive RMSE':<12} | {'GRU RMSE':<12} | {'Naive MAE':<12} | {'GRU MAE':<12} | {'Error Reduction':<15}", flush=True)
print("-" * 85, flush=True)
for cm in cell_metrics:
    c = cm["cell"]
    n_rmse = cm["rmse"]
    g_rmse = gru_summary.loc[c, "rmse"]
    n_mae = cm["mae"]
    g_mae = gru_summary.loc[c, "mae"]
    reduc = (1.0 - g_rmse / n_rmse) * 100
    print(f"{c:<8} | {n_rmse:<12.4f} | {g_rmse:<12.4f} | {n_mae:<12.4f} | {g_mae:<12.4f} | -{reduc:<14.1f}%", flush=True)

print("=" * 85, flush=True)
overall_gru_rmse = float(df_gru["rmse"].mean())
overall_gru_mae = float(df_gru["mae"].mean())
overall_reduc = (1.0 - overall_gru_rmse / overall_rmse) * 100
print(f"OVERALL  | {overall_rmse:<12.4f} | {overall_gru_rmse:<12.4f} | {overall_mae:<12.4f} | {overall_gru_mae:<12.4f} | -{overall_reduc:<14.1f}%", flush=True)
print("=" * 85, flush=True)
