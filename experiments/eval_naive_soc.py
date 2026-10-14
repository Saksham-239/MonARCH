"""
Zero-Learning Naive Coulomb-Counting Baseline for SoC
=====================================================

Calculates SoC via open-loop Coulomb integration normalized by RATED capacity (2.0 Ah):
    SoC_naive(t) = 1.0 - (∫₀ᵗ |I(τ)| dτ) / 2.0 Ah

Evaluated across all 4 NASA cells (B0005, B0006, B0007, B0018) on the exact same
intra-cycle discharge profiles as the GRU module.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

# Comparison with GRU and SOH-Conditioned CC across aging tiers
print("\n" + "=" * 90, flush=True)
print("AGING-STRATIFIED COMPARISON: NAIVE DRIFT vs SOH-CONDITIONED / GRU ESTIMATION", flush=True)
print("=" * 90, flush=True)
results_csv = Path("artifacts/intracycle_soc_results.csv")
if results_csv.exists():
    df_soc = pd.read_csv(results_csv)
    tier_summary = df_soc.groupby(["dataset", "aging_tier", "method"])[["rmse", "mae"]].mean().reset_index()
    print(f"{'Dataset':<10} | {'Aging Tier':<20} | {'Method':<25} | {'RMSE':<10} | {'MAE':<10}", flush=True)
    print("-" * 90, flush=True)
    for _, row in tier_summary.iterrows():
        print(f"{row['dataset']:<10} | {row['aging_tier']:<20} | {row['method']:<25} | {row['rmse']:<10.4f} | {row['mae']:<10.4f}", flush=True)
    print("=" * 90, flush=True)
else:
    print("Run `python experiments/run_intracycle_soc.py` first to generate artifacts/intracycle_soc_results.csv")

