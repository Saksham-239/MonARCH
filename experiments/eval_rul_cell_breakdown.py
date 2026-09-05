"""
Per-Cell Win/Loss and Breakdown Analysis for RUL Methods
========================================================
"""

import pandas as pd
import numpy as np

df_rul = pd.read_csv("artifacts/rul_validation_results.csv")
df_comp = pd.read_csv("artifacts/comparison_table.csv")

methods_order = [
    "Raw_SoH_Extrap",
    "Stat_Decomp",
    "RandomForest",
    "XGBoost",
    "Neural_RULHead",
    "Latent_Dt_Extrap",
]

print("=" * 95)
print("PER-CELL RUL BENCHMARK BREAKDOWN (MAE & RMSE in cycles)")
print("=" * 95)

table_data = []

for fold in ["B0005", "B0006", "B0018", "B0007"]:
    sub_rul = df_rul[df_rul["fold"] == fold]
    sub_comp = df_comp[df_comp["fold"] == fold]

    row_data = {"fold": fold}

    for m in ["Raw_SoH_Extrap", "Stat_Decomp", "Neural_RULHead", "Latent_Dt_Extrap"]:
        v = sub_rul[sub_rul["method"] == m]["mae"].dropna()
        r = sub_rul[sub_rul["method"] == m]["rmse"].dropna()
        if len(v) > 0:
            row_data[f"{m}_mae"] = f"{v.mean():5.2f} ± {v.std():4.2f}"
            row_data[f"{m}_rmse"] = f"{r.mean():5.2f} ± {r.std():4.2f}"
        else:
            row_data[f"{m}_mae"] = "NaN"
            row_data[f"{m}_rmse"] = "NaN"

    for tm in ["RandomForest", "XGBoost"]:
        v = sub_comp[sub_comp["model"] == tm]["rul_mae"].dropna()
        r = sub_comp[sub_comp["model"] == tm]["rul_rmse"].dropna()
        if len(v) > 0:
            row_data[f"{tm}_mae"] = f"{v.mean():5.2f} ± {v.std():4.2f}"
            row_data[f"{tm}_rmse"] = f"{r.mean():5.2f} ± {r.std():4.2f}"
        else:
            row_data[f"{tm}_mae"] = "NaN"
            row_data[f"{tm}_rmse"] = "NaN"

    table_data.append(row_data)

# Print per fold
for r in table_data:
    fold = r["fold"]
    print(f"\n--- Held-Out Cell: {fold} ---")
    if fold == "B0007":
        print("  All methods: Right-censored (min capacity 1.4005 Ah > 1.40 Ah; never reaches EOL).")
        continue
    for m in methods_order:
        mae_val = r.get(f"{m}_mae", "N/A")
        rmse_val = r.get(f"{m}_rmse", "N/A")
        print(f"  {m:<20} | MAE: {mae_val:<15} | RMSE: {rmse_val:<15}")

# Head-to-head win/loss record of Latent_Dt_Extrap vs each baseline
print("\n" + "=" * 95)
print("HEAD-TO-HEAD WIN/LOSS RECORD FOR Latent_Dt_Extrap (Across 3 Valid EOL Cells)")
print("=" * 95)

# Calculate mean MAE per fold for each method
means_by_fold = {}
for fold in ["B0005", "B0006", "B0018"]:
    means_by_fold[fold] = {}
    sub_rul = df_rul[df_rul["fold"] == fold]
    sub_comp = df_comp[df_comp["fold"] == fold]
    for m in ["Raw_SoH_Extrap", "Stat_Decomp", "Neural_RULHead", "Latent_Dt_Extrap"]:
        means_by_fold[fold][m] = sub_rul[sub_rul["method"] == m]["mae"].mean()
    for tm in ["RandomForest", "XGBoost"]:
        means_by_fold[fold][tm] = sub_comp[sub_comp["model"] == tm]["rul_mae"].mean()

for baseline in ["Raw_SoH_Extrap", "Stat_Decomp", "Neural_RULHead", "RandomForest", "XGBoost"]:
    wins = 0
    losses = 0
    details = []
    for fold in ["B0005", "B0006", "B0018"]:
        dt_mae = means_by_fold[fold]["Latent_Dt_Extrap"]
        base_mae = means_by_fold[fold][baseline]
        if dt_mae < base_mae:
            wins += 1
            details.append(f"{fold}: WIN ({dt_mae:.2f} vs {base_mae:.2f})")
        else:
            losses += 1
            details.append(f"{fold}: LOSS ({dt_mae:.2f} vs {base_mae:.2f})")
    print(f"Latent_Dt_Extrap vs {baseline:<18}: {wins} W - {losses} L | {', '.join(details)}")

print("=" * 95)
