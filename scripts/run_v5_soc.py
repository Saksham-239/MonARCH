"""V5 Verification Script: Part E1 (Continuous Intra-Cycle SoC Benchmark with Model-Free Comparator).

Evaluates across NASA and Oxford:
- Naive_Static_CC: Q_nom constant
- SOH_Conditioned_CC: Q_aged = PI_MDCNet_SoH * Q_nom
- IntraCycle_GRU: GRU trained on voltage, current, time
- ModelFree_Lag_CC: Q_aged = SoH[t-1] * Q_nom (true previous SoH, NO network)
Reports RMSE per aging tier:
- Fresh (>=90%)
- Mid-life (80-90%)
- Near-EOL (<80%)
- Overall
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

from src.data_loader import load_all_cells
from experiments.protocol import BENCHMARK_SEEDS, CYCLE_NORMALIZATION_SCALE
from experiments.train import train_pimdcnet, predict_pimdcnet
from src.models.pi_mdcnet import SLACConfig
from src.models.intracycle_soc import (
    CycleTimeProfile,
    extract_intracycle_profiles,
    extract_oxford_intracycle_profiles,
    compute_intracycle_stats,
    IntraCycleDataset,
    estimate_naive_coulomb_counting,
    estimate_soh_conditioned_coulomb_counting,
)
from experiments.run_intracycle_soc import train_soc_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("V5_SoC")

NASA_CELLS = ["B0005", "B0006", "B0007", "B0018"]
OXFORD_CELLS = ["Cell1", "Cell2", "Cell3", "Cell4"]

def estimate_model_free_lag_coulomb_counting(profile: CycleTimeProfile, lag_soh_val: float) -> np.ndarray:
    """Model-free comparator using true previous cycle's SoH: Q_aged = SoH[t-1] * Q_nom."""
    q_aged = max(lag_soh_val, 0.10) * profile.nominal_capacity_ah
    dt = np.diff(profile.time_s)
    curr_mid = np.abs(profile.current[1:] + profile.current[:-1]) / 2.0
    dq = curr_mid * dt / 3600.0
    q = np.concatenate([[0.0], np.cumsum(dq)]).astype(np.float32)
    return np.clip(1.0 - (q / q_aged), 0.0, 1.0)

def run_soc():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Running SoC benchmark on device: %s", device)

    data_dir = Path("data/raw") if Path("data/raw").exists() else Path("data")
    output_dir = Path("artifacts/v5")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load NASA Profiles
    nasa_profiles = {}
    for cell in NASA_CELLS:
        mat_file = data_dir / f"{cell}.mat"
        if mat_file.exists():
            nasa_profiles[cell] = extract_intracycle_profiles(mat_file, nominal_capacity=2.0)
            logger.info("Loaded NASA %s: %d profiles", cell, len(nasa_profiles[cell]))

    # 2. Load Oxford Profiles
    oxford_profiles = {}
    oxford_dir = data_dir / "oxford"
    for cell in OXFORD_CELLS:
        csv_file = oxford_dir / f"{cell}.csv"
        if csv_file.exists():
            oxford_profiles[cell] = extract_oxford_intracycle_profiles(csv_file, nominal_capacity=0.740)
            logger.info("Loaded Oxford %s: %d profiles", cell, len(oxford_profiles[cell]))

    datasets_to_run = [("NASA", nasa_profiles, NASA_CELLS, "nasa"), ("Oxford", oxford_profiles, OXFORD_CELLS, "oxford")]

    tier_names = ["Fresh (>=90%)", "Mid-life (80-90%)", "Near-EOL (<80%)", "Overall"]
    methods = ["Naive_Static_CC", "SOH_Conditioned_CC", "ModelFree_Lag_CC", "IntraCycle_GRU"]

    records = []

    for ds_name, cell_dict, cell_list, ds_key in datasets_to_run:
        if not cell_dict:
            continue
        logger.info("=== EVALUATING SoC: %s ===", ds_name)
        cell_dfs = load_all_cells(data_dir, dataset=ds_key)

        for held_out in cell_list:
            if held_out not in cell_dict:
                continue

            train_profs = []
            for c in cell_list:
                if c != held_out and c in cell_dict:
                    train_profs.extend(cell_dict[c])

            stats = compute_intracycle_stats(train_profs)
            test_profs = cell_dict[held_out]
            test_df = cell_dfs[held_out]
            train_dfs = [cell_dfs[c] for c in cell_list if c != held_out]

            for seed in BENCHMARK_SEEDS:
                # 1. Train seed-specific PI-MDCNet
                pimdc_model, scaler, _ = train_pimdcnet(
                    train_dfs, cfg=SLACConfig(), n_epochs=30, device=device, seed=seed,
                    cycle_norm_scale=CYCLE_NORMALIZATION_SCALE,
                )
                pimdc_res = predict_pimdcnet(pimdc_model, test_df, scaler=scaler, device=device)
                soh_preds = pimdc_res.soh

                # 2. Train seed-specific IntraCycle GRU
                gru_model = train_soc_model(
                    train_profs, stats, n_epochs=20, batch_size=32, lr=5e-3, device=device, seed=seed,
                )
                gru_model.eval()

                # Dataset for GRU
                test_dataset = IntraCycleDataset(
                    test_profs,
                    v_mean=stats["v_mean"], v_std=stats["v_std"],
                    i_mean=stats["i_mean"], i_std=stats["i_std"],
                    t_scale=stats["t_scale"], include_soh=True,
                )

                # Storage for chunks
                eval_chunks = {m: {t: ([], []) for t in tier_names} for m in methods}

                with torch.no_grad():
                    for i, p in enumerate(test_profs):
                        true_soc = p.soc_true
                        true_soh = p.soh_true

                        if true_soh >= 0.90:
                            active_tiers = ["Fresh (>=90%)", "Overall"]
                        elif true_soh >= 0.80:
                            active_tiers = ["Mid-life (80-90%)", "Overall"]
                        else:
                            active_tiers = ["Near-EOL (<80%)", "Overall"]

                        # 1. Naive CC
                        naive_p = estimate_naive_coulomb_counting(p)
                        for t in active_tiers:
                            eval_chunks["Naive_Static_CC"][t][0].append(naive_p)
                            eval_chunks["Naive_Static_CC"][t][1].append(true_soc)

                        # 2. SOH-Conditioned CC (conditioned on PI-MDCNet prediction)
                        c_idx = min(int(p.cycle_idx), len(soh_preds) - 1)
                        soh_val = float(soh_preds[c_idx])
                        cond_p = estimate_soh_conditioned_coulomb_counting(p, soh_pred=soh_val)
                        for t in active_tiers:
                            eval_chunks["SOH_Conditioned_CC"][t][0].append(cond_p)
                            eval_chunks["SOH_Conditioned_CC"][t][1].append(true_soc)

                        # 3. Model-Free Lag CC: Q_aged = SoH[t-1] * Q_nom
                        if i == 0:
                            lag_soh_val = float(p.soh_true)
                        else:
                            lag_soh_val = float(test_profs[i - 1].soh_true)
                        modelfree_p = estimate_model_free_lag_coulomb_counting(p, lag_soh_val=lag_soh_val)
                        for t in active_tiers:
                            eval_chunks["ModelFree_Lag_CC"][t][0].append(modelfree_p)
                            eval_chunks["ModelFree_Lag_CC"][t][1].append(true_soc)

                        # 4. IntraCycle GRU
                        x_t, _ = test_dataset[i]
                        x_t = x_t.unsqueeze(0).to(device)
                        gru_p = gru_model(x_t).squeeze(0).squeeze(-1).detach().cpu().numpy()
                        for t in active_tiers:
                            eval_chunks["IntraCycle_GRU"][t][0].append(gru_p)
                            eval_chunks["IntraCycle_GRU"][t][1].append(true_soc)

                for m in methods:
                    for t in tier_names:
                        p_list, t_list = eval_chunks[m][t]
                        if not p_list:
                            continue
                        cat_p = np.concatenate(p_list)
                        cat_t = np.concatenate(t_list)
                        rmse = float(np.sqrt(np.mean((cat_p - cat_t) ** 2)))
                        mae = float(np.mean(np.abs(cat_p - cat_t)))
                        records.append({
                            "dataset": ds_name,
                            "fold": held_out,
                            "seed": seed,
                            "method": m,
                            "aging_tier": t,
                            "rmse": rmse,
                            "mae": mae,
                            "timesteps": len(cat_p),
                            "cycles": len(p_list),
                        })

    df_res = pd.DataFrame(records)
    raw_path = output_dir / "soc_benchmark_raw.csv"
    df_res.to_csv(raw_path, index=False)
    logger.info("Saved raw SoC benchmark results to %s", raw_path)

    # Summary table: aggregate per (dataset, method, aging_tier)
    # First mean over seeds per fold
    fold_df = df_res.groupby(["dataset", "method", "aging_tier", "fold"])[["rmse", "mae"]].mean().reset_index()

    sum_rows = []
    for (ds, method, tier), sub in fold_df.groupby(["dataset", "method", "aging_tier"]):
        sum_rows.append({
            "dataset": ds,
            "method": method,
            "aging_tier": tier,
            "rmse_mean": float(sub["rmse"].mean()),
            "rmse_std": float(sub["rmse"].std(ddof=1)) if len(sub) > 1 else 0.0,
            "mae_mean": float(sub["mae"].mean()),
            "mae_std": float(sub["mae"].std(ddof=1)) if len(sub) > 1 else 0.0,
        })

    sum_df = pd.DataFrame(sum_rows)
    sum_path = output_dir / "soc_comparator_summary.csv"
    sum_df.to_csv(sum_path, index=False)
    logger.info("Saved SoC comparator summary table to %s", sum_path)

    print("\n" + "=" * 90)
    print("CONTINUOUS SOC AGING DRIFT COMPARISON: RMSE (Mean ± Std over folds)")
    print("=" * 90)
    for ds in ["NASA", "Oxford"]:
        print(f"\n--- DATASET: {ds} ---")
        piv = sum_df[sum_df["dataset"] == ds].pivot(index="method", columns="aging_tier", values="rmse_mean")
        piv_std = sum_df[sum_df["dataset"] == ds].pivot(index="method", columns="aging_tier", values="rmse_std")
        formatted = pd.DataFrame(index=piv.index)
        for col in tier_names:
            if col in piv.columns:
                formatted[col] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(piv[col], piv_std[col])]
        print(formatted.to_string())
    print("=" * 90 + "\n")

if __name__ == "__main__":
    run_soc()
