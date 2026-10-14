"""
LOBO Evaluation of Intra-Cycle SoC Auxiliary Module & Aging Drift Benchmarking
==============================================================================

Evaluates continuous intra-cycle SoC estimation under Leave-One-Battery-Out (LOBO)
cross-validation across both NASA PCoE (B0005, B0006, B0007, B0018) and Oxford
Kokam dynamic urban Artemis drive cycles (Cell1, Cell2, Cell3, Cell4).

Compares:
1. Naive Static Coulomb Counting (assuming static Q_nominal)
2. SOH-Conditioned Coulomb Counting (Q_aged = SOH_pred * Q_nominal)
3. SOH-Conditioned IntraCycle GRU ([V, I, t, SOH_pred] -> continuous SoC)

Stratified across aging stages:
- Fresh (SOH >= 0.90)
- Mid-life (0.80 <= SOH < 0.90)
- Near-EOL (SOH < 0.80)
- Overall
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from experiments.protocol import BENCHMARK_SEEDS
from src.data_loader import load_all_cells
from src.models.pi_mdcnet import PIMDCNet, SLACConfig
from experiments.train import FeatureScaler, CycleSequenceDataset, train_pimdcnet, predict_pimdcnet
from src.models.intracycle_soc import (
    extract_intracycle_profiles,
    extract_oxford_intracycle_profiles,
    estimate_naive_coulomb_counting,
    estimate_soh_conditioned_coulomb_counting,
    compute_intracycle_stats,
    IntraCycleDataset,
    IntraCycleSoCGRU,
    CycleTimeProfile,
)

NASA_CELLS = ["B0005", "B0006", "B0007", "B0018"]
OXFORD_CELLS = ["Cell1", "Cell2", "Cell3", "Cell4"]


def pad_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad batch of sequences to max length in batch and return mask."""
    x_list = [item[0] for item in batch]  # each is (T_i, 4)
    y_list = [item[1] for item in batch]  # each is (T_i, 1)

    lengths = [len(x) for x in x_list]
    max_len = max(lengths)

    x_padded = pad_sequence(x_list, batch_first=True, padding_value=0.0)  # (B, max_len, 4)
    y_padded = pad_sequence(y_list, batch_first=True, padding_value=0.0)  # (B, max_len, 1)

    mask = torch.zeros((len(batch), max_len, 1), dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l, 0] = True

    return x_padded, y_padded, mask


def train_soc_model(
    train_profiles: list[CycleTimeProfile],
    stats: dict[str, float],
    n_epochs: int = 20,
    batch_size: int = 32,
    lr: float = 5e-3,
    device: str = "cpu",
    seed: int = 42,
) -> IntraCycleSoCGRU:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = IntraCycleSoCGRU(input_dim=4, hidden_dim=32, num_layers=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    dataset = IntraCycleDataset(
        train_profiles,
        v_mean=stats["v_mean"],
        v_std=stats["v_std"],
        i_mean=stats["i_mean"],
        i_std=stats["i_std"],
        t_scale=stats["t_scale"],
        include_soh=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate_fn,
        drop_last=False,
    )

    model.train()
    for epoch in range(n_epochs):
        for x_pad, y_pad, mask in loader:
            x_pad = x_pad.to(device)
            y_pad = y_pad.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            pred = model(x_pad)
            loss = torch.mean((pred[mask] - y_pad[mask]) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    return model


def evaluate_profiles(
    model: IntraCycleSoCGRU | None,
    profiles: list[CycleTimeProfile],
    stats: dict[str, float],
    soh_preds: np.ndarray | None = None,
    device: str = "cpu",
) -> dict[str, dict[str, dict[str, float]]]:
    """Evaluate Naive CC, SOH-conditioned CC, and GRU across aging tiers."""
    dataset = IntraCycleDataset(
        profiles,
        v_mean=stats["v_mean"],
        v_std=stats["v_std"],
        i_mean=stats["i_mean"],
        i_std=stats["i_std"],
        t_scale=stats["t_scale"],
        include_soh=True,
    )

    if model is not None:
        model.eval()

    tier_names = ["Fresh (>=90%)", "Mid-life (80-90%)", "Near-EOL (<80%)", "Overall"]
    methods = ["Naive_Static_CC", "SOH_Conditioned_CC", "IntraCycle_GRU"]

    # Storage: method -> tier -> list of (pred_chunk, true_chunk)
    eval_chunks = {m: {t: ([], []) for t in tier_names} for m in methods}

    with torch.no_grad():
        for i, p in enumerate(profiles):
            true = p.soc_true
            soh = p.soh_true

            # Categorize tier
            if soh >= 0.90:
                tiers_for_p = ["Fresh (>=90%)", "Overall"]
            elif soh >= 0.80:
                tiers_for_p = ["Mid-life (80-90%)", "Overall"]
            else:
                tiers_for_p = ["Near-EOL (<80%)", "Overall"]

            # 1. Naive CC
            naive_pred = estimate_naive_coulomb_counting(p)
            for t in tiers_for_p:
                eval_chunks["Naive_Static_CC"][t][0].append(naive_pred)
                eval_chunks["Naive_Static_CC"][t][1].append(true)

            # 2. SOH-Conditioned CC (conditioned on seeded PI-MDCNet prediction)
            if soh_preds is not None and len(soh_preds) > 0:
                c_idx = min(int(p.cycle_idx), len(soh_preds) - 1)
                soh_val = float(soh_preds[c_idx])
            else:
                soh_val = p.soh_pred
            cond_pred = estimate_soh_conditioned_coulomb_counting(p, soh_pred=soh_val)
            for t in tiers_for_p:
                eval_chunks["SOH_Conditioned_CC"][t][0].append(cond_pred)
                eval_chunks["SOH_Conditioned_CC"][t][1].append(true)

            # 3. GRU
            if model is not None:
                x, _ = dataset[i]
                x = x.unsqueeze(0).to(device)
                gru_pred = model(x).squeeze(0).squeeze(-1).cpu().numpy()
                for t in tiers_for_p:
                    eval_chunks["IntraCycle_GRU"][t][0].append(gru_pred)
                    eval_chunks["IntraCycle_GRU"][t][1].append(true)

    # Compute metrics per method and tier
    results = {}
    for m in methods:
        results[m] = {}
        for t in tier_names:
            preds_list, trues_list = eval_chunks[m][t]
            if len(preds_list) == 0:
                results[m][t] = {"rmse": np.nan, "mae": np.nan, "max_err": np.nan, "timesteps": 0, "cycles": 0}
                continue
            cat_p = np.concatenate(preds_list)
            cat_t = np.concatenate(trues_list)
            rmse = float(np.sqrt(np.mean((cat_p - cat_t) ** 2)))
            mae = float(np.mean(np.abs(cat_p - cat_t)))
            max_err = float(np.max(np.abs(cat_p - cat_t)))
            results[m][t] = {
                "rmse": rmse,
                "mae": mae,
                "max_err": max_err,
                "timesteps": len(cat_t),
                "cycles": len(preds_list),
            }

    return results


def run_full_soc_benchmark(data_dir: Path, seeds: list[int] = BENCHMARK_SEEDS) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 90, flush=True)
    print(f"INTRA-CYCLE CONTINUOUS SoC BENCHMARK (Device: {device})", flush=True)
    print("Coupled Aging SOH vs Naive Open-Loop Static Drift across NASA and Oxford", flush=True)
    print("=" * 90, flush=True)

    records = []

    # 1. Benchmark NASA Cells
    nasa_profiles = {}
    for cell in NASA_CELLS:
        mat_file = data_dir / f"{cell}.mat"
        if mat_file.exists():
            nasa_profiles[cell] = extract_intracycle_profiles(mat_file, nominal_capacity=2.0)
            print(f"Loaded NASA {cell}: {len(nasa_profiles[cell])} cycles", flush=True)

    # 2. Benchmark Oxford Cells
    oxford_profiles = {}
    oxford_dir = data_dir / "oxford"
    for cell in OXFORD_CELLS:
        csv_file = oxford_dir / f"{cell}.csv"
        if csv_file.exists():
            oxford_profiles[cell] = extract_oxford_intracycle_profiles(csv_file, nominal_capacity=0.740)
            print(f"Loaded Oxford {cell}: {len(oxford_profiles[cell])} dynamic drive cycles", flush=True)

    datasets_to_run = [("NASA", nasa_profiles, NASA_CELLS), ("Oxford", oxford_profiles, OXFORD_CELLS)]

    for dataset_name, cell_dict, cell_list in datasets_to_run:
        if not cell_dict:
            continue
        print("\n" + "-" * 90, flush=True)
        print(f"EVALUATING DATASET: {dataset_name} (LOBO Cross-Validation)", flush=True)
        print("-" * 90, flush=True)

        for held_out in cell_list:
            if held_out not in cell_dict:
                continue
            train_profs = []
            for c in cell_list:
                if c != held_out and c in cell_dict:
                    train_profs.extend(cell_dict[c])

            stats = compute_intracycle_stats(train_profs)
            test_profs = cell_dict[held_out]

            d_key = "nasa" if dataset_name == "NASA" else "oxford"
            cell_dfs = load_all_cells(data_dir, dataset=d_key)
            train_dfs = [cell_dfs[c] for c in cell_list if c != held_out and c in cell_dfs]
            test_df = cell_dfs[held_out]

            for seed in seeds:
                # 1. Train seed-specific PI-MDCNet for realistic SOH-conditioning
                pimdc_model, scaler, _ = train_pimdcnet(
                    train_dfs, cfg=SLACConfig(), n_epochs=30, device=device, seed=seed
                )
                pimdc_res = predict_pimdcnet(pimdc_model, test_df, scaler=scaler, device=device)
                soh_preds = pimdc_res.soh

                # 2. Train seed-specific IntraCycle GRU
                model = train_soc_model(
                    train_profs,
                    stats,
                    n_epochs=20,
                    batch_size=32,
                    lr=5e-3,
                    device=device,
                    seed=seed,
                )
                metrics = evaluate_profiles(model, test_profs, stats, soh_preds=soh_preds, device=device)

                for method, tier_dict in metrics.items():
                    for tier, m_vals in tier_dict.items():
                        if np.isnan(m_vals["rmse"]):
                            continue
                        records.append({
                            "dataset": dataset_name,
                            "fold": held_out,
                            "seed": seed,
                            "method": method,
                            "aging_tier": tier,
                            "rmse": m_vals["rmse"],
                            "mae": m_vals["mae"],
                            "max_err": m_vals["max_err"],
                            "timesteps": m_vals["timesteps"],
                            "cycles": m_vals["cycles"],
                        })

    df_res = pd.DataFrame(records)
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    df_res.to_csv(artifacts_dir / "intracycle_soc_results.csv", index=False)
    print(f"\n[OK] Saved complete intra-cycle benchmark to {artifacts_dir / 'intracycle_soc_results.csv'}", flush=True)

    # Print publication summary table
    print("\n" + "=" * 90, flush=True)
    print("QUANTITATIVE AGING DRIFT COMPARISON ACROSS SOH TIERS (Mean ± Std)", flush=True)
    print("=" * 90, flush=True)
    summary = df_res.groupby(["dataset", "aging_tier", "method"])[["rmse", "mae"]].agg(["mean", "std"])
    print(summary.to_string(), flush=True)
    print("=" * 90, flush=True)

    return df_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--seeds", type=int, nargs="+", default=BENCHMARK_SEEDS)
    args = parser.parse_args()
    run_full_soc_benchmark(args.data_dir, seeds=args.seeds)

