"""
LOBO Evaluation of Intra-Cycle SoC Auxiliary Module
===================================================

Evaluates the lightweight intra-cycle SoC GRU under Leave-One-Battery-Out (LOBO)
cross-validation across all 4 NASA cells (B0005, B0006, B0007, B0018).

Optimized with batched training and sequence masking for fast convergence.
Reports:
- Per-cell intra-cycle SoC RMSE, MAE, and Max Error
- Mean ± Std across LOBO folds
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.models.intracycle_soc import (
    extract_intracycle_profiles,
    compute_intracycle_stats,
    IntraCycleDataset,
    IntraCycleSoCGRU,
)

CELL_IDS = ["B0005", "B0006", "B0007", "B0018"]


def pad_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad batch of sequences to max length in batch and return mask."""
    x_list = [item[0] for item in batch]  # each is (T_i, 3)
    y_list = [item[1] for item in batch]  # each is (T_i, 1)

    lengths = [len(x) for x in x_list]
    max_len = max(lengths)

    x_padded = pad_sequence(x_list, batch_first=True, padding_value=0.0)  # (B, max_len, 3)
    y_padded = pad_sequence(y_list, batch_first=True, padding_value=0.0)  # (B, max_len, 1)

    mask = torch.zeros((len(batch), max_len, 1), dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l, 0] = True

    return x_padded, y_padded, mask


def train_soc_model(
    train_profiles,
    stats,
    n_epochs: int = 25,
    batch_size: int = 32,
    lr: float = 5e-3,
    device: str = "cpu",
    seed: int = 42,
) -> IntraCycleSoCGRU:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = IntraCycleSoCGRU(input_dim=3, hidden_dim=32, num_layers=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    dataset = IntraCycleDataset(
        train_profiles,
        v_mean=stats["v_mean"],
        v_std=stats["v_std"],
        i_mean=stats["i_mean"],
        i_std=stats["i_std"],
        t_scale=stats["t_scale"],
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
            # Masked MSE loss
            loss = torch.mean((pred[mask] - y_pad[mask]) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    return model


def evaluate_soc_model(
    model: IntraCycleSoCGRU,
    test_profiles,
    stats,
    device: str = "cpu",
) -> dict[str, float]:
    dataset = IntraCycleDataset(
        test_profiles,
        v_mean=stats["v_mean"],
        v_std=stats["v_std"],
        i_mean=stats["i_mean"],
        i_std=stats["i_std"],
        t_scale=stats["t_scale"],
    )

    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for i in range(len(dataset)):
            x, y = dataset[i]
            x = x.unsqueeze(0).to(device)
            pred = model(x).squeeze(0).squeeze(-1).cpu().numpy()  # (T,)
            true = y.squeeze(-1).cpu().numpy()                    # (T,)
            all_preds.append(pred)
            all_trues.append(true)

    concat_pred = np.concatenate(all_preds)
    concat_true = np.concatenate(all_trues)

    rmse = float(np.sqrt(np.mean((concat_pred - concat_true) ** 2)))
    mae = float(np.mean(np.abs(concat_pred - concat_true)))
    max_err = float(np.max(np.abs(concat_pred - concat_true)))

    return {
        "rmse": rmse,
        "mae": mae,
        "max_err": max_err,
        "n_timesteps": len(concat_true),
        "n_cycles": len(test_profiles),
    }


def run_lobo_soc_benchmark(data_dir: Path, seeds: list[int] = [42, 43, 44]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80, flush=True)
    print(f"INTRA-CYCLE CONTINUOUS SoC LOBO BENCHMARK (Device: {device})", flush=True)
    print("=" * 80, flush=True)

    # 1. Load profiles for all cells
    cell_profiles = {}
    for cell in CELL_IDS:
        mat_file = data_dir / f"{cell}.mat"
        profiles = extract_intracycle_profiles(mat_file)
        cell_profiles[cell] = profiles
        all_socs = np.concatenate([p.soc_true for p in profiles])
        print(f"Cell {cell:5s}: {len(profiles):3d} cycles, {len(all_socs):6d} timesteps | "
              f"SoC min={all_socs.min():.4f}, max={all_socs.max():.4f}, ptp={np.ptp(all_socs):.4f}, std={all_socs.std():.4f}", flush=True)

    print("-" * 80, flush=True)
    print(f"{'Fold (Held-Out)':<15} | {'Seed':<5} | {'RMSE':<10} | {'MAE':<10} | {'Max Err':<10} | {'Timesteps':<10}", flush=True)
    print("-" * 80, flush=True)

    records = []
    fold_results = {c: [] for c in CELL_IDS}

    for held_out in CELL_IDS:
        train_cells = [c for c in CELL_IDS if c != held_out]
        train_profs = []
        for c in train_cells:
            train_profs.extend(cell_profiles[c])

        stats = compute_intracycle_stats(train_profs)
        test_profs = cell_profiles[held_out]

        for seed in seeds:
            model = train_soc_model(
                train_profs,
                stats,
                n_epochs=25,
                batch_size=32,
                lr=5e-3,
                device=device,
                seed=seed,
            )
            res = evaluate_soc_model(model, test_profs, stats, device=device)
            fold_results[held_out].append(res)
            records.append({
                "fold": held_out,
                "seed": seed,
                "rmse": res["rmse"],
                "mae": res["mae"],
                "max_err": res["max_err"],
                "timesteps": res["n_timesteps"],
                "cycles": res["n_cycles"],
            })
            print(f"{held_out:<15} | {seed:<5} | {res['rmse']:<10.4f} | {res['mae']:<10.4f} | {res['max_err']:<10.4f} | {res['n_timesteps']:<10}", flush=True)

    print("=" * 80, flush=True)
    print("SUMMARY: INTRA-CYCLE CONTINUOUS SoC LOBO RESULTS (Mean ± Std)", flush=True)
    print("=" * 80, flush=True)
    all_rmses = []
    all_maes = []
    for held_out in CELL_IDS:
        rmses = [r["rmse"] for r in fold_results[held_out]]
        maes = [r["mae"] for r in fold_results[held_out]]
        all_rmses.extend(rmses)
        all_maes.extend(maes)
        print(f"Held-out {held_out:5s} : RMSE = {np.mean(rmses):.4f} ± {np.std(rmses):.4f} | MAE = {np.mean(maes):.4f} ± {np.std(maes):.4f}", flush=True)

    print("-" * 80, flush=True)
    print(f"Overall LOBO SoC RMSE : {np.mean(all_rmses):.4f} ± {np.std(all_rmses):.4f} ({np.mean(all_rmses)*100:.2f}%)", flush=True)
    print(f"Overall LOBO SoC MAE  : {np.mean(all_maes):.4f} ± {np.std(all_maes):.4f} ({np.mean(all_maes)*100:.2f}%)", flush=True)
    print("=" * 80, flush=True)

    # Save to artifacts
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    df_res = pd.DataFrame(records)
    df_res.to_csv(artifacts_dir / "intracycle_soc_results.csv", index=False)
    print(f"Saved artifacts to {artifacts_dir / 'intracycle_soc_results.csv'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    run_lobo_soc_benchmark(args.data_dir, seeds=args.seeds)
