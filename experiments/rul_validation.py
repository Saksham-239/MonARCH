"""
RUL Validation Engine: Four-Way Comparative Study
=================================================

Implements the four-way Remaining Useful Life (RUL) benchmark under
identical Leave-One-Battery-Out (LOBO) cross-validation across all NASA cells:

1. Raw-SoH-extrapolation RUL:
   Direct linear regression of preceding raw measured SoH to EOL threshold (0.70).
2. Statistical-decomposition RUL:
   Secular degradation trend (Savitzky-Golay / moving average smoothed) extrapolated to 0.70.
3. SLAC Neural RUL:
   Trained neural RULHead from PI-MDCNet (windowed over D_t + cycle index).
4. Non-learned Linear Extrapolation of D_t to D_fail:
   Closed-form physical projection of the monotonic latent damage D_t:
   RUL = max(0, (D_fail - D_t) / dot{D}_t) with D_fail = SoH_0 - 0.70.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.data_loader import load_all_cells, CELL_IDS
from src.models.pi_mdcnet import PIMDCNet, SLACConfig
from experiments.train import FeatureScaler, CycleSequenceDataset, train_pimdcnet
from experiments.protocol import BENCHMARK_SEEDS

EOL_SOH_THRESHOLD = 0.70  # 1.4 Ah / 2.0 Ah


def extrapolate_linear_soh(soh_history: np.ndarray, target_soh: float = EOL_SOH_THRESHOLD) -> float:
    """Extrapolate raw SoH series linearly to target_soh.

    Returns estimated remaining cycles from current cycle.
    """
    n = len(soh_history)
    if n < 5:
        # Prior fallback
        return max(0.0, 100.0 - n)
    x = np.arange(n)
    # Fit line: y = a*x + b
    p = np.polyfit(x, soh_history, deg=1)
    slope, intercept = p[0], p[1]
    if slope >= -1e-6:
        # Fallback to cumulative average slope from start
        avg_slope = (soh_history[-1] - soh_history[0]) / max(n - 1, 1)
        if avg_slope >= -1e-6:
            return 0.0
        slope = avg_slope

    # target = slope * x_fail + intercept => x_fail = (target - intercept) / slope
    x_fail = (target_soh - intercept) / slope
    rul = max(0.0, float(x_fail - (n - 1)))
    return rul


def extrapolate_stat_decomp(soh_history: np.ndarray, rest_flags: np.ndarray, target_soh: float = EOL_SOH_THRESHOLD) -> float:
    """Statistical decomposition RUL: remove rest-rebound jumps and extrapolate secular trend."""
    n = len(soh_history)
    if n < 5:
        return max(0.0, 100.0 - n)

    # Filter out cycles that had rest rebounds or smooth with moving median/mean
    # To obtain secular degradation, compute exponential smoothing or rolling median
    s = pd.Series(soh_history)
    smooth_soh = s.ewm(span=min(10, n), adjust=False).mean().values

    # Fit linear or quadratic trend on secular smoothed series
    x = np.arange(n)
    p = np.polyfit(x, smooth_soh, deg=1)
    slope, intercept = p[0], p[1]
    if slope >= -1e-6:
        avg_slope = (smooth_soh[-1] - smooth_soh[0]) / max(n - 1, 1)
        if avg_slope >= -1e-6:
            return 0.0
        slope = avg_slope

    x_fail = (target_soh - intercept) / slope
    rul = max(0.0, float(x_fail - (n - 1)))
    return rul


def extrapolate_damage_latent(
    D_history: np.ndarray,
    soh_0: float = 1.0,
    target_soh: float = EOL_SOH_THRESHOLD,
    window_k: int = 10,
) -> float:
    """Non-learned linear extrapolation of monotonic damage latent D_t to D_fail.

    D_fail = soh_0 - target_soh
    Since D_t is strictly monotonic non-decreasing, dot{D}_t > 0.
    RUL = max(0, (D_fail - D_t) / dot{D}_t).
    """
    n = len(D_history)
    D_curr = D_history[-1]
    D_fail = max(0.05, soh_0 - target_soh)

    if D_curr >= D_fail:
        return 0.0

    # Estimate damage rate dot{D}_t from recent window (or cumulative if short)
    k = min(n, window_k)
    if k >= 5:
        x = np.arange(k)
        p = np.polyfit(x, D_history[-k:], deg=1)
        dot_D = p[0]
    else:
        dot_D = (D_curr - D_history[0]) / max(n - 1, 1)

    if dot_D <= 1e-7:
        # Fallback to cumulative average damage rate
        dot_D = max(D_curr / max(n, 1), 1e-5)

    rul = max(0.0, float((D_fail - D_curr) / dot_D))
    return rul


def run_rul_benchmark(
    data_dir: Path,
    seeds: list[int] = [42, 43, 44],
    burn_in: int = 10,
):
    print("=" * 90, flush=True)
    print("FOUR-WAY RUL VALIDATION BENCHMARK UNDER LOBO PROTOCOL", flush=True)
    print("=" * 90, flush=True)

    cell_data = load_all_cells(data_dir)

    # Check EOL for each cell
    print("Dataset EOL ground truth verification:", flush=True)
    for c in CELL_IDS:
        df = cell_data[c]
        reaches_eol = (df["capacity_ah"] <= 1.40).any()
        if reaches_eol:
            eol_idx = df.loc[df["capacity_ah"] <= 1.40, "cycle_index"].index[0]
            print(f"  Cell {c}: reaches EOL at cycle index {eol_idx} (capacity = {df['capacity_ah'].iloc[eol_idx]:.4f} Ah)", flush=True)
        else:
            print(f"  Cell {c}: min capacity = {df['capacity_ah'].min():.4f} Ah > 1.40 Ah. RIGHT-CENSORED (RUL is NaN)", flush=True)

    # Methods:
    # 1. Raw_SoH_Extrap
    # 2. Stat_Decomp
    # 3. Neural_RULHead
    # 4. Latent_Dt_Extrap

    results = []

    for held_out in CELL_IDS:
        test_df = cell_data[held_out]
        train_dfs = [cell_data[c] for c in CELL_IDS if c != held_out]

        # Check if held-out cell has valid RUL labels
        has_valid_rul = not np.isnan(test_df["rul"].values).all()
        true_rul = test_df["rul"].values

        eval_indices = [t for t in range(burn_in, len(test_df)) if not np.isnan(true_rul[t])] if has_valid_rul else []

        for seed in seeds:
            # 1. Train PI-MDCNet to get both the Neural RULHead and the latent D_t
            model, scaler, _ = train_pimdcnet(
                train_dfs,
                cfg=SLACConfig(),
                n_epochs=100,
                device="auto",
                seed=seed,
                model_cls=PIMDCNet,
            )

            # Evaluate PI-MDCNet on test cell
            test_ds = CycleSequenceDataset(test_df, scaler=scaler)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            test_batch = {k: v.unsqueeze(0).to(device) for k, v in test_ds[0].items()}
            model.eval()
            with torch.no_grad():
                out = model(
                    x_elec=test_batch["x_elec"],
                    x_therm=test_batch["x_therm"],
                    s_t=test_batch["s_t"],
                    rest_features=test_batch["rest_features"],
                    rest_flag=test_batch["rest_flag"],
                    cycle_idx=test_batch["cycle_idx"],
                )

            neural_rul_pred = out["rul"][0, :, 0].cpu().numpy()
            D_t_pred = out["D_t"][0, :, 0].cpu().numpy()
            soh_0_pred = float(model.soh_0.item())

            # Now compute predictions for all 4 methods across cycles
            pred_raw_soh = np.zeros(len(test_df))
            pred_stat_decomp = np.zeros(len(test_df))
            pred_latent_dt = np.zeros(len(test_df))

            soh_history_full = test_df["soh"].values
            rest_flags_full = test_df["rest_flag"].values

            for t in range(burn_in, len(test_df)):
                pred_raw_soh[t] = extrapolate_linear_soh(soh_history_full[: t + 1])
                pred_stat_decomp[t] = extrapolate_stat_decomp(soh_history_full[: t + 1], rest_flags_full[: t + 1])
                pred_latent_dt[t] = extrapolate_damage_latent(D_t_pred[: t + 1], soh_0=soh_0_pred)

            if len(eval_indices) > 0:
                idx = np.array(eval_indices)
                y_true = true_rul[idx]

                for name, y_p in [
                    ("Raw_SoH_Extrap", pred_raw_soh[idx]),
                    ("Stat_Decomp", pred_stat_decomp[idx]),
                    ("Neural_RULHead", neural_rul_pred[idx]),
                    ("Latent_Dt_Extrap", pred_latent_dt[idx]),
                ]:
                    mae = float(np.mean(np.abs(y_p - y_true)))
                    rmse = float(np.sqrt(np.mean((y_p - y_true) ** 2)))
                    results.append({
                        "method": name,
                        "fold": held_out,
                        "seed": seed,
                        "rmse": rmse,
                        "mae": mae,
                        "n_eval": len(idx),
                    })
            else:
                for name in ["Raw_SoH_Extrap", "Stat_Decomp", "Neural_RULHead", "Latent_Dt_Extrap"]:
                    results.append({
                        "method": name,
                        "fold": held_out,
                        "seed": seed,
                        "rmse": np.nan,
                        "mae": np.nan,
                        "n_eval": 0,
                    })

    df_res = pd.DataFrame(results)

    print("\n" + "=" * 90, flush=True)
    print("DETAILED PER-FOLD RESULTS (RAW STDOUT)", flush=True)
    print("=" * 90, flush=True)
    print(f"{'Method':<20} | {'Fold':<8} | {'Seed':<5} | {'RMSE (cycles)':<15} | {'MAE (cycles)':<15}", flush=True)
    print("-" * 90, flush=True)
    for _, row in df_res.iterrows():
        rmse_str = f"{row['rmse']:.2f}" if not np.isnan(row['rmse']) else "NaN (Censored)"
        mae_str = f"{row['mae']:.2f}" if not np.isnan(row['mae']) else "NaN (Censored)"
        print(f"{row['method']:<20} | {row['fold']:<8} | {int(row['seed']):<5} | {rmse_str:<15} | {mae_str:<15}", flush=True)

    print("\n" + "=" * 90, flush=True)
    print("SUMMARY BY METHOD (Across Valid LOBO Folds: B0005, B0006, B0018)", flush=True)
    print("=" * 90, flush=True)
    valid_df = df_res.dropna(subset=["mae"])
    summary = valid_df.groupby("method").agg(
        mean_mae=("mae", "mean"),
        std_mae=("mae", "std"),
        mean_rmse=("rmse", "mean"),
        std_rmse=("rmse", "std"),
    ).reset_index()

    # Sort by mean_mae
    summary = summary.sort_values(by="mean_mae")
    print(f"{'Method':<25} | {'Mean MAE ± Std (cycles)':<25} | {'Mean RMSE ± Std (cycles)':<25}", flush=True)
    print("-" * 90, flush=True)
    for _, row in summary.iterrows():
        print(f"{row['method']:<25} | {row['mean_mae']:6.2f} ± {row['std_mae']:5.2f} cycles    | {row['mean_rmse']:6.2f} ± {row['std_rmse']:5.2f} cycles", flush=True)
    print("=" * 90, flush=True)

    # Save to artifacts
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    df_res.to_csv(artifacts_dir / "rul_validation_results.csv", index=False)
    summary.to_csv(artifacts_dir / "rul_validation_summary.csv", index=False)
    print(f"\nArtifacts saved to {artifacts_dir / 'rul_validation_results.csv'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--burn-in", type=int, default=10)
    args = parser.parse_args()
    run_rul_benchmark(args.data_dir, seeds=args.seeds, burn_in=args.burn_in)
