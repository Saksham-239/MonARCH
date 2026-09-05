"""
PI-MDCNet v4 -- Ablation Study
===============================

Validates the three key architectural contributions of the Split-Latent
Aging Core (SLAC):

1. Cross-Attention Fusion:
   - Full: Bidirectional cross-attention between electrical & thermal latents
   - Ablated (w/o Cross-Attn): Simple concatenation [h_e; h_T] projected to latent dim

2. Hard Gating on Regeneration (R_t):
   - Full: R_t = g_t * R_raw where g_t = rest_flag_t (identically 0 during active cycling)
   - Ablated (w/o Hard Gating): R_t = R_raw (regeneration branch always active)

3. Monotonic Irreversible Damage (D_t):
    - Full: D_t = cumsum(softplus(W_d * s_t + b_d)) * damage_scale (strictly monotonic)
   - Ablated (w/o Monotonicity): D_t = linear projection of h_t (unconstrained)

Runs LOBO evaluation across all 4 cells for each variant.
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data_loader import load_all_cells
from src.models.pi_mdcnet import (
    PIMDCNet,
    SLACConfig,
    SLACLoss,
    ElectricalEncoder,
    ThermalEncoder,
    CrossAttentionBlock,
    DamageBranch,
    RegenerationBranch,
    SoCHead,
    RULHead,
)
from experiments.train import (
    FeatureScaler,
    CycleSequenceDataset,
    train_pimdcnet,
    predict_pimdcnet,
    compute_metrics,
    _compute_rt_correlation,
)
from experiments.protocol import BENCHMARK_EPOCHS, BENCHMARK_SEEDS, benchmark_folds

logger = logging.getLogger(__name__)


# =============================================================================
# ABLATED ARCHITECTURAL VARIANTS
# =============================================================================

# =============================================================================
# ABLATED ARCHITECTURAL VARIANTS
# =============================================================================

class AblationNoCrossAttn(PIMDCNet):
    """Variant 1: Simple concatenation [h_e; h_T] projected to latent dim (w/o Cross-Attn)."""

    def __init__(self, cfg: SLACConfig = SLACConfig()) -> None:
        super().__init__(cfg)
        self.concat_proj = nn.Linear(cfg.elec_embed_dim + cfg.therm_embed_dim, cfg.attn_dim)

    def forward(
        self,
        x_elec: torch.Tensor,
        x_therm: torch.Tensor,
        s_t: torch.Tensor,
        rest_features: torch.Tensor,
        rest_flag: torch.Tensor,
        cycle_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        B, T, _ = x_elec.shape
        e_seq = self.elec_encoder(x_elec)
        th_seq = self.therm_encoder(x_therm)

        e_flat = e_seq.reshape(B * T, self.cfg.elec_embed_dim)
        th_flat = th_seq.reshape(B * T, self.cfg.therm_embed_dim)
        # Direct concatenation projection instead of cross-attention
        h_flat = F.relu(self.concat_proj(torch.cat([e_flat, th_flat], dim=-1)))

        raw_damage = self.damage_branch.mlp(s_t)
        inc_seq = self.damage_branch.damage_scale * F.softplus(raw_damage)
        D_seq = torch.cumsum(inc_seq, dim=1)

        rest_flat = rest_features.reshape(B * T, self.cfg.rest_input_dim)
        rest_fl_flat = rest_flag.reshape(B * T, 1)
        R_flat = self.regen_branch(rest_flat, h_flat, rest_fl_flat)
        R_seq = R_flat.reshape(B, T, 1)

        soh_seq = self.soh_0 - D_seq + R_seq
        soc_seq = self.soc_head(h_flat).reshape(B, T, 1)

        k = self.cfg.rul_window_k
        D_padded = F.pad(D_seq.squeeze(-1), (k - 1, 0))
        D_windows = D_padded.unfold(dimension=1, size=k, step=1)
        rul_in = torch.cat([D_windows, cycle_idx], dim=-1)
        rul_seq = self.rul_head.mlp(rul_in)

        return {
            "soh": soh_seq, "soc": soc_seq, "rul": rul_seq,
            "D_t": D_seq, "R_t": R_seq, "damage_increments": inc_seq,
        }


class AblationNoHardGating(PIMDCNet):
    """Variant 2: Regeneration branch without hard gating (always active, soft/unconstrained)."""

    def forward(
        self,
        x_elec: torch.Tensor,
        x_therm: torch.Tensor,
        s_t: torch.Tensor,
        rest_features: torch.Tensor,
        rest_flag: torch.Tensor,
        cycle_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        B, T, _ = x_elec.shape
        e_seq = self.elec_encoder(x_elec)
        th_seq = self.therm_encoder(x_therm)

        e_flat = e_seq.reshape(B * T, self.cfg.elec_embed_dim)
        th_flat = th_seq.reshape(B * T, self.cfg.therm_embed_dim)
        h_flat = self.cross_attn(e_flat, th_flat)

        raw_damage = self.damage_branch.mlp(s_t)
        inc_seq = self.damage_branch.damage_scale * F.softplus(raw_damage)
        D_seq = torch.cumsum(inc_seq, dim=1)

        rest_flat = rest_features.reshape(B * T, self.cfg.rest_input_dim)
        # Always-on gating: remove hard binary mask
        always_on = torch.ones(B * T, 1, device=x_elec.device)
        R_flat = self.regen_branch(rest_flat, h_flat, always_on)
        R_seq = R_flat.reshape(B, T, 1)

        soh_seq = self.soh_0 - D_seq + R_seq
        soc_seq = self.soc_head(h_flat).reshape(B, T, 1)

        k = self.cfg.rul_window_k
        D_padded = F.pad(D_seq.squeeze(-1), (k - 1, 0))
        D_windows = D_padded.unfold(dimension=1, size=k, step=1)
        rul_in = torch.cat([D_windows, cycle_idx], dim=-1)
        rul_seq = self.rul_head.mlp(rul_in)

        return {
            "soh": soh_seq, "soc": soc_seq, "rul": rul_seq,
            "D_t": D_seq, "R_t": R_seq, "damage_increments": inc_seq,
        }


class AblationNoMonotonicity(PIMDCNet):
    """Variant 3: Unconstrained damage estimation without cumulative sum or positivity."""

    def __init__(self, cfg: SLACConfig = SLACConfig()) -> None:
        super().__init__(cfg)
        self.unconstrained_damage = nn.Sequential(
            nn.Linear(cfg.stressor_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x_elec: torch.Tensor,
        x_therm: torch.Tensor,
        s_t: torch.Tensor,
        rest_features: torch.Tensor,
        rest_flag: torch.Tensor,
        cycle_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        B, T, _ = x_elec.shape
        e_seq = self.elec_encoder(x_elec)
        th_seq = self.therm_encoder(x_therm)

        e_flat = e_seq.reshape(B * T, self.cfg.elec_embed_dim)
        th_flat = th_seq.reshape(B * T, self.cfg.therm_embed_dim)
        h_flat = self.cross_attn(e_flat, th_flat)

        # Unconstrained damage: can decrease over time (violating physical irreversible damage)
        D_seq = self.unconstrained_damage(s_t)

        rest_flat = rest_features.reshape(B * T, self.cfg.rest_input_dim)
        rest_fl_flat = rest_flag.reshape(B * T, 1)
        R_flat = self.regen_branch(rest_flat, h_flat, rest_fl_flat)
        R_seq = R_flat.reshape(B, T, 1)

        soh_seq = self.soh_0 - D_seq + R_seq
        soc_seq = self.soc_head(h_flat).reshape(B, T, 1)

        k = self.cfg.rul_window_k
        D_padded = F.pad(D_seq.squeeze(-1), (k - 1, 0))
        D_windows = D_padded.unfold(dimension=1, size=k, step=1)
        rul_in = torch.cat([D_windows, cycle_idx], dim=-1)
        rul_seq = self.rul_head.mlp(rul_in)

        return {
            "soh": soh_seq, "soc": soc_seq, "rul": rul_seq,
            "D_t": D_seq, "R_t": R_seq, "damage_increments": torch.zeros_like(D_seq),
        }


# =============================================================================
# ABLATION RUNNER
# =============================================================================

ABLATION_VARIANTS = {
    "Full_SLAC": PIMDCNet,
    "Ablation_NoCrossAttn": AblationNoCrossAttn,
    "Ablation_NoHardGating": AblationNoHardGating,
    "Ablation_NoMonotonicity": AblationNoMonotonicity,
}


def run_ablation_study(
    data_dir: Path | str = "data/raw",
    output_dir: Path | str = "artifacts",
    n_epochs: int = BENCHMARK_EPOCHS,
    seeds: tuple[int, ...] = BENCHMARK_SEEDS,
    device: str = "auto",
) -> pd.DataFrame:
    """Run full LOBO ablation study across all 4 cells and 3 random seeds.

    Parameters
    ----------
    data_dir : Path | str
        Path to raw .mat files.
    output_dir : Path | str
        Directory to save results CSV.
    n_epochs : int
        Training epochs per fold.
    seeds : list[int]
        List of random seeds (default [42, 43, 44]).
    device : str
        PyTorch device ('auto', 'cpu', 'cuda').

    Returns
    -------
    pd.DataFrame
        Ablation results.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cell_data = load_all_cells(data_dir)
    cell_ids = list(cell_data.keys())
    folds = benchmark_folds(cell_ids)

    all_results = []

    for variant_name, model_cls in ABLATION_VARIANTS.items():
        logger.info("=" * 60)
        logger.info("RUNNING ABLATION: %s", variant_name)
        logger.info("=" * 60)

        for fold_idx, (train_ids, test_id) in enumerate(folds):
            train_dfs = [cell_data[cid] for cid in train_ids]
            test_df = cell_data[test_id]

            for seed in seeds:
                model, scaler, wall_clock = train_pimdcnet(
                    train_dfs,
                    cfg=SLACConfig(),
                    n_epochs=n_epochs,
                    device=device,
                    seed=seed,
                    model_cls=model_cls,
                )

                # Test evaluation
                result = predict_pimdcnet(model, test_df, scaler=scaler, device=device)
                soh_m = compute_metrics(test_df["soh"].values, result.soh, "soh_")
                soc_m = compute_metrics(test_df["soc_eod"].values, result.soc, "soc_")
                rul_m = compute_metrics(test_df["rul"].values, result.rul, "rul_")

                # Monotonicity check on test set
                test_ds = CycleSequenceDataset(test_df, scaler=scaler)
                test_batch = {k: v.unsqueeze(0).to(device) for k, v in test_ds[0].items()}
                model.eval()
                with torch.no_grad():
                    test_out = model(
                        x_elec=test_batch["x_elec"],
                        x_therm=test_batch["x_therm"],
                        s_t=test_batch["s_t"],
                        rest_features=test_batch["rest_features"],
                        rest_flag=test_batch["rest_flag"],
                        cycle_idx=test_batch["cycle_idx"],
                    )
                D_t = test_out["D_t"][0, :, 0].cpu().numpy()
                d_diff = np.diff(D_t)
                is_monotonic = bool(np.all(d_diff >= -1e-6))
                max_negative_violation = float(np.min(d_diff)) if len(d_diff) > 0 else 0.0

                row = {
                    "variant": variant_name,
                    "fold": test_id,
                    "seed": seed,
                    "wall_clock_s": wall_clock,
                    "is_damage_monotonic": is_monotonic,
                    "max_monotonicity_violation": max_negative_violation,
                    **soh_m,
                    **soc_m,
                    **rul_m,
                }
                all_results.append(row)
                logger.info(
                    "  [%s] Fold %s (seed %d): SoH RMSE=%.4f, Monotonic D_t=%s (viol=%.6f)",
                    variant_name, test_id, seed, soh_m.get("soh_rmse", np.nan),
                    is_monotonic, max_negative_violation
                )

    ablation_df = pd.DataFrame(all_results)
    out_path = output_dir / "ablation_results.csv"
    tbl_path = output_dir / "ablation_table.csv"
    ablation_df.to_csv(out_path, index=False)
    ablation_df.to_csv(tbl_path, index=False)
    logger.info("Ablation results saved to %s and %s", out_path, tbl_path)

    # Print summary
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY (Mean across seeds & folds)")
    print("=" * 80)
    summary = ablation_df.groupby("variant").agg(
        soh_rmse_mean=("soh_rmse", "mean"),
        soh_rmse_std=("soh_rmse", "std"),
        soh_mae_mean=("soh_mae", "mean"),
        soh_mae_std=("soh_mae", "std"),
        monotonic=("is_damage_monotonic", "all"),
    )
    print(summary.to_string())
    print("=" * 80 + "\n")

    return ablation_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_ablation_study()
