"""
Intra-Cycle State of Charge (SoC) Auxiliary Module
=================================================

Two-timescale battery state estimation architecture:
- Core aging macro-model (SLAC / PI-MDCNet): operates per-cycle on degradation/regeneration
- Intra-cycle micro-model (this module): operates on raw high-frequency time-series within
  a discharge cycle to estimate real-time continuous SoC(t).

Retrospective Evaluation Reference Definition:
    Cycle-normalized Coulomb integration over Time and Current_measured:
        q(t) = ∫₀ᵗ |I(τ)| dτ
        Q_cycle = ∫₀^{T_end} |I(τ)| dτ
        SoC_ref(t) = 1.0 - q(t) / Q_cycle
    Note: Q_cycle is computed retrospectively for reference label construction,
    but is strictly withheld from model inputs during inference.

Architecture:
    Lightweight GRU sequence-to-sequence model predicting SoC(t) from
    [Voltage, Current, Normalized_Time].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


# =============================================================================
# DATA EXTRACTION & COULOMB INTEGRATION
# =============================================================================

@dataclass
class CycleTimeProfile:
    """Intra-cycle raw telemetry and continuous SoC label."""
    cell_id: str
    cycle_idx: int
    voltage: np.ndarray        # (T,) in Volts
    current: np.ndarray        # (T,) in Amperes
    time_s: np.ndarray         # (T,) in seconds from cycle start
    soc_true: np.ndarray       # (T,) continuous SoC in [0, 1]
    capacity_ah: float         # discharge capacity in Ah


def extract_intracycle_profiles(mat_path: Path) -> list[CycleTimeProfile]:
    """Extract raw intra-cycle discharge curves and compute continuous SoC(t).

    Parameters
    ----------
    mat_path : Path
        Path to NASA PCoE .mat file (e.g. data/raw/B0005.mat).

    Returns
    -------
    list[CycleTimeProfile]
        One profile per discharge cycle.
    """
    cell_name = mat_path.stem
    mat = sio.loadmat(str(mat_path), struct_as_record=True)
    cycle_arr = mat[cell_name][0, 0]["cycle"]

    profiles: list[CycleTimeProfile] = []
    discharge_idx = 0

    for i in range(cycle_arr.shape[1]):
        cyc = cycle_arr[0, i]
        type_str = str(cyc["type"].flatten()[0]).strip()
        if type_str != "discharge":
            continue

        d = cyc["data"][0, 0]
        v = d["Voltage_measured"].flatten().astype(np.float32)
        curr = d["Current_measured"].flatten().astype(np.float32)
        t = d["Time"].flatten().astype(np.float32)

        if len(t) < 5:
            continue

        # Zero-base the time
        t_elapsed = t - t[0]

        # Coulomb integration (trapezoidal) over seconds -> Ah
        # q(t) = ∫₀ᵗ |I(u)| du / 3600.0
        dt = np.diff(t_elapsed)
        curr_mid = np.abs(curr[1:] + curr[:-1]) / 2.0
        dq = curr_mid * dt / 3600.0
        q_coulomb = np.concatenate([[0.0], np.cumsum(dq)]).astype(np.float32)
        q_total = float(q_coulomb[-1])

        if q_total <= 0.01:
            continue

        # Continuous SoC(t) = 1.0 - (q(t) / Q_discharge)
        soc_true = np.clip(1.0 - (q_coulomb / q_total), 0.0, 1.0).astype(np.float32)

        cap_field = float(d["Capacity"].flatten()[0]) if "Capacity" in d.dtype.names else q_total

        profiles.append(
            CycleTimeProfile(
                cell_id=cell_name,
                cycle_idx=discharge_idx,
                voltage=v,
                current=curr,
                time_s=t_elapsed,
                soc_true=soc_true,
                capacity_ah=cap_field,
            )
        )
        discharge_idx += 1

    return profiles


# =============================================================================
# DATASET & NORMALIZATION
# =============================================================================

class IntraCycleDataset(Dataset):
    """PyTorch Dataset for intra-cycle sequence samples."""

    def __init__(
        self,
        profiles: list[CycleTimeProfile],
        v_mean: float = 3.6,
        v_std: float = 0.4,
        i_mean: float = -2.0,
        i_std: float = 0.5,
        t_scale: float = 3600.0,
    ) -> None:
        self.samples = []
        for p in profiles:
            v_norm = (p.voltage - v_mean) / (v_std + 1e-6)
            i_norm = (p.current - i_mean) / (i_std + 1e-6)
            t_norm = p.time_s / t_scale

            # Feature shape: (T, 3) -> [v_norm, i_norm, t_norm]
            feats = np.stack([v_norm, i_norm, t_norm], axis=-1).astype(np.float32)
            targets = p.soc_true[:, None].astype(np.float32)  # (T, 1)

            self.samples.append((
                torch.tensor(feats, dtype=torch.float32),
                torch.tensor(targets, dtype=torch.float32),
            ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def compute_intracycle_stats(profiles: list[CycleTimeProfile]) -> dict[str, float]:
    """Compute normalization statistics from training profiles only."""
    all_v = np.concatenate([p.voltage for p in profiles])
    all_i = np.concatenate([p.current for p in profiles])
    all_t = np.concatenate([p.time_s for p in profiles])

    return {
        "v_mean": float(np.mean(all_v)),
        "v_std": float(np.std(all_v)),
        "i_mean": float(np.mean(all_i)),
        "i_std": float(np.std(all_i)),
        "t_scale": float(max(np.max(all_t), 1.0)),
    }


# =============================================================================
# LIGHTWEIGHT AUXILIARY SoC GRU MODULE
# =============================================================================

class IntraCycleSoCGRU(nn.Module):
    """Lightweight GRU for online continuous SoC(t) estimation.

    Parameters
    ----------
    input_dim : int
        Number of intra-cycle signals (default 3: Voltage, Current, Time).
    hidden_dim : int
        GRU hidden dimension.
    num_layers : int
        Number of stacked GRU layers.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 32,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),  # SoC strictly in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor, shape (B, T, input_dim)

        Returns
        -------
        soc_pred : Tensor, shape (B, T, 1) in [0, 1]
        """
        out, _ = self.gru(x)
        soc_pred = self.head(out)
        return soc_pred
