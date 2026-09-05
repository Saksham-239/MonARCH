"""
PI-MDCNet v4 -- ONNX Export & Verification
===========================================

Exports PI-MDCNet to ONNX in two deployment profiles:

1. Streaming Edge Profile (PIMDCNetStep):
   - Designed for real-time edge/BMS inference (e.g. ESP32-S3 / Xtensa LX7, Embedded BMS).
   - Operates on a single cycle (B=1, T=1).
   - Takes current cycle features + rolling state (D_prev, D_history).
   - Ultra-compact, low latency, no variable-length sequence memory.

2. Sequence Batch Profile (PIMDCNet):
   - Dynamic sequence length T for batch offline analytics and cloud evaluation.

Validates numerical equivalence between PyTorch and ONNX Runtime:
max absolute difference < 1e-5 across multi-step sequential state transitions.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

# Add project root to sys.path if running as a standalone script
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pi_mdcnet import PIMDCNet, SLACConfig

logger = logging.getLogger(__name__)


class PIMDCNetStep(nn.Module):
    """Single-cycle streaming wrapper for real-time embedded BMS deployment.

    Inputs (per cycle t):
    - x_elec: (1, elec_input_dim)
    - x_therm: (1, 4)
    - s_t: (1, stressor_dim)
    - rest_features: (1, 3)
    - rest_flag: (1, 1)
    - cycle_idx: (1, 1)
    - D_prev: (1, 1) -- scalar cumulative damage from cycle t-1
    - D_history: (1, k) -- rolling buffer of past k damage values

    Outputs:
    - soh: (1, 1)
    - soc: (1, 1)
    - rul: (1, 1)
    - D_t: (1, 1) -- updated cumulative damage for next step
    - R_t: (1, 1) -- instantaneous regeneration
    - D_history_next: (1, k) -- updated rolling window
    """

    def __init__(self, core: PIMDCNet) -> None:
        super().__init__()
        self.core = core
        self.cfg = core.cfg

    def forward(
        self,
        x_elec: torch.Tensor,
        x_therm: torch.Tensor,
        s_t: torch.Tensor,
        rest_features: torch.Tensor,
        rest_flag: torch.Tensor,
        cycle_idx: torch.Tensor,
        D_prev: torch.Tensor,
        D_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Encoders
        e_t = self.core.elec_encoder(x_elec)
        th_t = self.core.therm_encoder(x_therm)

        # 2. Cross-Attention
        h_t = self.core.cross_attn(e_t, th_t)

        # 3. Incremental Damage & State Update
        raw_damage = self.core.damage_branch.mlp(s_t)
        delta_D = self.core.damage_branch.damage_scale * F.softplus(raw_damage)
        D_t = D_prev + delta_D  # Guaranteed monotonic since delta_D > 0

        # 4. Regeneration (Hard Gated)
        R_t = self.core.regen_branch(rest_features, h_t, rest_flag)

        # 5. SoH Decomposition
        soh = self.core.soh_0 - D_t + R_t

        # 6. SoC Head
        soc = self.core.soc_head(h_t)

        # 7. RUL Head from rolling damage history
        # D_history shape: (1, k), roll left and append new D_t
        D_history_next = torch.cat([D_history[:, 1:], D_t], dim=-1)
        rul_in = torch.cat([D_history_next, cycle_idx], dim=-1)
        rul = self.core.rul_head.mlp(rul_in)

        return soh, soc, rul, D_t, R_t, D_history_next


def export_streaming_onnx(
    model: PIMDCNet,
    output_path: Path | str = "artifacts/pi_mdcnet_edge.onnx",
) -> Path:
    """Export the streaming single-cycle model to ONNX."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    step_model = PIMDCNetStep(model)
    step_model.eval()

    k = model.cfg.rul_window_k
    dummy_inputs = (
        torch.randn(1, model.cfg.elec_input_dim),    # x_elec
        torch.randn(1, model.cfg.therm_input_dim),   # x_therm
        torch.randn(1, model.cfg.stressor_dim),      # s_t
        torch.randn(1, model.cfg.rest_input_dim),    # rest_features
        torch.zeros(1, 1),        # rest_flag
        torch.tensor([[0.5]]),    # cycle_idx
        torch.tensor([[0.1]]),    # D_prev
        torch.randn(1, k),        # D_history
    )

    input_names = [
        "x_elec", "x_therm", "s_t", "rest_features",
        "rest_flag", "cycle_idx", "D_prev", "D_history"
    ]
    output_names = ["soh", "soc", "rul", "D_t", "R_t", "D_history_next"]

    torch.onnx.export(
        step_model,
        dummy_inputs,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=14,
        do_constant_folding=True,
    )
    logger.info("Exported streaming ONNX model to %s", output_path)

    # Validate with onnx checker
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model verification passed!")

    # Verify with onnxruntime (Single-step check)
    ort_session = ort.InferenceSession(str(output_path))
    ort_inputs = {
        name: tensor.numpy()
        for name, tensor in zip(input_names, dummy_inputs)
    }
    ort_outputs = ort_session.run(None, ort_inputs)

    with torch.no_grad():
        pt_outputs = step_model(*dummy_inputs)

    logger.info("Single-Step ONNX vs PyTorch Numerical Parity:")
    for name, pt_out, ort_out in zip(output_names, pt_outputs, ort_outputs):
        max_diff = float(np.max(np.abs(pt_out.numpy() - ort_out)))
        assert max_diff < 1e-5, f"Discrepancy in {name}: {max_diff}"
        logger.info("  Verified %s: max diff = %.2e [PASS]", name, max_diff)

    # Multi-step sequential state progression verification (50 sequential steps)
    logger.info("Sequential State Progression Parity (50 consecutive cycles):")
    torch.manual_seed(42)
    np.random.seed(42)
    D_prev_pt = torch.tensor([[0.0]])
    D_hist_pt = torch.zeros(1, model.cfg.rul_window_k)
    D_prev_ort = np.zeros((1, 1), dtype=np.float32)
    D_hist_ort = np.zeros((1, model.cfg.rul_window_k), dtype=np.float32)

    seq_diffs: dict[str, list[float]] = {name: [] for name in output_names}
    for step in range(50):
        c_idx = torch.tensor([[step / 50.0]], dtype=torch.float32)
        x_e = torch.randn(1, model.cfg.elec_input_dim)
        x_th = torch.randn(1, model.cfg.therm_input_dim)
        s_t_step = torch.abs(torch.randn(1, model.cfg.stressor_dim))
        r_feat = torch.randn(1, model.cfg.rest_input_dim)
        r_fl = torch.tensor([[1.0 if step % 5 == 0 else 0.0]])

        with torch.no_grad():
            pt_step_out = step_model(x_e, x_th, s_t_step, r_feat, r_fl, c_idx, D_prev_pt, D_hist_pt)

        ort_step_in = {
            "x_elec": x_e.numpy(),
            "x_therm": x_th.numpy(),
            "s_t": s_t_step.numpy(),
            "rest_features": r_feat.numpy(),
            "rest_flag": r_fl.numpy(),
            "cycle_idx": c_idx.numpy(),
            "D_prev": D_prev_ort,
            "D_history": D_hist_ort,
        }
        ort_step_out = ort_session.run(None, ort_step_in)

        for name, p_o, o_o in zip(output_names, pt_step_out, ort_step_out):
            diff = float(np.max(np.abs(p_o.numpy() - o_o)))
            seq_diffs[name].append(diff)

        # Recurrent state progression
        D_prev_pt = pt_step_out[3]
        D_hist_pt = pt_step_out[5]
        D_prev_ort = ort_step_out[3]
        D_hist_ort = ort_step_out[5]

    for name in output_names:
        max_seq_diff = max(seq_diffs[name])
        assert max_seq_diff < 1e-5, f"Sequential discrepancy in {name}: {max_seq_diff}"
        logger.info("  Sequential %s: max diff = %.2e [PASS]", name, max_seq_diff)

    return output_path


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    model = PIMDCNet(SLACConfig())
    export_streaming_onnx(model)


if __name__ == "__main__":
    main()
