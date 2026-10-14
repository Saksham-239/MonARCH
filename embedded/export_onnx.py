"""
PI-MDCNet v4 -- ONNX Export, Quantization & C Header Generation
==============================================================

Exports PI-MDCNet to edge artifacts:
1. Streaming FP32 ONNX model (`artifacts/pi_mdcnet_edge.onnx`).
2. INT8 dynamically quantized ONNX model (`artifacts/pi_mdcnet_edge_int8.onnx`).
3. Pure C header with static parameters (`embedded/esp32/main/pi_mdcnet_weights.h`).

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
from onnxruntime.quantization import QuantType, quantize_dynamic
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
        D_history_next = torch.cat([D_history[:, 1:], D_t], dim=-1)
        rul_in = torch.cat([D_history_next, cycle_idx], dim=-1)
        rul = self.core.rul_head.mlp(rul_in)

        return soh, soc, rul, D_t, R_t, D_history_next


def export_streaming_onnx(
    model: PIMDCNet,
    output_path: Path | str = "artifacts/pi_mdcnet_edge.onnx",
) -> Path:
    """Export the streaming single-cycle model to FP32 ONNX."""
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
    logger.info("Exported streaming FP32 ONNX model to %s", output_path)

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

        D_prev_pt = pt_step_out[3]
        D_hist_pt = pt_step_out[5]
        D_prev_ort = ort_step_out[3]
        D_hist_ort = ort_step_out[5]

    for name in output_names:
        max_seq_diff = max(seq_diffs[name])
        assert max_seq_diff < 1e-5, f"Sequential discrepancy in {name}: {max_seq_diff}"
        logger.info("  Sequential %s: max diff = %.2e [PASS]", name, max_seq_diff)

    return output_path


def export_quantized_onnx(
    fp32_onnx_path: Path | str = "artifacts/pi_mdcnet_edge.onnx",
    quant_onnx_path: Path | str = "artifacts/pi_mdcnet_edge_int8.onnx",
) -> Path:
    """Quantize FP32 ONNX model to INT8 dynamic quantization."""
    fp32_onnx_path = Path(fp32_onnx_path)
    quant_onnx_path = Path(quant_onnx_path)
    quant_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=str(fp32_onnx_path),
        model_output=str(quant_onnx_path),
        weight_type=QuantType.QInt8,
    )
    fp32_size = fp32_onnx_path.stat().st_size / 1024.0
    int8_size = quant_onnx_path.stat().st_size / 1024.0
    logger.info(
        "Exported INT8 quantized ONNX model to %s (FP32: %.1f KB -> INT8: %.1f KB, %.1f%% reduction)",
        quant_onnx_path, fp32_size, int8_size, (1.0 - int8_size / fp32_size) * 100.0
    )

    # Verify INT8 inference session
    ort_session = ort.InferenceSession(str(quant_onnx_path))
    dummy_inputs = {
        "x_elec": np.random.randn(1, 13).astype(np.float32),
        "x_therm": np.random.randn(1, 4).astype(np.float32),
        "s_t": np.random.randn(1, 2).astype(np.float32),
        "rest_features": np.random.randn(1, 3).astype(np.float32),
        "rest_flag": np.zeros((1, 1), dtype=np.float32),
        "cycle_idx": np.array([[0.5]], dtype=np.float32),
        "D_prev": np.array([[0.1]], dtype=np.float32),
        "D_history": np.random.randn(1, 10).astype(np.float32),
    }
    ort_outputs = ort_session.run(None, dummy_inputs)
    assert len(ort_outputs) == 6, "INT8 ONNX model output count mismatch"
    logger.info("INT8 ONNX execution verified successfully!")
    return quant_onnx_path


def format_c_array(name: str, tensor: torch.Tensor | np.ndarray) -> str:
    """Format a tensor/array as a C static const float array."""
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy().flatten()
    else:
        arr = np.asarray(tensor).flatten()
    lines = [f"static const float {name}[{len(arr)}] = {{"]
    for i in range(0, len(arr), 8):
        chunk = arr[i : i + 8]
        chunk_str = ", ".join(f"{x:.8e}f" for x in chunk)
        if i + 8 < len(arr):
            chunk_str += ","
        lines.append(f"    {chunk_str}")
    lines.append("};\n")
    return "\n".join(lines)


def export_c_weights(
    model: PIMDCNet,
    output_header: Path | str = "embedded/esp32/main/pi_mdcnet_weights.h",
) -> Path:
    """Export model weights to a standalone C99 header file for embedded compilation."""
    output_header = Path(output_header)
    output_header.parent.mkdir(parents=True, exist_ok=True)
    cfg = model.cfg

    lines = [
        "/**",
        " * @file pi_mdcnet_weights.h",
        " * @brief PI-MDCNet v4 Pre-Trained Weights and Dilation/Architecture Config",
        " * Auto-generated by embedded/export_onnx.py. Zero dynamic heap allocation.",
        " */",
        "",
        "#ifndef PI_MDCNET_WEIGHTS_H",
        "#define PI_MDCNET_WEIGHTS_H",
        "",
        f"#define PIMDCNET_ELEC_IN_DIM      {cfg.elec_input_dim}",
        f"#define PIMDCNET_ELEC_HIDDEN      64",
        f"#define PIMDCNET_ELEC_OUT         {cfg.elec_embed_dim}",
        f"#define PIMDCNET_THERM_IN_DIM     {cfg.therm_input_dim}",
        f"#define PIMDCNET_THERM_HIDDEN     32",
        f"#define PIMDCNET_THERM_OUT        {cfg.therm_embed_dim}",
        f"#define PIMDCNET_ATTN_DIM         {cfg.attn_dim}",
        f"#define PIMDCNET_NUM_HEADS        {cfg.n_attn_heads}",
        f"#define PIMDCNET_STRESSOR_DIM     {cfg.stressor_dim}",
        f"#define PIMDCNET_DAMAGE_HIDDEN    {cfg.damage_hidden_dim}",
        f"#define PIMDCNET_REST_IN_DIM      {cfg.rest_input_dim}",
        f"#define PIMDCNET_REST_EMBED       {cfg.rest_embed_dim}",
        f"#define PIMDCNET_REGEN_HIDDEN     {cfg.regen_hidden_dim}",
        f"#define PIMDCNET_SOC_HIDDEN       {cfg.soc_hidden_dim}",
        f"#define PIMDCNET_RUL_WINDOW_K     {cfg.rul_window_k}",
        f"#define PIMDCNET_RUL_HIDDEN       {cfg.rul_hidden_dim}",
        "",
        f"#define PIMDCNET_DAMAGE_SCALE     {cfg.damage_scale:.8e}f",
        f"#define PIMDCNET_R_MAX            {cfg.r_max:.8e}f",
        f"#define PIMDCNET_SOH_0            {model.soh_0.item():.8e}f",
        "",
    ]

    # 1. Electrical Encoder
    lines.append("// 1. Electrical Encoder: Linear(13->64) -> GELU -> Linear(64->32) -> LayerNorm(32)")
    lines.append(format_c_array("W_elec_1", model.elec_encoder.mlp[0].weight))
    lines.append(format_c_array("b_elec_1", model.elec_encoder.mlp[0].bias))
    lines.append(format_c_array("W_elec_2", model.elec_encoder.mlp[2].weight))
    lines.append(format_c_array("b_elec_2", model.elec_encoder.mlp[2].bias))
    lines.append(format_c_array("gamma_elec_ln", model.elec_encoder.mlp[3].weight))
    lines.append(format_c_array("beta_elec_ln", model.elec_encoder.mlp[3].bias))

    # 2. Thermal Encoder
    lines.append("// 2. Thermal Encoder: Linear(4->32) -> GELU -> Linear(32->32) -> LayerNorm(32)")
    lines.append(format_c_array("W_therm_1", model.therm_encoder.mlp[0].weight))
    lines.append(format_c_array("b_therm_1", model.therm_encoder.mlp[0].bias))
    lines.append(format_c_array("W_therm_2", model.therm_encoder.mlp[2].weight))
    lines.append(format_c_array("b_therm_2", model.therm_encoder.mlp[2].bias))
    lines.append(format_c_array("gamma_therm_ln", model.therm_encoder.mlp[3].weight))
    lines.append(format_c_array("beta_therm_ln", model.therm_encoder.mlp[3].bias))

    # 3. Cross-Attention
    lines.append("// 3. Cross-Attention Backbone: Projections + Cross-Attention + LayerNorm(64)")
    lines.append(format_c_array("W_proj_e", model.cross_attn.proj_e.weight))
    lines.append(format_c_array("b_proj_e", model.cross_attn.proj_e.bias))
    lines.append(format_c_array("W_proj_th", model.cross_attn.proj_th.weight))
    lines.append(format_c_array("b_proj_th", model.cross_attn.proj_th.bias))
    lines.append(format_c_array("W_mha_e2th_in", model.cross_attn.cross_attn_e2th.in_proj_weight))
    lines.append(format_c_array("b_mha_e2th_in", model.cross_attn.cross_attn_e2th.in_proj_bias))
    lines.append(format_c_array("W_mha_e2th_out", model.cross_attn.cross_attn_e2th.out_proj.weight))
    lines.append(format_c_array("b_mha_e2th_out", model.cross_attn.cross_attn_e2th.out_proj.bias))
    lines.append(format_c_array("W_mha_th2e_in", model.cross_attn.cross_attn_th2e.in_proj_weight))
    lines.append(format_c_array("b_mha_th2e_in", model.cross_attn.cross_attn_th2e.in_proj_bias))
    lines.append(format_c_array("W_mha_th2e_out", model.cross_attn.cross_attn_th2e.out_proj.weight))
    lines.append(format_c_array("b_mha_th2e_out", model.cross_attn.cross_attn_th2e.out_proj.bias))
    lines.append(format_c_array("gamma_attn_ln", model.cross_attn.norm.weight))
    lines.append(format_c_array("beta_attn_ln", model.cross_attn.norm.bias))

    # 4. Damage Branch
    lines.append("// 4. Irreversible Damage Branch: Linear(2->32) -> GELU -> Linear(32->32) -> GELU -> Linear(32->1)")
    lines.append(format_c_array("W_damage_1", model.damage_branch.mlp[0].weight))
    lines.append(format_c_array("b_damage_1", model.damage_branch.mlp[0].bias))
    lines.append(format_c_array("W_damage_2", model.damage_branch.mlp[2].weight))
    lines.append(format_c_array("b_damage_2", model.damage_branch.mlp[2].bias))
    lines.append(format_c_array("W_damage_3", model.damage_branch.mlp[4].weight))
    lines.append(format_c_array("b_damage_3", model.damage_branch.mlp[4].bias))

    # 5. Regeneration Branch
    lines.append("// 5. Reversible Regeneration Branch: Rest MLP(3->32->64) + MHA + MLP(64->32->1)")
    lines.append(format_c_array("W_rest_1", model.regen_branch.rest_embed[0].weight))
    lines.append(format_c_array("b_rest_1", model.regen_branch.rest_embed[0].bias))
    lines.append(format_c_array("W_rest_2", model.regen_branch.rest_embed[2].weight))
    lines.append(format_c_array("b_rest_2", model.regen_branch.rest_embed[2].bias))
    lines.append(format_c_array("W_mha_regen_in", model.regen_branch.cross_attn.in_proj_weight))
    lines.append(format_c_array("b_mha_regen_in", model.regen_branch.cross_attn.in_proj_bias))
    lines.append(format_c_array("W_mha_regen_out", model.regen_branch.cross_attn.out_proj.weight))
    lines.append(format_c_array("b_mha_regen_out", model.regen_branch.cross_attn.out_proj.bias))
    lines.append(format_c_array("W_regen_1", model.regen_branch.mlp_regen[0].weight))
    lines.append(format_c_array("b_regen_1", model.regen_branch.mlp_regen[0].bias))
    lines.append(format_c_array("W_regen_2", model.regen_branch.mlp_regen[2].weight))
    lines.append(format_c_array("b_regen_2", model.regen_branch.mlp_regen[2].bias))

    # 6. SoC Head
    lines.append("// 6. SoC Head: Linear(64->32) -> GELU -> Linear(32->1) -> Sigmoid")
    lines.append(format_c_array("W_soc_1", model.soc_head.mlp[0].weight))
    lines.append(format_c_array("b_soc_1", model.soc_head.mlp[0].bias))
    lines.append(format_c_array("W_soc_2", model.soc_head.mlp[2].weight))
    lines.append(format_c_array("b_soc_2", model.soc_head.mlp[2].bias))

    # 7. RUL Head
    lines.append("// 7. RUL Head: Linear(11->32) -> GELU -> Linear(32->32) -> GELU -> Linear(32->1) -> Softplus")
    lines.append(format_c_array("W_rul_1", model.rul_head.mlp[0].weight))
    lines.append(format_c_array("b_rul_1", model.rul_head.mlp[0].bias))
    lines.append(format_c_array("W_rul_2", model.rul_head.mlp[2].weight))
    lines.append(format_c_array("b_rul_2", model.rul_head.mlp[2].bias))
    lines.append(format_c_array("W_rul_3", model.rul_head.mlp[4].weight))
    lines.append(format_c_array("b_rul_3", model.rul_head.mlp[4].bias))

    lines.append("#endif /* PI_MDCNET_WEIGHTS_H */")
    lines.append("")

    with open(output_header, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    param_count = sum(p.numel() for p in model.parameters())
    flash_kb = (param_count * 4) / 1024.0
    logger.info(
        "Exported C weights to %s (%d params, %.1f KB Flash .rodata footprint)",
        output_header, param_count, flash_kb
    )
    return output_header


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    torch.manual_seed(42)
    np.random.seed(42)
    model = PIMDCNet(SLACConfig())
    model.eval()

    fp32_onnx = export_streaming_onnx(model, "artifacts/pi_mdcnet_edge.onnx")
    export_quantized_onnx(fp32_onnx, "artifacts/pi_mdcnet_edge_int8.onnx")
    export_c_weights(model, "embedded/esp32/main/pi_mdcnet_weights.h")


if __name__ == "__main__":
    main()
