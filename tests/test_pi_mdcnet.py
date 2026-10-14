"""
Unit tests for PI-MDCNet v4 SLAC Architecture (Phase 2).
"""

import pytest
import torch
import numpy as np

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


@pytest.fixture
def dummy_inputs():
    """Create random dummy inputs matching expected tensor shapes."""
    torch.manual_seed(42)
    cfg = SLACConfig()
    B, T = 2, 50
    x_elec = torch.randn(B, T, cfg.elec_input_dim)
    x_therm = torch.randn(B, T, cfg.therm_input_dim)
    s_t = torch.randn(B, T, cfg.stressor_dim)
    rest_features = torch.randn(B, T, cfg.rest_input_dim)

    # Rest flags: mostly 0, few 1s
    rest_flag = torch.zeros(B, T, 1)
    rest_flag[:, [10, 25, 40], 0] = 1.0

    cycle_idx = torch.linspace(0, 1, T).unsqueeze(0).repeat(B, 1).unsqueeze(-1)
    return {
        "x_elec": x_elec,
        "x_therm": x_therm,
        "s_t": s_t,
        "rest_features": rest_features,
        "rest_flag": rest_flag,
        "cycle_idx": cycle_idx,
    }


def test_forward_shapes_and_decomposition(dummy_inputs):
    """Verify output shapes, exact decomposition, monotonicity of D_t, and hard gating."""
    model = PIMDCNet(SLACConfig())
    model.eval()

    with torch.no_grad():
        out = model(**dummy_inputs)

    B, T = dummy_inputs["x_elec"].shape[:2]

    # Shape checks
    assert out["soh"].shape == (B, T, 1)
    assert out["soc"].shape == (B, T, 1)
    assert out["rul"].shape == (B, T, 1)
    assert out["D_t"].shape == (B, T, 1)
    assert out["R_t"].shape == (B, T, 1)

    # Exact decomposition check: SoH_t = SoH_0 - D_t + R_t
    soh_0 = model.soh_0
    reconstructed = soh_0 - out["D_t"] + out["R_t"]
    err = torch.max(torch.abs(out["soh"] - reconstructed)).item()
    assert err < 1e-6, f"Decomposition error: {err}"

    # Monotonicity of D_t: delta_D >= 0
    diffs = out["D_t"][:, 1:, :] - out["D_t"][:, :-1, :]
    assert torch.all(diffs >= -1e-6), "D_t must be monotonically non-decreasing"

    # Hard gating of R_t: R_t == 0 whenever rest_flag == 0
    zero_rest_mask = dummy_inputs["rest_flag"] == 0.0
    r_t_at_zero = out["R_t"][zero_rest_mask]
    assert torch.all(r_t_at_zero == 0.0), "R_t must be exactly 0 when rest_flag is 0"


def test_loss_computation(dummy_inputs):
    """Verify SLACLoss calculates finite scalar loss and gradients flow."""
    model = PIMDCNet(SLACConfig())
    criterion = SLACLoss()

    out = model(**dummy_inputs)

    # Fake targets
    targets = {
        "soc": torch.rand_like(out["soc"]),
        "soh": torch.rand_like(out["soh"]),
        "rul": torch.rand_like(out["rul"]) * 100,
        "rul_mask": torch.ones_like(out["rul"]),
    }

    losses = criterion(out, targets, targets["rul_mask"])
    assert torch.isfinite(losses["total"])
    assert "soc_loss" in losses
    assert "soh_loss" in losses
    assert "rul_loss" in losses

    # Backward pass check
    losses["total"].backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


def test_damage_depends_only_on_stressors(dummy_inputs):
    """Changing the backbone inputs cannot change the damage trajectory."""
    model = PIMDCNet(SLACConfig()).eval()
    altered = dict(dummy_inputs)
    altered["x_elec"] = torch.randn_like(dummy_inputs["x_elec"])
    altered["x_therm"] = torch.randn_like(dummy_inputs["x_therm"])

    with torch.no_grad():
        original = model(**dummy_inputs)["D_t"]
        changed_backbone = model(**altered)["D_t"]

    assert torch.equal(original, changed_backbone)


def test_soh0_is_tanh_bounded_and_differentiable():
    """SoH_0 remains bounded without clamp-induced zero-gradient saturation."""
    cfg = SLACConfig()
    model = PIMDCNet(cfg)
    model._soh_0_raw.data.fill_(1.0)
    upper = model.soh_0
    upper.backward()
    assert cfg.soh_0_min < upper.item() < cfg.soh_0_max
    assert model._soh_0_raw.grad is not None
    assert model._soh_0_raw.grad.abs().item() > 0.0


def test_parameter_budget():
    """Verify model parameter count fits well within embedded budget (< 100k params)."""
    model = PIMDCNet(SLACConfig())
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 100_000, f"Model has {n_params} parameters, expected < 100,000"


def test_c_inference_numerical_parity():
    """Verify complete C99 forward pass matches PyTorch within < 1e-4 across sequential cycles."""
    import ctypes
    import subprocess
    import shutil
    from pathlib import Path
    from embedded.export_onnx import PIMDCNetStep

    repo_root = Path(__file__).resolve().parent.parent
    dll_path = repo_root / "embedded" / "libpi_mdcnet.dll"
    c_src = repo_root / "embedded" / "esp32" / "main" / "pi_mdcnet_inference.c"
    inc_dir = repo_root / "embedded" / "esp32" / "main"

    # Compile shared library if needed
    if not dll_path.exists() and shutil.which("gcc"):
        cmd = [
            "gcc", "-O3", "-shared", "-fPIC", "-Wall", "-Wextra",
            f"-I{inc_dir}", str(c_src), "-o", str(dll_path), "-lm"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"GCC compilation failed: {res.stderr}"

    if not dll_path.exists():
        pytest.skip("GCC shared library not available on this platform")

    lib = ctypes.CDLL(str(dll_path))

    class CycleInput(ctypes.Structure):
        _fields_ = [
            ("x_elec", ctypes.c_float * 13),
            ("x_therm", ctypes.c_float * 4),
            ("s_t", ctypes.c_float * 2),
            ("rest_features", ctypes.c_float * 3),
            ("rest_flag", ctypes.c_float),
            ("cycle_idx", ctypes.c_float),
        ]

    class State(ctypes.Structure):
        _fields_ = [
            ("D_prev", ctypes.c_float),
            ("D_history", ctypes.c_float * 10),
            ("cycle_count", ctypes.c_uint32),
            ("initial_soh_0", ctypes.c_float),
        ]

    class Output(ctypes.Structure):
        _fields_ = [
            ("soh", ctypes.c_float),
            ("soc", ctypes.c_float),
            ("rul", ctypes.c_float),
            ("D_t", ctypes.c_float),
            ("R_t", ctypes.c_float),
            ("delta_D", ctypes.c_float),
            ("inference_us", ctypes.c_uint32),
        ]

    lib.pimdcnet_init.argtypes = [ctypes.POINTER(State), ctypes.c_float]
    lib.pimdcnet_init.restype = None
    lib.pimdcnet_infer_step.argtypes = [ctypes.POINTER(CycleInput), ctypes.POINTER(State), ctypes.POINTER(Output)]
    lib.pimdcnet_infer_step.restype = ctypes.c_int

    torch.manual_seed(42)
    py_model = PIMDCNet(SLACConfig())
    py_model.eval()
    step_model = PIMDCNetStep(py_model)

    state = State()
    lib.pimdcnet_init(ctypes.byref(state), 1.0)

    D_prev_pt = torch.tensor([[0.0]])
    D_hist_pt = torch.zeros(1, 10)

    for step in range(15):
        c_in = CycleInput()
        x_e = torch.randn(1, 13)
        x_th = torch.randn(1, 4)
        s_t = torch.abs(torch.randn(1, 2))
        r_feat = torch.randn(1, 3)
        r_fl = torch.tensor([[1.0 if step % 4 == 0 else 0.0]])
        c_idx = torch.tensor([[step / 100.0]])

        for i in range(13): c_in.x_elec[i] = x_e[0, i].item()
        for i in range(4): c_in.x_therm[i] = x_th[0, i].item()
        for i in range(2): c_in.s_t[i] = s_t[0, i].item()
        for i in range(3): c_in.rest_features[i] = r_feat[0, i].item()
        c_in.rest_flag = r_fl[0, 0].item()
        c_in.cycle_idx = c_idx[0, 0].item()

        out_c = Output()
        ret = lib.pimdcnet_infer_step(ctypes.byref(c_in), ctypes.byref(state), ctypes.byref(out_c))
        assert ret == 0, f"Inference step failed at step {step}"

        with torch.no_grad():
            soh_pt, soc_pt, rul_pt, D_t_pt, R_t_pt, D_hist_next_pt = step_model(
                x_e, x_th, s_t, r_feat, r_fl, c_idx, D_prev_pt, D_hist_pt
            )

        D_prev_pt = D_t_pt
        D_hist_pt = D_hist_next_pt

        # Verify numerical equivalence within 1e-4
        assert abs(out_c.soh - soh_pt.item()) < 1e-4, f"SoH mismatch at step {step}"
        assert abs(out_c.soc - soc_pt.item()) < 1e-4, f"SoC mismatch at step {step}"
        assert abs(out_c.rul - rul_pt.item()) < 1e-4, f"RUL mismatch at step {step}"
        assert abs(out_c.D_t - D_t_pt.item()) < 1e-4, f"D_t mismatch at step {step}"
        assert abs(out_c.R_t - R_t_pt.item()) < 1e-4, f"R_t mismatch at step {step}"


def test_onnx_fp32_and_int8_verification():
    """Verify exported ONNX FP32 and INT8 models run and match expected outputs."""
    import onnxruntime as ort
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    fp32_path = repo_root / "artifacts" / "pi_mdcnet_edge.onnx"
    int8_path = repo_root / "artifacts" / "pi_mdcnet_edge_int8.onnx"

    if not fp32_path.exists():
        pytest.skip("FP32 ONNX model not exported yet")

    session_fp32 = ort.InferenceSession(str(fp32_path))
    dummy_in = {
        "x_elec": np.random.randn(1, 13).astype(np.float32),
        "x_therm": np.random.randn(1, 4).astype(np.float32),
        "s_t": np.random.randn(1, 2).astype(np.float32),
        "rest_features": np.random.randn(1, 3).astype(np.float32),
        "rest_flag": np.zeros((1, 1), dtype=np.float32),
        "cycle_idx": np.array([[0.5]], dtype=np.float32),
        "D_prev": np.array([[0.1]], dtype=np.float32),
        "D_history": np.random.randn(1, 10).astype(np.float32),
    }
    out_fp32 = session_fp32.run(None, dummy_in)
    assert len(out_fp32) == 6

    if int8_path.exists():
        session_int8 = ort.InferenceSession(str(int8_path))
        out_int8 = session_int8.run(None, dummy_in)
        assert len(out_int8) == 6
        # INT8 vs FP32 parity within 0.05
        diff_soh = float(np.max(np.abs(out_fp32[0] - out_int8[0])))
        assert diff_soh < 0.05, f"INT8 quantization error too large: {diff_soh}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
