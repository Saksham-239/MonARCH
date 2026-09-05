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


if __name__ == "__main__":
    pytest.main(["-v", __file__])
