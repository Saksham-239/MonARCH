"""
PI-MDCNet v4 -- Split-Latent Aging Core (SLAC) Architecture
=============================================================

Core neural architecture for battery degradation/regeneration decomposition.

Architecture overview (§2):
    SoH_t = SoH_0 - D_t + R_t

Where:
    D_t : cumulative irreversible damage, monotonically non-decreasing by
          construction (architecturally-guaranteed monotonicity of D_t).
          Fed from the stressor vector s_t (throughput, temperature, C-rate),
          NOT from the cross-attention backbone h_t, to prevent leakage of
          rest-related signals.
    R_t : reversible regeneration capacity, hard-gated by rest_flag_t.
          Zero whenever rest_flag_t = 0 -- the model is architecturally
          incapable of using R_t on non-rest cycles.

Submodules:
    ElectricalEncoder   -> e_t  (dim 32)
    ThermalEncoder      -> th_t (dim 32)
    CrossAttentionBlock -> h_t  (dim 64)
    DamageBranch        -> D_t  (cumulative, monotonic)
    RegenerationBranch  -> R_t  (gated by rest_flag)
    SoCHead             -> SoC prediction
    RULHead             -> RUL prediction (from D_t window)

References:
    - Orchard, Saha & Goebel, IEEE Trans. I&M 2013
    - Qin et al. 2016/2019 -- rest-time-based prognostic framework
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# HYPERPARAMETER CONFIG
# =============================================================================

@dataclass
class SLACConfig:
    """Hyperparameter configuration for PI-MDCNet v4 SLAC architecture.

    All dimensions and hyperparameters are documented here to avoid
    magic numbers scattered through the model code.
    """

    # -- Encoder dims --
    elec_input_dim: int = 13
    """Number of electrical features: v_mean, v_min, v_max, v_std, v_slope,
    i_mean, i_std, dq_dv_peak_v, dq_dv_peak_mag, v_ocv, r0, r1, c1.
    Labels and capacity-equivalent inputs are excluded."""

    therm_input_dim: int = 4
    """Number of thermal features: t_mean, t_max, t_std, t_rise."""

    elec_embed_dim: int = 32
    """Electrical encoder output dimension (e_t)."""

    therm_embed_dim: int = 32
    """Thermal encoder output dimension (th_t)."""

    # -- Cross-attention --
    attn_dim: int = 64
    """Fused embedding dimension (h_t) after cross-attention."""

    n_attn_heads: int = 4
    """Number of attention heads in the cross-attention block."""

    attn_dropout: float = 0.1
    """Dropout rate for attention weights."""

    # -- Damage branch --
    stressor_dim: int = 2
    """Stressor vector dimension: [mean_temp_cycle, c_rate]."""

    damage_hidden_dim: int = 32
    """Hidden dimension for the damage MLP."""

    damage_scale: float = 0.003
    """Scaling factor for per-cycle damage increments. Controls the magnitude
    of softplus output. With ~168 cycles and ~0.3 total fade, we need
    per-cycle damage of ~0.0018. softplus(0) * 0.003 ≈ 0.00208, matching
    the empirical degradation slope at initialization."""

    # -- Regeneration branch --
    rest_input_dim: int = 3
    """Rest feature dimension: [rest_duration_hours, soc_at_rest_onset,
    ambient_temp_rest]."""

    rest_embed_dim: int = 32
    """Embedding dimension for rest features before attention."""

    r_max: float = 0.03
    """Maximum regeneration magnitude (SoH units). Caps R_t via tanh scaling.
    0.03 means R_t can recover up to 3% of rated capacity per rest event."""

    regen_hidden_dim: int = 32
    """Hidden dimension for regeneration MLP."""

    # -- SoH_0 --
    soh_0_init: float = 1.0
    """Initial value for learnable SoH_0, smoothly bounded with tanh."""

    soh_0_min: float = 0.95
    """Lower bound for SoH_0. Cells start near rated capacity."""

    soh_0_max: float = 1.05
    """Upper bound for SoH_0. Allows slight overcapacity."""

    # -- SoC head --
    soc_hidden_dim: int = 32
    """Hidden dimension for SoC prediction MLP."""

    # -- RUL head --
    rul_window_k: int = 10
    """Number of recent D_t values used as RUL predictor input."""

    rul_hidden_dim: int = 32
    """Hidden dimension for RUL prediction MLP."""

    # -- Loss weights (§2.7) --
    lambda_rul: float = 0.01
    """Weight for RUL MAE loss term. Scaled to balance against SoH MSE."""

    lambda_sparse: float = 0.01
    """Weight for L1 sparsity on R_t (belt-and-suspenders, not primary
    mechanism -- the hard gate is the primary control)."""

    # -- Training --
    learning_rate: float = 2e-3
    """Initial learning rate for Adam/AdamW."""


# =============================================================================
# ELECTRICAL ENCODER
# =============================================================================

class ElectricalEncoder(nn.Module):
    """Encodes voltage/current summary stats + dQ/dV + ECM params -> e_t.

    Input features (13):
        v_mean, v_min, v_max, v_std, v_slope,
        i_mean, i_std,
        dq_dv_peak_v, dq_dv_peak_mag,
        v_ocv, r0, r1, c1

    Output: e_t in R^{elec_embed_dim}
    """

    def __init__(self, cfg: SLACConfig) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.elec_input_dim, 64),
            nn.GELU(),
            nn.Linear(64, cfg.elec_embed_dim),
            nn.LayerNorm(cfg.elec_embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor, shape (B, elec_input_dim)

        Returns
        -------
        e_t : Tensor, shape (B, elec_embed_dim)
        """
        return self.mlp(x)


# =============================================================================
# THERMAL ENCODER
# =============================================================================

class ThermalEncoder(nn.Module):
    """Encodes temperature summary stats -> th_t.

    Input features (4): t_mean, t_max, t_std, t_rise

    Output: th_t in R^{therm_embed_dim}
    """

    def __init__(self, cfg: SLACConfig) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.therm_input_dim, 32),
            nn.GELU(),
            nn.Linear(32, cfg.therm_embed_dim),
            nn.LayerNorm(cfg.therm_embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor, shape (B, therm_input_dim)

        Returns
        -------
        th_t : Tensor, shape (B, therm_embed_dim)
        """
        return self.mlp(x)


# =============================================================================
# CROSS-ATTENTION BLOCK
# =============================================================================

class CrossAttentionBlock(nn.Module):
    """Fuses electrical and thermal embeddings via cross-attention -> h_t.

    Uses standard multi-head attention where:
    - Query: one modality's embedding
    - Key/Value: other modality's embedding
    Then concatenates both cross-attended outputs and projects down to attn_dim.

    This allows the model to learn which thermal features are relevant for
    interpreting which electrical features, and vice versa.
    """

    def __init__(self, cfg: SLACConfig) -> None:
        super().__init__()
        # Project both to same dimension for attention
        self.proj_e = nn.Linear(cfg.elec_embed_dim, cfg.attn_dim)
        self.proj_th = nn.Linear(cfg.therm_embed_dim, cfg.attn_dim)

        # Cross-attention: electrical queries thermal
        self.cross_attn_e2th = nn.MultiheadAttention(
            embed_dim=cfg.attn_dim,
            num_heads=cfg.n_attn_heads,
            dropout=cfg.attn_dropout,
            batch_first=True,
        )
        # Cross-attention: thermal queries electrical
        self.cross_attn_th2e = nn.MultiheadAttention(
            embed_dim=cfg.attn_dim,
            num_heads=cfg.n_attn_heads,
            dropout=cfg.attn_dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(cfg.attn_dim)

    def forward(self, e_t: Tensor, th_t: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        e_t : Tensor, shape (B, elec_embed_dim)
        th_t : Tensor, shape (B, therm_embed_dim)

        Returns
        -------
        h_t : Tensor, shape (B, attn_dim)
        """
        # Project to attention dimension and add sequence dim (length 1)
        e_proj = self.proj_e(e_t).unsqueeze(1)   # (B, 1, attn_dim)
        th_proj = self.proj_th(th_t).unsqueeze(1)  # (B, 1, attn_dim)

        # Cross attention: e queries th, th queries e
        e_attended, _ = self.cross_attn_e2th(e_proj, th_proj, th_proj)
        th_attended, _ = self.cross_attn_th2e(th_proj, e_proj, e_proj)

        # Combine via sum + residual and squeeze
        h_t = self.norm(e_attended + th_attended).squeeze(1)  # (B, attn_dim)
        return h_t


# =============================================================================
# DAMAGE BRANCH (§2.2)
# =============================================================================

class DamageBranch(nn.Module):
    """Architecturally-guaranteed monotonicity of D_t (irreversible damage).

    Damage is cumulative and non-decreasing BY CONSTRUCTION:
        damage_increment_t = softplus(MLP(s_t))    # always >= 0
        D_t = D_{t-1} + damage_increment_t         # non-decreasing

    The stressor vector s_t = [throughput_cycle_ah, mean_temp_cycle, c_rate]
    is computed directly from raw electrical/thermal measurements in the
    data pipeline (data_loader.py). It is NOT derived from the cross-attention
    backbone h_t, preventing any rest-related signal from leaking into the
    damage branch.

    This decoupling is critical for identifiability: if D_t could see rest
    features (even indirectly via h_t), the model could absorb regeneration
    effects into the damage branch, defeating the decomposition.

    D_0 = 0 (initial damage is zero for a fresh cell).
    """

    def __init__(self, cfg: SLACConfig) -> None:
        super().__init__()
        self.damage_scale = cfg.damage_scale
        self.mlp = nn.Sequential(
            nn.Linear(cfg.stressor_dim, cfg.damage_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.damage_hidden_dim, cfg.damage_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.damage_hidden_dim, 1),
            # No activation -- softplus applied after MLP output
        )
        # Initialize last layer with small weights for stable training
        with torch.no_grad():
            self.mlp[-1].weight.mul_(0.1)
            self.mlp[-1].bias.zero_()

    def forward(self, s_t: Tensor, D_prev: Tensor) -> tuple[Tensor, Tensor]:
        """Compute cumulative damage D_t.

        Parameters
        ----------
        s_t : Tensor, shape (B, stressor_dim)
            Stressor vector for this cycle.
        D_prev : Tensor, shape (B, 1)
            Cumulative damage from previous cycle.

        Returns
        -------
        D_t : Tensor, shape (B, 1)
            Updated cumulative damage (non-decreasing).
        damage_increment : Tensor, shape (B, 1)
            Per-cycle damage increment (always >= 0).
        """
        raw = self.mlp(s_t)  # (B, 1)
        damage_increment = self.damage_scale * F.softplus(raw)  # >= 0, scaled
        D_t = D_prev + damage_increment  # cumulative, non-decreasing
        return D_t, damage_increment


# =============================================================================
# REGENERATION BRANCH (§2.3)
# =============================================================================

class RegenerationBranch(nn.Module):
    """Hard-gated reversible regeneration R_t.

    R_t is nonzero ONLY when rest_flag_t = 1:
        rest_emb = MLP_rest([rest_duration, soc_at_rest, ambient_temp_rest])
        regen_attn = CrossAttention(query=rest_emb, key=h_t, value=h_t)
        R_t = rest_flag_t * R_MAX * tanh(MLP_regen(regen_attn))

    The hard gate (multiplication by rest_flag_t) makes the model
    architecturally incapable of producing regeneration on non-rest cycles.
    This is not a soft penalty -- it's a hard architectural constraint.

    SoH = SoH_0 - D_t + R_t is NOT monotonic since R_t can increase
    after rest. Only D_t is monotonically non-decreasing.
    """

    def __init__(self, cfg: SLACConfig) -> None:
        super().__init__()
        self.r_max = cfg.r_max

        # Embed rest features
        self.rest_embed = nn.Sequential(
            nn.Linear(cfg.rest_input_dim, cfg.rest_embed_dim),
            nn.GELU(),
            nn.Linear(cfg.rest_embed_dim, cfg.attn_dim),
        )

        # Cross attention: rest queries backbone
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.attn_dim,
            num_heads=cfg.n_attn_heads,
            dropout=cfg.attn_dropout,
            batch_first=True,
        )

        # Predict regeneration magnitude
        self.mlp_regen = nn.Sequential(
            nn.Linear(cfg.attn_dim, cfg.regen_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.regen_hidden_dim, 1),
            nn.Tanh(),  # output in [-1, 1], scaled by R_MAX
        )

    def forward(
        self,
        rest_features: Tensor,
        h_t: Tensor,
        rest_flag: Tensor,
    ) -> Tensor:
        """Compute gated regeneration R_t.

        Parameters
        ----------
        rest_features : Tensor, shape (B, rest_input_dim)
            [rest_duration_hours, soc_at_rest_onset, ambient_temp_rest]
        h_t : Tensor, shape (B, attn_dim)
            Backbone embedding from cross-attention.
        rest_flag : Tensor, shape (B, 1)
            Binary flag: 1.0 if rest occurred, 0.0 otherwise.

        Returns
        -------
        R_t : Tensor, shape (B, 1)
            Regeneration estimate. Exactly 0.0 when rest_flag = 0.
        """
        # Embed rest features
        rest_emb = self.rest_embed(rest_features).unsqueeze(1)  # (B, 1, attn_dim)
        h_kv = h_t.unsqueeze(1)  # (B, 1, attn_dim)

        # Cross-attention: rest queries backbone state
        regen_attn, _ = self.cross_attn(rest_emb, h_kv, h_kv)
        regen_attn = regen_attn.squeeze(1)  # (B, attn_dim)

        # Predict regeneration magnitude, scale, and gate
        R_raw = self.mlp_regen(regen_attn)  # (B, 1), in [-1, 1]
        R_t = rest_flag * self.r_max * R_raw  # hard gate

        return R_t


# =============================================================================
# SoC HEAD (§2.5)
# =============================================================================

class SoCHead(nn.Module):
    """Predict end-of-discharge State of Charge from backbone embedding h_t.

    Simple MLP projecting h_t -> SoC_pred in [0, 1].
    """

    def __init__(self, cfg: SLACConfig) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.attn_dim, cfg.soc_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.soc_hidden_dim, 1),
            nn.Sigmoid(),  # SoC in [0, 1]
        )

    def forward(self, h_t: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        h_t : Tensor, shape (B, attn_dim)

        Returns
        -------
        soc_pred : Tensor, shape (B, 1), values in [0, 1]
        """
        return self.mlp(h_t)


# =============================================================================
# RUL HEAD (§2.6)
# =============================================================================

class RULHead(nn.Module):
    """Predict Remaining Useful Life from a window of recent damage values.

    Input: [D_{t-k+1}, ..., D_t, t] where k = rul_window_k
    Output: RUL prediction (non-negative, in cycles)

    Using D_t (not raw SoH) for RUL prediction makes the prediction
    invariant to regeneration artifacts -- it tracks the irreversible
    degradation trajectory.
    """

    def __init__(self, cfg: SLACConfig) -> None:
        super().__init__()
        self.window_k = cfg.rul_window_k
        self.mlp = nn.Sequential(
            nn.Linear(cfg.rul_window_k + 1, cfg.rul_hidden_dim),  # +1 for cycle index
            nn.GELU(),
            nn.Linear(cfg.rul_hidden_dim, cfg.rul_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.rul_hidden_dim, 1),
            nn.Softplus(),  # RUL >= 0
        )

    def forward(self, D_window: Tensor, cycle_idx: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        D_window : Tensor, shape (B, rul_window_k)
            Recent D_t values (zero-padded for early cycles).
        cycle_idx : Tensor, shape (B, 1)
            Normalized cycle index (t / max_cycles).

        Returns
        -------
        rul_pred : Tensor, shape (B, 1), values >= 0
        """
        x = torch.cat([D_window, cycle_idx], dim=-1)  # (B, k+1)
        return self.mlp(x)


# =============================================================================
# PI-MDCNet v4 -- FULL MODEL
# =============================================================================

class PIMDCNet(nn.Module):
    """PI-MDCNet v4: Split-Latent Aging Core (SLAC).

    Full model combining all submodules.

    Forward pass (single time step):
        1. e_t = ElectricalEncoder(x_elec)
        2. th_t = ThermalEncoder(x_therm)
        3. h_t = CrossAttentionBlock(e_t, th_t)
        4. D_t, delta_D = DamageBranch(s_t, D_prev)
        5. R_t = RegenerationBranch(rest_features, h_t, rest_flag)
        6. SoH_t = SoH_0 - D_t + R_t
        7. SoC_t = SoCHead(h_t)
        8. RUL_t = RULHead(D_window, cycle_idx)

    The model processes a SEQUENCE of cycles for each cell, maintaining
    cumulative damage D_t across the sequence.
    """

    def __init__(self, cfg: SLACConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or SLACConfig()

        # Map the configured initial value into the unconstrained tanh space.
        midpoint = (self.cfg.soh_0_min + self.cfg.soh_0_max) / 2.0
        half_span = (self.cfg.soh_0_max - self.cfg.soh_0_min) / 2.0
        normalized_init = (self.cfg.soh_0_init - midpoint) / half_span
        normalized_init = min(max(normalized_init, -0.999999), 0.999999)
        self._soh_0_raw = nn.Parameter(
            torch.tensor(math.atanh(normalized_init), dtype=torch.float32)
        )

        # Submodules
        self.elec_encoder = ElectricalEncoder(self.cfg)
        self.therm_encoder = ThermalEncoder(self.cfg)
        self.cross_attn = CrossAttentionBlock(self.cfg)
        self.damage_branch = DamageBranch(self.cfg)
        self.regen_branch = RegenerationBranch(self.cfg)
        self.soc_head = SoCHead(self.cfg)
        self.rul_head = RULHead(self.cfg)

    @property
    def soh_0(self) -> Tensor:
        """Learnable SoH_0 smoothly constrained to [soh_0_min, soh_0_max]."""
        midpoint = (self.cfg.soh_0_min + self.cfg.soh_0_max) / 2.0
        half_span = (self.cfg.soh_0_max - self.cfg.soh_0_min) / 2.0
        return midpoint + half_span * torch.tanh(self._soh_0_raw)

    def forward(
        self,
        x_elec: Tensor,
        x_therm: Tensor,
        s_t: Tensor,
        rest_features: Tensor,
        rest_flag: Tensor,
        cycle_idx: Tensor,
    ) -> dict[str, Tensor]:
        """Forward pass over a SEQUENCE of cycles.

        All inputs have shape (B, T, *) where T is the sequence length
        (number of cycles). The model processes each time step sequentially
        to maintain the cumulative damage state.

        Parameters
        ----------
        x_elec : Tensor, shape (B, T, elec_input_dim)
            Electrical features per cycle.
        x_therm : Tensor, shape (B, T, therm_input_dim)
            Thermal features per cycle.
        s_t : Tensor, shape (B, T, stressor_dim)
            Stressor vector per cycle.
        rest_features : Tensor, shape (B, T, rest_input_dim)
            Rest metadata per cycle.
        rest_flag : Tensor, shape (B, T, 1)
            Binary rest flag per cycle.
        cycle_idx : Tensor, shape (B, T, 1)
            Normalized cycle index per cycle.

        Returns
        -------
        dict with keys:
            'soh': (B, T, 1) -- predicted SoH per cycle
            'soc': (B, T, 1) -- predicted end-of-discharge SoC
            'rul': (B, T, 1) -- predicted RUL
            'D_t': (B, T, 1) -- cumulative damage
            'R_t': (B, T, 1) -- regeneration
            'damage_increments': (B, T, 1) -- per-cycle damage increments
        """
        B, T, _ = x_elec.shape

        # 1. Encoders (natively support arbitrary leading dimensions)
        e_seq = self.elec_encoder(x_elec)      # (B, T, elec_embed_dim)
        th_seq = self.therm_encoder(x_therm)  # (B, T, therm_embed_dim)

        # 2. Cross-attention fusion per cycle
        e_flat = e_seq.reshape(B * T, self.cfg.elec_embed_dim)
        th_flat = th_seq.reshape(B * T, self.cfg.therm_embed_dim)
        h_flat = self.cross_attn(e_flat, th_flat)  # (B * T, attn_dim)
        h_seq = h_flat.reshape(B, T, self.cfg.attn_dim)

        # 3. Damage branch (architecturally monotonic via cumsum)
        raw_damage = self.damage_branch.mlp(s_t)  # (B, T, 1)
        inc_seq = self.damage_branch.damage_scale * F.softplus(raw_damage)
        D_seq = torch.cumsum(inc_seq, dim=1)  # (B, T, 1), monotonic

        # 4. Regeneration branch (hard-gated)
        rest_flat = rest_features.reshape(B * T, self.cfg.rest_input_dim)
        rest_fl_flat = rest_flag.reshape(B * T, 1)
        R_flat = self.regen_branch(rest_flat, h_flat, rest_fl_flat)
        R_seq = R_flat.reshape(B, T, 1)

        # 5. SoH decomposition
        soh_seq = self.soh_0 - D_seq + R_seq  # (B, T, 1)

        # 6. SoC head
        soc_seq = self.soc_head(h_flat).reshape(B, T, 1)

        # 7. RUL head (windowed over D_t)
        k = self.cfg.rul_window_k
        D_padded = F.pad(D_seq.squeeze(-1), (k - 1, 0))  # (B, T + k - 1)
        D_windows = D_padded.unfold(dimension=1, size=k, step=1)  # (B, T, k)
        rul_in = torch.cat([D_windows, cycle_idx], dim=-1)  # (B, T, k + 1)
        rul_seq = self.rul_head.mlp(rul_in)  # (B, T, 1)

        return {
            "soh": soh_seq,
            "soc": soc_seq,
            "rul": rul_seq,
            "D_t": D_seq,
            "R_t": R_seq,
            "damage_increments": inc_seq,
        }


# =============================================================================
# LOSS FUNCTION (§2.7)
# =============================================================================

class SLACLoss(nn.Module):
    """Combined loss for PI-MDCNet v4 SLAC training.

    L = MSE(SoC_pred, SoC_true)
      + MSE(SoH_pred, SoH_true)
      + lambda_rul * MAE(RUL_pred, RUL_true)
      + lambda_sparse * mean(|R_t|)

    The L1 sparsity on R_t is belt-and-suspenders -- the hard gate is the
    primary control mechanism. The sparsity term just discourages the model
    from predicting large R_t even on rest cycles where the data doesn't
    support regeneration.
    """

    def __init__(self, cfg: SLACConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or SLACConfig()

    def forward(
        self,
        predictions: dict[str, Tensor],
        targets: dict[str, Tensor],
        rul_mask: Tensor | None = None,
        soc_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute combined loss.

        Parameters
        ----------
        predictions : dict
            Model outputs from PIMDCNet.forward().
        targets : dict
            Ground truth with keys: 'soh', 'soc', 'rul'.
        rul_mask : Tensor, optional, shape (B, T, 1)
            Mask for valid RUL labels (1 = valid, 0 = censored/NaN).
        soc_mask : Tensor, optional, shape (B, T, 1)
            Mask for informative SoC labels. Degenerate labels are masked out.

        Returns
        -------
        dict with keys:
            'total': total scalar loss
            'soc_loss': MSE on SoC
            'soh_loss': MSE on SoH
            'rul_loss': MAE on RUL (masked)
            'sparse_loss': L1 on R_t
        """
        soh_loss = F.mse_loss(predictions["soh"], targets["soh"])
        if soc_mask is not None and soc_mask.any():
            soc_loss = F.mse_loss(
                predictions["soc"][soc_mask.bool()], targets["soc"][soc_mask.bool()]
            )
        elif soc_mask is not None:
            soc_loss = torch.tensor(0.0, device=predictions["soc"].device)
        else:
            soc_loss = F.mse_loss(predictions["soc"], targets["soc"])

        # RUL loss: MAE, masked for right-censored cells
        if rul_mask is not None and rul_mask.any():
            rul_pred_valid = predictions["rul"][rul_mask.bool()]
            rul_true_valid = targets["rul"][rul_mask.bool()]
            rul_loss = F.l1_loss(rul_pred_valid, rul_true_valid)
        elif rul_mask is not None:
            rul_loss = torch.tensor(0.0, device=predictions["rul"].device)
        else:
            rul_loss = F.l1_loss(predictions["rul"], targets["rul"])

        # L1 sparsity on R_t
        sparse_loss = predictions["R_t"].abs().mean()

        total = (
            soc_loss
            + soh_loss
            + self.cfg.lambda_rul * rul_loss
            + self.cfg.lambda_sparse * sparse_loss
        )

        return {
            "total": total,
            "soc_loss": soc_loss,
            "soh_loss": soh_loss,
            "rul_loss": rul_loss,
            "sparse_loss": sparse_loss,
        }


# =============================================================================
# UNIT TESTS (spec §2 definition of done)
# =============================================================================

def _run_assertions() -> None:
    """Run unit-test assertions for the SLAC architecture.

    Definition of done (spec §2):
    1. Forward pass runs on a batch without error.
    2. D_t is non-decreasing across a synthetic sequence.
    3. R_t == 0 wherever rest_flag_t == 0.
    """
    print("=" * 70)
    print("PI-MDCNet v4 SLAC -- Unit Test Assertions")
    print("=" * 70)

    cfg = SLACConfig()
    model = PIMDCNet(cfg)
    model.eval()

    B, T = 4, 20  # 4 cells, 20 cycles

    # Create synthetic inputs
    torch.manual_seed(42)
    x_elec = torch.randn(B, T, cfg.elec_input_dim)
    x_therm = torch.randn(B, T, cfg.therm_input_dim)
    s_t = torch.abs(torch.randn(B, T, cfg.stressor_dim))  # stressors are positive
    rest_features = torch.randn(B, T, cfg.rest_input_dim)
    cycle_idx = torch.linspace(0, 1, T).unsqueeze(0).unsqueeze(-1).expand(B, T, 1)

    # Create rest flags: only a few cycles have rest
    rest_flag = torch.zeros(B, T, 1)
    rest_flag[:, 5, :] = 1.0   # rest at cycle 5
    rest_flag[:, 12, :] = 1.0  # rest at cycle 12
    rest_flag[:, 18, :] = 1.0  # rest at cycle 18

    # --- Test 1: Forward pass ---
    print("\n[Test 1] Forward pass on synthetic batch...")
    with torch.no_grad():
        outputs = model(x_elec, x_therm, s_t, rest_features, rest_flag, cycle_idx)

    for key, val in outputs.items():
        print(f"  {key}: shape={list(val.shape)}, "
              f"range=[{val.min().item():.6f}, {val.max().item():.6f}]")

    assert outputs["soh"].shape == (B, T, 1), f"SoH shape mismatch: {outputs['soh'].shape}"
    assert outputs["soc"].shape == (B, T, 1), f"SoC shape mismatch: {outputs['soc'].shape}"
    assert outputs["rul"].shape == (B, T, 1), f"RUL shape mismatch: {outputs['rul'].shape}"
    assert outputs["D_t"].shape == (B, T, 1), f"D_t shape mismatch: {outputs['D_t'].shape}"
    assert outputs["R_t"].shape == (B, T, 1), f"R_t shape mismatch: {outputs['R_t'].shape}"
    print("  [PASS] Forward pass shapes correct.")

    # --- Test 2: D_t monotonicity ---
    print("\n[Test 2] D_t monotonicity (architecturally guaranteed)...")
    D_t = outputs["D_t"]  # (B, T, 1)
    for b in range(B):
        D_seq = D_t[b, :, 0]  # (T,)
        diffs = D_seq[1:] - D_seq[:-1]
        assert torch.all(diffs >= -1e-7), (
            f"D_t is NOT monotonic for batch {b}! "
            f"Min diff: {diffs.min().item():.8f}"
        )
    print(f"  D_t range across all batches: "
          f"[{D_t.min().item():.6f}, {D_t.max().item():.6f}]")
    print("  [PASS] D_t is non-decreasing for all sequences.")

    # --- Test 3: R_t hard gating ---
    print("\n[Test 3] R_t hard gating (zero when rest_flag=0)...")
    R_t = outputs["R_t"]  # (B, T, 1)
    non_rest_mask = (rest_flag == 0.0)  # (B, T, 1)
    R_at_non_rest = R_t[non_rest_mask]
    assert torch.all(R_at_non_rest == 0.0), (
        f"R_t is nonzero at non-rest cycles! "
        f"Max |R_t| at non-rest: {R_at_non_rest.abs().max().item():.8f}"
    )

    # Check that R_t CAN be nonzero at rest cycles
    rest_mask = (rest_flag == 1.0)
    R_at_rest = R_t[rest_mask]
    print(f"  R_t at non-rest cycles: all exactly 0.0")
    print(f"  R_t at rest cycles: range=[{R_at_rest.min().item():.6f}, "
          f"{R_at_rest.max().item():.6f}]")
    print("  [PASS] R_t is exactly 0 wherever rest_flag=0.")

    # --- Test 4: SoH decomposition ---
    print("\n[Test 4] SoH = SoH_0 - D_t + R_t (decomposition check)...")
    soh_pred = outputs["soh"]
    soh_reconstructed = model.soh_0 - D_t + R_t
    err = (soh_pred - soh_reconstructed).abs().max().item()
    assert err < 1e-5, f"SoH decomposition error: {err:.8f}"
    print(f"  Max reconstruction error: {err:.8f}")
    print(f"  SoH_0 = {model.soh_0.item():.6f}")
    print("  [PASS] SoH decomposition is exact.")

    # --- Test 5: Loss computation ---
    print("\n[Test 5] Loss function computation...")
    loss_fn = SLACLoss(cfg)
    targets_dict = {
        "soh": torch.rand(B, T, 1),
        "soc": torch.rand(B, T, 1),
        "rul": torch.rand(B, T, 1) * 100,
    }
    rul_mask = torch.ones(B, T, 1)
    rul_mask[:, -5:, :] = 0  # last 5 cycles censored

    losses = loss_fn(outputs, targets_dict, rul_mask)
    for key, val in losses.items():
        print(f"  {key}: {val.item():.6f}")
    assert not torch.isnan(losses["total"]), "Total loss is NaN!"
    assert losses["total"].item() > 0, "Total loss should be positive!"
    print("  [PASS] Loss computation successful.")

    # --- Test 6: Parameter count ---
    print("\n[Test 6] Model parameter count...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size (float32): {total_params * 4 / 1024:.1f} KB")

    print("\n" + "=" * 70)
    print("[ALL PASS] PI-MDCNet v4 SLAC architecture verified.")
    print("=" * 70)


if __name__ == "__main__":
    _run_assertions()
