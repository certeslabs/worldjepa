"""
worldjepa/model.py — Phase 0.5

Architecture:
  Encoder : frozen V-JEPA 2.1 ViT-B (80M) or ViT-L (300M) via torch.hub.
             The hub call returns (encoder, predictor); we discard the predictor
             deliberately.  V-JEPA 2.1's predictor is trained on spatial-masking
             with mask-distance-weighted Lctx — a different factorisation from
             WorldJEPA's temporal-causal objective.  Mixing them would be
             meaningless.  What we test is SIGReg applied to a temporal prediction
             head atop frozen V-JEPA 2.1 patch features, not "V-JEPA 2.1 + SIGReg"
             as a joint system.  See PRE_REGISTRATION.md §framing.

  Projector: Linear → BatchNorm1d  (LeWM §3.1; BN counteracts ViT's final
             LayerNorm and prevents SIGReg from fighting the encoder's normalization).
             No LayerNorm here — see plan v3 D2 Fix 2.

  Predictor: Causal transformer with hidden_dim = latent_dim (= encoder's
             embed_dim = 768 for ViT-B, 1024 for ViT-L).  Setting hidden_dim
             smaller caps achievable rank regardless of SIGReg — see plan D2 Fix 1.

  SIGReg:   lejepa.multivariate.SlicingUnivariateTest with
             EppsPulley(n_points=17).
             NOTE: the README contains a stale kwarg spelling; the installed
             package uses n_points=17.  Verified by inspection.
             See plan v3 D2 Fix 4 and tests/test_lejepa_api.py.

  Loss:     L = MSE(Z_pred, sg(Z_target)) + λ · SIGReg(Z_pred_flat)
             where sg() is stop-gradient applied to the encoder targets.

Geometry (ViT-B, T input frames, 384×384):
  tubelet_size = 2  →  T_eff = T // 2  temporal positions
  patch_size   = 16 →  24×24 = 576 spatial patches per position
  total tokens      →  T_eff × 576
  encoder output    →  (B, T_eff × 576, 768)
  after mean pool   →  (B, T_eff, 768)

Usage:
    model = WorldJEPA(encoder_variant="vitb", freeze_encoder=True)
    out   = model(video)  # video: (B, T, C, H, W)
    loss  = out["loss"]
"""

from __future__ import annotations

import math
import warnings
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

import lejepa

# ---------------------------------------------------------------------------
# Constants derived from V-JEPA 2.1 architecture
# ---------------------------------------------------------------------------

_VARIANT_CONFIG = {
    "vitb": {
        "hub_name": "vjepa2_1_vit_base_384",
        "embed_dim": 768,
        "default_predictor_layers": 6,
    },
    "vitl": {
        "hub_name": "vjepa2_1_vit_large_384",
        "embed_dim": 1024,
        "default_predictor_layers": 12,
    },
}

_TUBELET_SIZE = 2  # V-JEPA 2.1 default; patchifies consecutive frame pairs
_PATCH_SIZE = 16  # spatial patch size in pixels
_RESOLUTION = 384  # expected input resolution
_N_SPATIAL = (_RESOLUTION // _PATCH_SIZE) ** 2  # 576 patches per temporal position


# ---------------------------------------------------------------------------
# Encoder wrapper
# ---------------------------------------------------------------------------


def _load_vjepa21_encoder(variant: str) -> nn.Module:
    """
    Load the V-JEPA 2.1 encoder via torch.hub.

    The hub call returns a (encoder, predictor) tuple.  We discard the
    predictor — see module docstring for the rationale.

    Returns:
        encoder set to eval() with all parameters frozen.
    """
    cfg = _VARIANT_CONFIG[variant]
    encoder, _predictor_discarded = torch.hub.load(
        "facebookresearch/vjepa2",
        cfg["hub_name"],
        trust_repo=True,
    )
    # Freeze all encoder parameters.  This is non-negotiable for Phase 0.5:
    # we are not end-to-end fine-tuning; we characterise SIGReg in the
    # frozen-encoder regime.
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


class FrozenVJEPA21Encoder(nn.Module):
    """
    Thin wrapper around a frozen V-JEPA 2.1 backbone.

    Handles the patch-token reshape and pooling so that the rest of the
    model sees clean (B, T_eff, D) tensors.
    """

    def __init__(
        self,
        variant: str = "vitb",
        feature_mode: Literal["mean", "cls"] = "mean",
    ):
        super().__init__()
        assert (
            variant in _VARIANT_CONFIG
        ), f"Unknown variant '{variant}'. Choices: {list(_VARIANT_CONFIG)}"
        self.variant = variant
        self.embed_dim = _VARIANT_CONFIG[variant]["embed_dim"]
        self.feature_mode = feature_mode
        self.backbone = _load_vjepa21_encoder(variant)

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            videos: (B, T, C, H, W)  — T must be even (tubelet_size=2).

        Returns:
            Z: (B, T_eff, D)  where T_eff = T // 2.
        """
        B, T, C, H, W = videos.shape
        assert (
            T % _TUBELET_SIZE == 0
        ), f"T={T} must be divisible by tubelet_size={_TUBELET_SIZE}"
        assert (
            H == W == _RESOLUTION
        ), f"Expected {_RESOLUTION}×{_RESOLUTION} input, got {H}×{W}"

        T_eff = T // _TUBELET_SIZE

        # Encoder is frozen — no grad needed here.
        with torch.no_grad():
            videos_bcthw = videos.permute(
                0, 2, 1, 3, 4
            ).contiguous()  # (B,C,T,H,W) pour V-JEPA 2.1
            out = self.backbone(videos_bcthw)  # (B, T_eff × N_spatial, D)

        B_out, N_tot, D = out.shape
        assert (
            D == self.embed_dim
        ), f"embed_dim mismatch: expected {self.embed_dim}, got {D}"
        expected_tokens = T_eff * _N_SPATIAL
        assert N_tot == expected_tokens, (
            f"Token count mismatch: expected {expected_tokens} "
            f"(T_eff={T_eff} × N_spatial={_N_SPATIAL}), got {N_tot}. "
            "Check that input resolution is 384 and T is correct."
        )

        out = out.reshape(B, T_eff, _N_SPATIAL, D)

        if self.feature_mode == "mean":
            Z = out.mean(dim=2)  # (B, T_eff, D) — mean over spatial patches
        elif self.feature_mode == "cls":
            # "cls" here means first patch token — V-JEPA 2.1 has no dedicated
            # CLS token.  Micro-exp 1 should confirm mean >> cls.
            Z = out[:, :, 0, :]  # (B, T_eff, D)
        else:
            raise ValueError(f"Unknown feature_mode: {self.feature_mode!r}")

        return Z  # (B, T_eff, D)


# ---------------------------------------------------------------------------
# Projector
# ---------------------------------------------------------------------------


class Projector(nn.Module):
    """
    Linear → BatchNorm1d projector.

    BatchNorm is intentional (LeWM §3.1): it counteracts the ViT's final
    LayerNorm, which otherwise prevents SIGReg from shaping the distribution.
    Do NOT replace with LayerNorm.

    The input/output dimensionality can differ (e.g. project 768 → 768),
    but both are required; no silent defaults.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., in_dim) — any leading batch dimensions.

        Returns:
            y: (..., out_dim)
        """
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1])  # (N, in_dim)
        y_flat = self.bn(self.proj(x_flat))  # (N, out_dim)
        return y_flat.reshape(*shape[:-1], -1)  # (..., out_dim)


# ---------------------------------------------------------------------------
# Temporal causal predictor
# ---------------------------------------------------------------------------


class TemporalCausalPredictor(nn.Module):
    """
    Autoregressive causal transformer that predicts Z_{t+1} given Z_{0:t}.

    hidden_dim MUST equal latent_dim (the encoder's embed_dim).  A smaller
    hidden_dim caps achievable rank regardless of SIGReg — see plan D2 Fix 1
    and arXiv:2512.24497 §predictor.

    The causal mask is built once in __init__ and registered as a buffer
    so it moves to the correct device automatically.
    """

    def __init__(
        self,
        latent_dim: int,
        num_layers: int | None = None,
        num_heads: int = 8,
        ffn_mult: float = 4.0,
        dropout: float = 0.0,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # hidden_dim == latent_dim is enforced (plan D2 Fix 1)
        hidden_dim = latent_dim

        # Default num_layers by encoder size (plan §Fix 1 comment)
        if num_layers is None:
            num_layers = 6 if latent_dim == 768 else 12

        # Sinusoidal position embedding
        self.register_buffer("pos_emb", _sinusoidal_pos_emb(max_seq_len, latent_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(hidden_dim * ffn_mult),
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm (more stable)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Causal mask: upper-triangular with -inf above diagonal
        causal = torch.full((max_seq_len, max_seq_len), float("-inf"))
        causal = torch.triu(causal, diagonal=1)
        self.register_buffer("causal_mask", causal)

        self.out_norm = nn.LayerNorm(latent_dim)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z: (B, T, D)  — projected encoder features.

        Returns:
            Z_pred: (B, T, D)  — predicted representation for position t+1
                    (i.e. Z_pred[:, t, :] is the prediction of Z[:, t+1, :]).
        """
        B, T, D = Z.shape
        assert D == self.latent_dim, f"Input dim {D} ≠ latent_dim {self.latent_dim}"

        x = Z + self.pos_emb[:T]  # (B, T, D)
        mask = self.causal_mask[:T, :T]
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.out_norm(x)  # (B, T, D)


# ---------------------------------------------------------------------------
# Full WorldJEPA model
# ---------------------------------------------------------------------------


class WorldJEPA(nn.Module):
    """
    WorldJEPA Phase 0.5.

    Training objective:
        L = MSE(Z_pred, sg(Z_target)) + λ · SIGReg(Z_pred_flat)

    where:
        Z        = FrozenEncoder(video)        projected (B, T_eff, D)
        Z_pred   = Predictor(Z)                (B, T_eff, D)
        Z_target = Z shifted by 1              (B, T_eff-1, D)
        sg()     = stop-gradient (encoder targets are not differentiated)
        SIGReg   = lejepa.multivariate.SlicingUnivariateTest

    The SIGReg term regularises Z_pred towards an isotropic Gaussian,
    preventing collapse without teacher-student or momentum encoders.

    sigreg_reduction:
        "per_timestep" — SIGReg is computed separately for each temporal
                         position and averaged (LeWM-style).  Requires B≥32.
        "flatten"      — Z_pred is reshaped to (B×T, D) before SIGReg.
                         Viable at small batch sizes (e.g. T4 batch 16).
    """

    def __init__(
        self,
        encoder_variant: str = "vitb",
        freeze_encoder: bool = True,
        feature_mode: Literal["mean", "cls"] = "mean",
        lambda_sigreg: float = 0.1,
        sigreg_reduction: Literal["per_timestep", "flatten"] = "per_timestep",
        # predictor kwargs
        predictor_layers: int | None = None,
        predictor_heads: int = 8,
        predictor_ffn_mult: float = 4.0,
        predictor_dropout: float = 0.0,
    ):
        super().__init__()

        assert freeze_encoder, (
            "freeze_encoder=False is not supported in Phase 0.5. "
            "End-to-end fine-tuning is Phase 2+."
        )

        cfg = _VARIANT_CONFIG[encoder_variant]
        self.embed_dim = cfg["embed_dim"]
        self.lambda_sigreg = lambda_sigreg
        self.sigreg_reduction = sigreg_reduction

        # ── Encoder (frozen) ──────────────────────────────────────────────
        self.encoder = FrozenVJEPA21Encoder(
            variant=encoder_variant, feature_mode=feature_mode
        )

        # ── Projector (encoder → predictor space) ─────────────────────────
        # Dimension is preserved (in = out = embed_dim).  The projector's
        # BatchNorm is the critical component — see LeWM §3.1.
        self.projector = Projector(in_dim=self.embed_dim, out_dim=self.embed_dim)

        # ── Temporal causal predictor ─────────────────────────────────────
        self.predictor = TemporalCausalPredictor(
            latent_dim=self.embed_dim,
            num_layers=predictor_layers or cfg["default_predictor_layers"],
            num_heads=predictor_heads,
            ffn_mult=predictor_ffn_mult,
            dropout=predictor_dropout,
        )

        # ── SIGReg (lejepa official library) ──────────────────────────────
        # IMPORTANT: the constructor kwarg is `n_points`, not the variant
        # spelled with an underscore-less prefix that appears in the README.
        # The README is outdated; the installed code uses n_points.
        # Verified: inspect.signature(lejepa.univariate.EppsPulley.__init__)
        # → (self, t_max: float = 3, n_points: int = 17, integration: str = 'trapezoid')
        # See tests/test_lejepa_api.py for a regression guard.
        self.sigreg = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=lejepa.univariate.EppsPulley(n_points=17),
            num_slices=1024,
        )

    # ── Convenience: encode only ──────────────────────────────────────────

    def encode(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Run only the frozen encoder + projector.

        Args:
            videos: (B, T, C, H, W)

        Returns:
            Z: (B, T_eff, D)  — projected encoder features.
        """
        Z_raw = self.encoder(videos)  # (B, T_eff, D)
        return self.projector(Z_raw)  # (B, T_eff, D)

    # ── Full forward ──────────────────────────────────────────────────────

    def forward(self, videos: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            videos: (B, T, C, H, W)

        Returns dict with keys:
            loss          — total training loss (scalar)
            loss_mse      — temporal prediction MSE component
            loss_sigreg   — SIGReg regularisation component
            Z             — encoder features (B, T_eff, D), detached
            Z_pred        — predictor output  (B, T_eff, D)
        """
        # ── 1. Encode (no grad — encoder is frozen) ──────────────────────
        Z = self.encode(videos)  # (B, T_eff, D)
        B, T, D = Z.shape

        # ── 2. Predict ───────────────────────────────────────────────────
        Z_pred = self.predictor(Z)  # (B, T, D)

        # ── 3. MSE loss ──────────────────────────────────────────────────
        # Z_pred[:, t] predicts Z[:, t+1]  (causal, so shift by 1).
        # Stop-gradient on targets: encoder is frozen, but we make this
        # explicit so nothing back-props through the encoder path.
        pred_for_loss = Z_pred[:, :-1, :]  # (B, T-1, D)
        target_for_loss = Z[:, 1:, :].detach()  # (B, T-1, D) — stop-grad

        loss_mse = F.mse_loss(pred_for_loss, target_for_loss)

        # ── 4. SIGReg loss ───────────────────────────────────────────────
        if self.sigreg_reduction == "per_timestep":
            if B < 32:
                warnings.warn(
                    f"SIGReg per_timestep with B={B} < 32 is noisy. "
                    "Switching to flatten reduction automatically. "
                    "Use --sigreg_reduction flatten to silence this warning.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                loss_sigreg = self.sigreg(Z_pred.reshape(B * T, D))
            else:
                # Compute SIGReg per temporal position and average
                loss_sigreg = torch.stack(
                    [self.sigreg(Z_pred[:, t, :]) for t in range(T)]
                ).mean()
        else:
            # "flatten": treat (B, T, D) as (B*T, D)
            loss_sigreg = self.sigreg(Z_pred.reshape(B * T, D))

        # ── 5. Total loss ─────────────────────────────────────────────────
        loss = loss_mse + self.lambda_sigreg * loss_sigreg

        return {
            "loss": loss,
            "loss_mse": loss_mse,
            "loss_sigreg": loss_sigreg,
            "Z": Z.detach(),
            "Z_pred": Z_pred,
        }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _sinusoidal_pos_emb(max_len: int, dim: int) -> torch.Tensor:
    """Standard sinusoidal position embedding, shape (max_len, dim)."""
    position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
    div_term = torch.exp(
        torch.arange(0, dim, 2) * (-math.log(10000.0) / dim)
    )  # (dim//2,)
    pe = torch.zeros(max_len, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
