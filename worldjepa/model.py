"""
WorldJEPA v0.1 — Core Model
CertesLabs · April 2026

Architecture:
  - Encoder: ViT-L (~300M) initialized from V-JEPA 2 pretrained weights
  - Predictor: ViT-S (~22M) trained from scratch with causal attention
  - Training objective: MSE prediction + SIGReg (no EMA, no stop-grad)

Based on principles from:
  - V-JEPA 2 (Assran et al., Meta FAIR, 2025) — MIT License
  - LeWorldModel (Maes et al., 2026) — MIT License
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from worldjepa.sigreg import SIGRegFast


# ─── Encoder ──────────────────────────────────────────────────────────────

class WorldJEPAEncoder(nn.Module):
    """
    ViT-L encoder initialized from V-JEPA 2 pretrained weights.

    For v0.1, can be:
    - Frozen: only predictor trains (fast, fits M4 Pro)
    - Fine-tuned: full end-to-end (requires more compute)

    Output: CLS token + projection → latent z of dimension `latent_dim`
    """

    def __init__(
        self,
        vit_dim: int = 1024,          # ViT-L hidden dimension
        latent_dim: int = 1024,        # Output latent dimension
        freeze: bool = True,           # Freeze encoder for v0.1
        use_cls_token: bool = True,    # Use CLS token as frame representation
    ):
        super().__init__()
        self.vit_dim = vit_dim
        self.latent_dim = latent_dim
        self.freeze = freeze

        # Projection head: maps ViT-L output → latent space
        # 1-layer MLP with BatchNorm, critical for SIGReg (see LeWM paper §3)
        self.projector = nn.Sequential(
            nn.Linear(vit_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
        )

        # Backbone will be loaded separately via load_vjepa2_encoder()
        self.backbone = None

    def load_vjepa2_encoder(self, model_name: str = "vjepa2_vit_large"):
        """
        Load pretrained V-JEPA 2 ViT-L encoder via torch.hub.

        Args:
            model_name: one of 'vjepa2_vit_large', 'vjepa2_vit_huge', 'vjepa2_vit_giant'
        """
        print(f"[WorldJEPA] Loading {model_name} from facebookresearch/vjepa2...")
        self.backbone = torch.hub.load(
            "facebookresearch/vjepa2",
            model_name,
            trust_repo=True,
        )

        if self.freeze:
            print("[WorldJEPA] Encoder frozen — only predictor will train.")
            for param in self.backbone.parameters():
                param.requires_grad_(False)
        else:
            print("[WorldJEPA] Encoder unfrozen — full end-to-end training.")

        return self

    def load_mock_encoder(self, device="cpu"):
        """
        Lightweight mock encoder for testing pipeline without downloading weights.
        Replaces ViT-L with a small random network — same API, different capacity.
        """
        print("[WorldJEPA] Using MOCK encoder (for testing only)")

        class MockViT(nn.Module):
            def __init__(self, out_dim):
                super().__init__()
                self.out_dim = out_dim
                # Minimal conv + pooling to simulate ViT output
                self.conv = nn.Conv2d(3, 64, kernel_size=16, stride=16)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.proj = nn.Linear(64, out_dim)

            def forward(self, x):
                # x: (B, C, H, W)
                feat = self.conv(x)           # (B, 64, H', W')
                feat = self.pool(feat)        # (B, 64, 1, 1)
                feat = feat.flatten(1)        # (B, 64)
                return self.proj(feat)        # (B, out_dim)

        self.backbone = MockViT(self.vit_dim).to(device)
        if self.freeze:
            for param in self.backbone.parameters():
                param.requires_grad_(False)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a single frame.

        Args:
            x: (B, C, H, W) — single video frame

        Returns:
            z: (B, latent_dim) — latent embedding
        """
        assert self.backbone is not None, \
            "Call load_vjepa2_encoder() or load_mock_encoder() first."

        if self.freeze:
            with torch.no_grad():
                feat = self.backbone(x)  # (B, vit_dim)
        else:
            feat = self.backbone(x)

        z = self.projector(feat)  # (B, latent_dim)
        return z

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode all frames of a video clip independently.

        Args:
            video: (B, T, C, H, W)

        Returns:
            Z: (B, T, latent_dim)
        """
        B, T, C, H, W = video.shape
        # Reshape to process all frames in one batch
        frames = video.reshape(B * T, C, H, W)
        Z_flat = self.forward(frames)          # (B*T, latent_dim)
        Z = Z_flat.reshape(B, T, self.latent_dim)
        return Z


# ─── Predictor ────────────────────────────────────────────────────────────

class WorldJEPAPredictor(nn.Module):
    """
    Causal transformer predictor.

    Takes a history of N latent frames and predicts the next frame embedding.
    Architecture: ViT-S equivalent with causal masking.

    Key differences from LeWM:
    - No action conditioning (action-free for v0.1)
    - Operates on higher-dim latents (1024 vs 192 in LeWM)
    - Trained from scratch (random init)
    """

    def __init__(
        self,
        latent_dim: int = 1024,    # Input/output dimension (matches encoder)
        hidden_dim: int = 384,     # Internal predictor dimension (ViT-S)
        num_layers: int = 12,      # Transformer depth
        num_heads: int = 6,        # Attention heads
        dropout: float = 0.1,     # Critical: 0.1 is optimal (LeWM ablation Table 9)
        max_seq_len: int = 16,     # Max number of frames in history
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Project from latent_dim → hidden_dim
        self.input_proj = nn.Linear(latent_dim, hidden_dim)

        # Learned positional embeddings
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)

        # Causal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Project back from hidden_dim → latent_dim
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),  # BatchNorm on output too
        )

        # Causal mask — will be built lazily
        self._causal_mask = None

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for training stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Build causal attention mask — cached for efficiency."""
        if self._causal_mask is None or self._causal_mask.shape[0] != seq_len:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device) * float("-inf"),
                diagonal=1,
            )
            self._causal_mask = mask
        return self._causal_mask.to(device)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Predict next latent states for all timesteps (teacher forcing).

        Args:
            Z: (B, T, latent_dim) — sequence of encoded frame embeddings

        Returns:
            Z_pred: (B, T, latent_dim) — predicted next-frame embeddings
                    Z_pred[:, t, :] predicts Z[:, t+1, :]
        """
        B, T, D = Z.shape
        device = Z.device

        # Project to hidden dim
        h = self.input_proj(Z)  # (B, T, hidden_dim)

        # Add positional embeddings
        positions = torch.arange(T, device=device)
        h = h + self.pos_embed(positions).unsqueeze(0)  # (B, T, hidden_dim)

        # Causal self-attention
        causal_mask = self._get_causal_mask(T, device)
        h = self.transformer(h, mask=causal_mask)  # (B, T, hidden_dim)

        # Project back: reshape for BatchNorm, then restore
        h_flat = h.reshape(B * T, self.hidden_dim)
        Z_pred_flat = self.output_proj(h_flat)        # (B*T, latent_dim)
        Z_pred = Z_pred_flat.reshape(B, T, D)

        return Z_pred


# ─── Full WorldJEPA Model ─────────────────────────────────────────────────

class WorldJEPA(nn.Module):
    """
    WorldJEPA v0.1 — Complete world model.

    Training objective:
        L = L_pred + λ · SIGReg(Z)

        L_pred  = MSE(Z_pred[:, :-1], Z[:, 1:].detach())
        SIGReg  = Sketched Isotropic Gaussian Regularizer (step-wise)

    No EMA. No stop-gradient tricks. No action conditioning.
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        predictor_hidden: int = 384,
        predictor_layers: int = 12,
        predictor_heads: int = 6,
        predictor_dropout: float = 0.1,
        lambda_sigreg: float = 0.1,
        num_projections: int = 1024,
        freeze_encoder: bool = True,
        max_seq_len: int = 16,
    ):
        super().__init__()

        self.lambda_sigreg = lambda_sigreg

        # Components
        self.encoder = WorldJEPAEncoder(
            vit_dim=1024,
            latent_dim=latent_dim,
            freeze=freeze_encoder,
        )

        self.predictor = WorldJEPAPredictor(
            latent_dim=latent_dim,
            hidden_dim=predictor_hidden,
            num_layers=predictor_layers,
            num_heads=predictor_heads,
            dropout=predictor_dropout,
            max_seq_len=max_seq_len,
        )

        self.sigreg = SIGRegFast(
            num_projections=num_projections,
        )

    def forward(self, video: torch.Tensor) -> dict:
        """
        Full forward pass with loss computation.

        Args:
            video: (B, T, C, H, W) — video clip

        Returns:
            dict with keys:
                'loss'        — total training loss
                'loss_pred'   — prediction MSE
                'loss_sigreg' — SIGReg regularization
                'isotropy'    — isotropy score (monitoring metric)
                'Z'           — latent embeddings (B, T, D)
                'Z_pred'      — predicted embeddings (B, T, D)
        """
        B, T, C, H, W = video.shape

        # 1. Encode all frames
        Z = self.encoder.encode_video(video)    # (B, T, D)

        # 2. Predict next frames (teacher forcing)
        Z_pred = self.predictor(Z)              # (B, T, D)

        # 3. Prediction loss — target detached from gradient flow
        # Z_pred[:, :-1] predicts Z[:, 1:]
        Z_target = Z[:, 1:, :].detach()        # (B, T-1, D)
        Z_source = Z_pred[:, :-1, :]           # (B, T-1, D)
        loss_pred = F.mse_loss(Z_source, Z_target)

        # 4. SIGReg — applied step-wise, averaged over timesteps
        # Reshape: (B, T, D) → T × (B, D) for per-timestep computation
        sigreg_losses = []
        for t in range(T):
            Z_t = Z[:, t, :]                   # (B, D)
            sigreg_losses.append(self.sigreg(Z_t))
        loss_sigreg = torch.stack(sigreg_losses).mean()

        # 5. Total loss
        loss = loss_pred + self.lambda_sigreg * loss_sigreg

        # 6. Isotropy score (monitoring only, not gradient)
        with torch.no_grad():
            isotropy = self._isotropy_score(Z.reshape(-1, Z.shape[-1]))

        return {
            "loss": loss,
            "loss_pred": loss_pred.detach(),
            "loss_sigreg": loss_sigreg.detach(),
            "isotropy": isotropy,
            "Z": Z.detach(),
            "Z_pred": Z_pred.detach(),
        }

    @torch.no_grad()
    def _isotropy_score(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Compute isotropy score = λ_min / λ_max of covariance matrix.
        Score of 1.0 = perfectly isotropic. Score < 0.1 = collapsed.

        Args:
            Z: (N, D)

        Returns:
            scalar in [0, 1]
        """
        # Subsample if too large (covariance is D×D regardless)
        if Z.shape[0] > 2048:
            idx = torch.randperm(Z.shape[0])[:2048]
            Z = Z[idx]

        Z_centered = Z - Z.mean(dim=0, keepdim=True)
        cov = (Z_centered.T @ Z_centered) / (Z.shape[0] - 1)

        try:
            eigvals = torch.linalg.eigvalsh(cov)
            eigvals = eigvals.clamp(min=0)  # Numerical safety
            score = eigvals.min() / (eigvals.max() + 1e-10)
        except Exception:
            score = torch.tensor(0.0, device=Z.device)

        return score

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        params = self.parameters() if not trainable_only else \
                 filter(lambda p: p.requires_grad, self.parameters())
        return sum(p.numel() for p in params)
