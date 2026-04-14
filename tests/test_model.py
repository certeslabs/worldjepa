"""
Tests — worldjepa.model
pytest tests/test_model.py
"""

import pytest
import torch
from worldjepa.model import WorldJEPA, WorldJEPAEncoder, WorldJEPAPredictor


# ─── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def small_model():
    """Minimal WorldJEPA for fast testing."""
    model = WorldJEPA(
        latent_dim=128,
        predictor_hidden=64,
        predictor_layers=2,
        predictor_heads=4,
        predictor_dropout=0.0,   # No dropout in tests
        lambda_sigreg=0.1,
        num_projections=32,
        freeze_encoder=False,
        max_seq_len=8,
    )
    model.encoder.load_mock_encoder()
    return model


@pytest.fixture
def mock_video():
    """Small mock video batch: (B=2, T=8, C=3, H=64, W=64)."""
    torch.manual_seed(0)
    return torch.randn(2, 8, 3, 64, 64)


# ─── Encoder tests ────────────────────────────────────────────────────────

class TestWorldJEPAEncoder:

    def test_encode_single_frame(self):
        enc = WorldJEPAEncoder(vit_dim=64, latent_dim=128, freeze=False)
        enc.load_mock_encoder()
        frame = torch.randn(4, 3, 64, 64)     # (B, C, H, W)
        z = enc(frame)
        assert z.shape == (4, 128), f"Expected (4, 128), got {z.shape}"

    def test_encode_video(self):
        enc = WorldJEPAEncoder(vit_dim=64, latent_dim=128, freeze=False)
        enc.load_mock_encoder()
        video = torch.randn(2, 8, 3, 64, 64)  # (B, T, C, H, W)
        Z = enc.encode_video(video)
        assert Z.shape == (2, 8, 128), f"Expected (2, 8, 128), got {Z.shape}"

    def test_frozen_no_grad(self):
        enc = WorldJEPAEncoder(vit_dim=64, latent_dim=128, freeze=True)
        enc.load_mock_encoder()
        # Backbone should have no grad when frozen
        for param in enc.backbone.parameters():
            assert not param.requires_grad, "Frozen encoder must not require grad"

    def test_projector_trainable(self):
        enc = WorldJEPAEncoder(vit_dim=64, latent_dim=128, freeze=True)
        enc.load_mock_encoder()
        # Projector always trainable
        for param in enc.projector.parameters():
            assert param.requires_grad, "Projector must always require grad"


# ─── Predictor tests ──────────────────────────────────────────────────────

class TestWorldJEPAPredictor:

    def test_output_shape(self):
        pred = WorldJEPAPredictor(
            latent_dim=128, hidden_dim=64, num_layers=2, num_heads=4
        )
        Z = torch.randn(4, 8, 128)   # (B, T, D)
        Z_pred = pred(Z)
        assert Z_pred.shape == Z.shape, (
            f"Predictor output must match input shape, got {Z_pred.shape}"
        )

    def test_causal_masking(self):
        """
        Causal masking: prediction at t must not depend on future frames.
        We verify by checking that changing frame t+1 doesn't affect prediction at t.
        """
        pred = WorldJEPAPredictor(
            latent_dim=64, hidden_dim=32, num_layers=2, num_heads=4, dropout=0.0
        )
        pred.eval()

        Z = torch.randn(1, 6, 64)
        Z_modified = Z.clone()
        Z_modified[:, 3:, :] += 100.0   # Modify frames 3, 4, 5

        with torch.no_grad():
            out_original = pred(Z)
            out_modified = pred(Z_modified)

        # Predictions for t=0,1,2 must be identical (causal)
        assert torch.allclose(out_original[:, :3], out_modified[:, :3], atol=1e-5), \
            "Causal masking broken: future frames affect past predictions"

    def test_differentiable(self):
        pred = WorldJEPAPredictor(
            latent_dim=64, hidden_dim=32, num_layers=2, num_heads=4
        )
        Z = torch.randn(2, 4, 64, requires_grad=True)
        out = pred(Z)
        loss = out.mean()
        loss.backward()
        assert Z.grad is not None


# ─── Full WorldJEPA tests ─────────────────────────────────────────────────

class TestWorldJEPA:

    def test_forward_pass_shapes(self, small_model, mock_video):
        out = small_model(mock_video)
        B, T, D = 2, 8, 128

        assert out["Z"].shape == (B, T, D)
        assert out["Z_pred"].shape == (B, T, D)
        assert out["loss"].ndim == 0
        assert out["loss_pred"].ndim == 0
        assert out["loss_sigreg"].ndim == 0
        assert out["isotropy"].ndim == 0

    def test_loss_is_positive(self, small_model, mock_video):
        out = small_model(mock_video)
        assert out["loss"].item() > 0, "Total loss must be positive"
        assert out["loss_pred"].item() > 0, "Pred loss must be positive"
        assert out["loss_sigreg"].item() >= 0, "SIGReg loss must be non-negative"

    def test_loss_components_sum(self, small_model, mock_video):
        """Total loss = pred_loss + lambda * sigreg_loss."""
        out = small_model(mock_video)
        expected = (
            out["loss_pred"] +
            small_model.lambda_sigreg * out["loss_sigreg"]
        ).item()
        # Note: loss is detached for pred/sigreg, so compare approximately
        assert abs(out["loss"].item() - expected) < 0.01, \
            "Loss components don't sum correctly"

    def test_backward_pass(self, small_model, mock_video):
        """Training step must complete without errors."""
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, small_model.parameters()),
            lr=1e-4
        )
        optimizer.zero_grad()
        out = small_model(mock_video)
        out["loss"].backward()
        optimizer.step()
        # Check no NaN in parameters
        for name, param in small_model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), \
                    f"NaN gradient in {name}"

    def test_no_nan_in_output(self, small_model, mock_video):
        out = small_model(mock_video)
        assert not torch.isnan(out["loss"]), "NaN in total loss"
        assert not torch.isnan(out["Z"]).any(), "NaN in embeddings"
        assert not torch.isnan(out["Z_pred"]).any(), "NaN in predictions"

    def test_isotropy_score_range(self, small_model, mock_video):
        out = small_model(mock_video)
        iso = out["isotropy"].item()
        assert 0.0 <= iso <= 1.0, f"Isotropy score out of [0,1]: {iso}"

    def test_parameter_count(self, small_model):
        trainable = small_model.num_parameters(trainable_only=True)
        total = small_model.num_parameters(trainable_only=False)
        assert trainable <= total
        assert trainable > 0, "Must have trainable parameters"

    def test_frozen_encoder_not_updated(self, mock_video):
        """When encoder is frozen, its weights must not change after backward."""
        model = WorldJEPA(
            latent_dim=128,
            predictor_hidden=64,
            predictor_layers=2,
            predictor_heads=4,
            freeze_encoder=True,
            num_projections=16,
        )
        model.encoder.load_mock_encoder()

        # Save initial backbone weights
        initial_weights = {
            name: param.clone()
            for name, param in model.encoder.backbone.named_parameters()
        }

        # Training step
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
        )
        optimizer.zero_grad()
        out = model(mock_video)
        out["loss"].backward()
        optimizer.step()

        # Backbone weights must be unchanged
        for name, param in model.encoder.backbone.named_parameters():
            assert torch.allclose(param, initial_weights[name]), \
                f"Frozen encoder weight '{name}' was modified!"
