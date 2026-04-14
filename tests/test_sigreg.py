"""
Tests — worldjepa.sigreg
pytest tests/test_sigreg.py
"""

import pytest
import torch
from worldjepa.sigreg import SIGReg, SIGRegFast


# ─── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def normal_embeddings():
    """Well-distributed Gaussian embeddings — SIGReg should give low loss."""
    torch.manual_seed(42)
    return torch.randn(64, 192)


@pytest.fixture
def collapsed_embeddings():
    """Collapsed embeddings (near-constant) — SIGReg should give high loss."""
    return torch.ones(64, 192) + torch.randn(64, 192) * 0.001


# ─── SIGReg tests ─────────────────────────────────────────────────────────

class TestSIGReg:

    def test_output_is_scalar(self, normal_embeddings):
        sigreg = SIGReg(num_projections=32)
        loss = sigreg(normal_embeddings)
        assert loss.ndim == 0, "SIGReg must return a scalar"

    def test_differentiable(self, normal_embeddings):
        sigreg = SIGReg(num_projections=32)
        Z = normal_embeddings.requires_grad_(True)
        loss = sigreg(Z)
        loss.backward()
        assert Z.grad is not None, "SIGReg must be differentiable"
        assert not torch.isnan(Z.grad).any(), "Gradients must not be NaN"

    def test_collapse_detection(self, normal_embeddings, collapsed_embeddings):
        """Collapsed embeddings must have higher SIGReg loss."""
        sigreg = SIGReg(num_projections=64)
        # Use rank-1 embeddings for a clear collapse signal
        v = torch.randn(1, 192)
        rank1 = v.expand(64, -1)  # All identical rows — true collapse
        loss_normal = sigreg(normal_embeddings).item()
        loss_collapsed = sigreg(rank1).item()
        assert loss_collapsed > loss_normal * 2, (
            f"Collapsed loss ({loss_collapsed:.4f}) should be >> "
            f"normal loss ({loss_normal:.4f})"
        )

    def test_loss_non_negative(self, normal_embeddings):
        sigreg = SIGReg(num_projections=32)
        loss = sigreg(normal_embeddings)
        assert loss.item() >= 0, "SIGReg loss must be non-negative"

    def test_different_batch_sizes(self):
        sigreg = SIGReg(num_projections=32)
        for batch_size in [8, 32, 128]:
            Z = torch.randn(batch_size, 192)
            loss = sigreg(Z)
            assert loss.ndim == 0, f"Failed for batch_size={batch_size}"

    def test_different_dims(self):
        sigreg = SIGReg(num_projections=32)
        for dim in [64, 192, 512, 1024]:
            Z = torch.randn(32, dim)
            loss = sigreg(Z)
            assert not torch.isnan(loss), f"NaN for dim={dim}"


class TestSIGRegFast:
    """SIGRegFast must match SIGReg behavior (not exact values, same behavior)."""

    def test_output_is_scalar(self, normal_embeddings):
        sigreg = SIGRegFast(num_projections=64)
        loss = sigreg(normal_embeddings)
        assert loss.ndim == 0

    def test_differentiable(self, normal_embeddings):
        sigreg = SIGRegFast(num_projections=64)
        Z = normal_embeddings.requires_grad_(True)
        loss = sigreg(Z)
        loss.backward()
        assert Z.grad is not None
        assert not torch.isnan(Z.grad).any()

    def test_collapse_detection(self, normal_embeddings, collapsed_embeddings):
        sigreg = SIGRegFast(num_projections=128)
        v = torch.randn(1, 192)
        rank1 = v.expand(64, -1)
        loss_normal = sigreg(normal_embeddings).item()
        loss_collapsed = sigreg(rank1).item()
        assert loss_collapsed > loss_normal * 2, (
            f"Fast: collapsed ({loss_collapsed:.4f}) vs normal ({loss_normal:.4f})"
        )

    def test_lambda_sensitivity(self):
        """
        SIGReg should be robust across lambda values [0.01, 0.2].
        We test that the loss itself is not NaN/Inf in this range.
        """
        sigreg = SIGRegFast(num_projections=64)
        Z = torch.randn(32, 192)
        loss = sigreg(Z)
        for lam in [0.01, 0.05, 0.1, 0.2]:
            scaled = lam * loss
            assert not torch.isnan(scaled), f"NaN at lambda={lam}"
            assert not torch.isinf(scaled), f"Inf at lambda={lam}"
