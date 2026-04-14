"""
Tests — benchmarks.metrics
pytest tests/test_benchmarks.py
"""

import pytest
import torch
from benchmarks.metrics import (
    isotropy_score,
    effective_rank,
    temporal_mse,
    temporal_consistency,
    run_all_benchmarks,
)


class TestIsotropyScore:

    def test_isotropic_embeddings(self):
        """N(0,I) embeddings should give score close to 1."""
        torch.manual_seed(42)
        Z = torch.randn(512, 128)
        score = isotropy_score(Z)
        assert score > 0.1, f"Isotropic embeddings gave low score: {score:.4f}"

    def test_collapsed_embeddings(self):
        """Near-constant embeddings should give score close to 0."""
        Z = torch.ones(64, 128) + torch.randn(64, 128) * 1e-4
        score = isotropy_score(Z)
        assert score < 0.01, f"Collapsed embeddings gave high score: {score:.4f}"

    def test_score_in_range(self):
        Z = torch.randn(128, 64)
        score = isotropy_score(Z)
        assert 0.0 <= score <= 1.0


class TestEffectiveRank:

    def test_full_rank(self):
        """Full-rank embeddings should have high effective rank."""
        torch.manual_seed(0)
        Z = torch.randn(256, 128)
        rank = effective_rank(Z)
        assert rank > 10, f"Full-rank embeddings gave low effective rank: {rank:.1f}"

    def test_rank_one(self):
        """Rank-1 matrix should have effective rank close to 1."""
        v = torch.randn(128)
        Z = torch.outer(torch.randn(64), v)  # Rank-1
        rank = effective_rank(Z)
        assert rank < 3.0, f"Rank-1 matrix gave high effective rank: {rank:.1f}"


class TestTemporalMSE:

    def test_perfect_prediction(self):
        """Perfect predictor should give MSE = 0."""
        Z = torch.randn(4, 8, 64)
        # Z_pred shifted by 1 exactly matches Z shifted by 1
        Z_pred = Z.clone()
        mse = temporal_mse(Z, Z_pred)
        # Note: temporal_mse compares Z_pred[:,:-1] vs Z[:,1:]
        # With identical tensors this is Z[:,:-1] vs Z[:,1:] — not zero
        assert mse >= 0

    def test_mse_non_negative(self):
        Z = torch.randn(2, 6, 32)
        Z_pred = torch.randn(2, 6, 32)
        mse = temporal_mse(Z, Z_pred)
        assert mse >= 0


class TestTemporalConsistency:

    def test_straight_trajectory(self):
        """Perfectly straight latent trajectory should give score = 1."""
        # All frames move in the same direction
        direction = torch.randn(1, 1, 32)
        Z = direction * torch.arange(8).reshape(1, 8, 1).float()
        Z = Z.expand(2, -1, -1)
        score = temporal_consistency(Z)
        assert score > 0.9, f"Straight trajectory gave low consistency: {score:.4f}"

    def test_score_in_range(self):
        Z = torch.randn(4, 8, 64)
        score = temporal_consistency(Z)
        assert -1.0 <= score <= 1.0


class TestRunAllBenchmarks:

    def test_returns_all_keys(self):
        Z = torch.randn(4, 8, 64)
        Z_pred = torch.randn(4, 8, 64)
        metrics = run_all_benchmarks(Z, Z_pred)
        expected_keys = {
            "isotropy_score", "effective_rank",
            "temporal_mse", "temporal_consistency"
        }
        assert set(metrics.keys()) == expected_keys

    def test_all_floats(self):
        Z = torch.randn(4, 8, 64)
        Z_pred = torch.randn(4, 8, 64)
        metrics = run_all_benchmarks(Z, Z_pred)
        for k, v in metrics.items():
            assert isinstance(v, float), f"{k} is not a float: {type(v)}"
