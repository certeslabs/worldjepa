"""
SIGReg — Sketched Isotropic Gaussian Regularizer
Based on: Balestriero & LeCun, "LeJEPA: Provable and Scalable SSL without Heuristics" (2025)
         Maes et al., "LeWorldModel" (2026)

Used in WorldJEPA to prevent representation collapse
without EMA or stop-gradient heuristics.
"""

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """
    Sketched Isotropic Gaussian Regularizer.

    Forces latent embeddings to match an isotropic Gaussian N(0, I)
    via the Cramér-Wold theorem:
    - Project embeddings onto M random unit-norm directions
    - Apply univariate Epps-Pulley normality test on each projection
    - Aggregate: matching all 1D marginals → matching full joint distribution

    Args:
        num_projections (int): Number of random directions M. Default 1024.
                               Performance insensitive to this (see LeWM paper, App. G).
        num_knots (int): Integration knots for Epps-Pulley. Default 17.
                         Also insensitive — λ is the only real hyperparameter.
    """

    def __init__(self, num_projections: int = 1024, num_knots: int = 17):
        super().__init__()
        self.num_projections = num_projections
        self.num_knots = num_knots

        # Fixed integration knots for Epps-Pulley statistic
        # Uniform in [0.2, 4.0] as in LeWM paper
        knots = torch.linspace(0.2, 4.0, num_knots)
        self.register_buffer("knots", knots)

    def epps_pulley(self, h: torch.Tensor) -> torch.Tensor:
        """
        Univariate Epps-Pulley test statistic.

        Args:
            h: (N,) — 1D projection of embeddings

        Returns:
            scalar — normality test statistic (lower = more Gaussian)
        """
        N = h.shape[0]
        t = self.knots  # (K,)

        # Empirical characteristic function: ECF(t) = mean(exp(i*t*h))
        # = mean(cos(t*h)) since we're comparing to N(0,1)
        th = t.unsqueeze(0) * h.unsqueeze(1)  # (N, K)
        ecf_real = torch.cos(th).mean(dim=0)  # (K,)
        ecf_imag = torch.sin(th).mean(dim=0)  # (K,)

        # Target: characteristic function of N(0,1) = exp(-t²/2)
        target = torch.exp(-0.5 * t ** 2)  # (K,)

        # Weight function w(t) = exp(-t²/(2λ²)), λ=1 standard
        weight = torch.exp(-0.5 * t ** 2)  # (K,)

        # Epps-Pulley statistic: ∫ w(t)|ECF(t) - φ₀(t)|² dt
        # Approximated via trapezoid rule on knots
        integrand = weight * ((ecf_real - target) ** 2 + ecf_imag ** 2)
        dt = (self.knots[-1] - self.knots[0]) / (self.num_knots - 1)
        stat = integrand.sum() * dt

        return stat

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Compute SIGReg on a batch of embeddings.

        Args:
            Z: (N, D) — batch of latent embeddings
                        where N = batch_size × history_length

        Returns:
            scalar — SIGReg loss value
        """
        N, D = Z.shape
        device = Z.device

        # Sample M random unit-norm projection directions u ~ Uniform(S^{D-1})
        U = torch.randn(D, self.num_projections, device=device)
        U = U / U.norm(dim=0, keepdim=True)  # (D, M) — unit norm columns

        # Project embeddings: H = Z @ U → (N, M)
        H = Z @ U  # (N, M)

        # Compute Epps-Pulley for each projection
        stats = []
        for m in range(self.num_projections):
            h_m = H[:, m]  # (N,)
            # Normalize to zero mean, unit variance before test
            h_m = (h_m - h_m.mean()) / (h_m.std() + 1e-8)
            stats.append(self.epps_pulley(h_m))

        sigreg_loss = torch.stack(stats).mean()
        return sigreg_loss


# ─── Fast approximation for large-scale training ───────────────────────────

class SIGRegFast(nn.Module):
    """
    Faster SIGReg using vectorized operations.
    Trades slight numerical precision for speed — recommended for v0.1 training.
    """

    def __init__(self, num_projections: int = 1024, num_knots: int = 17):
        super().__init__()
        self.M = num_projections
        knots = torch.linspace(0.2, 4.0, num_knots)
        self.register_buffer("knots", knots)
        self.register_buffer("target", torch.exp(-0.5 * knots ** 2))
        self.register_buffer("weight", torch.exp(-0.5 * knots ** 2))

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Fully vectorized SIGReg computation.

        Args:
            Z: (N, D)

        Returns:
            scalar
        """
        N, D = Z.shape
        device = Z.device

        # Random projections
        U = torch.randn(D, self.M, device=device, dtype=Z.dtype)
        U = U / U.norm(dim=0, keepdim=True)  # (D, M)

        # Project: (N, M)
        H = Z @ U

        # Normalize each projection
        mu = H.mean(dim=0, keepdim=True)    # (1, M)
        std = H.std(dim=0, keepdim=True)    # (1, M)
        H = (H - mu) / (std + 1e-8)         # (N, M) — normalized

        # Vectorized ECF over all projections and all knots simultaneously
        # t: (K,), H: (N, M) → th: (N, M, K)
        t = self.knots  # (K,)
        th = H.unsqueeze(-1) * t.unsqueeze(0).unsqueeze(0)  # (N, M, K)

        ecf_real = torch.cos(th).mean(dim=0)  # (M, K)
        ecf_imag = torch.sin(th).mean(dim=0)  # (M, K)

        # Target and weight: (K,) → broadcast over M
        target = self.target.unsqueeze(0)  # (1, K)
        weight = self.weight.unsqueeze(0)  # (1, K)

        # Integrand: (M, K)
        integrand = weight * ((ecf_real - target) ** 2 + ecf_imag ** 2)

        # Trapezoid integration over knots
        dt = (self.knots[-1] - self.knots[0]) / (len(self.knots) - 1)
        stats = integrand.sum(dim=-1) * dt  # (M,)

        return stats.mean()
