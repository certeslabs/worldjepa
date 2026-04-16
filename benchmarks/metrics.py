"""
WorldJEPA Benchmarks — v0.1
Métriques d'évaluation pour valider la thèse SIGReg.

3 métriques core:
  1. Isotropy Score     — qualité de SIGReg (target > 0.1, idéal ~1.0)
  2. Temporal MSE       — précision de prédiction
  3. Temporal Consistency — stabilité sur N frames
"""

import torch
import torch.nn.functional as F
from typing import Dict


@torch.no_grad()
def isotropy_score(Z: torch.Tensor) -> float:
    """
    λ_min / λ_max du covariance matrix.

    Score = 1.0 → parfaitement isotropique (idéal après SIGReg)
    Score < 0.1 → collapse détecté

    Args:
        Z: (N, D) — embeddings

    Returns:
        float in [0, 1]
    """
    Z = Z.float()
    if Z.shape[0] > 4096:
        idx = torch.randperm(Z.shape[0])[:4096]
        Z = Z[idx]

    Z_c = Z - Z.mean(dim=0)
    cov = (Z_c.T @ Z_c) / (Z.shape[0] - 1)

    try:
        eigvals = torch.linalg.eigvalsh(cov).clamp(min=0)
        # Filtre les valeurs propres nulles (espace sous-dimensionné)
        # quand N < D, seules N valeurs propres sont non-nulles
        threshold = eigvals.max() * 1e-4
        nonzero = eigvals[eigvals > threshold]
        if len(nonzero) < 2:
            score = 0.0
        else:
            score = (nonzero.min() / (nonzero.max() + 1e-10)).item()
    except Exception:
        score = 0.0

    return score


@torch.no_grad()
def effective_rank(Z: torch.Tensor) -> float:
    """
    Effective rank = exp(entropy of normalized singular values).
    Measures how many dimensions are actually used.
    Higher = better (less collapsed).

    Args:
        Z: (N, D)

    Returns:
        float in [1, D]
    """
    Z = Z.float()
    if Z.shape[0] > 2048:
        idx = torch.randperm(Z.shape[0])[:2048]
        Z = Z[idx]

    Z_c = Z - Z.mean(dim=0)
    singular_vals = torch.linalg.svdvals(Z_c)
    probs = singular_vals / (singular_vals.sum() + 1e-10)
    probs = probs.clamp(min=1e-10)
    entropy = -(probs * torch.log(probs)).sum()
    return torch.exp(entropy).item()


@torch.no_grad()
def temporal_mse(Z: torch.Tensor, Z_pred: torch.Tensor) -> float:
    """
    MSE entre embeddings prédits et réels.
    Mesure la qualité de prédiction du predictor.

    Args:
        Z:      (B, T, D) — embeddings réels
        Z_pred: (B, T, D) — embeddings prédits

    Returns:
        float — MSE moyen
    """
    # On compare Z_pred[:, :-1] avec Z[:, 1:]
    target = Z[:, 1:, :]
    pred = Z_pred[:, :-1, :]
    return F.mse_loss(pred, target).item()


@torch.no_grad()
def temporal_consistency(Z: torch.Tensor) -> float:
    """
    Cosine similarity moyenne entre embeddings consécutifs.
    Mesure la régularité/smoothness des trajectoires latentes.
    Valeur haute = trajectoires droites (temporal straightening emergent).

    Args:
        Z: (B, T, D) — embeddings réels

    Returns:
        float in [-1, 1]
    """
    v = Z[:, 1:, :] - Z[:, :-1, :]         # velocities (B, T-1, D)
    v_norm = F.normalize(v, dim=-1)          # unit vectors

    # Cosine sim between consecutive velocity vectors
    cos_sim = (v_norm[:, :-1, :] * v_norm[:, 1:, :]).sum(dim=-1)  # (B, T-2)
    return cos_sim.mean().item()


@torch.no_grad()
def run_all_benchmarks(
    Z: torch.Tensor,
    Z_pred: torch.Tensor,
) -> Dict[str, float]:
    """
    Run all WorldJEPA v0.1 benchmarks.

    Args:
        Z:      (B, T, D) — encoded embeddings
        Z_pred: (B, T, D) — predicted embeddings

    Returns:
        dict of metric_name → float
    """
    Z_flat = Z.reshape(-1, Z.shape[-1])

    return {
        "isotropy_score":      isotropy_score(Z_flat),
        "effective_rank":      effective_rank(Z_flat),
        "temporal_mse":        temporal_mse(Z, Z_pred),
        "temporal_consistency": temporal_consistency(Z),
    }


def print_benchmark_report(metrics: Dict[str, float], title: str = "Benchmark"):
    """Pretty-print benchmark results."""
    print(f"\n  ── {title} ──────────────────────────────")
    print(f"  isotropy_score      : {metrics['isotropy_score']:.4f}  "
          f"{'✓ healthy' if metrics['isotropy_score'] > 0.1 else '✗ collapsed'}")
    print(f"  effective_rank      : {metrics['effective_rank']:.1f}")
    print(f"  temporal_mse        : {metrics['temporal_mse']:.4f}")
    print(f"  temporal_consistency: {metrics['temporal_consistency']:.4f}")
    print()
