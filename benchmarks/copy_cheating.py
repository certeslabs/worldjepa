import torch
import torch.nn.functional as F


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)


@torch.no_grad()
def compute_copy_metrics(Z_pred: torch.Tensor, Z: torch.Tensor):
    """
    Z_pred: (B, T-1, D)
    Z:      (B, T,   D)
    """

    B, T, D = Z.shape

    assert Z_pred.shape == (
        B,
        T - 1,
        D,
    ), f"Expected Z_pred shape {(B, T-1, D)}, got {tuple(Z_pred.shape)}"

    Z_now = Z[:, :-1]
    Z_next = Z[:, 1:]

    cos_now = cosine_sim(Z_pred, Z_now)
    cos_next = cosine_sim(Z_pred, Z_next)

    return {
        "cos_pred_now": cos_now.mean().item(),
        "cos_pred_next": cos_next.mean().item(),
        "copy_gap": (cos_next - cos_now).mean().item(),
    }


@torch.no_grad()
def recall_at_1_hard(Z_pred: torch.Tensor, Z: torch.Tensor, min_gap: int = 4):
    """
    Hard negatives: same video, temporally distant
    """

    B, T, D = Z.shape

    assert Z_pred.shape == (
        B,
        T - 1,
        D,
    ), f"Expected Z_pred shape {(B, T-1, D)}, got {tuple(Z_pred.shape)}"

    Z_pred = F.normalize(Z_pred, dim=-1)
    Z = F.normalize(Z, dim=-1)

    correct = 0
    total = 0

    for b in range(B):
        for t in range(T - 1):
            anchor = Z_pred[b, t]
            pos = Z[b, t + 1]

            valid_idx = [k for k in range(T) if abs(k - (t + 1)) >= min_gap]

            if not valid_idx:
                continue

            negs = Z[b, valid_idx]

            sim_pos = torch.matmul(anchor, pos)
            sim_negs = torch.matmul(negs, anchor)

            if sim_pos > sim_negs.max():
                correct += 1

            total += 1

    return correct / max(total, 1)


class CopyCheatingLogger:
    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {
            "cos_pred_now": [],
            "cos_pred_next": [],
            "copy_gap": [],
            "r1_hard": [],
            "sim_t_t1": [],
            "sim_t_t2": [],
        }

    @torch.no_grad()
    def update(self, Z_pred, Z):
        """
        Z_pred: (B, T, D) OR (B, T-1, D)
        Z:      (B, T, D)
        """

        B, T, D = Z.shape

        # Align Z_pred
        if Z_pred.shape[1] == T:
            Z_pred = Z_pred[:, :-1]

        assert Z_pred.shape == (
            B,
            T - 1,
            D,
        ), f"After alignment expected {(B, T-1, D)}, got {tuple(Z_pred.shape)}"

        # Copy metrics
        m = compute_copy_metrics(Z_pred, Z)
        r1 = recall_at_1_hard(Z_pred, Z)

        # Temporal similarity decay (encoder)
        Z_norm = F.normalize(Z, dim=-1)
        for gap in [1, 2]:
            if T > gap:
                sim = (Z_norm[:, :-gap] * Z_norm[:, gap:]).sum(dim=-1).mean().item()
                self.metrics[f"sim_t_t{gap}"].append(sim)

        # Store
        self.metrics["cos_pred_now"].append(m["cos_pred_now"])
        self.metrics["cos_pred_next"].append(m["cos_pred_next"])
        self.metrics["copy_gap"].append(m["copy_gap"])
        self.metrics["r1_hard"].append(r1)

    def summarize(self):
        return {
            k: float(torch.tensor(v).mean()) if len(v) > 0 else 0.0
            for k, v in self.metrics.items()
        }

    def verdict(self, baseline: dict | None = None) -> str:
        s = self.summarize()
        gap = s["copy_gap"]
        r1 = s["r1_hard"]

        # Failure
        if gap <= 0:
            return "❌ RED — copy behavior (predictor copie Z_t)"

        if gap < 0.01 and r1 < 0.02:
            return "❌ RED — signal quasi nul"

        # With baseline
        if baseline is not None:
            baseline_gap = baseline.get("copy_gap", 0.0)
            baseline_r1 = baseline.get("r1_hard", 0.0)

            delta_gap = gap - baseline_gap
            delta_r1 = r1 - baseline_r1

            if delta_gap > 0.01 and delta_r1 > 0.02:
                return (
                    f"✅ GREEN — gap={gap:.3f} (Δ+{delta_gap:.3f}), "
                    f"R@1={r1:.3f} (Δ+{delta_r1:.3f})"
                )
            elif delta_gap > 0 or delta_r1 > 0:
                return (
                    f"⚠️ YELLOW — gap={gap:.3f}, R@1={r1:.3f} "
                    f"(Δgap={delta_gap:+.3f}, ΔR@1={delta_r1:+.3f})"
                )
            else:
                return (
                    f"❌ RED — gap={gap:.3f}, R@1={r1:.3f} "
                    f"(Δgap={delta_gap:+.3f}, ΔR@1={delta_r1:+.3f})"
                )

        # Fallback
        if gap > 0.02 and r1 > 0.05:
            return f"✅ GREEN — gap={gap:.3f}, R@1={r1:.3f} (sans baseline)"

        return f"⚠️ YELLOW — gap={gap:.3f}, R@1={r1:.3f} (incertain)"
