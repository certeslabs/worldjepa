"""Phase 0.5 D0.0 — Micro-exp 1: CLS vs mean/max pooling diagnostics."""

from __future__ import annotations

import torch

from benchmarks.metrics import effective_rank, isotropy_score
from micro_exp_common import (
    build_arg_parser,
    build_loader,
    collect_encoder_outputs,
    pick_device,
    save_json,
)


def main() -> None:
    parser = build_arg_parser("Micro-exp 1: pooling diagnostics")
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"[micro-exp-1] device={device}")

    encoder, _ = torch.hub.load(
        "facebookresearch/vjepa2", args.encoder_hub_name, trust_repo=True
    )
    encoder = encoder.to(device).eval()

    loader = build_loader(args)
    out = collect_encoder_outputs(encoder, loader, device)

    if out.ndim != 3:
        raise ValueError(f"Expected (B, N, D) patch tokens, got {tuple(out.shape)}")

    print(f"[micro-exp-1] encoder output shape = {tuple(out.shape)}")

    z_views = {
        "cls": out[:, 0, :],
        "mean": out.mean(dim=1),
        "max": out.max(dim=1).values,
    }

    results = {
        "output_shape": list(out.shape),
        "pooling": {},
    }

    for name, z in z_views.items():
        iso = isotropy_score(z)
        rank = effective_rank(z)
        var = z.var(dim=0)
        row = {
            "isotropy": float(iso),
            "effective_rank": float(rank),
            "var_mean": float(var.mean().item()),
            "var_std": float(var.std().item()),
        }
        results["pooling"][name] = row
        print(
            f"[micro-exp-1] {name:>4}: "
            f"iso={row['isotropy']:.6f} rank={row['effective_rank']:.2f} "
            f"var_mean={row['var_mean']:.6f} var_std={row['var_std']:.6f}"
        )

    out_path = save_json(args.output_dir, "micro_exp_1_pooling.json", results)
    print(f"[micro-exp-1] wrote {out_path}")


if __name__ == "__main__":
    main()
