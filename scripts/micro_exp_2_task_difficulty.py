"""Phase 0.5 D0.0 — Micro-exp 2: task difficulty via adjacent cosine."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from micro_exp_common import (
    build_arg_parser,
    build_loader,
    collect_encoder_outputs,
    infer_layout,
    pick_device,
    save_json,
)


@torch.no_grad()
def cosine_adjacent(z_per_frame: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(z_per_frame[:, :-1, :], z_per_frame[:, 1:, :], dim=-1)
        .mean()
        .item()
    )


@torch.no_grad()
def cosine_random(z_per_frame: torch.Tensor, n_pairs: int = 200) -> float:
    n = z_per_frame.shape[0]
    n_pairs = min(n_pairs, n)
    idx1 = torch.randperm(n)[:n_pairs]
    idx2 = torch.randperm(n)[:n_pairs]
    return float(
        F.cosine_similarity(z_per_frame[idx1, 0, :], z_per_frame[idx2, 0, :], dim=-1)
        .mean()
        .item()
    )


def main() -> None:
    parser = build_arg_parser("Micro-exp 2: task difficulty")
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"[micro-exp-2] device={device}")

    encoder, _ = torch.hub.load(
        "facebookresearch/vjepa2", args.encoder_hub_name, trust_repo=True
    )
    encoder = encoder.to(device).eval()

    loader = build_loader(args)
    out = collect_encoder_outputs(encoder, loader, device)

    if out.ndim != 3:
        raise ValueError(f"Expected (B, N, D), got {tuple(out.shape)}")

    bsz, n_tokens_total, dim = out.shape
    t_eff, n_spatial = infer_layout(n_tokens_total, args.num_frames, args.resolution)
    z_per_frame = out.reshape(bsz, t_eff, n_spatial, dim).mean(dim=2)

    cos_adj = cosine_adjacent(z_per_frame)
    cos_rand = cosine_random(z_per_frame)
    ratio = float((1.0 - cos_adj) / max(1e-8, (1.0 - cos_rand)))

    print(f"[micro-exp-2] output shape: {tuple(out.shape)}")
    print(f"[micro-exp-2] z_per_frame shape: {tuple(z_per_frame.shape)}")
    print(f"[micro-exp-2] cos_sim(Z_t, Z_t+1) = {cos_adj:.6f}")
    print(f"[micro-exp-2] cos_sim(random pairs) = {cos_rand:.6f}")
    print(f"[micro-exp-2] task difficulty ratio = {ratio:.6f}")

    payload = {
        "output_shape": [bsz, n_tokens_total, dim],
        "z_per_frame_shape": list(z_per_frame.shape),
        "t_eff": int(t_eff),
        "n_spatial": int(n_spatial),
        "cos_adjacent": cos_adj,
        "cos_random": cos_rand,
        "task_difficulty_ratio": ratio,
    }
    out_path = save_json(args.output_dir, "micro_exp_2_task_difficulty.json", payload)
    print(f"[micro-exp-2] wrote {out_path}")


if __name__ == "__main__":
    main()
