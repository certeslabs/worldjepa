"""Phase 0.5 D0.0 — Micro-exp 3: batch-size sensitivity of SIGReg."""

from __future__ import annotations

import statistics

import torch
import lejepa

from micro_exp_common import (
    build_arg_parser,
    build_loader,
    collect_encoder_outputs,
    infer_layout,
    pick_device,
    save_json,
)


def main() -> None:
    parser = build_arg_parser("Micro-exp 3: batch effect on SIGReg")
    parser.add_argument("--num_repeats", type=int, default=10)
    parser.add_argument("--num_slices", type=int, default=1024)
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"[micro-exp-3] device={device}")

    sigreg = lejepa.multivariate.SlicingUnivariateTest(
        univariate_test=lejepa.univariate.EppsPulley(n_points=17),
        num_slices=args.num_slices,
    ).to(device)

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

    one_timestep = z_per_frame[:, 0, :].to(device)

    batch_sizes = [16, 32, 64, 128]
    results = {
        "input_shape": list(one_timestep.shape),
        "num_repeats": int(args.num_repeats),
        "sigreg": {},
    }

    for b in batch_sizes:
        if one_timestep.shape[0] < b:
            continue

        z = one_timestep[:b]
        losses = [float(sigreg(z).item()) for _ in range(args.num_repeats)]
        mean_val = statistics.mean(losses)
        std_val = statistics.pstdev(losses)
        ratio = std_val / max(mean_val, 1e-8)

        results["sigreg"][str(b)] = {
            "mean": mean_val,
            "std": std_val,
            "std_over_mean": ratio,
            "all": losses,
        }

        print(
            f"[micro-exp-3] B={b}: "
            f"sigreg_mean={mean_val:.6f} sigreg_std={std_val:.6f} "
            f"std/mean={ratio:.6f}"
        )

    out_path = save_json(args.output_dir, "micro_exp_3_batch_effect.json", results)
    print(f"[micro-exp-3] wrote {out_path}")


if __name__ == "__main__":
    main()
