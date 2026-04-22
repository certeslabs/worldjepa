"""
WorldJEPA v0.5 — Training Script
Phase 0.5: V-JEPA 2.1 + SIGReg (lejepa) + CopyCheatingLogger

Usage:
    # Smoke test avec mock data (Mac M4 Pro)
    python scripts/train.py --mock --epochs 2 --batch_size 4 --log_every 1

    # Entraînement réel SSv2 — Mac M4 Pro (run court)
    python scripts/train.py \
        --data_dir ~/Data/ssv2 \
        --max_train_samples 500 \
        --batch_size 8 \
        --epochs 3 \
        --log_every 5 \
        --seed 0

    # Run A — baseline (Kaggle T4 / RunPod A100)
    python scripts/train.py \
        --data_dir /path/to/ssv2 \
        --max_train_samples 20000 \
        --batch_size 32 \
        --epochs 3 \
        --lambda_sigreg 0.0 \
        --seed 0 \
        --save_dir checkpoints/run_A_seed0

    # Run B — SIGReg (Kaggle T4 / RunPod A100)
    python scripts/train.py \
        --data_dir /path/to/ssv2 \
        --max_train_samples 20000 \
        --batch_size 32 \
        --epochs 3 \
        --lambda_sigreg 0.1 \
        --seed 0 \
        --save_dir checkpoints/run_B_seed0

References:
    LeWM       arXiv:2603.19312  — SIGReg justification
    LeJEPA     arXiv:2511.08544  — SIGReg origin
    V-JEPA 2.1 arXiv:2603.14482  — backbone
    JEPA-WMs   arXiv:2512.24497  — predictor_hidden = latent_dim
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim

from benchmarks.copy_cheating import CopyCheatingLogger
from benchmarks.metrics import (
    effective_rank,
    isotropy_score,
    run_all_benchmarks,
    print_benchmark_report,
)
from worldjepa.model import WorldJEPA
from worldjepa.utils.seeding import set_all_seeds, seed_worker


# ─── Device ──────────────────────────────────────────────────────────────────


def get_device() -> torch.device:
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"[Device] CUDA — {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        d = torch.device("mps")
        print("[Device] Apple MPS — M4 Pro")
    else:
        d = torch.device("cpu")
        print("[Device] CPU")
    return d


# ─── Mock dataset ─────────────────────────────────────────────────────────────


class MockVideoDataset(torch.utils.data.Dataset):
    """Synthetic dataset for smoke tests — no SSv2 required."""

    def __init__(self, n: int = 256, T: int = 8, H: int = 384, W: int = 384):
        self.n = n
        self.T = T
        self.H = H
        self.W = W

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        return {"video": torch.rand(self.T, 3, self.H, self.W)}


# ─── Logger ───────────────────────────────────────────────────────────────────


class EpochLogger:
    """Tracks per-step metrics and prints epoch summaries."""

    def __init__(self, log_every: int = 10):
        self.log_every = log_every
        self.step = 0
        self.t0 = time.time()
        self.history: dict[str, list] = {
            "loss": [],
            "loss_pred": [],
            "loss_sigreg": [],
            "isotropy": [],
        }

    def log(self, m: dict) -> None:
        self.step += 1
        for k in self.history:
            if k in m:
                v = m[k]
                self.history[k].append(v.item() if torch.is_tensor(v) else v)

        if self.step % self.log_every == 0:
            iso = m.get("isotropy", 0)
            iso = iso.item() if torch.is_tensor(iso) else iso
            flag = "OK" if iso > 0.1 else "low"
            print(
                f"  step {self.step:5d} | loss {m['loss']:.4f} | "
                f"pred {m['loss_pred']:.4f} | sig {m['loss_sigreg']:.4f} | "
                f"iso {iso:.4f} [{flag}] | {time.time() - self.t0:.0f}s"
            )

    def epoch_summary(self, epoch: int, val: dict | None = None) -> None:
        n = min(50, len(self.history["loss"]))
        if n == 0:
            return
        avg_loss = sum(self.history["loss"][-n:]) / n
        avg_iso = sum(self.history["isotropy"][-n:]) / n
        status = "healthy" if avg_iso > 0.1 else "low"
        print(
            f"Epoch {epoch} — loss: {avg_loss:.4f} | isotropy: {avg_iso:.4f} [{status}]"
        )
        if val:
            print_benchmark_report(val, "Val")
            if "iso_pred" in val:
                print(f"  iso_pred  (Z_pred) : {val['iso_pred']:.4f}  <- SIGReg target")
                print(f"  rank_pred (Z_pred) : {val['rank_pred']:.1f}")


# ─── Validation ───────────────────────────────────────────────────────────────


@torch.no_grad()
def validate(
    model: WorldJEPA,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> tuple[dict, CopyCheatingLogger] | None:
    """
    Run validation loop.

    Returns (metrics_dict, cheat_logger) or None if no batches.

    metrics_dict includes:
        - standard benchmarks (isotropy, effective_rank, temporal_mse, temporal_consistency)
        - iso_pred, rank_pred on Z_pred
        - copy cheating metrics (cos_pred_now, cos_pred_next, copy_gap, r1_hard, sim_t_t1/t2)
    """
    model.eval()
    cheat_logger = CopyCheatingLogger()
    Zs: list[torch.Tensor] = []
    Zps: list[torch.Tensor] = []

    for i, batch in enumerate(loader):
        if i >= max_batches or batch is None:
            break
        video = batch["video"].to(device)
        out = model(video)

        Z = out["Z"]
        Zp = out["Z_pred"]

        Zs.append(Z.cpu())
        Zps.append(Zp.cpu())

        # Anti-triviality: update per batch (cpu to avoid OOM on large val sets)
        cheat_logger.update(Zp.detach().cpu(), Z.detach().cpu())

    model.train()

    if not Zs:
        return None

    Z = torch.cat(Zs)  # (N, T, D)
    Zp = torch.cat(Zps)  # (N, T, D)

    # ── Standard benchmarks ───────────────────────────────────────────────────
    metrics = run_all_benchmarks(Z, Zp)

    # ── Z_pred geometry (SIGReg target) ───────────────────────────────────────
    # Flatten, centre, and measure isotropy on Z_pred
    Zp_flat = Zp.reshape(-1, Zp.shape[-1]).float()
    Zp_flat = Zp_flat - Zp_flat.mean(dim=0, keepdim=True)
    Zp_flat = Zp_flat / (Zp_flat.std(dim=0, keepdim=True) + 1e-6)

    metrics["iso_pred"] = isotropy_score(Zp_flat)
    metrics["rank_pred"] = effective_rank(Zp_flat)

    # ── Copy cheating metrics ─────────────────────────────────────────────────
    cheat_metrics = cheat_logger.summarize()
    metrics.update(cheat_metrics)

    return metrics, cheat_logger


# ─── Main training loop ───────────────────────────────────────────────────────


def train(args: argparse.Namespace) -> None:
    print("WorldJEPA v0.5 — CertesLabs")

    # ── Seeding (Fix 5 — reproducibility) ────────────────────────────────────
    set_all_seeds(args.seed)
    print(f"[Seed] {args.seed}")

    device = get_device()

    # ── Model ─────────────────────────────────────────────────────────────────
    # Phase 0.5 defaults:
    #   latent_dim=768    (V-JEPA 2.1 ViT-B embed_dim)
    #   predictor_hidden  must equal latent_dim (JEPA-WMs §4)
    #   BatchNorm kept in projectors (LeWM §3.1)
    #   SIGReg via lejepa (official library)
    model = WorldJEPA(
        latent_dim=args.latent_dim,
        predictor_layers=args.predictor_layers,
        predictor_heads=args.predictor_heads,
        predictor_dropout=args.predictor_dropout,
        lambda_sigreg=args.lambda_sigreg,
        freeze_encoder=args.freeze_encoder,
        max_seq_len=args.num_frames,
    )
    # sigreg_reduction is passed via model attribute if supported
    if hasattr(model, "sigreg_reduction"):
        model.sigreg_reduction = args.sigreg_reduction

    if args.mock:
        model.encoder.load_mock_encoder(device=str(device))
        print("[WorldJEPA] Mock encoder loaded (no V-JEPA 2.1 weights)")
    else:
        model.encoder.load_vjepa2_encoder(args.encoder_hub_name)
        print(
            f"[WorldJEPA] V-JEPA 2.1 loaded — encoder_hub_name={args.encoder_hub_name}"
        )

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable: {n_params:,}")

    # ── Data ──────────────────────────────────────────────────────────────────
    if args.mock:
        g = torch.Generator()
        g.manual_seed(args.seed)

        train_dataset = MockVideoDataset(
            args.mock_samples, args.num_frames, args.resolution, args.resolution
        )
        val_dataset = MockVideoDataset(
            64, args.num_frames, args.resolution, args.resolution
        )

        tr = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            generator=g,
        )
        vl = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=0,
        )
    else:
        from data.ssv2_dataset import build_ssv2_loaders

        tr, vl = build_ssv2_loaders(
            data_root=args.data_dir,
            num_frames=args.num_frames,
            resolution=args.resolution,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            seed=args.seed,
        )

    print(
        f"[SSv2] train: {len(tr.dataset):,} videos | {args.num_frames} frames @ {args.resolution}px"
    )
    print(
        f"[SSv2] validation: {len(vl.dataset):,} videos | {args.num_frames} frames @ {args.resolution}px"
    )
    print(f"[Data] {len(tr)} train batches | {len(vl)} val batches")

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    total_steps = args.epochs * len(tr)
    warmup_steps = min(500, total_steps // 10)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training ──────────────────────────────────────────────────────────────
    epoch_logger = EpochLogger(args.log_every)
    model.train()

    print(
        f"[Train] {args.epochs} epochs | batch={args.batch_size} | "
        f"λ_sigreg={args.lambda_sigreg} | sigreg_reduction={args.sigreg_reduction}"
    )

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")

        for batch in tr:
            if batch is None:
                continue

            video = batch["video"].to(device)
            optimizer.zero_grad()
            out = model(video)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_logger.log(out)

        # ── Validation ────────────────────────────────────────────────────────
        val_result = validate(model, vl, device, max_batches=args.val_batches)

        if val_result is not None:
            val_metrics, cheat_logger = val_result

            epoch_logger.epoch_summary(epoch, val_metrics)

            # ── Anti-triviality verdict (no cross-run comparison in train loop)
            print(f"\n  ── Anti-triviality ──────────────────────────────")
            print(f"  cos_pred_now  : {val_metrics.get('cos_pred_now', 'N/A'):.4f}")
            print(f"  cos_pred_next : {val_metrics.get('cos_pred_next', 'N/A'):.4f}")
            print(f"  copy_gap      : {val_metrics.get('copy_gap', 'N/A'):.4f}")
            print(f"  R@1_hard      : {val_metrics.get('r1_hard', 'N/A'):.4f}")
            print(f"  sim_t_t1      : {val_metrics.get('sim_t_t1', 'N/A'):.4f}")
            print(f"  verdict       : {cheat_logger.verdict()}")
            print()

            # ── Checkpoint ────────────────────────────────────────────────────
            if args.save_dir and epoch % args.save_every == 0:
                os.makedirs(args.save_dir, exist_ok=True)
                ckpt_path = os.path.join(
                    args.save_dir, f"worldjepa_epoch{epoch:03d}.pt"
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "seed": args.seed,
                        "lambda_sigreg": args.lambda_sigreg,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "history": epoch_logger.history,
                        "val_metrics": val_metrics,
                    },
                    ckpt_path,
                )
                print(f"[Checkpoint] {ckpt_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    if epoch_logger.history["isotropy"]:
        print(f"\nDone. Final isotropy: {epoch_logger.history['isotropy'][-1]:.4f}")
    else:
        print("\nDone.")


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WorldJEPA Phase 0.5 training")

    # ── Data ──────────────────────────────────────────────────────────────────
    p.add_argument(
        "--mock", action="store_true", help="Use synthetic mock data (no SSv2 required)"
    )
    p.add_argument(
        "--mock_samples",
        type=int,
        default=256,
        help="Number of synthetic videos for mock mode",
    )
    p.add_argument(
        "--data_dir", default="~/Data/ssv2", help="Path to SSv2 dataset root"
    )
    p.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Limit training set size (None = full dataset)",
    )
    p.add_argument(
        "--max_val_samples", type=int, default=500, help="Limit validation set size"
    )

    # ── Video ─────────────────────────────────────────────────────────────────
    p.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help="Frames per clip (Phase 0.5 default: 8 → T_eff=4)",
    )
    p.add_argument(
        "--resolution",
        type=int,
        default=384,
        help="Spatial resolution (V-JEPA 2.1 requires 384)",
    )

    # ── Encoder ───────────────────────────────────────────────────────────────
    p.add_argument(
        "--encoder_hub_name",
        default="vjepa2_1_vit_base_384",
        choices=["vjepa2_1_vit_base_384", "vjepa2_1_vit_large_384"],
        help="V-JEPA 2.1 variant (ViT-B=768, ViT-L=1024)",
    )
    p.add_argument(
        "--freeze_encoder",
        action="store_true",
        default=True,
        help="Freeze V-JEPA 2.1 encoder (Phase 0.5 default)",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    # Phase 0.5: latent_dim must match encoder embed_dim
    # ViT-B → 768, ViT-L → 1024
    p.add_argument(
        "--latent_dim",
        type=int,
        default=768,
        help="Latent dimension (must match encoder: 768=ViT-B, 1024=ViT-L)",
    )
    p.add_argument(
        "--predictor_layers",
        type=int,
        default=6,
        help="Number of predictor transformer layers (6 for ViT-B)",
    )
    p.add_argument(
        "--predictor_heads",
        type=int,
        default=6,
        help="Number of attention heads (must divide latent_dim)",
    )
    p.add_argument(
        "--predictor_dropout",
        type=float,
        default=0.1,
        help="Dropout in predictor (LeWM optimal: 0.1)",
    )

    # ── SIGReg ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--lambda_sigreg",
        type=float,
        default=0.1,
        help="SIGReg weight (0.0=baseline, 0.1=LeWM default)",
    )
    p.add_argument(
        "--sigreg_reduction",
        default="flatten",
        choices=["per_timestep", "flatten"],
        help="per_timestep=LeWM-style (needs B≥32), flatten=T4 fallback",
    )

    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs (Phase 0.5: 3)"
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (32 on T4, 64 on A100 for SIGReg per_timestep)",
    )
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate (predictor)")
    p.add_argument(
        "--weight_decay", type=float, default=1e-5, help="AdamW weight decay"
    )
    p.add_argument(
        "--num_workers", type=int, default=2, help="DataLoader worker threads"
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (Phase 0.5: run with seeds 0-4)",
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    p.add_argument("--log_every", type=int, default=50, help="Log every N steps")
    p.add_argument(
        "--val_batches",
        type=int,
        default=20,
        help="Number of validation batches per epoch",
    )

    # ── Checkpointing ─────────────────────────────────────────────────────────
    p.add_argument(
        "--save_dir", default="checkpoints", help="Directory for model checkpoints"
    )
    p.add_argument(
        "--save_every", type=int, default=1, help="Save checkpoint every N epochs"
    )

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
