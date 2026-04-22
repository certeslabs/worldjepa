"""
WorldJEPA v0.5 — Training Script
Phase 0.5: V-JEPA 2.1 + SIGReg (lejepa) + CopyCheatingLogger

Usage:
    # Smoke test
    python scripts/train.py --mock --epochs 2 --batch_size 4 --log_every 1

    # SSv2 court (Mac)
    python scripts/train.py \
        --data_dir ~/Data/ssv2 \
        --max_train_samples 500 \
        --batch_size 8 --epochs 1 --log_every 5 --seed 0

    # Run A — baseline (Kaggle/RunPod)
    python scripts/train.py \
        --data_dir /path/to/ssv2 \
        --max_train_samples 20000 \
        --batch_size 32 --epochs 3 \
        --lambda_sigreg 0.0 --seed 0 \
        --save_dir checkpoints/run_A_seed0

    # Run B — SIGReg (Kaggle/RunPod)
    python scripts/train.py \
        --data_dir /path/to/ssv2 \
        --max_train_samples 20000 \
        --batch_size 32 --epochs 3 \
        --lambda_sigreg 0.1 --seed 0 \
        --save_dir checkpoints/run_B_seed0
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
from worldjepa.utils.seeding import set_all_seeds


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
    def __init__(self, n=256, T=8, H=384, W=384):
        self.n, self.T, self.H, self.W = n, T, H, W

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"video": torch.rand(self.T, 3, self.H, self.W)}


# ─── Logger ───────────────────────────────────────────────────────────────────


class EpochLogger:
    def __init__(self, log_every=10):
        self.log_every = log_every
        self.step = 0
        self.t0 = time.time()
        self.history = {"loss": [], "loss_pred": [], "loss_sigreg": [], "isotropy": []}

    def log(self, m):
        self.step += 1
        # Normalize key: loss_mse (Phase 0.5 model) → loss_pred (logger)
        if "loss_mse" in m and "loss_pred" not in m:
            m = dict(m)
            m["loss_pred"] = m["loss_mse"]
        for k in self.history:
            if k in m:
                v = m[k]
                self.history[k].append(v.item() if torch.is_tensor(v) else v)
        if self.step % self.log_every == 0:
            iso = m.get("isotropy", 0)
            iso = iso.item() if torch.is_tensor(iso) else iso
            flag = "OK" if iso > 0.1 else "low"
            loss_pred = m.get("loss_pred", m.get("loss_mse", 0))
            if torch.is_tensor(loss_pred):
                loss_pred = loss_pred.item()
            print(
                f"  step {self.step:5d} | loss {m['loss']:.4f} | "
                f"pred {loss_pred:.4f} | sig {m['loss_sigreg']:.4f} | "
                f"iso {iso:.4f} [{flag}] | {time.time()-self.t0:.0f}s"
            )

    def epoch_summary(self, epoch, val=None):
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
def validate(model, loader, device, max_batches=20):
    model.eval()
    cheat_logger = CopyCheatingLogger()
    Zs, Zps = [], []

    for i, batch in enumerate(loader):
        if i >= max_batches or batch is None:
            break
        video = batch["video"].to(device)
        out = model(video)
        Z, Zp = out["Z"], out["Z_pred"]
        Zs.append(Z.cpu())
        Zps.append(Zp.cpu())
        cheat_logger.update(Zp.detach().cpu(), Z.detach().cpu())

    model.train()
    if not Zs:
        return None

    Z = torch.cat(Zs)
    Zp = torch.cat(Zps)

    metrics = run_all_benchmarks(Z, Zp)

    Zp_flat = Zp.reshape(-1, Zp.shape[-1]).float()
    Zp_flat = Zp_flat - Zp_flat.mean(dim=0, keepdim=True)
    Zp_flat = Zp_flat / (Zp_flat.std(dim=0, keepdim=True) + 1e-6)
    metrics["iso_pred"] = isotropy_score(Zp_flat)
    metrics["rank_pred"] = effective_rank(Zp_flat)
    metrics.update(cheat_logger.summarize())

    return metrics, cheat_logger


# ─── Training ─────────────────────────────────────────────────────────────────


def train(args):
    print("WorldJEPA v0.5 — CertesLabs")
    set_all_seeds(args.seed)
    print(f"[Seed] {args.seed}")
    device = get_device()

    # ── Model ─────────────────────────────────────────────────────────────────
    # Phase 0.5 WorldJEPA signature:
    #   encoder_variant, freeze_encoder, feature_mode,
    #   lambda_sigreg, sigreg_reduction,
    #   predictor_layers, predictor_heads, predictor_dropout
    model = WorldJEPA(
        encoder_variant=args.encoder_variant,
        freeze_encoder=True,
        feature_mode=args.feature_mode,
        lambda_sigreg=args.lambda_sigreg,
        sigreg_reduction=args.sigreg_reduction,
        predictor_layers=args.predictor_layers,
        predictor_heads=args.predictor_heads,
        predictor_dropout=args.predictor_dropout,
    )

    if args.mock:
        # Load mock encoder to skip V-JEPA 2.1 weight download
        if hasattr(model, "encoder") and hasattr(model.encoder, "load_mock_encoder"):
            model.encoder.load_mock_encoder(device=str(device))
            print("[WorldJEPA] Mock encoder loaded (no V-JEPA 2.1 weights)")
        else:
            print(
                "[WorldJEPA] Warning: load_mock_encoder not found, using real encoder"
            )
    else:
        print(f"[WorldJEPA] V-JEPA 2.1 loaded — variant={args.encoder_variant}")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable: {n_params:,}")

    # ── Data ──────────────────────────────────────────────────────────────────
    if args.mock:
        g = torch.Generator()
        g.manual_seed(args.seed)
        tr = torch.utils.data.DataLoader(
            MockVideoDataset(
                args.mock_samples, args.num_frames, args.resolution, args.resolution
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            generator=g,
        )
        vl = torch.utils.data.DataLoader(
            MockVideoDataset(64, args.num_frames, args.resolution, args.resolution),
            batch_size=args.batch_size,
            num_workers=0,
        )
        print(
            f"[SSv2] train: {args.mock_samples} videos | {args.num_frames} frames @ {args.resolution}px"
        )
        print(
            f"[SSv2] validation: 64 videos | {args.num_frames} frames @ {args.resolution}px"
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

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    total_steps = args.epochs * len(tr)
    warmup_steps = min(500, total_steps // 10)

    def lr_lambda(s):
        if s < warmup_steps:
            return s / max(1, warmup_steps)
        p = (s - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * p))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Loop ──────────────────────────────────────────────────────────────────
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

        val_result = validate(model, vl, device, max_batches=args.val_batches)

        if val_result is not None:
            val_metrics, cheat_logger = val_result
            epoch_logger.epoch_summary(epoch, val_metrics)

            print(f"\n  ── Anti-triviality ──────────────────────────────")
            print(f"  cos_pred_now  : {val_metrics.get('cos_pred_now', 0):.4f}")
            print(f"  cos_pred_next : {val_metrics.get('cos_pred_next', 0):.4f}")
            print(f"  copy_gap      : {val_metrics.get('copy_gap', 0):.4f}")
            print(f"  R@1_hard      : {val_metrics.get('r1_hard', 0):.4f}")
            print(f"  sim_t_t1      : {val_metrics.get('sim_t_t1', 0):.4f}")
            print(f"  verdict       : {cheat_logger.verdict()}")
            print()

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

    if epoch_logger.history["isotropy"]:
        print(f"\nDone. Final isotropy: {epoch_logger.history['isotropy'][-1]:.4f}")
    else:
        print("\nDone.")


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="WorldJEPA Phase 0.5 training")

    # Data
    p.add_argument("--mock", action="store_true")
    p.add_argument("--mock_samples", type=int, default=256)
    p.add_argument("--data_dir", default="~/Data/ssv2")
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_val_samples", type=int, default=500)
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--resolution", type=int, default=384)
    p.add_argument("--num_workers", type=int, default=2)

    # Encoder — matches Phase 0.5 WorldJEPA.__init__ signature
    p.add_argument("--encoder_variant", default="vitb", choices=["vitb", "vitl"])
    p.add_argument("--feature_mode", default="mean", choices=["mean", "cls"])

    # Predictor
    p.add_argument("--predictor_layers", type=int, default=None)
    p.add_argument("--predictor_heads", type=int, default=8)
    p.add_argument("--predictor_dropout", type=float, default=0.1)

    # SIGReg
    p.add_argument("--lambda_sigreg", type=float, default=0.1)
    p.add_argument(
        "--sigreg_reduction", default="flatten", choices=["per_timestep", "flatten"]
    )

    # Training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)

    # Logging
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--val_batches", type=int, default=20)

    # Checkpointing
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--save_every", type=int, default=1)

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
