"""
WorldJEPA v0.1 — Training Script
Optimized for Apple M4 Pro (MPS backend) and single GPU

Usage:
    # Test pipeline with mock data (no downloads needed):
    python scripts/train.py --mock --epochs 2 --batch_size 4

    # Full training on SSv2:
    python scripts/train.py --data_dir /path/to/ssv2 --epochs 50

    # Cloud (A100):
    python scripts/train.py --data_dir /path/to/ssv2 --epochs 100 --batch_size 32
"""

import argparse
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from worldjepa.model import WorldJEPA


# ─── Device selection ─────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Auto-select best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] CUDA — {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] Apple MPS (Metal) — M4 Pro")
    else:
        device = torch.device("cpu")
        print("[Device] CPU only")
    return device


# ─── Mock dataloader ──────────────────────────────────────────────────────

class MockVideoDataset(torch.utils.data.Dataset):
    """
    Synthetic video dataset for testing the pipeline.
    No downloads required. Generates random tensors.
    """

    def __init__(self, num_samples=256, T=16, H=224, W=224):
        self.num_samples = num_samples
        self.T = T
        self.H = H
        self.W = W

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random video clip (T, C, H, W) normalized to [0, 1]
        video = torch.rand(self.T, 3, self.H, self.W)
        return {"video": video}


# ─── Logging ──────────────────────────────────────────────────────────────

class Logger:
    def __init__(self, log_every: int = 10):
        self.log_every = log_every
        self.step = 0
        self.start_time = time.time()
        self.history = {
            "loss": [], "loss_pred": [],
            "loss_sigreg": [], "isotropy": []
        }

    def log(self, metrics: dict):
        self.step += 1
        for k, v in metrics.items():
            if k in self.history:
                val = v.item() if torch.is_tensor(v) else v
                self.history[k].append(val)

        if self.step % self.log_every == 0:
            elapsed = time.time() - self.start_time
            print(
                f"  step {self.step:5d} | "
                f"loss {metrics['loss']:.4f} | "
                f"pred {metrics['loss_pred']:.4f} | "
                f"sig {metrics['loss_sigreg']:.4f} | "
                f"iso {metrics['isotropy']:.4f} | "
                f"{elapsed:.0f}s"
            )

    def summary(self, epoch: int):
        """Print epoch summary."""
        if not self.history["loss"]:
            return
        recent = -min(100, len(self.history["loss"]))
        print(
            f"\n  ── Epoch {epoch} Summary ──────────────────────\n"
            f"  loss      : {sum(self.history['loss'][recent:]) / abs(recent):.4f}\n"
            f"  isotropy  : {sum(self.history['isotropy'][recent:]) / abs(recent):.4f}"
            f"  (target > 0.1)\n"
        )


# ─── Trainer ──────────────────────────────────────────────────────────────

def train(args):
    print("\n╔══════════════════════════════════════════╗")
    print("║  WorldJEPA v0.1 — CertesLabs             ║")
    print("╚══════════════════════════════════════════╝\n")

    device = get_device()

    # ── Model ──────────────────────────────────────────────────────────────
    print("[Model] Initializing WorldJEPA...")
    model = WorldJEPA(
        latent_dim=args.latent_dim,
        predictor_hidden=args.predictor_hidden,
        predictor_layers=args.predictor_layers,
        predictor_heads=args.predictor_heads,
        predictor_dropout=0.1,          # Fixed: optimal from LeWM paper
        lambda_sigreg=args.lambda_sigreg,
        num_projections=args.num_projections,
        freeze_encoder=args.freeze_encoder,
        max_seq_len=args.num_frames,
    )

    # Load encoder
    if args.mock:
        model.encoder.load_mock_encoder(device=device)
    else:
        model.encoder.load_vjepa2_encoder(args.encoder_model)

    model = model.to(device)

    # Parameter counts
    total = model.num_parameters(trainable_only=False)
    trainable = model.num_parameters(trainable_only=True)
    print(f"[Model] Total params: {total:,} | Trainable: {trainable:,}")

    # ── Data ───────────────────────────────────────────────────────────────
    if args.mock:
        print(f"[Data] Mock dataset ({args.mock_samples} samples)")
        dataset = MockVideoDataset(
            num_samples=args.mock_samples,
            T=args.num_frames,
            H=args.resolution,
            W=args.resolution,
        )
    else:
        raise NotImplementedError(
            "Real SSv2 dataloader coming in next step. "
            "Use --mock for now."
        )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,          # 0 for MPS compatibility
        pin_memory=False,       # False for MPS
        drop_last=True,
    )

    # ── Optimizer ──────────────────────────────────────────────────────────
    # Only optimize trainable parameters (predictor + projector)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Warmup + cosine decay
    total_steps = args.epochs * len(loader)
    warmup_steps = min(1000, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(3.14159 * progress)).item())

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP — CUDA only (MPS doesn't support float16 AMP well yet)
    use_amp = device.type == "cuda"
    scaler = GradScaler() if use_amp else None

    # ── Training loop ──────────────────────────────────────────────────────
    logger = Logger(log_every=args.log_every)
    model.train()

    print(f"\n[Train] Starting — {args.epochs} epochs, "
          f"batch={args.batch_size}, lr={args.lr}\n")

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")

        for batch in loader:
            video = batch["video"].to(device)  # (B, T, C, H, W)

            optimizer.zero_grad()

            if use_amp:
                with autocast():
                    out = model(video)
                loss = out["loss"]
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(video)
                loss = out["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            logger.log(out)

        logger.summary(epoch)

        # Save checkpoint
        if args.save_dir and epoch % args.save_every == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            ckpt_path = os.path.join(args.save_dir, f"worldjepa_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "args": vars(args),
                "history": logger.history,
            }, ckpt_path)
            print(f"[Checkpoint] Saved → {ckpt_path}")

    print("\n[Done] Training complete.")
    print(f"  Final isotropy: {logger.history['isotropy'][-1]:.4f}")
    print(f"  Final loss:     {logger.history['loss'][-1]:.4f}")


# ─── CLI ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="WorldJEPA v0.1 Training")

    # Data
    p.add_argument("--mock", action="store_true",
                   help="Use mock data (no downloads required)")
    p.add_argument("--mock_samples", type=int, default=256)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--resolution", type=int, default=224)

    # Model
    p.add_argument("--encoder_model", type=str, default="vjepa2_vit_large",
                   choices=["vjepa2_vit_large", "vjepa2_vit_huge", "vjepa2_vit_giant"])
    p.add_argument("--freeze_encoder", action="store_true", default=True,
                   help="Freeze encoder (recommended for M4 Pro)")
    p.add_argument("--latent_dim", type=int, default=1024)
    p.add_argument("--predictor_hidden", type=int, default=384)
    p.add_argument("--predictor_layers", type=int, default=12)
    p.add_argument("--predictor_heads", type=int, default=6)

    # SIGReg
    p.add_argument("--lambda_sigreg", type=float, default=0.1,
                   help="SIGReg weight λ (stable in [0.01, 0.2])")
    p.add_argument("--num_projections", type=int, default=1024)

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8,
                   help="Recommended: 8 for M4 Pro, 32 for A100")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)

    # Logging & saving
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--save_every", type=int, default=10)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
