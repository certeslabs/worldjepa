"""
WorldJEPA v0.1 — Training Script
Optimized for Apple M4 Pro (MPS) + CUDA

Usage:
    # Test avec mock data
    python scripts/train.py --mock --epochs 2 --batch_size 4 --log_every 1

    # Entraînement réel SSv2 — M4 Pro (run court pour tester)
    python scripts/train.py \
        --data_dir ~/Data/ssv2 \
        --max_train_samples 5000 \
        --freeze_encoder \
        --batch_size 8 \
        --epochs 5 \
        --log_every 5

    # Entraînement complet SSv2
    python scripts/train.py \
        --data_dir ~/Data/ssv2 \
        --freeze_encoder \
        --batch_size 8 \
        --epochs 50
"""

import argparse
import time
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim

from worldjepa.model import WorldJEPA
from benchmarks.metrics import run_all_benchmarks, print_benchmark_report


def get_device():
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


class MockVideoDataset(torch.utils.data.Dataset):
    def __init__(self, n=256, T=16, H=224, W=224):
        self.n = n
        self.T = T
        self.H = H
        self.W = W

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"video": torch.rand(self.T, 3, self.H, self.W)}


class Logger:
    def __init__(self, log_every=10):
        self.log_every = log_every
        self.step = 0
        self.t0 = time.time()
        self.history = {"loss": [], "loss_pred": [], "loss_sigreg": [], "isotropy": []}

    def log(self, m):
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
                f"iso {iso:.4f} [{flag}] | {time.time()-self.t0:.0f}s"
            )

    def epoch_summary(self, epoch, val=None):
        n = min(50, len(self.history["loss"]))
        if n == 0:
            return
        avg_loss = sum(self.history["loss"][-n:]) / n
        avg_iso = sum(self.history["isotropy"][-n:]) / n
        print(
            f"Epoch {epoch} — loss: {avg_loss:.4f} | isotropy: {avg_iso:.4f} "
            f"[{"healthy" if avg_iso > 0.1 else "low"  }]"
        )
        if val:
            print_benchmark_report(val, "Val")
            if "iso_pred" in val:
                print(f"  iso_pred  (Z_pred) : {val['iso_pred']:.4f}  <- SIGReg target")
                print(f"  rank_pred (Z_pred) : {val['rank_pred']:.1f}")


@torch.no_grad()
def validate(model, loader, device, max_batches=20):
    model.eval()
    Zs, Zps = [], []
    for i, batch in enumerate(loader):
        if i >= max_batches or batch is None:
            break
        out = model(batch["video"].to(device))
        Zs.append(out["Z"].cpu())
        Zps.append(out["Z_pred"].cpu())
    model.train()
    if not Zs:
        return None
    Z  = torch.cat(Zs)
    Zp = torch.cat(Zps)
    metrics = run_all_benchmarks(Z, Zp)
    from benchmarks.metrics import isotropy_score, effective_rank
    Zp_flat = Zp.reshape(-1, Zp.shape[-1])
    Zp_flat = Zp_flat - Zp_flat.mean(dim=0, keepdim=True)
    Zp_flat = Zp_flat / (Zp_flat.std(dim=0, keepdim=True) + 1e-6)
    metrics["iso_pred"]  = isotropy_score(Zp_flat)
    metrics["rank_pred"] = effective_rank(Zp_flat)
    return metrics


def train(args):
    print("WorldJEPA v0.1 — CertesLabs")
    device = get_device()

    model = WorldJEPA(
        latent_dim=args.latent_dim,
        predictor_hidden=args.predictor_hidden,
        predictor_layers=args.predictor_layers,
        predictor_heads=args.predictor_heads,
        predictor_dropout=0.1,
        lambda_sigreg=args.lambda_sigreg,
        num_projections=args.num_projections,
        freeze_encoder=args.freeze_encoder,
        max_seq_len=args.num_frames,
    )

    if args.mock:
        model.encoder.load_mock_encoder(device=str(device))
    else:
        model.encoder.load_vjepa2_encoder(args.encoder_model)

    # Unfreeze partiel si demandé
    if hasattr(args, "unfreeze_last_n_layers") and args.unfreeze_last_n_layers > 0:
        model.encoder.unfreeze_last_n_layers(args.unfreeze_last_n_layers)

    model = model.to(device)
    print(f"[Model] Trainable: {model.num_parameters():,}")

    if args.mock:
        tr = torch.utils.data.DataLoader(
            MockVideoDataset(
                args.mock_samples, args.num_frames, args.resolution, args.resolution
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        vl = torch.utils.data.DataLoader(
            MockVideoDataset(64, args.num_frames, args.resolution, args.resolution),
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
            num_workers=0,
            max_train_samples=args.max_train_samples,
            max_val_samples=500,
        )

    print(f"[Data] {len(tr)} train batches | {len(vl)} val batches")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    total_steps = args.epochs * len(tr)
    warmup = min(500, total_steps // 10)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (
            s / max(1, warmup)
            if s < warmup
            else 0.5
            * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup)))
        ),
    )

    logger = Logger(args.log_every)
    model.train()
    print(
        f"[Train] {args.epochs} epochs | batch={args.batch_size} | λ={args.lambda_sigreg}"
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
            logger.log(out)

        val = validate(model, vl, device)
        logger.epoch_summary(epoch, val)

        if args.save_dir and epoch % args.save_every == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            path = os.path.join(args.save_dir, f"worldjepa_epoch{epoch:03d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "history": logger.history,
                    "val": val,
                },
                path,
            )
            print(f"[Checkpoint] {path}")

    print(f"Done. Final isotropy: {logger.history['isotropy'][-1]:.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mock", action="store_true")
    p.add_argument("--mock_samples", type=int, default=256)
    p.add_argument("--data_dir", default="~/Data/ssv2")
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--encoder_model", default="vjepa2_vit_large")
    p.add_argument("--freeze_encoder", action="store_true", default=True)
    p.add_argument("--unfreeze_last_n_layers", type=int, default=0,
                   help="Dégèle les n dernières couches encoder (0=frozen)")
    p.add_argument("--lr_encoder", type=float, default=1e-5,
                   help="LR pour les couches encoder dégelées (10x plus petit)")
    p.add_argument("--latent_dim", type=int, default=1024)
    p.add_argument("--predictor_hidden", type=int, default=384)
    p.add_argument("--predictor_layers", type=int, default=12)
    p.add_argument("--predictor_heads", type=int, default=6)
    p.add_argument("--lambda_sigreg", type=float, default=0.1)
    p.add_argument("--num_projections", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--save_every", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
