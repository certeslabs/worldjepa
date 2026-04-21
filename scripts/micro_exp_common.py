"""Utilities for Phase 0.5 Day-1 micro-experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from data.ssv2_dataset import SSv2Dataset, collate_fn


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data_root", type=str, default="~/Data/ssv2")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_dir", type=str, default="results/phase_0_5/micro_exps"
    )
    parser.add_argument("--encoder_hub_name", type=str, default="vjepa2_1_vit_base_384")
    return parser


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_arg == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_loader(args: argparse.Namespace) -> DataLoader:
    dataset = SSv2Dataset(
        data_root=args.data_root,
        split="train",
        num_frames=args.num_frames,
        resolution=args.resolution,
        sampling="uniform",
        max_samples=args.max_samples,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=False,
        collate_fn=collate_fn,
    )


def load_encoder(hub_name: str, device: torch.device):
    encoder, _ = torch.hub.load("facebookresearch/vjepa2", hub_name, trust_repo=True)
    return encoder.to(device).eval()


@torch.no_grad()
def collect_encoder_outputs(
    encoder,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> torch.Tensor:
    outs = []
    for batch_idx, batch in enumerate(loader):
        if batch is None:
            continue
        videos = batch["video"].to(device)
        out = encoder(videos)
        if not isinstance(out, torch.Tensor):
            raise TypeError(f"Expected tensor output, got {type(out)}")
        outs.append(out.cpu())
        if max_batches is not None and batch_idx + 1 >= max_batches:
            break
    if not outs:
        raise RuntimeError("No valid batches were loaded from SSv2.")
    return torch.cat(outs, dim=0)


def infer_layout(
    n_tokens_total: int, num_frames: int, resolution: int, tubelet_size: int = 2
) -> Tuple[int, int]:
    t_eff = num_frames // tubelet_size
    patches_side = resolution // 16
    n_spatial = patches_side * patches_side
    expected = t_eff * n_spatial
    if n_tokens_total != expected:
        raise ValueError(
            f"Token layout mismatch: got N={n_tokens_total}, expected {expected} "
            f"(t_eff={t_eff}, n_spatial={n_spatial})."
        )
    return t_eff, n_spatial


def save_json(output_dir: str, filename: str, payload: Dict) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path
