"""
WorldJEPA — Something-Something V2 DataLoader
CertesLabs · 2026

Compatible avec la structure exacte du téléchargement Qualcomm SSv2 :
    ~/Data/ssv2/
    ├── videos/          ← fichiers .webm (220k vidéos)
    └── labels/
        ├── train.json
        ├── validation.json
        └── labels.json

Usage:
    from data.ssv2_dataset import SSv2Dataset, build_ssv2_loaders
    train_loader, val_loader = build_ssv2_loaders(
        data_root="~/Data/ssv2",
        num_frames=16,
        batch_size=8,
    )
"""

import os
import json
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.io as io


# ─── Transforms ───────────────────────────────────────────────────────────

def build_transforms(
    resolution: int = 256,
    is_train: bool = True,
) -> T.Compose:
    """
    Standard video transforms for JEPA pretraining.
    Minimal augmentation — we want stable representations.
    """
    transforms = []

    if is_train:
        transforms += [
            T.RandomResizedCrop(
                resolution,
                scale=(0.3, 1.0),
                ratio=(0.75, 1.35),
                antialias=True,
            ),
            T.RandomHorizontalFlip(p=0.5),
        ]
    else:
        transforms += [
            T.Resize(int(resolution * 1.15), antialias=True),
            T.CenterCrop(resolution),
        ]

    transforms += [
        T.ConvertImageDtype(torch.float32),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]

    return T.Compose(transforms)


# ─── Frame sampling ───────────────────────────────────────────────────────

def sample_frames(
    total_frames: int,
    num_frames: int,
    strategy: str = "uniform",
) -> list:
    """
    Sample frame indices from a video.

    Args:
        total_frames: total number of frames in video
        num_frames: how many frames to sample
        strategy: 'uniform' or 'random_window'

    Returns:
        list of frame indices
    """
    if total_frames <= num_frames:
        # Video too short — repeat last frame
        indices = list(range(total_frames))
        while len(indices) < num_frames:
            indices.append(indices[-1])
        return indices

    if strategy == "uniform":
        # Evenly spaced frames across full video
        step = total_frames / num_frames
        return [int(i * step) for i in range(num_frames)]

    elif strategy == "random_window":
        # Random contiguous window — better for temporal learning
        max_start = total_frames - num_frames
        start = random.randint(0, max_start)
        return list(range(start, start + num_frames))

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ─── SSv2 Dataset ─────────────────────────────────────────────────────────

class SSv2Dataset(Dataset):
    """
    Something-Something V2 dataset for WorldJEPA pretraining.

    WorldJEPA is action-free — we use video clips only,
    no label supervision during pretraining.
    Labels are loaded but only used for downstream evaluation.

    Args:
        data_root: path to ~/Data/ssv2/
        split: 'train' or 'validation'
        num_frames: frames to sample per clip
        resolution: spatial resolution (224 for ViT-L)
        sampling: 'uniform' or 'random_window'
        max_samples: limit dataset size (for debugging)
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        num_frames: int = 16,
        resolution: int = 256,
        sampling: str = "random_window",
        max_samples: Optional[int] = None,
    ):
        super().__init__()

        self.data_root = Path(data_root).expanduser()
        self.video_dir = self.data_root / "videos"
        self.split = split
        self.num_frames = num_frames
        self.resolution = resolution
        self.sampling = sampling

        # Transforms
        self.transform = build_transforms(
            resolution=resolution,
            is_train=(split == "train"),
        )

        # Load annotations
        ann_path = self.data_root / "labels" / f"{split}.json"
        assert ann_path.exists(), f"Annotations not found: {ann_path}"

        with open(ann_path) as f:
            annotations = json.load(f)

        # annotations format: list of {"id": "1234", "label": "...", "template": "..."}
        self.samples = annotations

        if max_samples is not None:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

        # Load label mapping
        labels_path = self.data_root / "labels" / "labels.json"
        if labels_path.exists():
            with open(labels_path) as f:
                self.label_map = json.load(f)
        else:
            self.label_map = {}

        print(f"[SSv2] {split}: {len(self.samples):,} videos | "
              f"{num_frames} frames @ {resolution}px")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_video(self, video_id: str) -> Optional[torch.Tensor]:
        """
        Load a video file and return frames tensor.

        SSv2 videos are .webm files named by ID.

        Returns:
            (T, C, H, W) float32 tensor, or None if loading fails
        """
        video_path = self.video_dir / f"{video_id}.webm"

        if not video_path.exists():
            return None

        try:
            import av
            import numpy as np
            container = av.open(str(video_path))
            raw = []
            for frame in container.decode(video=0):
                img = frame.to_ndarray(format="rgb24")  # (H, W, 3)
                raw.append(img)
            container.close()
            if not raw:
                return None
            # Stack and convert to (T, C, H, W) uint8
            arr = np.stack(raw)                        # (T, H, W, 3)
            frames = torch.from_numpy(arr).permute(0, 3, 1, 2)  # (T, C, H, W)
            return frames
        except Exception as e:
            return None

    def __getitem__(self, idx: int) -> Optional[dict]:
        """
        Returns:
            dict with keys:
                'video'    : (T, C, H, W) float32 tensor
                'video_id' : str
                'label'    : str (action label)
        """
        sample = self.samples[idx]
        video_id = str(sample["id"])
        label = sample.get("label", "")

        # Load video
        frames = self._load_video(video_id)

        if frames is None:
            # Skip bad video — try up to 10 next ones
            for skip in range(1, 10):
                alt = self.samples[(idx + skip) % len(self)]
                frames = self._load_video(str(alt["id"]))
                if frames is not None:
                    break
            if frames is None:
                # Return zeros as last resort
                return {
                    "video": __import__("torch").zeros(self.num_frames, 3, self.resolution, self.resolution),
                    "video_id": str(self.samples[idx]["id"]),
                    "label": "",
                }

        total_frames = frames.shape[0]

        # Sample frame indices
        indices = sample_frames(total_frames, self.num_frames, self.sampling)
        frames = frames[indices]  # (T, C, H, W)

        # Apply spatial transforms frame by frame
        transformed = []
        for t in range(frames.shape[0]):
            frame = frames[t]  # (C, H, W) uint8
            frame = self.transform(frame)  # (C, H, W) float32
            transformed.append(frame)

        video = torch.stack(transformed)  # (T, C, H, W)

        return {
            "video": video,
            "video_id": video_id,
            "label": label,
        }


# ─── Collate function ─────────────────────────────────────────────────────

def collate_fn(batch):
    """Filter None samples and stack batch."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    videos = torch.stack([b["video"] for b in batch])  # (B, T, C, H, W)
    video_ids = [b["video_id"] for b in batch]
    labels = [b["label"] for b in batch]

    return {
        "video": videos,
        "video_id": video_ids,
        "label": labels,
    }


# ─── Build loaders ────────────────────────────────────────────────────────

def build_ssv2_loaders(
    data_root: str = "~/Data/ssv2",
    num_frames: int = 16,
    resolution: int = 256,
    batch_size: int = 8,
    num_workers: int = 0,       # 0 for MPS compatibility
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = 1000,
) -> tuple:
    """
    Build SSv2 train and validation DataLoaders.

    Args:
        data_root: path to SSv2 root directory
        num_frames: frames per clip
        resolution: spatial resolution
        batch_size: batch size (8 recommended for M4 Pro)
        num_workers: 0 for MPS, 4+ for CUDA
        max_train_samples: limit train size (None = full dataset)
        max_val_samples: limit val size for fast evaluation

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = SSv2Dataset(
        data_root=data_root,
        split="train",
        num_frames=num_frames,
        resolution=resolution,
        sampling="random_window",
        max_samples=max_train_samples,
    )

    val_dataset = SSv2Dataset(
        data_root=data_root,
        split="validation",
        num_frames=num_frames,
        resolution=resolution,
        sampling="uniform",
        max_samples=max_val_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,       # False for MPS
        drop_last=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader
