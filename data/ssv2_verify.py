"""
WorldJEPA — SSv2 Verification Script
Vérifie que l'extraction est correcte avant de lancer l'entraînement.

Usage:
    python data/ssv2_verify.py --data_root ~/Data/ssv2
"""

import os
import json
import argparse
from pathlib import Path


def verify_ssv2(data_root: str):
    root = Path(data_root).expanduser()
    print(f"\n{'='*50}")
    print(f"  SSv2 Verification — WorldJEPA")
    print(f"{'='*50}")
    print(f"  Root: {root}\n")

    errors = []
    ok = []

    # ── 1. Check structure ─────────────────────────────────────────────────
    video_dir = root / "videos"
    labels_dir = root / "labels"

    if video_dir.exists():
        ok.append(f"✅  videos/ found")
    else:
        errors.append(f"❌  videos/ NOT found at {video_dir}")

    if labels_dir.exists():
        ok.append(f"✅  labels/ found")
    else:
        errors.append(f"❌  labels/ NOT found at {labels_dir}")

    # ── 2. Check label files ───────────────────────────────────────────────
    for fname in ["train.json", "validation.json", "labels.json"]:
        fpath = labels_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
            count = len(data) if isinstance(data, list) else len(data.keys())
            ok.append(f"✅  {fname}: {count:,} entries")
        else:
            errors.append(f"❌  {fname} NOT found")

    # ── 3. Count videos ────────────────────────────────────────────────────
    if video_dir.exists():
        webm_files = list(video_dir.glob("*.webm"))
        count = len(webm_files)

        if count == 0:
            errors.append("❌  No .webm files found in videos/ — extraction may have failed")
        elif count < 100000:
            errors.append(f"⚠️   Only {count:,} videos found (expected ~220,847) — extraction incomplete?")
        else:
            ok.append(f"✅  {count:,} .webm videos found")

        # Check a few random videos are readable
        if webm_files:
            print("\n  Checking 3 random videos...")
            import random
            sample = random.sample(webm_files, min(3, len(webm_files)))
            for vp in sample:
                size_kb = vp.stat().st_size / 1024
                print(f"    {vp.name}: {size_kb:.0f} KB")

    # ── 4. Disk space ──────────────────────────────────────────────────────
    if video_dir.exists():
        total_size = sum(f.stat().st_size for f in video_dir.glob("*.webm"))
        size_gb = total_size / (1024**3)
        print(f"\n  Total video size: {size_gb:.1f} GB")

    # ── 5. Print summary ───────────────────────────────────────────────────
    print("\n  Results:")
    for msg in ok:
        print(f"  {msg}")
    for msg in errors:
        print(f"  {msg}")

    if not errors:
        print(f"\n  ✅ SSv2 ready for WorldJEPA training!")
        print(f"\n  Next step:")
        print(f"    python scripts/train.py \\")
        print(f"      --data_dir {root} \\")
        print(f"      --freeze_encoder \\")
        print(f"      --batch_size 8 \\")
        print(f"      --epochs 50")
    else:
        print(f"\n  ⚠️  Fix the errors above before training.")

    print(f"\n{'='*50}\n")
    return len(errors) == 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="~/Data/ssv2")
    args = p.parse_args()
    verify_ssv2(args.data_root)
