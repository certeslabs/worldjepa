# WorldJEPA

> **Open, trustworthy world models with explicit simulation and controllable reasoning.**

**CertesLabs** · April 2026 · MIT License

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange)](https://pytorch.org)

---

## Overview

WorldJEPA is a JEPA-based world model that scales the principles of
[LeWorldModel](https://arxiv.org/abs/2603.19312) to large-scale internet video.

**Core idea:** V-JEPA 2's powerful video representations + SIGReg's provable
anti-collapse guarantee = stable, isotropic latent world model — no EMA, no
stop-gradient hacks.

**v0.1 thesis:** SIGReg stabilizes JEPA at 300M+ parameter scale on real-world
video, producing more isotropic and temporally consistent latent representations
than V-JEPA 2 alone.

---

## Architecture

```
Video (B, T, C, H, W)
      │
      ▼
┌─────────────────────┐
│  Encoder: ViT-L     │  ~300M params · Pretrained V-JEPA 2 · Frozen (v0.1)
│  1024-dim latents   │
└─────────┬───────────┘
          │  Z: (B, T, 1024)
          │
          ├──────────────────────────→  SIGReg(Z)  [anti-collapse]
          │
          ▼
┌─────────────────────┐
│  Predictor: ViT-S   │  ~22M params · Trained from scratch · Causal attention
│  Causal, no actions │
└─────────┬───────────┘
          │  Ẑ: (B, T, 1024)
          │
          ▼
  L = MSE(Ẑ[:,:-1], Z[:,1:]) + λ · SIGReg(Z)
         └── λ = 0.1  (only hyperparameter)
```

**No EMA. No stop-gradient. No action labels.**

---

## Quickstart

```bash
# Clone
git clone https://github.com/certeslabs/worldjepa
cd worldjepa

# Install
pip install -r requirements.txt

# Test pipeline — no downloads, runs on any machine
python scripts/train.py --mock --epochs 2 --batch_size 4 --log_every 1

# Training on Apple M4 Pro (MPS)
python scripts/train.py \
    --mock \
    --epochs 50 \
    --batch_size 8 \
    --freeze_encoder \
    --log_every 10

# Training on A100 (full SSv2 — coming in v0.2)
python scripts/train.py \
    --data_dir /path/to/ssv2 \
    --epochs 100 \
    --batch_size 32
```

---

## Benchmarks (v0.1 targets)

| Metric | Target | Description |
|---|---|---|
| Isotropy Score | > 0.1 | λ_min/λ_max of covariance — SIGReg quality |
| Effective Rank | > 50 | Dimensions actually used in latent space |
| Temporal MSE | < V-JEPA 2 baseline | Prediction quality |
| Temporal Consistency | > 0.4 | Trajectory smoothness (temporal straightening) |

---

## Hardware

| Setup | Hardware | Batch | Notes |
|---|---|---|---|
| Dev / testing | MacBook M4 Pro 24GB | 4–8 | MPS backend, frozen encoder |
| Full training | 8×A100 80GB | 128 | ~2–3 weeks |
| Cloud minimal | 1×A100 | 32 | Proof-of-concept |

---

## Repository Structure

```
worldjepa/
├── worldjepa/
│   ├── __init__.py
│   ├── model.py          # WorldJEPA, Encoder, Predictor
│   └── sigreg.py         # SIGReg + SIGRegFast
├── scripts/
│   └── train.py          # Training loop (MPS + CUDA)
├── benchmarks/
│   ├── __init__.py
│   └── metrics.py        # Isotropy, MSE, Temporal Consistency
├── configs/
│   ├── v01_mac.yaml      # M4 Pro config
│   └── v01_cloud.yaml    # A100 config
├── data/                 # Dataset utilities (v0.2)
├── requirements.txt
├── LICENSE               # MIT
└── README.md
```

---

## Roadmap

- **v0.1** — Encoder (frozen V-JEPA 2) + Predictor + SIGReg, action-free, SSv2
- **v0.2** — SSv2 dataloader, full benchmark suite, arXiv preprint
- **v0.3** — Action conditioning, gaming environments (MuJoCo, Minecraft)
- **v1.0** — Scientific simulation domains, public checkpoints on HuggingFace

---

## Citations

```bibtex
@misc{worldjepa2026,
  title        = {WorldJEPA: Scaling JEPA with SIGReg to Internet-Scale Video},
  author       = {CertesLabs},
  year         = {2026},
  url          = {https://github.com/certeslabs/worldjepa}
}

@article{maes_lelidec2026lewm,
  title        = {LeWorldModel: Stable End-to-End Joint-Embedding Predictive
                  Architecture from Pixels},
  author       = {Maes, Lucas and Le Lidec, Quentin and Scieur, Damien
                  and LeCun, Yann and Balestriero, Randall},
  journal      = {arXiv preprint arXiv:2603.19312},
  year         = {2026}
}

@article{assran2025vjepa2,
  title        = {V-JEPA 2: Self-Supervised Video Models Enable
                  Understanding, Prediction and Planning},
  author       = {Assran, Mahmoud and others},
  journal      = {arXiv preprint arXiv:2506.09985},
  year         = {2025}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

Built on [V-JEPA 2](https://github.com/facebookresearch/vjepa2) (MIT, Meta FAIR)
and [LeWorldModel](https://github.com/lucas-maes/le-wm) (MIT, Maes et al. 2026).
