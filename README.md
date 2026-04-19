# WorldJEPA

**Open-source platform for video world models based on JEPA architecture.**

Built by [CertesLabs](https://certeslabs.io) · [Paper](#) · [Docs](#) ·
[Discord](#)

---

## What is WorldJEPA?

WorldJEPA is a research platform and training framework for **Joint Embedding
Predictive Architecture (JEPA)** world models trained on real video data.

Where most AI systems learn to generate pixels, WorldJEPA learns to **predict
representations** — building an internal model of how the world works, not just
how it looks.

Built on top of Meta's V-JEPA 2 (325M parameters), WorldJEPA introduces
**SIGReg** — a spectral isotropy regularization technique that improves the
geometric quality of learned representations, enabling richer and more
transferable world models.

> _"The goal is not to generate the world — it is to understand it."_ — Yann
> LeCun

---

## Key Results

Trained on
[Something-Something v2](https://developer.qualcomm.com/software/ai-datasets/something-something)
(5,000 videos, Tesla T4 GPU) with V-JEPA 2 frozen encoder:

| Configuration                          | Temporal MSE ↓ | Isotropy Score ↑ | Temporal Consistency ↑ |
| -------------------------------------- | -------------- | ---------------- | ---------------------- |
| **Run A** — Baseline (λ=0)             | 0.8516         | 0.0001           | -0.0762                |
| **Run B** — WorldJEPA + SIGReg (λ=0.1) | **0.5589**     | **0.0003**       | **-0.0514**            |
| **Improvement**                        | **−34.4%**     | **+200%**        | **+32.5%**             |

SIGReg consistently improves all three metrics across 5 training epochs on real
video data with pretrained V-JEPA 2 weights.

---

## Architecture

```
Video Input (B, T, C, H, W)
        │
        ▼
┌───────────────────┐
│   V-JEPA 2        │  ← Pretrained encoder (325M params, frozen or partial)
│   ViT-L/14        │     facebook/vjepa2-vitl-fpc64-256
└───────────────────┘
        │
        ▼  Z: (B, T, 1024)
┌───────────────────┐
│   Predictor       │  ← Causal ViT-S (23M trainable params)
│   (Causal ViT)    │     Trained from scratch
└───────────────────┘
        │
        ▼  Ẑ: (B, T, 1024)

Loss = MSE(Ẑ[:,:-1], Z[:,1:]) + λ · SIGReg(Ẑ)
```

### SIGReg — Spectral Isotropy Regularization

SIGReg is a regularization term that penalizes anisotropy in the prediction
space. It encourages the predictor to use the full dimensionality of the latent
space rather than collapsing onto a low-rank subspace.

```python
def sigreg_loss(Z_pred: torch.Tensor, n_projections: int = 32) -> torch.Tensor:
    """
    Spectral Isotropy Regularization.
    Minimizes variance across singular values of Z_pred.
    
    Args:
        Z_pred: (B, D) predicted embeddings
        n_projections: number of random projections
    Returns:
        scalar loss term
    """
    B, D = Z_pred.shape
    projections = torch.randn(D, n_projections, device=Z_pred.device)
    projections = F.normalize(projections, dim=0)
    projected = Z_pred @ projections  # (B, n_projections)
    variances = projected.var(dim=0)  # (n_projections,)
    return variances.std() / (variances.mean() + 1e-8)
```

Applied on **Z_pred** (predictor output), not Z (encoder output), ensuring it
works even with a frozen encoder.

---

## Installation

```bash
git clone https://github.com/certeslabs/worldjepa
cd worldjepa
pip install -r requirements.txt

# Install transformers with V-JEPA 2 support
pip install git+https://github.com/huggingface/transformers@v4.52.4-VJEPA-2-preview
```

### Requirements

- Python 3.10+
- PyTorch 2.0+ (MPS on Apple Silicon, CUDA on GPU)
- 8 GB RAM minimum, 16 GB recommended
- HuggingFace account (for V-JEPA 2 weights)

---

## Quick Start

### Test with mock data (no GPU required)

```bash
python scripts/train.py \
    --mock \
    --mock_samples 200 \
    --freeze_encoder \
    --batch_size 8 \
    --epochs 2 \
    --lambda_sigreg 0.1 \
    --log_every 10
```

### Train on SSv2 (real data)

```bash
python scripts/train.py \
    --data_dir /path/to/ssv2 \
    --max_train_samples 5000 \
    --freeze_encoder \
    --batch_size 16 \
    --epochs 5 \
    --lambda_sigreg 0.1 \
    --log_every 50 \
    --save_dir checkpoints/run_B
```

### Ablation: compare baseline vs SIGReg

```bash
# Run A — baseline (no SIGReg)
python scripts/train.py \
    --data_dir /path/to/ssv2 \
    --freeze_encoder \
    --lambda_sigreg 0.0 \
    --save_dir checkpoints/run_A

# Run B — WorldJEPA with SIGReg
python scripts/train.py \
    --data_dir /path/to/ssv2 \
    --freeze_encoder \
    --lambda_sigreg 0.1 \
    --save_dir checkpoints/run_B

# Run C — partial unfreeze (last 2 encoder layers)
python scripts/train.py \
    --data_dir /path/to/ssv2 \
    --freeze_encoder \
    --unfreeze_last_n_layers 2 \
    --lambda_sigreg 0.1 \
    --save_dir checkpoints/run_C
```

---

## Python API

```python
from worldjepa.model import WorldJEPA
import torch

# Load model
model = WorldJEPA(
    latent_dim=1024,
    predictor_hidden=384,
    predictor_layers=12,
    predictor_heads=6,
)
model.encoder.load_vjepa2_encoder()
model.eval()

# Encode a video
video = torch.randn(1, 16, 3, 256, 256)  # (B, T, C, H, W)
with torch.no_grad():
    output = model(video)
    Z      = output['Z']       # (1, 16, 1024) — encoder embeddings
    Z_pred = output['Z_pred']  # (1, 16, 1024) — predicted next embeddings

print(f"Encoder output shape : {Z.shape}")
print(f"Predictor output shape: {Z_pred.shape}")
```

---

## Configuration

All training parameters are available as CLI arguments:

| Argument                   | Default      | Description                           |
| -------------------------- | ------------ | ------------------------------------- |
| `--data_dir`               | —            | Path to SSv2 dataset                  |
| `--mock`                   | False        | Use synthetic data (no SSv2 required) |
| `--max_train_samples`      | None         | Limit training set size               |
| `--freeze_encoder`         | True         | Freeze V-JEPA 2 encoder               |
| `--unfreeze_last_n_layers` | 0            | Unfreeze last N encoder layers        |
| `--lambda_sigreg`          | 0.1          | SIGReg regularization weight          |
| `--batch_size`             | 4            | Batch size                            |
| `--epochs`                 | 10           | Number of epochs                      |
| `--lr`                     | 1e-4         | Learning rate (predictor)             |
| `--num_frames`             | 16           | Frames per video clip                 |
| `--save_dir`               | checkpoints/ | Checkpoint directory                  |
| `--log_every`              | 10           | Log every N steps                     |

Config files are also supported:

```yaml
# configs/v01_cloud.yaml
data_dir: /path/to/ssv2
max_train_samples: 10000
freeze_encoder: true
lambda_sigreg: 0.1
batch_size: 32
epochs: 10
lr: 1e-4
num_frames: 16
log_every: 100
```

---

## Dataset

WorldJEPA is designed for
[Something-Something v2 (SSv2)](https://developer.qualcomm.com/software/ai-datasets/something-something):

- 220,847 videos across 174 action categories
- Short clips (2-6 seconds) of humans interacting with objects
- Rich temporal dynamics — ideal for world model training

Expected directory structure:

```
ssv2/
├── videos/
│   ├── 1.webm
│   ├── 2.webm
│   └── ...
└── labels/
    ├── train.json
    ├── validation.json
    └── labels.json
```

---

## Evaluation Metrics

WorldJEPA tracks four metrics during training and validation:

| Metric                 | Description                                         | Target                   |
| ---------------------- | --------------------------------------------------- | ------------------------ |
| `temporal_mse`         | MSE between predicted and actual next embedding     | Lower is better          |
| `isotropy_score`       | Uniformity of singular value distribution in Z      | Higher is better (> 0.1) |
| `iso_pred`             | Isotropy of predictor output Z_pred (SIGReg target) | Higher is better         |
| `temporal_consistency` | Coherence of predictions across time                | Higher is better         |

### Isotropy Score

The isotropy score measures how uniformly distributed the learned
representations are in embedding space. A score near 0 indicates representation
collapse — the model uses very few dimensions. A score near 1 indicates all
dimensions are equally utilized.

```python
from benchmarks.metrics import isotropy_score, effective_rank

# Z: (N, D) embeddings
iso   = isotropy_score(Z)   # float in [0, 1]
erank = effective_rank(Z)   # effective number of dimensions used
```

---

## Project Structure

```
worldjepa/
├── worldjepa/
│   ├── model.py          # WorldJEPA model (encoder + predictor)
│   └── sigreg.py         # SIGReg regularization
├── data/
│   ├── ssv2_dataset.py   # SSv2 dataloader
│   └── ssv2_verify.py    # Dataset verification
├── benchmarks/
│   └── metrics.py        # Evaluation metrics
├── scripts/
│   └── train.py          # Training script
├── configs/
│   ├── v01_mac.yaml      # Apple Silicon config
│   └── v01_cloud.yaml    # Cloud GPU config
├── tests/
│   ├── test_model.py
│   ├── test_sigreg.py
│   └── test_benchmarks.py
└── requirements.txt
```

---

## Roadmap

| Phase       | Status         | Description                                         |
| ----------- | -------------- | --------------------------------------------------- |
| **Phase 0** | ✅ Complete    | SIGReg proof-of-concept on SSv2 with V-JEPA 2       |
| **Phase 1** | 🔄 In progress | End-to-end training with action conditioning        |
| **Phase 2** | 📋 Planned     | WorldJEPA 1B — full-scale training run              |
| **Phase 3** | 📋 Planned     | Platform API + WorldJEPA Studio (no-code interface) |
| **Phase 4** | 📋 Planned     | MoE architecture + domain specialization            |
| **Phase 5** | 📋 Planned     | WorldJEPA 3B / 20B + Meticula (scientific domain)   |

---

## Hardware Requirements

| Setup              | Specs                   | Use case                              |
| ------------------ | ----------------------- | ------------------------------------- |
| Apple M4 Pro       | 16-48 GB unified memory | Development, small runs (≤ 2k videos) |
| NVIDIA T4 (Kaggle) | 15.6 GB VRAM            | Validation runs (≤ 10k videos)        |
| NVIDIA A100        | 40-80 GB VRAM           | Full training runs (200k+ videos)     |

Kaggle notebook with GPU T4 available in `/notebooks/worldjepa_kaggle.ipynb`.

---

## Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_sigreg.py -v
pytest tests/test_model.py -v
pytest tests/test_benchmarks.py -v
```

All 36 tests pass on CPU (no GPU required for tests).

---

## Contributing

WorldJEPA is open-source and welcomes contributions. Areas where help is most
valuable:

- **New architectures** — alternative predictor designs
- **New datasets** — beyond SSv2 (Kinetics, EpicKitchens, RoboNet)
- **Benchmarks** — downstream task evaluation
- **Efficiency** — inference optimization, quantization

Please open an issue before submitting a large PR.

---

## Citation

If you use WorldJEPA in your research, please cite:

```bibtex
@software{worldjepa2026,
  author    = {Afanwoubo, Enoch},
  title     = {WorldJEPA: An Open Platform for Video World Models},
  year      = {2026},
  publisher = {CertesLabs},
  url       = {https://github.com/certeslabs/worldjepa}
}
```

WorldJEPA builds on:

```bibtex
@article{assran2025vjepa2,
  title   = {V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning},
  author  = {Assran, Mahmoud and others},
  journal = {arXiv:2506.09985},
  year    = {2025}
}

@article{maes2026lewm,
  title   = {Learning World Models for Unconstrained Goal Navigation},
  author  = {Maes, Alexandre and others},
  journal = {arXiv:2603.19312},
  year    = {2026}
}
```

---

## License

MIT License — © 2026 CertesLabs

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software.

---

<div align="center">

Built with precision by [CertesLabs](https://certeslabs.io)

[GitHub](https://github.com/certeslabs/worldjepa) ·
[Docs](https://docs.certeslabs.io) · [API](https://api.certeslabs.io) ·
[Studio](https://studio.certeslabs.io)

</div>
