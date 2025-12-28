# WrinkleFree-Fairy2

Complex-valued LLM quantization with Fairy2i - achieving effective 2-bit precision with weights in {+1, -1, +i, -i}.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

WrinkleFree-Fairy2 implements the [Fairy2i paper](https://arxiv.org/abs/2512.02901) for extreme LLM quantization. The approach converts pre-trained real-valued models to complex-valued representations, then quantizes to the fourth roots of unity {+1, -1, +i, -i}.

**Key Benefits:**
- **2-bit effective precision** with W2 mode (2-stage residual quantization)
- **Multiplication-free inference** - only additions and bit operations
- **Commercially friendly** - Apache 2.0 licensed models and training data
- **Reuses pre-trained weights** - no training from scratch required

## Installation

```bash
# Clone the repository
git clone https://github.com/DeepOpt-com/WrinkleFree-Fairy2.git
cd WrinkleFree-Fairy2

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Quick Start

### Smoke Test

```bash
# Run a quick 10-step training test
uv run python scripts/smoke_test.py --model smollm2_135m --mode w2 --steps 10
```

### Full Training

```bash
# Train SmolLM2-135M with W2 (2-bit) mode
uv run python scripts/train.py \
    model=smollm2_135m \
    training=fairy2_w2 \
    data=mixed_pretrain

# Train with W1 (1-bit) mode for more aggressive quantization
uv run python scripts/train.py \
    model=smollm2_135m \
    training=fairy2_w1 \
    data=mixed_pretrain
```

### Model Conversion

```python
from transformers import AutoModelForCausalLM
from fairy2.models import convert_to_fairy2

# Load pre-trained model
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")

# Convert to Fairy2 format (W2 = 2 stages)
fairy2_model = convert_to_fairy2(model, num_stages=2)
```

## Algorithm

Fairy2i consists of three key components:

### 1. Widely-Linear Complex Conversion

Convert real-valued Linear layers to complex widely-linear form:

```
y = U·x + W·conj(x)
```

Where U and W are derived from the original real weight matrix R:
- Re(U) = 0.5 × (R₁₁ + R₂₂)
- Im(U) = 0.5 × (R₂₁ - R₁₂)
- Re(W) = 0.5 × (R₁₁ - R₂₂)
- Im(W) = 0.5 × (R₁₂ + R₂₁)

### 2. Phase-Aware Quantization

Quantize complex weights to the nearest fourth root of unity:

```
k = ⌊2θ/π + 0.5⌋ mod 4  where θ = Arg(w)
```

Mapping: k=0→+1, k=1→+i, k=2→-1, k=3→-i

### 3. Recursive Residual Quantization

For W2 mode, apply two-stage residual quantization:

```
W_q ≈ (s₁_re·q₁_re + i·s₁_im·q₁_im) + (s₂_re·q₂_re + i·s₂_im·q₂_im)
```

## Supported Models

All models use commercially-friendly Apache 2.0 licenses:

| Model | Config | Parameters | Recommended GPU |
|-------|--------|------------|-----------------|
| SmolLM2-135M | `smollm2_135m` | 135M | Any GPU (4GB+) |
| Qwen3-4B | `qwen3_4b` | 4B | H100/A100 (24GB+) |

## Training Modes

| Mode | Stages | Bits/Weight | Quality | Speed |
|------|--------|-------------|---------|-------|
| W1 | 1 | ~1 bit | Lower | Faster |
| W2 | 2 | ~2 bits | Higher | Standard |

## Configuration

Training is configured using Hydra. Key configuration files:

```
configs/
├── config.yaml           # Main config
├── model/
│   ├── smollm2_135m.yaml
│   └── qwen3_4b.yaml
├── training/
│   ├── fairy2_w1.yaml    # 1-stage quantization
│   └── fairy2_w2.yaml    # 2-stage quantization
├── data/
│   └── mixed_pretrain.yaml
└── distributed/
    ├── single_gpu.yaml
    └── fsdp_multi.yaml
```

### Common Overrides

```bash
# Limit training steps
training.max_steps=100

# Change batch size
training.batch_size=16

# Enable GCS checkpointing
gcs.enabled=true gcs.bucket=your-bucket

# Disable W&B logging
training.logging.wandb.enabled=false
```

## Cloud Deployment

Integration with WrinkleFree-Deployer for SkyPilot-based cloud training:

```bash
# Via CLI
cd ../WrinkleFree-Deployer
wf fairy2 -m smollm2_135m --mode w2

# Via Python
from wf_deployer import train_fairy2
run_id = train_fairy2("smollm2_135m", mode="w2")
```

## Evaluation

Integration with WrinkleFree-Eval for benchmarking:

```bash
# Run evaluation
uv run python -m wrinklefree_eval \
    model_path=outputs/fairy2_smollm2_135m/checkpoint \
    benchmark=smoke_test
```

## Project Structure

```
WrinkleFree-Fairy2/
├── src/fairy2/
│   ├── models/
│   │   ├── widely_linear.py    # Real-to-complex conversion
│   │   ├── fairy2_linear.py    # Full Fairy2 layer
│   │   └── converter.py        # Model conversion
│   ├── quantization/
│   │   ├── phase_aware.py      # Phase-aware quantization
│   │   ├── residual.py         # Residual quantization
│   │   └── ste.py              # Straight-Through Estimator
│   ├── training/
│   │   ├── trainer.py          # QAT trainer
│   │   └── loss.py             # Loss functions
│   └── inference/
│       └── table_lookup.py     # Multiplication-free inference
├── scripts/
│   ├── train.py                # Training entry point
│   ├── convert.py              # Model conversion script
│   └── smoke_test.py           # Quick validation
├── configs/                    # Hydra configs
├── tests/                      # Unit tests
└── skypilot/                   # Cloud deployment configs
```

## Development

```bash
# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=fairy2

# Lint
uv run ruff check src/

# Type check
uv run mypy src/
```

## Training Data

All training data sources are commercially friendly:

| Dataset | License | Weight |
|---------|---------|--------|
| DCLM | CC-BY-4.0 | 25% |
| FineWeb-Edu | ODC-By | 30% |
| GitHub Code 2025 | MIT | 15% |
| FineMath | ODC-By | 15% |
| SlimPajama | Apache 2.0 | 15% |

## Citation

If you use this code, please cite the original Fairy2i paper:

```bibtex
@article{wang2024fairy2i,
  title={Fairy2i: Training Complex LLMs from Real LLMs with All Parameters in $\{\pm 1, \pm i\}$},
  author={Wang, Feiyu and Tan, Xinyu and Huang, Bokai and Zhang, Yihao and Wang, Guoan and Cong, Peizhuang and Yang, Tong},
  journal={arXiv preprint arXiv:2512.02901},
  year={2024}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Fairy2i paper](https://arxiv.org/abs/2512.02901) for the algorithm
- [HuggingFace](https://huggingface.co/) for model hosting
- [WrinkleFree](https://github.com/WrinkleFree) project for the infrastructure
