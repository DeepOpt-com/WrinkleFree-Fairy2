# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

WrinkleFree-Fairy2 implements the Fairy2i paper (arxiv:2512.02901) for complex-valued LLM quantization:
- **Algorithm**: Fairy2i - weights quantized to fourth roots of unity {+1, -1, +i, -i}
- **Training**: QAT (Quantization-Aware Training) with STE
- **Config**: Hydra
- **Package management**: uv
- **Distributed**: FSDP for multi-GPU
- **License**: Apache 2.0 (commercially friendly)

## Key Concepts

### Fairy2i Algorithm (3 Components)

1. **Widely-Linear Complex Conversion**: Real Linear layer → Complex (U, W) matrices
   ```
   y = U*x + W*conj(x)  where U, W ∈ ℂ^(n×m)
   ```

2. **Phase-Aware Quantization**: Quantize complex weights to {+1, -1, +i, -i}
   ```
   k = floor(2*θ/π + 0.5) mod 4  where θ = Arg(w)
   k=0→1, k=1→i, k=2→-1, k=3→-i
   ```

3. **Recursive Residual Quantization**: Multi-stage for W2 (2-bit) precision
   ```
   W_q ≈ Σ_{t=0}^{T-1} (s_re^t * q_re^t + i * s_im^t * q_im^t)
   ```

## Quick Start

```bash
# Install dependencies
uv sync

# Run smoke test (10 steps, SmolLM2-135M)
uv run python scripts/smoke_test.py --model smollm2_135m --mode w2 --steps 10

# Full training
uv run python scripts/train.py model=smollm2_135m training=fairy2_w2 data=mixed_pretrain

# Run tests
uv run pytest
```

## Training Modes

| Mode | Stages | Bits/Param | Description |
|------|--------|------------|-------------|
| W1 | 1 | ~1 bit | Single-stage, most aggressive |
| W2 | 2 | ~2 bits | Two-stage residual, better quality |

## Supported Models (Apache 2.0 Licensed)

| Model | Config | Params | License |
|-------|--------|--------|---------|
| SmolLM2-135M | `smollm2_135m` | 135M | Apache 2.0 |
| Qwen3-4B | `qwen3_4b` | 4B | Apache 2.0 |

## Architecture

### Core Modules
- `src/fairy2/models/widely_linear.py` - Real-to-complex widely-linear conversion
- `src/fairy2/quantization/phase_aware.py` - Phase-aware quantization to {1,i,-1,-i}
- `src/fairy2/quantization/residual.py` - Recursive residual quantization
- `src/fairy2/models/fairy2_linear.py` - Full Fairy2Linear layer with STE
- `src/fairy2/models/converter.py` - Convert HuggingFace models to Fairy2

### Training
- `src/fairy2/training/trainer.py` - QAT trainer with FSDP, W&B, GCS
- `scripts/train.py` - Main training entry point

### Inference
- `src/fairy2/inference/table_lookup.py` - Multiplication-free inference

## Configuration

All configs in `configs/` using Hydra:
- `model/` - Model configs (smollm2_135m, qwen3_4b)
- `training/` - Training configs (fairy2_w1, fairy2_w2)
- `data/` - Dataset configs (mixed_pretrain)
- `distributed/` - FSDP/DDP settings

### Hydra Override Examples

```bash
# Change mode (1-bit vs 2-bit)
uv run python scripts/train.py training=fairy2_w1

# Limit training steps (for testing)
uv run python scripts/train.py training.max_steps=100

# Disable wandb
uv run python scripts/train.py training.logging.wandb.enabled=false

# Enable GCS checkpointing
uv run python scripts/train.py gcs.enabled=true gcs.bucket=wrinklefree-checkpoints
```

## Cloud Deployment (SkyPilot)

Training via WrinkleFree-Deployer:

```bash
cd ../WrinkleFree-Deployer

# Launch Fairy2 training
wf fairy2 -m smollm2_135m --mode w2

# Or use Python API
from wf_deployer import train_fairy2
run_id = train_fairy2("smollm2_135m", mode="w2")
```

## Testing

```bash
# All tests
uv run pytest

# Smoke tests only
uv run pytest -m smoke

# With coverage
uv run pytest --cov=fairy2

# Type check
uv run mypy src/
```

## Key Implementation Details

### Edge Cases to Handle

1. **Zero magnitude weights (|w| = 0)**: Default to +1, set scale to 0
2. **Numerical precision**: Use float32/float64 for conversion step
3. **Real input handling**: When x is real, y = (U+W)x

### STE (Straight-Through Estimator)

Gradients flow through quantization using detach trick:
```python
w_quant = w + (quantize(w) - w).detach()
```

### FSDP Compatibility

Store complex weights as separate real tensors for FSDP:
- `U_re, U_im, W_re, W_im` instead of complex tensors

## Data (Commercially Friendly)

Training uses mixed_pretrain with influence-based remixing:
- DCLM (25%) - CC-BY-4.0
- FineWeb-Edu (30%) - ODC-By
- GitHub Code 2025 (15%) - MIT
- FineMath (15%) - ODC-By
- SlimPajama (15%) - Apache 2.0

## References

- Paper: [Fairy2i: Training Complex LLMs from Real LLMs](https://arxiv.org/abs/2512.02901)
- Related: [BitDistill](https://arxiv.org/abs/2510.13998), [BitNet](https://arxiv.org/abs/2310.11453)
