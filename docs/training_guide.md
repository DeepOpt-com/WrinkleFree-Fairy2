# Training Guide

This document explains training parameters and recommendations for Fairy2i.

## Key Parameters

### T (Number of Stages) - The Most Important Parameter

T controls the number of residual quantization stages. This is set via `--mode`:

| Mode | T | Bits/Weight | Quality | Speed |
|------|---|-------------|---------|-------|
| `w1` | 1 | ~1 bit | Lower | Faster |
| `w2` | 2 | ~2 bits | Higher | Slower |

**Recommendation**: Start with `w2` (T=2) for better quality. Only use `w1` if memory is critical.

```bash
# W2 mode (T=2, recommended)
wf fairy2 -m smollm2_135m --mode w2

# W1 mode (T=1, more aggressive)
wf fairy2 -m smollm2_135m --mode w1
```

In code, this maps to `num_stages`:
```python
from fairy2.models import convert_to_fairy2

# T=2 (W2 mode)
model = convert_to_fairy2(model, num_stages=2)

# T=1 (W1 mode)
model = convert_to_fairy2(model, num_stages=1)
```

### Continued Training Duration

**How much training is needed?** This depends on model size:

| Model Size | Recommended Tokens | Approx. Steps (batch=8, seq=2048) |
|------------|-------------------|-----------------------------------|
| 135M | 500M - 1B | 30K - 60K |
| 1B | 1B - 2B | 60K - 120K |
| 4B | 2B - 5B | 120K - 300K |
| 7B+ | 5B - 10B | 300K - 600K |

**Rule of thumb**: Train for ~1-5% of original pretraining tokens.

The default config trains for 10B tokens:
```yaml
# configs/training/fairy2_w2.yaml
training:
  total_tokens: 10_000_000_000  # 10B tokens
```

Override for shorter runs:
```bash
# Quick test: 100M tokens
wf fairy2 -m smollm2_135m training.total_tokens=100000000

# Or limit by steps
wf fairy2 -m smollm2_135m training.max_steps=10000
```

### Learning Rate

| Model Size | Recommended LR |
|------------|---------------|
| < 500M | 1e-4 |
| 500M - 2B | 5e-5 |
| 2B - 7B | 2e-5 |
| > 7B | 1e-5 |

Override:
```bash
wf fairy2 -m smollm2_135m training.optimizer.lr=5e-5
```

### Batch Size

Larger is generally better for stability. Defaults:

| Scale | Batch Size | Gradient Accum | Effective Batch |
|-------|------------|----------------|-----------------|
| small (1x H100) | 8 | 8 | 64 |
| medium (2x H100) | 8 | 4 | 64 |
| large (4x H100) | 8 | 2 | 64 |

## Training Phases

Fairy2i QAT happens in one phase (unlike BitDistill which has 3 stages):

```
┌─────────────────────────────────────────────────┐
│           Quantization-Aware Training            │
│                                                  │
│  Start: Pretrained model                        │
│    ↓                                            │
│  Convert: Linear → Fairy2Linear                 │
│    ↓                                            │
│  Train: ~1-10B tokens with quantization active  │
│    ↓                                            │
│  End: Quantized model ready for inference       │
│                                                  │
└─────────────────────────────────────────────────┘
```

## Scheduler: WSD (Warmup-Stable-Decay)

The default scheduler has three phases:

```
Learning Rate
    ↑
    │    ┌────────────────────┐
    │   /│                    │\
    │  / │      Stable        │ \
    │ /  │                    │  \
    │/   │                    │   \
    └────┴────────────────────┴────→ Steps
     Warm      80% of training    Decay
     (5%)                         (15%)
```

## What Layers Get Quantized?

By default:
- **Quantized**: All `nn.Linear` layers in transformer blocks
- **Excluded**: `embed_tokens`, `lm_head`, layer norms

You can customize exclusions:
```python
model = convert_to_fairy2(
    model,
    num_stages=2,
    exclude_names=["embed_tokens", "lm_head", "my_special_layer"]
)
```

## Monitoring Training

### Key Metrics to Watch

1. **Loss**: Should decrease steadily
   - Initial: ~10-20 (random quantized weights)
   - After training: ~2-4 (depends on data)

2. **Gradient Norm**: Should be stable
   - Spikes may indicate learning rate too high

3. **Quantization Error**: Available in W&B
   - Lower is better
   - W2 should have lower error than W1

### W&B Dashboard

Training logs to `wrinklefree-fairy2` project:

```bash
# Check recent runs
wf wandb-status -p wrinklefree-fairy2

# Or open dashboard
# https://wandb.ai/<your-entity>/wrinklefree-fairy2
```

## Example Training Commands

### Quick Smoke Test (5 minutes)
```bash
wf fairy2 -m smollm2_135m --mode w2 training.max_steps=100
```

### Full SmolLM2-135M Training (~2 hours)
```bash
wf fairy2 -m smollm2_135m --mode w2
```

### Qwen3-4B Training (~8 hours)
```bash
wf fairy2 -m qwen3_4b --mode w2 --scale large
```

### Custom Configuration
```bash
wf fairy2 -m smollm2_135m --mode w2 \
  training.optimizer.lr=1e-4 \
  training.batch_size=4 \
  training.total_tokens=1000000000
```

## Checkpointing

Checkpoints are saved:
- Every 1000 steps (configurable)
- At the end of training
- To GCS: `gs://wrinklefree-checkpoints/fairy2/{model}/`

Resume from checkpoint:
```bash
wf fairy2 -m smollm2_135m \
  training.resume_from=gs://wrinklefree-checkpoints/fairy2/smollm2_135m/step_5000.pt
```

## Expected Results

After training, expect:

| Metric | W1 (T=1) | W2 (T=2) |
|--------|----------|----------|
| Perplexity increase | 10-20% | 2-5% |
| Memory savings | ~16x | ~8x |
| Inference speedup | ~4x | ~3x |

*Actual results vary by model and training duration*
