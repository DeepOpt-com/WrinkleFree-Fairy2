# Fairy2i Documentation

Welcome to the WrinkleFree-Fairy2 documentation.

## Quick Start

```bash
# Install
cd WrinkleFree-Fairy2
uv sync

# Run smoke test (local, ~5 min)
uv run python scripts/smoke_test.py --model smollm2_135m --mode w2 --steps 10

# Deploy training to cloud (requires SkyPilot + credentials)
cd ../WrinkleFree-Deployer
wf fairy2 -m smollm2_135m --mode w2
```

## Documentation Index

| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | How Fairy2i works (algorithm, STE, quantization) |
| [Training Guide](training_guide.md) | Training parameters, T, duration, recommendations |
| [Deployment](deployment.md) | Where training runs, SkyPilot, GCS, costs |

## Key Concepts

### What is Fairy2i?

Fairy2i is a **quantization method** that compresses LLM weights to complex numbers with only 4 possible values: {+1, -1, +i, -i}.

### What is "Training" Here?

This is **Quantization-Aware Training (QAT)**, NOT training from scratch:
1. Take a pretrained model (SmolLM2-135M, Qwen3-4B)
2. Convert Linear layers → Fairy2Linear layers
3. Continue training so model adapts to quantized weights

### What is T?

T is the number of residual quantization stages:
- **T=1** (`--mode w1`): ~1 bit per weight, more aggressive
- **T=2** (`--mode w2`): ~2 bits per weight, better quality (recommended)

### How Long to Train?

| Model Size | Tokens | Time (1x H100) |
|------------|--------|----------------|
| 135M | 500M-1B | ~1-2 hours |
| 4B | 2B-5B | ~8-12 hours |

### Where Does Training Run?

Training runs on **cloud GPUs** via SkyPilot (Nebius H100 by default).
Checkpoints save to **GCS** (`gs://wrinklefree-checkpoints/fairy2/`).

## FAQ

**Q: Do I need a GPU locally?**
A: For smoke tests, yes. For full training, no - it runs in the cloud.

**Q: What models are supported?**
A: Any HuggingFace causal LM. Pre-configured: SmolLM2-135M, Qwen3-4B (both Apache 2.0).

**Q: How do I monitor training?**
A: Via Weights & Biases (`wf wandb-status`) or SkyPilot logs (`wf logs wf-fairy2-train`).

**Q: How much does training cost?**
A: SmolLM2-135M ~$6, Qwen3-4B ~$100 (varies by cloud).
