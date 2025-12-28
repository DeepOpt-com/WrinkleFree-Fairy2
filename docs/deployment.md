# Deployment Guide

This document explains where and how Fairy2 training jobs are deployed.

## Deployment Overview

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Your PC    │ ──→  │   SkyPilot   │ ──→  │  Cloud GPU   │
│  (launcher)  │      │ (orchestrator)│      │  (training)  │
└──────────────┘      └──────────────┘      └──────────────┘
                                                   │
                                                   ↓
                                            ┌──────────────┐
                                            │     GCS      │
                                            │ (checkpoints)│
                                            └──────────────┘
```

## What is SkyPilot?

[SkyPilot](https://skypilot.readthedocs.io/) is a framework for running ML workloads on any cloud. It:
- Finds the cheapest available GPUs across clouds
- Handles provisioning, setup, and teardown
- Manages spot instance interruptions
- Syncs code to the cloud automatically

## Where Does Training Run?

Training runs on **cloud GPUs**, not your local machine. Current configuration:

| Cloud | GPU | Default |
|-------|-----|---------|
| Nebius | H100 | ✓ Primary |
| GCP | H100/A100 | Fallback |
| AWS | H100/A100 | Fallback |

The SkyPilot YAML (`skypilot/fairy2_train.yaml`) specifies:
```yaml
resources:
  accelerators: H100:1  # 1x H100 GPU
  use_spot: false       # On-demand (reliable)
  disk_size: 200        # 200GB disk
```

## Where Do Checkpoints Go?

Checkpoints are saved to **Google Cloud Storage (GCS)**:

```
gs://wrinklefree-checkpoints/
└── fairy2/
    └── smollm2_135m/
        ├── checkpoint_step_1000.pt
        ├── checkpoint_step_2000.pt
        └── final_checkpoint.pt
```

This happens automatically via the training script.

## How to Launch Training

### Option 1: Via WrinkleFree-Deployer CLI

```bash
cd WrinkleFree-Deployer

# Launch Fairy2 training (returns immediately)
wf fairy2 -m smollm2_135m --mode w2

# Launch and wait for completion
wf fairy2 -m smollm2_135m --mode w2 --no-detach

# View logs
wf logs wf-fairy2-train

# Cancel job
wf cancel wf-fairy2-train
```

### Option 2: Via Python API

```python
from wf_deployer import train_fairy2

# Launch training
run_id = train_fairy2("smollm2_135m", mode="w2")

# With custom overrides
run_id = train_fairy2(
    "smollm2_135m",
    mode="w2",
    scale="large",  # 4x H100
    overrides=["training.max_steps=10000"],
)
```

### Option 3: Direct SkyPilot

```bash
cd WrinkleFree-Fairy2

# Launch directly with SkyPilot
sky launch skypilot/fairy2_train.yaml \
  --env MODEL=smollm2_135m \
  --env MODE=w2
```

## GPU Scale Profiles

| Scale | GPUs | Use Case |
|-------|------|----------|
| dev | 1x A10G | Quick testing |
| small | 1x H100 | Default, small models |
| medium | 2x H100 | Medium models |
| large | 4x H100 | Qwen3-4B |
| xlarge | 8x H100 | Large-scale training |

Example:
```bash
# Use 4x H100 for larger model
wf fairy2 -m qwen3_4b --mode w2 --scale large
```

## Cost Estimates

| Model | Scale | Est. Time | Est. Cost |
|-------|-------|-----------|-----------|
| SmolLM2-135M | small (1x H100) | ~2 hours | ~$6 |
| Qwen3-4B | large (4x H100) | ~8 hours | ~$100 |

*Costs vary by cloud provider and region*

## Prerequisites

1. **SkyPilot configured**: `sky check`
2. **Cloud credentials**: AWS/GCP/Nebius authenticated
3. **GCS access**: Service account with Storage Admin role
4. **W&B API key**: Set `WANDB_API_KEY` environment variable

## Monitoring

### W&B Dashboard
Training logs to Weights & Biases:
- Project: `wrinklefree-fairy2`
- Metrics: loss, learning rate, tokens processed

```bash
# Check W&B status
wf wandb-status -p wrinklefree-fairy2
```

### SkyPilot Dashboard
```bash
# List all jobs
wf runs

# View specific job logs
wf logs wf-fairy2-train -f  # -f for follow
```

## Troubleshooting

### Job Not Starting
```bash
# Check cloud quotas
sky check

# Check job status
sky jobs queue
```

### Checkpoint Not Uploading
```bash
# Verify GCS credentials
gcloud auth list
gsutil ls gs://wrinklefree-checkpoints/
```

### Out of Memory
```bash
# Reduce batch size
wf fairy2 -m smollm2_135m training.batch_size=4
```
