# Fairy2i Architecture

This document explains the Fairy2i algorithm and how training works.

## Overview

Fairy2i is a **quantization method** for Large Language Models that represents weights using complex numbers quantized to the fourth roots of unity: {+1, -1, +i, -i}.

This is NOT training from scratch. It's **Quantization-Aware Training (QAT)** where we:
1. Take a pretrained model (e.g., SmolLM2-135M)
2. Convert its Linear layers to quantized Fairy2Linear layers
3. Continue training so the model adapts to work with quantized weights

## What Problem Does This Solve?

Normal LLM weights are 16-bit or 32-bit floating point numbers. This is expensive for:
- **Memory**: A 7B model needs ~14GB in float16
- **Inference**: Multiplying floats is slow

Fairy2i quantizes weights to just 4 values: {+1, -1, +i, -i}. Benefits:
- **Memory**: ~2 bits per weight (W2 mode) or ~1 bit (W1 mode)
- **Inference**: Multiplication becomes table lookup (no actual multiplication)

## The Three Core Components

### 1. Widely-Linear Complex Transformation

Real-valued Linear layers are converted to complex form:

```
Real Linear:     y = W·x
Complex Widely:  y = U·x + W·conj(x)
```

Where U and W are complex matrices derived from the original real weights:
```python
# Given real weight matrix R partitioned as [[R11, R12], [R21, R22]]:
Re(U) = 0.5 * (R11 + R22)
Im(U) = 0.5 * (R21 - R12)
Re(W) = 0.5 * (R11 - R22)
Im(W) = 0.5 * (R12 + R21)
```

### 2. Phase-Aware Quantization

Each complex weight is quantized to the nearest fourth root of unity:

```
Input:  w = a + bi  (any complex number)
Output: q ∈ {+1, -1, +i, -i}
```

Algorithm:
1. Compute phase angle: θ = atan2(b, a)
2. Find nearest quadrant: k = floor(2θ/π + 0.5) mod 4
3. Map to codebook: k=0→+1, k=1→+i, k=2→-1, k=3→-i

### 3. Recursive Residual Quantization

To improve accuracy, we quantize in multiple stages:

```
Stage 0: Quantize original weight → get q₀, scale s₀
Stage 1: Quantize residual (w - s₀·q₀) → get q₁, scale s₁
...
Final: w ≈ s₀·q₀ + s₁·q₁ + ...
```

- **W1 mode**: 1 stage (~1 bit per weight)
- **W2 mode**: 2 stages (~2 bits per weight, better quality)

## Training Flow

```
┌─────────────────────────────────────────────────────────┐
│                    QAT Training Loop                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Load pretrained model (SmolLM2-135M, Qwen3-4B)      │
│                         ↓                                │
│  2. Convert Linear → Fairy2Linear                        │
│     - Keep master weights in FP32                        │
│     - 210 layers converted for SmolLM2-135M             │
│                         ↓                                │
│  3. Forward pass:                                        │
│     - Quantize weights to {+1,-1,+i,-i}                 │
│     - Use quantized weights for computation             │
│                         ↓                                │
│  4. Backward pass:                                       │
│     - STE: gradients flow through as if no quantization │
│     - Update master weights in FP32                     │
│                         ↓                                │
│  5. Repeat for N steps                                   │
│                         ↓                                │
│  6. Save quantized checkpoint                            │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Straight-Through Estimator (STE)

Quantization is non-differentiable (discrete). STE solves this:

```python
# Forward: use quantized weights
w_quant = quantize(w)  # {+1,-1,+i,-i}

# Backward: pretend quantization didn't happen
# Gradient flows through directly to master weights
grad_w = grad_w_quant  # No modification
```

Implementation trick:
```python
w_quant = w + (quantize(w) - w).detach()
# Forward: w_quant = quantize(w)
# Backward: grad flows to w directly
```

## File Structure

```
src/fairy2/
├── models/
│   ├── widely_linear.py    # Real → Complex conversion
│   ├── fairy2_linear.py    # Full layer with quantization + STE
│   └── converter.py        # Convert HuggingFace models
├── quantization/
│   ├── phase_aware.py      # Phase-based quantization
│   ├── residual.py         # Multi-stage residual quantization
│   └── ste.py              # Straight-through estimator
└── training/
    ├── trainer.py          # QAT training loop
    └── loss.py             # Cross-entropy loss
```

## References

- Paper: [Fairy2i: Training Complex LLMs from Real LLMs](https://arxiv.org/abs/2512.02901)
- Related: BitNet, BitDistill
