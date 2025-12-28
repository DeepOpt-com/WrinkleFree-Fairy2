"""Phase-aware quantization to fourth roots of unity.

This module implements the phase-aware quantization algorithm from Fairy2i,
which projects complex weights to the nearest element of {+1, -1, +i, -i}.

The key insight is that these four points (the fourth roots of unity) on the
complex unit circle provide optimal 2-bit encoding with full symmetry.

Mathematical Details:
    Given a complex weight w = |w| * e^(iθ), we quantize to:
        q = i^k where k = ⌊2θ/π + 0.5⌋ mod 4

    This creates decision boundaries at ±π/4 and ±3π/4, which are the
    optimal Voronoi regions for the codebook {1, i, -1, -i}.

Reference:
    Fairy2i: Training Complex LLMs from Real LLMs with All Parameters in {±1, ±i}
    https://arxiv.org/abs/2512.02901
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


def phase_aware_quantize(
    w_re: torch.Tensor,
    w_im: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Quantize complex weights to nearest fourth root of unity.

    Projects each complex weight w = w_re + i*w_im to the nearest element
    of the codebook {+1, -1, +i, -i}, with axis-wise scaling factors.

    Args:
        w_re: Real part of complex weights
        w_im: Imaginary part of complex weights
        eps: Small constant for numerical stability

    Returns:
        Tuple of:
            - (q_re, q_im): Quantized weights (each element in {-1, 0, 1})
              Note: Each weight has either q_re != 0 OR q_im != 0, not both
            - (s_re, s_im): Per-tensor axis-wise scaling factors

    Example:
        >>> w_re = torch.tensor([1.0, 0.0, -1.0, 0.0])
        >>> w_im = torch.tensor([0.0, 1.0, 0.0, -1.0])
        >>> (q_re, q_im), (s_re, s_im) = phase_aware_quantize(w_re, w_im)
        >>> q_re  # [1, 0, -1, 0]
        >>> q_im  # [0, 1, 0, -1]

    Note:
        For zero-magnitude weights (|w| < eps), defaults to +1 to avoid NaN.
    """
    # Compute phase angle: θ = atan2(im, re)
    theta = torch.atan2(w_im, w_re)

    # Quantize to nearest i^k using decision boundaries at ±π/4, ±3π/4
    # k = floor(2θ/π + 0.5) mod 4
    k = torch.floor(2 * theta / math.pi + 0.5).long() % 4

    # Map k to codebook:
    # k=0 → +1 (θ ∈ [-π/4, π/4))
    # k=1 → +i (θ ∈ [π/4, 3π/4))
    # k=2 → -1 (θ ∈ [3π/4, -3π/4) or equivalently [-π, -3π/4) ∪ [3π/4, π])
    # k=3 → -i (θ ∈ [-3π/4, -π/4))

    # Create output tensors
    # For +1 and -1: q_re = ±1, q_im = 0
    # For +i and -i: q_re = 0, q_im = ±1
    q_re = torch.zeros_like(w_re)
    q_im = torch.zeros_like(w_im)

    # k=0 → +1
    mask_0 = k == 0
    q_re = torch.where(mask_0, torch.ones_like(w_re), q_re)

    # k=1 → +i
    mask_1 = k == 1
    q_im = torch.where(mask_1, torch.ones_like(w_im), q_im)

    # k=2 → -1
    mask_2 = k == 2
    q_re = torch.where(mask_2, -torch.ones_like(w_re), q_re)

    # k=3 → -i
    mask_3 = k == 3
    q_im = torch.where(mask_3, -torch.ones_like(w_im), q_im)

    # Handle edge case: zero magnitude weights
    # For |w| ≈ 0, the phase is undefined, so we default to +1
    magnitude = torch.sqrt(w_re ** 2 + w_im ** 2)
    zero_mask = magnitude < eps
    q_re = torch.where(zero_mask, torch.ones_like(q_re), q_re)
    q_im = torch.where(zero_mask, torch.zeros_like(q_im), q_im)

    # Compute axis-wise scaling factors
    # s_re = mean(|w_re|) for weights that map to ±1
    # s_im = mean(|w_im|) for weights that map to ±i
    real_mask = mask_0 | mask_2  # Weights that quantize to ±1
    imag_mask = mask_1 | mask_3  # Weights that quantize to ±i

    # Compute scales (with fallback for empty masks)
    if real_mask.any():
        s_re = w_re.abs()[real_mask].mean()
    else:
        s_re = w_re.abs().mean()

    if imag_mask.any():
        s_im = w_im.abs()[imag_mask].mean()
    else:
        s_im = w_im.abs().mean()

    # Ensure scales are not zero
    s_re = torch.clamp(s_re, min=eps)
    s_im = torch.clamp(s_im, min=eps)

    return (q_re, q_im), (s_re, s_im)


class PhaseAwareSTE(torch.autograd.Function):
    """Straight-Through Estimator for phase-aware quantization.

    This autograd function enables gradient flow through the discrete
    quantization operation by using the straight-through estimator:
    during the forward pass, quantization is applied normally, but
    during the backward pass, gradients pass through unchanged.

    Usage:
        >>> w_re = torch.randn(10, 10, requires_grad=True)
        >>> w_im = torch.randn(10, 10, requires_grad=True)
        >>> q_re, q_im, s_re, s_im = PhaseAwareSTE.apply(w_re, w_im)
    """

    @staticmethod
    def forward(
        ctx,
        w_re: torch.Tensor,
        w_im: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with quantization.

        Args:
            ctx: Context for backward pass
            w_re: Real part of weights
            w_im: Imaginary part of weights

        Returns:
            Tuple of (q_re, q_im, s_re, s_im)
        """
        (q_re, q_im), (s_re, s_im) = phase_aware_quantize(w_re, w_im)

        # Save for potential use in backward (not strictly needed for STE)
        ctx.save_for_backward(w_re, w_im)

        return q_re, q_im, s_re, s_im

    @staticmethod
    def backward(
        ctx,
        grad_q_re: torch.Tensor,
        grad_q_im: torch.Tensor,
        grad_s_re: torch.Tensor,
        grad_s_im: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward pass with straight-through estimation.

        Gradients for q_re and q_im pass through unchanged to w_re and w_im.
        Scale gradients are accumulated appropriately.

        Args:
            ctx: Context from forward pass
            grad_q_re: Gradient w.r.t. quantized real part
            grad_q_im: Gradient w.r.t. quantized imaginary part
            grad_s_re: Gradient w.r.t. real scale
            grad_s_im: Gradient w.r.t. imaginary scale

        Returns:
            Tuple of gradients for (w_re, w_im)
        """
        # STE: Pass gradients through unchanged
        # This is the "detach trick" used in many quantization methods
        return grad_q_re, grad_q_im


def quantize_with_ste(
    w_re: torch.Tensor,
    w_im: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convenience function for phase-aware quantization with STE.

    This combines the quantization operation with the straight-through
    estimator for use in training.

    Args:
        w_re: Real part of weights (requires_grad=True for training)
        w_im: Imaginary part of weights (requires_grad=True for training)

    Returns:
        Tuple of (q_re, q_im, s_re, s_im) where:
            - q_re, q_im: Quantized weights
            - s_re, s_im: Scaling factors

    Example:
        >>> w_re = torch.randn(10, 10, requires_grad=True)
        >>> w_im = torch.randn(10, 10, requires_grad=True)
        >>> q_re, q_im, s_re, s_im = quantize_with_ste(w_re, w_im)
        >>> # Gradients will flow through during backward pass
    """
    return PhaseAwareSTE.apply(w_re, w_im)
