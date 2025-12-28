"""Straight-Through Estimator (STE) utilities for complex quantization.

This module provides general-purpose STE implementations for complex-valued
quantization operations used in Fairy2i training.

The straight-through estimator is a technique for enabling gradient-based
optimization through discrete operations: during forward pass, the discrete
operation is applied normally, but during backward pass, gradients pass
through unchanged (as if the operation were the identity).

Reference:
    Bengio, Y., Léonard, N., & Courville, A. (2013).
    Estimating or propagating gradients through stochastic neurons for
    conditional computation.
"""

from __future__ import annotations

import torch


class ComplexSTE(torch.autograd.Function):
    """General Straight-Through Estimator for complex weights.

    This autograd function provides STE behavior for any operation on
    complex weights represented as (real, imaginary) tensor pairs.

    The pattern is:
        w_quant = w + (quant(w) - w).detach()

    This makes w_quant equal to quant(w) in forward, but gradients
    flow as if w_quant = w.

    Usage:
        >>> w_re = torch.randn(10, 10, requires_grad=True)
        >>> w_im = torch.randn(10, 10, requires_grad=True)
        >>> w_re_q, w_im_q = my_quantize_fn(w_re, w_im)
        >>> # Apply STE
        >>> w_re_out, w_im_out = ComplexSTE.apply(w_re, w_im, w_re_q, w_im_q)
    """

    @staticmethod
    def forward(
        ctx,
        w_re: torch.Tensor,
        w_im: torch.Tensor,
        w_re_quant: torch.Tensor,
        w_im_quant: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: return quantized weights.

        Args:
            ctx: Autograd context
            w_re: Original real part (master weights)
            w_im: Original imaginary part (master weights)
            w_re_quant: Quantized real part
            w_im_quant: Quantized imaginary part

        Returns:
            Tuple of (w_re_quant, w_im_quant)
        """
        ctx.save_for_backward(w_re, w_im)
        return w_re_quant, w_im_quant

    @staticmethod
    def backward(
        ctx,
        grad_re: torch.Tensor,
        grad_im: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        """Backward pass: pass gradients through unchanged.

        Args:
            ctx: Autograd context
            grad_re: Gradient w.r.t. quantized real part
            grad_im: Gradient w.r.t. quantized imaginary part

        Returns:
            Tuple of gradients for (w_re, w_im, None, None)
            The None values are for the quantized inputs which don't need gradients.
        """
        # STE: gradients pass through unchanged
        return grad_re, grad_im, None, None


def apply_complex_ste(
    w_re: torch.Tensor,
    w_im: torch.Tensor,
    quantize_fn,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply STE to any complex quantization function.

    Convenience function that wraps any quantization function with STE
    behavior for gradient flow.

    Args:
        w_re: Real part of master weights
        w_im: Imaginary part of master weights
        quantize_fn: Function that takes (w_re, w_im, **kwargs) and returns
                     (w_re_quant, w_im_quant)
        **kwargs: Additional arguments passed to quantize_fn

    Returns:
        Tuple of (w_re_quant, w_im_quant) with STE gradient behavior

    Example:
        >>> def my_quant(w_re, w_im, threshold=0.5):
        ...     q_re = torch.round(w_re / threshold) * threshold
        ...     q_im = torch.round(w_im / threshold) * threshold
        ...     return q_re, q_im
        >>>
        >>> w_re_q, w_im_q = apply_complex_ste(w_re, w_im, my_quant, threshold=0.5)
    """
    w_re_quant, w_im_quant = quantize_fn(w_re, w_im, **kwargs)
    return ComplexSTE.apply(w_re, w_im, w_re_quant, w_im_quant)


def detach_ste(
    w: torch.Tensor,
    w_quant: torch.Tensor,
) -> torch.Tensor:
    """Simple STE using the detach trick.

    The classic pattern for STE:
        w_out = w + (w_quant - w).detach()

    This makes w_out = w_quant in forward pass, but gradients
    flow to w in backward pass.

    Args:
        w: Master weights (requires grad)
        w_quant: Quantized weights (can be detached)

    Returns:
        Tensor that equals w_quant but has gradients flowing to w
    """
    return w + (w_quant - w).detach()


def complex_detach_ste(
    w_re: torch.Tensor,
    w_im: torch.Tensor,
    w_re_quant: torch.Tensor,
    w_im_quant: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """STE for complex weights using the detach trick.

    Args:
        w_re: Real master weights
        w_im: Imaginary master weights
        w_re_quant: Quantized real part
        w_im_quant: Quantized imaginary part

    Returns:
        Tuple of tensors that equal (w_re_quant, w_im_quant) in forward
        but have gradients flowing to (w_re, w_im) in backward.
    """
    out_re = w_re + (w_re_quant - w_re).detach()
    out_im = w_im + (w_im_quant - w_im).detach()
    return out_re, out_im
