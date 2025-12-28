"""Recursive residual quantization for multi-stage precision.

This module implements the recursive residual quantization algorithm from
Fairy2i, which enables higher precision (W2 mode) by quantizing the residual
error across multiple stages.

Key Insight:
    Instead of a single quantization step, we iteratively quantize the
    residual (error) from the previous stage:

    Stage 0: q₀ = quant(w), residual₁ = w - s₀·q₀
    Stage 1: q₁ = quant(residual₁), residual₂ = residual₁ - s₁·q₁
    ...

    Final approximation: w ≈ Σ sₜ·qₜ

    Each stage adds approximately 1 bit of precision.

Reference:
    Fairy2i: Training Complex LLMs from Real LLMs with All Parameters in {±1, ±i}
    https://arxiv.org/abs/2512.02901
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

from fairy2.quantization.phase_aware import phase_aware_quantize


@dataclass
class QuantizationStage:
    """Container for a single quantization stage's outputs.

    Attributes:
        q_re: Quantized real component (values in {-1, 0, 1})
        q_im: Quantized imaginary component (values in {-1, 0, 1})
        s_re: Real axis scaling factor
        s_im: Imaginary axis scaling factor
    """

    q_re: torch.Tensor
    q_im: torch.Tensor
    s_re: torch.Tensor
    s_im: torch.Tensor

    def to_tuple(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert to tuple format."""
        return (self.q_re, self.q_im, self.s_re, self.s_im)


class ResidualQuantizer:
    """Recursive residual quantizer for multi-stage precision.

    This class implements the recursive residual quantization algorithm
    that enables W2 (2-bit) and higher precision modes.

    Args:
        num_stages: Number of quantization stages
            - W1 = 1 stage (~1 bit per weight)
            - W2 = 2 stages (~2 bits per weight)
            - W3 = 3 stages (~3 bits per weight)

    Example:
        >>> quantizer = ResidualQuantizer(num_stages=2)  # W2 mode
        >>> w_re = torch.randn(10, 10)
        >>> w_im = torch.randn(10, 10)
        >>> stages = quantizer.quantize(w_re, w_im)
        >>> len(stages)
        2
        >>> # Reconstruct
        >>> w_re_approx, w_im_approx = quantizer.dequantize(stages)
    """

    def __init__(self, num_stages: int = 2):
        if num_stages < 1:
            raise ValueError(f"num_stages must be >= 1, got {num_stages}")
        self.num_stages = num_stages

    def quantize(
        self,
        w_re: torch.Tensor,
        w_im: torch.Tensor,
    ) -> List[QuantizationStage]:
        """Perform recursive residual quantization.

        Quantizes the input weights across multiple stages, where each
        subsequent stage quantizes the residual error from the previous.

        IMPORTANT: Each stage quantizes the RESIDUAL, not the original weight.

        Args:
            w_re: Real part of complex weights
            w_im: Imaginary part of complex weights

        Returns:
            List of QuantizationStage objects, one per stage.
            The weight can be reconstructed as:
                w ≈ Σ (stages[t].s_re * stages[t].q_re + i * stages[t].s_im * stages[t].q_im)
        """
        stages: List[QuantizationStage] = []

        # Initialize residual with the original weights
        residual_re = w_re.clone()
        residual_im = w_im.clone()

        for t in range(self.num_stages):
            # Quantize the current residual (not the original!)
            (q_re, q_im), (s_re, s_im) = phase_aware_quantize(residual_re, residual_im)

            # Store this stage's results
            stages.append(QuantizationStage(
                q_re=q_re,
                q_im=q_im,
                s_re=s_re,
                s_im=s_im,
            ))

            # Update residual: r_{t+1} = r_t - s_t * q_t
            # This is the key insight: quantize the error, not the original
            residual_re = residual_re - s_re * q_re
            residual_im = residual_im - s_im * q_im

        return stages

    def dequantize(
        self,
        stages: List[QuantizationStage],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct weights from quantized stages.

        Combines all stages to approximate the original weights:
            w ≈ Σ (s_re^t * q_re^t + i * s_im^t * q_im^t)

        Args:
            stages: List of QuantizationStage objects from quantize()

        Returns:
            Tuple of (w_re_approx, w_im_approx)
        """
        if not stages:
            raise ValueError("stages list cannot be empty")

        # Initialize with zeros matching the shape of the first stage
        w_re = torch.zeros_like(stages[0].q_re)
        w_im = torch.zeros_like(stages[0].q_im)

        # Accumulate contributions from each stage
        for stage in stages:
            w_re = w_re + stage.s_re * stage.q_re
            w_im = w_im + stage.s_im * stage.q_im

        return w_re, w_im

    def quantization_error(
        self,
        w_re: torch.Tensor,
        w_im: torch.Tensor,
        stages: List[QuantizationStage],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the quantization error.

        Returns the difference between the original and reconstructed weights.

        Args:
            w_re: Original real part
            w_im: Original imaginary part
            stages: Quantization stages from quantize()

        Returns:
            Tuple of (error_re, error_im)
        """
        w_re_approx, w_im_approx = self.dequantize(stages)
        return w_re - w_re_approx, w_im - w_im_approx

    def mse(
        self,
        w_re: torch.Tensor,
        w_im: torch.Tensor,
        stages: List[QuantizationStage],
    ) -> torch.Tensor:
        """Compute mean squared error of quantization.

        Args:
            w_re: Original real part
            w_im: Original imaginary part
            stages: Quantization stages from quantize()

        Returns:
            Scalar MSE tensor
        """
        err_re, err_im = self.quantization_error(w_re, w_im, stages)
        return (err_re ** 2 + err_im ** 2).mean()


class ResidualQuantizerSTE(torch.autograd.Function):
    """Straight-through estimator for residual quantization.

    Enables gradient flow through the multi-stage quantization operation
    for use in quantization-aware training.
    """

    @staticmethod
    def forward(
        ctx,
        w_re: torch.Tensor,
        w_im: torch.Tensor,
        num_stages: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with residual quantization.

        Args:
            ctx: Autograd context
            w_re: Real part of weights
            w_im: Imaginary part of weights
            num_stages: Number of quantization stages

        Returns:
            Tuple of (w_re_quant, w_im_quant) - the dequantized approximation
        """
        quantizer = ResidualQuantizer(num_stages)
        stages = quantizer.quantize(w_re, w_im)
        w_re_quant, w_im_quant = quantizer.dequantize(stages)

        ctx.save_for_backward(w_re, w_im)
        return w_re_quant, w_im_quant

    @staticmethod
    def backward(
        ctx,
        grad_w_re: torch.Tensor,
        grad_w_im: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        """Backward pass with straight-through estimation.

        Gradients pass through unchanged (STE).

        Args:
            ctx: Autograd context
            grad_w_re: Gradient w.r.t. quantized real part
            grad_w_im: Gradient w.r.t. quantized imaginary part

        Returns:
            Tuple of gradients for (w_re, w_im, num_stages)
        """
        return grad_w_re, grad_w_im, None


def residual_quantize_with_ste(
    w_re: torch.Tensor,
    w_im: torch.Tensor,
    num_stages: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convenience function for residual quantization with STE.

    Performs multi-stage residual quantization and returns the dequantized
    approximation, with gradients flowing through via STE.

    Args:
        w_re: Real part of weights (requires_grad=True for training)
        w_im: Imaginary part of weights (requires_grad=True for training)
        num_stages: Number of quantization stages (1=W1, 2=W2)

    Returns:
        Tuple of (w_re_quant, w_im_quant)

    Example:
        >>> w_re = torch.randn(10, 10, requires_grad=True)
        >>> w_im = torch.randn(10, 10, requires_grad=True)
        >>> w_re_q, w_im_q = residual_quantize_with_ste(w_re, w_im, num_stages=2)
    """
    return ResidualQuantizerSTE.apply(w_re, w_im, num_stages)
