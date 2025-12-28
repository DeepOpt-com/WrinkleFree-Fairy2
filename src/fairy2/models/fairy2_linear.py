"""Fairy2Linear: Full Fairy2 quantized linear layer.

This module implements the complete Fairy2 quantized linear layer that
combines:
1. Widely-linear complex representation
2. Phase-aware quantization to {+1, -1, +i, -i}
3. Recursive residual quantization (for W2 mode)
4. Straight-through estimator for QAT training

The layer maintains master weights in full precision (FP32) for training
stability, and applies quantization during the forward pass with STE for
gradient flow.

Reference:
    Fairy2i: Training Complex LLMs from Real LLMs with All Parameters in {±1, ±i}
    https://arxiv.org/abs/2512.02901
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairy2.quantization.residual import ResidualQuantizer
from fairy2.quantization.ste import complex_detach_ste


class Fairy2Linear(nn.Module):
    """Fairy2 quantized linear layer.

    This layer combines:
    - Widely-linear complex representation (U·x + W·conj(x))
    - Phase-aware quantization to fourth roots of unity {+1, -1, +i, -i}
    - Recursive residual quantization for multi-stage precision (W1/W2/W3)
    - Straight-through estimator (STE) for gradient flow

    The complex weights U and W are stored as pairs of real tensors
    (U_re, U_im, W_re, W_im) for FSDP compatibility.

    Args:
        in_features: Input dimension (must be even)
        out_features: Output dimension (must be even)
        num_stages: Number of residual quantization stages
            - 1 = W1 mode (~1 bit per weight)
            - 2 = W2 mode (~2 bits per weight)
        bias: Whether to include bias (default: False)

    Attributes:
        U_re, U_im: Master weights for holomorphic component U
        W_re, W_im: Master weights for non-holomorphic component W
        num_stages: Number of quantization stages

    Example:
        >>> layer = Fairy2Linear(128, 256, num_stages=2)
        >>> x = torch.randn(2, 10, 128)
        >>> y = layer(x)
        >>> y.shape
        torch.Size([2, 10, 256])

    Training:
        During training, the layer uses STE to enable gradient flow through
        the discrete quantization operation. Master weights are maintained
        in FP32 for numerical stability.

    Inference:
        During inference (eval mode), quantized weights can be cached for
        multiplication-free computation using table lookup.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_stages: int = 2,
        bias: bool = False,
    ):
        super().__init__()

        if in_features % 2 != 0:
            raise ValueError(f"in_features must be even, got {in_features}")
        if out_features % 2 != 0:
            raise ValueError(f"out_features must be even, got {out_features}")

        self.in_features = in_features
        self.out_features = out_features
        self.num_stages = num_stages

        # Complex dimensions
        self.complex_in = in_features // 2
        self.complex_out = out_features // 2

        # Master weights stored as real tensors (for FSDP)
        # U is the holomorphic component, W is the non-holomorphic component
        self.U_re = nn.Parameter(torch.zeros(self.complex_out, self.complex_in))
        self.U_im = nn.Parameter(torch.zeros(self.complex_out, self.complex_in))
        self.W_re = nn.Parameter(torch.zeros(self.complex_out, self.complex_in))
        self.W_im = nn.Parameter(torch.zeros(self.complex_out, self.complex_in))

        if bias:
            self.bias_re = nn.Parameter(torch.zeros(self.complex_out))
            self.bias_im = nn.Parameter(torch.zeros(self.complex_out))
        else:
            self.register_parameter("bias_re", None)
            self.register_parameter("bias_im", None)

        # Residual quantizer for W2/W3 modes
        self.quantizer = ResidualQuantizer(num_stages)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters using Xavier uniform."""
        nn.init.xavier_uniform_(self.U_re)
        nn.init.xavier_uniform_(self.U_im)
        nn.init.xavier_uniform_(self.W_re)
        nn.init.xavier_uniform_(self.W_im)

        if self.bias_re is not None:
            nn.init.zeros_(self.bias_re)
            nn.init.zeros_(self.bias_im)

    @classmethod
    def from_real_linear(
        cls,
        linear: nn.Linear,
        num_stages: int = 2,
    ) -> Fairy2Linear:
        """Create a Fairy2Linear from a real-valued nn.Linear.

        This conversion is lossless before quantization - the complex layer
        produces identical outputs to the original real layer when given real
        inputs (before any quantization is applied).

        Args:
            linear: Source nn.Linear layer (must have even in/out features)
            num_stages: Number of quantization stages (1=W1, 2=W2)

        Returns:
            Fairy2Linear: Equivalent Fairy2 quantized layer
        """
        in_features = linear.in_features
        out_features = linear.out_features
        has_bias = linear.bias is not None

        if in_features % 2 != 0 or out_features % 2 != 0:
            raise ValueError(
                f"Linear dimensions must be even, got in={in_features}, out={out_features}"
            )

        # Create new layer
        layer = cls(in_features, out_features, num_stages=num_stages, bias=has_bias)

        # Get the real weight matrix and partition it
        # Use float32 for precision during conversion
        R = linear.weight.data.float()
        half_out = out_features // 2
        half_in = in_features // 2

        # Partition R into [[R11, R12], [R21, R22]]
        R11 = R[:half_out, :half_in]
        R12 = R[:half_out, half_in:]
        R21 = R[half_out:, :half_in]
        R22 = R[half_out:, half_in:]

        # Apply conversion formulas (from Fairy2i paper)
        layer.U_re.data = (0.5 * (R11 + R22)).to(linear.weight.dtype)
        layer.U_im.data = (0.5 * (R21 - R12)).to(linear.weight.dtype)
        layer.W_re.data = (0.5 * (R11 - R22)).to(linear.weight.dtype)
        layer.W_im.data = (0.5 * (R12 + R21)).to(linear.weight.dtype)

        # Convert bias if present
        if has_bias:
            bias = linear.bias.data
            layer.bias_re.data = bias[:half_out]
            layer.bias_im.data = bias[half_out:]

        return layer

    def _quantize_weights(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize U and W using residual quantization.

        Returns:
            Tuple of (U_re_q, U_im_q, W_re_q, W_im_q) quantized weights
        """
        # Quantize U
        U_stages = self.quantizer.quantize(self.U_re, self.U_im)
        U_re_q, U_im_q = self.quantizer.dequantize(U_stages)

        # Quantize W
        W_stages = self.quantizer.quantize(self.W_re, self.W_im)
        W_re_q, W_im_q = self.quantizer.dequantize(W_stages)

        return U_re_q, U_im_q, W_re_q, W_im_q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized widely-linear transformation.

        Training:
            Uses STE for gradient flow through quantization.
            w_out = w + (quantize(w) - w).detach()

        Inference:
            Uses quantized weights directly (can be optimized with table lookup).

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        # Quantize weights
        U_re_q, U_im_q, W_re_q, W_im_q = self._quantize_weights()

        # Apply STE for gradient flow during training
        if self.training:
            U_re_q, U_im_q = complex_detach_ste(self.U_re, self.U_im, U_re_q, U_im_q)
            W_re_q, W_im_q = complex_detach_ste(self.W_re, self.W_im, W_re_q, W_im_q)

        # Cast to input dtype for mixed precision
        U_re = U_re_q.to(x.dtype)
        U_im = U_im_q.to(x.dtype)
        W_re = W_re_q.to(x.dtype)
        W_im = W_im_q.to(x.dtype)

        # Split input into "real" and "imaginary" halves
        x_re = x[..., :self.complex_in]
        x_im = x[..., self.complex_in:]

        # Widely-linear transformation: y = U·x + W·conj(x)
        # Expanded (see widely_linear.py for derivation):
        # y_re = (U_re + W_re)·x_re + (W_im - U_im)·x_im
        # y_im = (U_im + W_im)·x_re + (U_re - W_re)·x_im

        y_re = F.linear(x_re, U_re + W_re) + F.linear(x_im, W_im - U_im)
        y_im = F.linear(x_re, U_im + W_im) + F.linear(x_im, U_re - W_re)

        # Add bias if present
        if self.bias_re is not None:
            y_re = y_re + self.bias_re.to(x.dtype)
            y_im = y_im + self.bias_im.to(x.dtype)

        # Concatenate to form output
        return torch.cat([y_re, y_im], dim=-1)

    def get_quantized_weights(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get quantized weights for inspection or export.

        Returns:
            Tuple of (U_re_q, U_im_q, W_re_q, W_im_q)
        """
        with torch.no_grad():
            return self._quantize_weights()

    def quantization_error(self) -> dict[str, float]:
        """Compute quantization error for debugging.

        Returns:
            Dict with MSE for U and W components
        """
        U_re_q, U_im_q, W_re_q, W_im_q = self._quantize_weights()

        with torch.no_grad():
            U_mse = (
                (self.U_re - U_re_q).pow(2).mean() +
                (self.U_im - U_im_q).pow(2).mean()
            ).item()

            W_mse = (
                (self.W_re - W_re_q).pow(2).mean() +
                (self.W_im - W_im_q).pow(2).mean()
            ).item()

        return {"U_mse": U_mse, "W_mse": W_mse}

    def extra_repr(self) -> str:
        """String representation for printing."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"num_stages={self.num_stages}, bias={self.bias_re is not None}"
        )
