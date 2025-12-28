"""Widely-Linear Complex transformation for Fairy2i.

This module implements the real-to-complex conversion that forms the foundation
of the Fairy2i algorithm. A real-valued linear layer is converted to an
equivalent widely-linear complex form.

Mathematical Background:
    A widely-linear transformation takes the form:
        y = U·x + W·conj(x)

    where U and W are complex matrices, x is the complex input, and conj(x)
    is the complex conjugate of x.

    Any even-dimensional real linear layer R can be losslessly converted to
    this form using the partition formulas derived in the Fairy2i paper.

Reference:
    Fairy2i: Training Complex LLMs from Real LLMs with All Parameters in {±1, ±i}
    https://arxiv.org/abs/2512.02901
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WidelyLinearComplex(nn.Module):
    """Widely-linear complex transformation layer.

    Converts a real-valued linear transformation to an equivalent complex
    widely-linear form: y = U·x + W·conj(x).

    The complex weights U and W are stored as pairs of real tensors for
    FSDP compatibility: (U_re, U_im) and (W_re, W_im).

    Args:
        in_features: Input dimension (must be even for complex interpretation)
        out_features: Output dimension (must be even for complex interpretation)
        bias: Whether to include bias (default: False)

    Attributes:
        U_re: Real part of holomorphic component U
        U_im: Imaginary part of holomorphic component U
        W_re: Real part of non-holomorphic component W
        W_im: Imaginary part of non-holomorphic component W

    Example:
        >>> layer = WidelyLinearComplex(128, 256)
        >>> x = torch.randn(2, 10, 128)  # Real input
        >>> y = layer(x)
        >>> y.shape
        torch.Size([2, 10, 256])

    Note:
        When the input x is real, the forward pass simplifies to:
        y = (U + W)·x since conj(x) = x for real x.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ):
        super().__init__()

        if in_features % 2 != 0:
            raise ValueError(f"in_features must be even, got {in_features}")
        if out_features % 2 != 0:
            raise ValueError(f"out_features must be even, got {out_features}")

        self.in_features = in_features
        self.out_features = out_features

        # Complex dimensions (half of real dimensions)
        self.complex_in = in_features // 2
        self.complex_out = out_features // 2

        # Store U and W as pairs of real tensors for FSDP compatibility
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
    def from_real_linear(cls, linear: nn.Linear) -> WidelyLinearComplex:
        """Create a WidelyLinearComplex from a real-valued nn.Linear.

        This conversion is lossless - the complex layer produces identical
        outputs to the original real layer when given real inputs.

        Conversion formulas (for R partitioned as [[R11, R12], [R21, R22]]):
            Re(U) = 0.5 * (R11 + R22)
            Im(U) = 0.5 * (R21 - R12)
            Re(W) = 0.5 * (R11 - R22)
            Im(W) = 0.5 * (R12 + R21)

        Args:
            linear: Source nn.Linear layer (must have even in/out features)

        Returns:
            WidelyLinearComplex: Equivalent complex widely-linear layer

        Raises:
            ValueError: If linear dimensions are not even
        """
        in_features = linear.in_features
        out_features = linear.out_features
        has_bias = linear.bias is not None

        if in_features % 2 != 0 or out_features % 2 != 0:
            raise ValueError(
                f"Linear layer dimensions must be even, got in={in_features}, out={out_features}"
            )

        # Create new layer
        layer = cls(in_features, out_features, bias=has_bias)

        # Get the real weight matrix and partition it
        # R is (out_features, in_features), partition into 2x2 blocks
        R = linear.weight.data.float()  # Use float32 for precision
        half_out = out_features // 2
        half_in = in_features // 2

        # Partition R into [[R11, R12], [R21, R22]]
        R11 = R[:half_out, :half_in]
        R12 = R[:half_out, half_in:]
        R21 = R[half_out:, :half_in]
        R22 = R[half_out:, half_in:]

        # Apply conversion formulas
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

    def to_real_linear(self) -> nn.Linear:
        """Convert back to a real-valued nn.Linear.

        This is the inverse of from_real_linear and is useful for validation.

        Inverse formulas:
            R11 = U_re + W_re
            R12 = W_im - U_im
            R21 = U_im + W_im
            R22 = U_re - W_re

        Returns:
            nn.Linear: Equivalent real-valued linear layer
        """
        has_bias = self.bias_re is not None
        linear = nn.Linear(self.in_features, self.out_features, bias=has_bias)

        # Reconstruct R from U and W
        R11 = self.U_re.data + self.W_re.data
        R12 = self.W_im.data - self.U_im.data
        R21 = self.U_im.data + self.W_im.data
        R22 = self.U_re.data - self.W_re.data

        # Assemble full weight matrix
        top = torch.cat([R11, R12], dim=1)
        bottom = torch.cat([R21, R22], dim=1)
        linear.weight.data = torch.cat([top, bottom], dim=0)

        # Reconstruct bias if present
        if has_bias:
            linear.bias.data = torch.cat([self.bias_re.data, self.bias_im.data])

        return linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with widely-linear complex transformation.

        For real input x, the transformation simplifies to:
            y = (U + W)·x

        The output is reconstructed as real values by interleaving
        the real and imaginary parts.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        # Cast weights to input dtype for mixed precision
        U_re = self.U_re.to(x.dtype)
        U_im = self.U_im.to(x.dtype)
        W_re = self.W_re.to(x.dtype)
        W_im = self.W_im.to(x.dtype)

        # Split input into "real" and "imaginary" halves
        # For a truly real input, these are just the first and second halves
        x_re = x[..., :self.complex_in]
        x_im = x[..., self.complex_in:]

        # Widely-linear transformation: y = U·x + W·conj(x)
        # where x = x_re + i*x_im and conj(x) = x_re - i*x_im
        #
        # Expanding:
        # y_re = U_re·x_re - U_im·x_im + W_re·x_re + W_im·x_im
        #      = (U_re + W_re)·x_re + (W_im - U_im)·x_im
        # y_im = U_im·x_re + U_re·x_im + W_im·x_re - W_re·x_im
        #      = (U_im + W_im)·x_re + (U_re - W_re)·x_im

        y_re = F.linear(x_re, U_re + W_re) + F.linear(x_im, W_im - U_im)
        y_im = F.linear(x_re, U_im + W_im) + F.linear(x_im, U_re - W_re)

        # Add bias if present
        if self.bias_re is not None:
            y_re = y_re + self.bias_re.to(x.dtype)
            y_im = y_im + self.bias_im.to(x.dtype)

        # Concatenate real and imaginary parts to form output
        return torch.cat([y_re, y_im], dim=-1)

    def extra_repr(self) -> str:
        """String representation for printing."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"complex_in={self.complex_in}, complex_out={self.complex_out}, "
            f"bias={self.bias_re is not None}"
        )
