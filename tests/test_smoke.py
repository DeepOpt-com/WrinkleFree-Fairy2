"""Smoke tests for Fairy2i.

These tests are designed to be fast and catch basic issues.
They can be run frequently during development.
"""

import pytest
import torch

from fairy2.models.fairy2_linear import Fairy2Linear
from fairy2.models.widely_linear import WidelyLinearComplex
from fairy2.quantization.phase_aware import phase_aware_quantize
from fairy2.quantization.residual import ResidualQuantizer


@pytest.mark.smoke
class TestSmokeTests:
    """Quick smoke tests for CI/CD."""

    def test_widely_linear_forward(self):
        """WidelyLinearComplex forward pass works."""
        layer = WidelyLinearComplex(64, 128)
        x = torch.randn(2, 10, 64)
        y = layer(x)
        assert y.shape == (2, 10, 128)

    def test_fairy2_linear_forward(self):
        """Fairy2Linear forward pass works."""
        layer = Fairy2Linear(64, 128, num_stages=2)
        x = torch.randn(2, 10, 64)
        y = layer(x)
        assert y.shape == (2, 10, 128)

    def test_phase_quantization_basic(self):
        """Phase-aware quantization produces valid outputs."""
        w_re = torch.randn(10, 10)
        w_im = torch.randn(10, 10)
        (q_re, q_im), (s_re, s_im) = phase_aware_quantize(w_re, w_im)

        # Check quantized values are in {-1, 0, 1}
        assert set(q_re.unique().tolist()).issubset({-1.0, 0.0, 1.0})
        assert set(q_im.unique().tolist()).issubset({-1.0, 0.0, 1.0})

        # Check scales are positive
        assert s_re > 0
        assert s_im > 0

    def test_residual_quantizer_stages(self):
        """Residual quantizer produces correct number of stages."""
        quantizer = ResidualQuantizer(num_stages=2)
        w_re = torch.randn(10, 10)
        w_im = torch.randn(10, 10)

        stages = quantizer.quantize(w_re, w_im)
        assert len(stages) == 2

    def test_fairy2_linear_from_linear(self):
        """Fairy2Linear.from_real_linear works."""
        import torch.nn as nn

        linear = nn.Linear(64, 128, bias=False)
        fairy2 = Fairy2Linear.from_real_linear(linear, num_stages=2)

        assert fairy2.in_features == 64
        assert fairy2.out_features == 128
        assert fairy2.num_stages == 2

    def test_fairy2_linear_backward(self):
        """Fairy2Linear backward pass works with STE."""
        layer = Fairy2Linear(64, 128, num_stages=2)
        x = torch.randn(2, 10, 64, requires_grad=True)

        y = layer(x)
        loss = y.sum()
        loss.backward()

        # Gradients should exist
        assert x.grad is not None
        assert layer.U_re.grad is not None

    def test_import_main_api(self):
        """Main API imports work."""
        from fairy2 import (
            Fairy2Linear,
            WidelyLinearComplex,
            convert_to_fairy2,
            phase_aware_quantize,
            ResidualQuantizer,
        )

        assert Fairy2Linear is not None
        assert WidelyLinearComplex is not None
        assert convert_to_fairy2 is not None
        assert phase_aware_quantize is not None
        assert ResidualQuantizer is not None
