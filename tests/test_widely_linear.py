"""Tests for WidelyLinearComplex layer."""

import pytest
import torch
import torch.nn as nn

from fairy2.models.widely_linear import WidelyLinearComplex


class TestWidelyLinearComplex:
    """Tests for the WidelyLinearComplex layer."""

    def test_init(self):
        """Test basic initialization."""
        layer = WidelyLinearComplex(64, 128)
        assert layer.in_features == 64
        assert layer.out_features == 128
        assert layer.complex_in == 32
        assert layer.complex_out == 64

    def test_init_odd_features_raises(self):
        """Test that odd dimensions raise ValueError."""
        with pytest.raises(ValueError):
            WidelyLinearComplex(63, 128)
        with pytest.raises(ValueError):
            WidelyLinearComplex(64, 127)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        layer = WidelyLinearComplex(64, 128)
        x = torch.randn(2, 10, 64)
        y = layer(x)
        assert y.shape == (2, 10, 128)

    def test_from_real_linear(self, simple_linear):
        """Test conversion from nn.Linear."""
        wl_layer = WidelyLinearComplex.from_real_linear(simple_linear)
        assert wl_layer.in_features == simple_linear.in_features
        assert wl_layer.out_features == simple_linear.out_features

    def test_from_real_linear_preserves_output(self, simple_linear):
        """Test that conversion preserves output for real inputs."""
        wl_layer = WidelyLinearComplex.from_real_linear(simple_linear)

        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            y_original = simple_linear(x)
            y_widely = wl_layer(x)

        # Outputs should match (within floating point tolerance)
        assert torch.allclose(y_original, y_widely, rtol=1e-4, atol=1e-6)

    def test_to_real_linear_roundtrip(self, simple_linear):
        """Test conversion to widely-linear and back."""
        wl_layer = WidelyLinearComplex.from_real_linear(simple_linear)
        reconstructed = wl_layer.to_real_linear()

        # Weights should match
        assert torch.allclose(
            simple_linear.weight.data,
            reconstructed.weight.data,
            rtol=1e-4,
            atol=1e-6,
        )

    def test_with_bias(self):
        """Test layer with bias."""
        linear = nn.Linear(64, 128, bias=True)
        wl_layer = WidelyLinearComplex.from_real_linear(linear)

        assert wl_layer.bias_re is not None
        assert wl_layer.bias_im is not None

        x = torch.randn(2, 10, 64)
        y = wl_layer(x)
        assert y.shape == (2, 10, 128)
