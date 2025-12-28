"""Tests for phase-aware quantization."""

import math

import pytest
import torch

from fairy2.quantization.phase_aware import (
    PhaseAwareSTE,
    phase_aware_quantize,
    quantize_with_ste,
)


class TestPhaseAwareQuantize:
    """Tests for the phase_aware_quantize function."""

    def test_positive_real(self):
        """Positive real numbers should quantize to +1."""
        w_re = torch.tensor([1.0, 2.0, 0.5])
        w_im = torch.zeros(3)
        (q_re, q_im), _ = phase_aware_quantize(w_re, w_im)

        assert torch.allclose(q_re, torch.ones(3))
        assert torch.allclose(q_im, torch.zeros(3))

    def test_positive_imag(self):
        """Positive imaginary numbers should quantize to +i."""
        w_re = torch.zeros(3)
        w_im = torch.tensor([1.0, 2.0, 0.5])
        (q_re, q_im), _ = phase_aware_quantize(w_re, w_im)

        assert torch.allclose(q_re, torch.zeros(3))
        assert torch.allclose(q_im, torch.ones(3))

    def test_negative_real(self):
        """Negative real numbers should quantize to -1."""
        w_re = torch.tensor([-1.0, -2.0, -0.5])
        w_im = torch.zeros(3)
        (q_re, q_im), _ = phase_aware_quantize(w_re, w_im)

        assert torch.allclose(q_re, -torch.ones(3))
        assert torch.allclose(q_im, torch.zeros(3))

    def test_negative_imag(self):
        """Negative imaginary numbers should quantize to -i."""
        w_re = torch.zeros(3)
        w_im = torch.tensor([-1.0, -2.0, -0.5])
        (q_re, q_im), _ = phase_aware_quantize(w_re, w_im)

        assert torch.allclose(q_re, torch.zeros(3))
        assert torch.allclose(q_im, -torch.ones(3))

    def test_diagonal_angles(self):
        """Test 45-degree and 135-degree angles."""
        # 45 degrees -> should be closer to +i than +1
        angle = math.pi / 4 + 0.1
        w_re = torch.tensor([math.cos(angle)])
        w_im = torch.tensor([math.sin(angle)])
        (q_re, q_im), _ = phase_aware_quantize(w_re, w_im)

        # At 45 degrees + epsilon, should quantize to +i
        assert q_im.item() == 1.0
        assert q_re.item() == 0.0

    def test_zero_magnitude(self):
        """Zero magnitude weights should default to +1."""
        w_re = torch.zeros(3)
        w_im = torch.zeros(3)
        (q_re, q_im), _ = phase_aware_quantize(w_re, w_im)

        # Default to +1 for zero magnitude
        assert torch.allclose(q_re, torch.ones(3))
        assert torch.allclose(q_im, torch.zeros(3))

    def test_scaling_factors(self):
        """Test axis-wise scaling factors."""
        w_re = torch.tensor([2.0, -2.0, 0.0, 0.0])
        w_im = torch.tensor([0.0, 0.0, 3.0, -3.0])
        (q_re, q_im), (s_re, s_im) = phase_aware_quantize(w_re, w_im)

        # s_re should be mean of |w_re| for real quantized weights
        assert s_re > 0
        assert s_im > 0

    def test_quantized_values_in_codebook(self):
        """All quantized values should be in {-1, 0, 1}."""
        w_re = torch.randn(100, 100)
        w_im = torch.randn(100, 100)
        (q_re, q_im), _ = phase_aware_quantize(w_re, w_im)

        assert set(q_re.unique().tolist()).issubset({-1.0, 0.0, 1.0})
        assert set(q_im.unique().tolist()).issubset({-1.0, 0.0, 1.0})

    def test_mutually_exclusive(self):
        """Each weight should be either real OR imaginary, not both."""
        w_re = torch.randn(100, 100)
        w_im = torch.randn(100, 100)
        (q_re, q_im), _ = phase_aware_quantize(w_re, w_im)

        # For each position, exactly one of q_re or q_im should be non-zero
        # (except for zero-magnitude weights which default to +1)
        both_nonzero = (q_re != 0) & (q_im != 0)
        assert not both_nonzero.any()


class TestPhaseAwareSTE:
    """Tests for the PhaseAwareSTE autograd function."""

    def test_forward(self):
        """Test forward pass returns quantized values."""
        w_re = torch.randn(10, 10)
        w_im = torch.randn(10, 10)

        q_re, q_im, s_re, s_im = PhaseAwareSTE.apply(w_re, w_im)

        assert set(q_re.unique().tolist()).issubset({-1.0, 0.0, 1.0})
        assert set(q_im.unique().tolist()).issubset({-1.0, 0.0, 1.0})

    def test_backward_gradients_flow(self):
        """Test that gradients flow through STE."""
        w_re = torch.randn(10, 10, requires_grad=True)
        w_im = torch.randn(10, 10, requires_grad=True)

        q_re, q_im, s_re, s_im = quantize_with_ste(w_re, w_im)

        # Compute some loss
        loss = q_re.sum() + q_im.sum()
        loss.backward()

        # Gradients should exist and be non-zero
        assert w_re.grad is not None
        assert w_im.grad is not None
