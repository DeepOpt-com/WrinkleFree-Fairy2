"""Tests for recursive residual quantization."""

import pytest
import torch

from fairy2.quantization.residual import (
    QuantizationStage,
    ResidualQuantizer,
    residual_quantize_with_ste,
)


class TestResidualQuantizer:
    """Tests for the ResidualQuantizer class."""

    def test_init(self):
        """Test initialization."""
        quantizer = ResidualQuantizer(num_stages=2)
        assert quantizer.num_stages == 2

    def test_init_invalid_stages(self):
        """Test that num_stages < 1 raises."""
        with pytest.raises(ValueError):
            ResidualQuantizer(num_stages=0)

    def test_quantize_w1(self, random_complex_weights):
        """Test W1 (1-stage) quantization."""
        w_re, w_im = random_complex_weights()
        quantizer = ResidualQuantizer(num_stages=1)

        stages = quantizer.quantize(w_re, w_im)

        assert len(stages) == 1
        assert isinstance(stages[0], QuantizationStage)

    def test_quantize_w2(self, random_complex_weights):
        """Test W2 (2-stage) quantization."""
        w_re, w_im = random_complex_weights()
        quantizer = ResidualQuantizer(num_stages=2)

        stages = quantizer.quantize(w_re, w_im)

        assert len(stages) == 2
        for stage in stages:
            assert isinstance(stage, QuantizationStage)

    def test_dequantize(self, random_complex_weights):
        """Test dequantization reconstructs weights."""
        w_re, w_im = random_complex_weights()
        quantizer = ResidualQuantizer(num_stages=2)

        stages = quantizer.quantize(w_re, w_im)
        w_re_q, w_im_q = quantizer.dequantize(stages)

        assert w_re_q.shape == w_re.shape
        assert w_im_q.shape == w_im.shape

    def test_more_stages_reduces_error(self, random_complex_weights):
        """Test that more stages reduces quantization error."""
        w_re, w_im = random_complex_weights()

        q1 = ResidualQuantizer(num_stages=1)
        q2 = ResidualQuantizer(num_stages=2)
        q3 = ResidualQuantizer(num_stages=3)

        mse1 = q1.mse(w_re, w_im, q1.quantize(w_re, w_im))
        mse2 = q2.mse(w_re, w_im, q2.quantize(w_re, w_im))
        mse3 = q3.mse(w_re, w_im, q3.quantize(w_re, w_im))

        # More stages should reduce error
        assert mse2 < mse1
        assert mse3 < mse2

    def test_quantization_error(self, random_complex_weights):
        """Test quantization error computation."""
        w_re, w_im = random_complex_weights()
        quantizer = ResidualQuantizer(num_stages=2)

        stages = quantizer.quantize(w_re, w_im)
        err_re, err_im = quantizer.quantization_error(w_re, w_im, stages)

        assert err_re.shape == w_re.shape
        assert err_im.shape == w_im.shape

    def test_residual_is_quantized(self, random_complex_weights):
        """Test that each stage quantizes the residual."""
        w_re, w_im = random_complex_weights()
        quantizer = ResidualQuantizer(num_stages=2)

        stages = quantizer.quantize(w_re, w_im)

        # Stage 0 should quantize original weights
        # Stage 1 should quantize the residual
        # Both should have values in {-1, 0, 1}
        for stage in stages:
            assert set(stage.q_re.unique().tolist()).issubset({-1.0, 0.0, 1.0})
            assert set(stage.q_im.unique().tolist()).issubset({-1.0, 0.0, 1.0})


class TestResidualQuantizerSTE:
    """Tests for the residual quantization STE."""

    def test_forward(self, random_complex_weights):
        """Test forward pass."""
        w_re, w_im = random_complex_weights()
        w_re_q, w_im_q = residual_quantize_with_ste(w_re, w_im, num_stages=2)

        assert w_re_q.shape == w_re.shape
        assert w_im_q.shape == w_im.shape

    def test_backward(self, random_complex_weights):
        """Test that gradients flow through."""
        w_re, w_im = random_complex_weights()
        w_re.requires_grad_(True)
        w_im.requires_grad_(True)

        w_re_q, w_im_q = residual_quantize_with_ste(w_re, w_im, num_stages=2)

        loss = w_re_q.sum() + w_im_q.sum()
        loss.backward()

        assert w_re.grad is not None
        assert w_im.grad is not None
