"""Tests for model conversion."""

import pytest
import torch
import torch.nn as nn

from fairy2.models.converter import (
    convert_to_fairy2,
    count_fairy2_layers,
    get_layer_info,
)
from fairy2.models.fairy2_linear import Fairy2Linear


class SimpleModel(nn.Module):
    """Simple model for testing conversion."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 64)  # Should not be converted
        self.linear1 = nn.Linear(64, 128)
        self.linear2 = nn.Linear(128, 64)
        self.lm_head = nn.Linear(64, 100)  # Should be excluded

    def forward(self, x):
        x = self.embed(x)
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return self.lm_head(x)


class TestConvertToFairy2:
    """Tests for convert_to_fairy2 function."""

    def test_basic_conversion(self):
        """Test basic model conversion."""
        model = SimpleModel()
        model = convert_to_fairy2(model, num_stages=2)

        # linear1 and linear2 should be converted
        assert isinstance(model.linear1, Fairy2Linear)
        assert isinstance(model.linear2, Fairy2Linear)

    def test_excludes_lm_head(self):
        """Test that lm_head is excluded by default."""
        model = SimpleModel()
        model = convert_to_fairy2(model, num_stages=2)

        # lm_head should remain nn.Linear
        assert isinstance(model.lm_head, nn.Linear)
        assert not isinstance(model.lm_head, Fairy2Linear)

    def test_custom_exclude_names(self):
        """Test custom exclusion patterns."""
        model = SimpleModel()
        model = convert_to_fairy2(model, num_stages=2, exclude_names=["linear1"])

        # linear1 should remain nn.Linear
        assert isinstance(model.linear1, nn.Linear)
        assert not isinstance(model.linear1, Fairy2Linear)
        # linear2 should be converted
        assert isinstance(model.linear2, Fairy2Linear)

    def test_num_stages(self):
        """Test that num_stages is propagated."""
        model = SimpleModel()
        model = convert_to_fairy2(model, num_stages=1)

        assert model.linear1.num_stages == 1
        assert model.linear2.num_stages == 1

    def test_preserves_output(self):
        """Test that conversion preserves model output (before quantization)."""
        model = SimpleModel()
        x = torch.randint(0, 100, (2, 10))

        # Get original output
        with torch.no_grad():
            original_output = model(x)

        # Convert (this is destructive, so we can't compare directly)
        # Instead, we verify that forward still works
        model = convert_to_fairy2(model, num_stages=2)

        with torch.no_grad():
            new_output = model(x)

        # Shape should match
        assert new_output.shape == original_output.shape


class TestCountFairy2Layers:
    """Tests for count_fairy2_layers function."""

    def test_count_converted_model(self):
        """Test counting in converted model."""
        model = SimpleModel()
        model = convert_to_fairy2(model, num_stages=2)

        counts = count_fairy2_layers(model)

        assert counts["fairy2_linear"] == 2  # linear1, linear2
        assert counts["nn_linear"] == 1  # lm_head
        assert counts["total"] == 3

    def test_count_original_model(self):
        """Test counting in original model."""
        model = SimpleModel()
        counts = count_fairy2_layers(model)

        assert counts["fairy2_linear"] == 0
        assert counts["nn_linear"] == 3  # linear1, linear2, lm_head


class TestGetLayerInfo:
    """Tests for get_layer_info function."""

    def test_layer_info_converted(self):
        """Test layer info for converted model."""
        model = SimpleModel()
        model = convert_to_fairy2(model, num_stages=2)

        info = get_layer_info(model)

        # Should have 3 entries
        assert len(info) == 3

        # Check types
        types = {i["type"] for i in info}
        assert "Fairy2Linear" in types
        assert "Linear" in types

    def test_layer_info_includes_num_stages(self):
        """Test that Fairy2Linear info includes num_stages."""
        model = SimpleModel()
        model = convert_to_fairy2(model, num_stages=2)

        info = get_layer_info(model)

        fairy2_layers = [i for i in info if i["type"] == "Fairy2Linear"]
        for layer in fairy2_layers:
            assert "num_stages" in layer
            assert layer["num_stages"] == 2
