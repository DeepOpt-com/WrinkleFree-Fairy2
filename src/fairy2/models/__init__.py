"""Fairy2 model components.

This module provides the core neural network layers for Fairy2i quantization:
- WidelyLinearComplex: Real-to-complex widely-linear transformation
- Fairy2Linear: Full Fairy2 quantized linear layer with STE
- convert_to_fairy2: Convert HuggingFace models to Fairy2 format
- count_fairy2_layers: Count Fairy2 vs standard Linear layers
- get_layer_info: Get detailed layer information
"""

from fairy2.models.converter import convert_to_fairy2, count_fairy2_layers, get_layer_info
from fairy2.models.fairy2_linear import Fairy2Linear
from fairy2.models.widely_linear import WidelyLinearComplex

__all__ = [
    "WidelyLinearComplex",
    "Fairy2Linear",
    "convert_to_fairy2",
    "count_fairy2_layers",
    "get_layer_info",
]
