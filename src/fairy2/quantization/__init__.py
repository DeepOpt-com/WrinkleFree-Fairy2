"""Fairy2 quantization algorithms.

This module provides the quantization primitives for Fairy2i:
- phase_aware_quantize: Quantize complex weights to {+1, -1, +i, -i}
- ResidualQuantizer: Multi-stage recursive residual quantization
- PhaseAwareSTE: Straight-Through Estimator for complex weights
"""

from fairy2.quantization.phase_aware import PhaseAwareSTE, phase_aware_quantize
from fairy2.quantization.residual import ResidualQuantizer
from fairy2.quantization.ste import ComplexSTE

__all__ = [
    "phase_aware_quantize",
    "PhaseAwareSTE",
    "ResidualQuantizer",
    "ComplexSTE",
]
