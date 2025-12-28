"""WrinkleFree-Fairy2: Complex-valued LLM quantization with Fairy2i.

This package implements the Fairy2i algorithm for extreme LLM quantization,
converting pre-trained real-valued models to complex-valued representations
with weights in {+1, -1, +i, -i}.

Example:
    >>> from fairy2.models import convert_to_fairy2, Fairy2Linear
    >>> from transformers import AutoModelForCausalLM
    >>>
    >>> # Load and convert model
    >>> model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    >>> fairy2_model = convert_to_fairy2(model, num_stages=2)

References:
    - Paper: https://arxiv.org/abs/2512.02901
    - Fairy2i: Training Complex LLMs from Real LLMs with All Parameters in {±1, ±i}
"""

from fairy2.models.converter import convert_to_fairy2
from fairy2.models.fairy2_linear import Fairy2Linear
from fairy2.models.widely_linear import WidelyLinearComplex
from fairy2.quantization.phase_aware import phase_aware_quantize
from fairy2.quantization.residual import ResidualQuantizer

__version__ = "0.1.0"

__all__ = [
    "convert_to_fairy2",
    "Fairy2Linear",
    "WidelyLinearComplex",
    "phase_aware_quantize",
    "ResidualQuantizer",
]
