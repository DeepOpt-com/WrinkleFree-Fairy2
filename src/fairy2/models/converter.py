"""Model converter for Fairy2i quantization.

This module provides utilities to convert pre-trained HuggingFace models
to Fairy2 format, replacing nn.Linear layers with Fairy2Linear layers.

The conversion process:
1. Iterate through all modules in the model
2. Replace each nn.Linear with Fairy2Linear.from_real_linear()
3. Preserve weights via the widely-linear conversion formulas
4. Optionally exclude specific layers (e.g., embed_tokens, lm_head)

Reference:
    Fairy2i: Training Complex LLMs from Real LLMs with All Parameters in {±1, ±i}
    https://arxiv.org/abs/2512.02901
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from fairy2.models.fairy2_linear import Fairy2Linear

logger = logging.getLogger(__name__)


def _set_module(model: nn.Module, name: str, module: nn.Module) -> None:
    """Set a nested module by its full path name.

    Args:
        model: Root model
        name: Dot-separated module path (e.g., "layer.0.attention.q_proj")
        module: Module to set
    """
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def convert_to_fairy2(
    model: nn.Module,
    num_stages: int = 2,
    exclude_names: Optional[list[str]] = None,
    skip_odd_dimensions: bool = True,
    verbose: bool = True,
) -> nn.Module:
    """Convert a pre-trained model to Fairy2 format.

    Replaces all nn.Linear layers with Fairy2Linear layers, enabling
    complex-valued quantization with weights in {+1, -1, +i, -i}.

    The conversion is lossless before quantization - the widely-linear
    complex representation produces identical outputs to the original
    real-valued layers.

    Args:
        model: Pre-trained model (e.g., from HuggingFace)
        num_stages: Number of residual quantization stages
            - 1 = W1 mode (~1 bit per weight)
            - 2 = W2 mode (~2 bits per weight, recommended)
        exclude_names: List of substrings to exclude from conversion.
            Default: ["embed_tokens", "lm_head", "embed_positions"]
            Layers containing any of these strings are kept as nn.Linear.
        skip_odd_dimensions: If True, skip layers with odd in/out features
            (they cannot be converted to widely-linear form)
        verbose: If True, log conversion progress

    Returns:
        The same model with Linear layers replaced by Fairy2Linear.
        (Modification is in-place, but model is returned for convenience)

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        >>> fairy2_model = convert_to_fairy2(model, num_stages=2)

    Note:
        The conversion modifies the model in-place to avoid memory overhead.
        If you need to preserve the original model, clone it first.
    """
    if exclude_names is None:
        exclude_names = ["embed_tokens", "lm_head", "embed_positions", "wte", "wpe"]

    converted_count = 0
    skipped_count = 0
    excluded_count = 0

    # Collect (name, module) pairs first to avoid modifying during iteration
    linear_modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and not isinstance(module, Fairy2Linear)
    ]

    for name, module in linear_modules:
        # Check if this layer should be excluded
        if any(excl in name for excl in exclude_names):
            if verbose:
                logger.debug(f"Excluding {name} (matches exclude pattern)")
            excluded_count += 1
            continue

        # Check for odd dimensions
        if module.in_features % 2 != 0 or module.out_features % 2 != 0:
            if skip_odd_dimensions:
                if verbose:
                    logger.warning(
                        f"Skipping {name}: odd dimensions "
                        f"({module.in_features}, {module.out_features})"
                    )
                skipped_count += 1
                continue
            else:
                raise ValueError(
                    f"Cannot convert {name}: dimensions must be even, "
                    f"got ({module.in_features}, {module.out_features})"
                )

        # Convert to Fairy2Linear
        fairy2_layer = Fairy2Linear.from_real_linear(module, num_stages=num_stages)
        _set_module(model, name, fairy2_layer)
        converted_count += 1

        if verbose:
            logger.debug(f"Converted {name}")

    if verbose:
        logger.info(
            f"Fairy2 conversion complete: "
            f"{converted_count} converted, "
            f"{excluded_count} excluded, "
            f"{skipped_count} skipped (odd dimensions)"
        )

    return model


def count_fairy2_layers(model: nn.Module) -> dict[str, int]:
    """Count Fairy2Linear and remaining Linear layers in a model.

    Args:
        model: Model to analyze

    Returns:
        Dict with counts:
            - "fairy2_linear": Number of Fairy2Linear layers
            - "nn_linear": Number of remaining nn.Linear layers
            - "total": Total linear layers
    """
    fairy2_count = 0
    linear_count = 0

    for module in model.modules():
        if isinstance(module, Fairy2Linear):
            fairy2_count += 1
        elif isinstance(module, nn.Linear):
            linear_count += 1

    return {
        "fairy2_linear": fairy2_count,
        "nn_linear": linear_count,
        "total": fairy2_count + linear_count,
    }


def verify_conversion(
    original_model: nn.Module,
    fairy2_model: nn.Module,
    input_shape: tuple[int, ...] = (2, 10, 128),
    device: str = "cpu",
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> tuple[bool, float]:
    """Verify that Fairy2 conversion preserves model output.

    Compares outputs of original and converted models on random input
    to ensure the conversion is lossless (before quantization).

    Args:
        original_model: Original model (before conversion)
        fairy2_model: Converted model (with Fairy2Linear layers)
        input_shape: Shape of test input (batch, seq_len, hidden_dim)
        device: Device to run test on
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        Tuple of (passed: bool, max_diff: float)

    Note:
        This test should PASS for the conversion (outputs should match).
        After training with quantization, outputs will differ.
    """
    original_model = original_model.to(device).eval()
    fairy2_model = fairy2_model.to(device).eval()

    with torch.no_grad():
        # Create random input
        x = torch.randn(input_shape, device=device)

        # Forward through both models
        out_original = original_model(x)
        out_fairy2 = fairy2_model(x)

        # Handle tuple outputs (e.g., from HuggingFace models)
        if isinstance(out_original, tuple):
            out_original = out_original[0]
        if isinstance(out_fairy2, tuple):
            out_fairy2 = out_fairy2[0]

        # Compute difference
        diff = (out_original - out_fairy2).abs()
        max_diff = diff.max().item()

        # Check if outputs match
        passed = torch.allclose(out_original, out_fairy2, rtol=rtol, atol=atol)

    return passed, max_diff


def get_layer_info(model: nn.Module) -> list[dict]:
    """Get information about all linear layers in a model.

    Args:
        model: Model to analyze

    Returns:
        List of dicts with layer information:
            - name: Full path name
            - type: "Fairy2Linear" or "Linear"
            - in_features: Input dimension
            - out_features: Output dimension
            - num_stages: Number of quantization stages (Fairy2Linear only)
    """
    info = []
    for name, module in model.named_modules():
        if isinstance(module, Fairy2Linear):
            info.append({
                "name": name,
                "type": "Fairy2Linear",
                "in_features": module.in_features,
                "out_features": module.out_features,
                "num_stages": module.num_stages,
            })
        elif isinstance(module, nn.Linear):
            info.append({
                "name": name,
                "type": "Linear",
                "in_features": module.in_features,
                "out_features": module.out_features,
            })
    return info
