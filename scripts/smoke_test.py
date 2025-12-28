#!/usr/bin/env python3
"""Smoke test for Fairy2i quantization.

This script runs a quick smoke test to verify that:
1. Model loading works
2. Fairy2 conversion works
3. Forward pass works
4. Training loop runs without errors
5. Checkpointing works

Usage:
    uv run python scripts/smoke_test.py --model smollm2_135m --mode w2 --steps 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_smoke_test(model_name: str, mode: str, steps: int) -> bool:
    """Run smoke test for Fairy2 training.

    Args:
        model_name: Model config name (e.g., "smollm2_135m")
        mode: Quantization mode ("w1" or "w2")
        steps: Number of training steps

    Returns:
        True if test passed, False otherwise
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from fairy2.models import convert_to_fairy2, count_fairy2_layers, Fairy2Linear
    from fairy2.quantization import phase_aware_quantize, ResidualQuantizer

    logger.info("=" * 60)
    logger.info("Fairy2i Smoke Test")
    logger.info(f"Model: {model_name}, Mode: {mode}, Steps: {steps}")
    logger.info("=" * 60)

    # Model mapping
    model_map = {
        "smollm2_135m": "HuggingFaceTB/SmolLM2-135M",
        "qwen3_4b": "Qwen/Qwen3-4B",
    }

    if model_name not in model_map:
        logger.error(f"Unknown model: {model_name}. Available: {list(model_map.keys())}")
        return False

    pretrained = model_map[model_name]
    num_stages = 1 if mode == "w1" else 2

    # Test 1: Phase-aware quantization
    logger.info("Test 1: Phase-aware quantization...")
    try:
        w_re = torch.randn(10, 10)
        w_im = torch.randn(10, 10)
        (q_re, q_im), (s_re, s_im) = phase_aware_quantize(w_re, w_im)

        # Verify quantized values are in {-1, 0, 1}
        assert set(q_re.unique().tolist()).issubset({-1.0, 0.0, 1.0})
        assert set(q_im.unique().tolist()).issubset({-1.0, 0.0, 1.0})
        logger.info("  PASSED: Phase-aware quantization")
    except Exception as e:
        logger.error(f"  FAILED: Phase-aware quantization - {e}")
        return False

    # Test 2: Residual quantization
    logger.info("Test 2: Residual quantization...")
    try:
        quantizer = ResidualQuantizer(num_stages=num_stages)
        stages = quantizer.quantize(w_re, w_im)
        assert len(stages) == num_stages
        w_re_q, w_im_q = quantizer.dequantize(stages)
        assert w_re_q.shape == w_re.shape
        logger.info(f"  PASSED: Residual quantization ({num_stages} stages)")
    except Exception as e:
        logger.error(f"  FAILED: Residual quantization - {e}")
        return False

    # Test 3: Fairy2Linear layer
    logger.info("Test 3: Fairy2Linear layer...")
    try:
        layer = Fairy2Linear(64, 128, num_stages=num_stages)
        x = torch.randn(2, 10, 64)
        y = layer(x)
        assert y.shape == (2, 10, 128)
        logger.info("  PASSED: Fairy2Linear forward pass")
    except Exception as e:
        logger.error(f"  FAILED: Fairy2Linear - {e}")
        return False

    # Test 4: Load and convert model
    logger.info(f"Test 4: Loading model {pretrained}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("  PASSED: Model loaded")
    except Exception as e:
        logger.error(f"  FAILED: Model loading - {e}")
        return False

    # Test 5: Convert to Fairy2
    logger.info("Test 5: Converting to Fairy2...")
    try:
        model = convert_to_fairy2(model, num_stages=num_stages)
        counts = count_fairy2_layers(model)
        logger.info(f"  Layer counts: {counts}")
        assert counts["fairy2_linear"] > 0
        logger.info("  PASSED: Model conversion")
    except Exception as e:
        logger.error(f"  FAILED: Model conversion - {e}")
        return False

    # Test 6: Forward pass
    logger.info("Test 6: Forward pass...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        input_ids = torch.randint(0, tokenizer.vocab_size, (2, 32), device=device)
        with torch.no_grad():
            outputs = model(input_ids)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        assert logits.shape[:2] == (2, 32)
        logger.info(f"  PASSED: Forward pass on {device}")
    except Exception as e:
        logger.error(f"  FAILED: Forward pass - {e}")
        return False

    # Test 7: Training loop
    logger.info(f"Test 7: Training for {steps} steps...")
    try:
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        for step in range(steps):
            input_ids = torch.randint(0, tokenizer.vocab_size, (2, 32), device=device)
            labels = input_ids.clone()

            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            # Compute loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % max(1, steps // 5) == 0:
                logger.info(f"  Step {step}/{steps}, Loss: {loss.item():.4f}")

        logger.info("  PASSED: Training loop")
    except Exception as e:
        logger.error(f"  FAILED: Training loop - {e}")
        return False

    # Test 8: Save checkpoint
    logger.info("Test 8: Saving checkpoint...")
    try:
        checkpoint_dir = Path("outputs/smoke_test_checkpoint")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_dir / "model.pt")
        logger.info(f"  PASSED: Checkpoint saved to {checkpoint_dir}")
    except Exception as e:
        logger.error(f"  FAILED: Checkpoint saving - {e}")
        return False

    logger.info("=" * 60)
    logger.info("ALL TESTS PASSED!")
    logger.info("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(description="Fairy2i Smoke Test")
    parser.add_argument(
        "--model",
        default="smollm2_135m",
        choices=["smollm2_135m", "qwen3_4b"],
        help="Model to test",
    )
    parser.add_argument(
        "--mode",
        default="w2",
        choices=["w1", "w2"],
        help="Quantization mode (w1=1-bit, w2=2-bit)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of training steps",
    )
    args = parser.parse_args()

    success = run_smoke_test(args.model, args.mode, args.steps)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
