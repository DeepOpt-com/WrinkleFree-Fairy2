#!/usr/bin/env python3
"""Main training script for Fairy2i QAT.

This script provides the main entry point for Fairy2i quantization-aware
training. It uses Hydra for configuration management.

Usage:
    # Basic training with SmolLM2-135M
    uv run python scripts/train.py model=smollm2_135m training=fairy2_w2

    # With W1 (1-bit) mode
    uv run python scripts/train.py model=smollm2_135m training=fairy2_w1

    # Limit training steps (for testing)
    uv run python scripts/train.py training.max_steps=100

    # Disable wandb
    uv run python scripts/train.py training.logging.wandb.enabled=false
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fairy2.models import convert_to_fairy2, count_fairy2_layers
from fairy2.training import Fairy2Trainer

logger = logging.getLogger(__name__)


def setup_logging(cfg: DictConfig) -> None:
    """Configure logging based on config."""
    level = logging.INFO
    if cfg.get("debug", False):
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloader(cfg: DictConfig, tokenizer):
    """Create training dataloader.

    For simplicity, we create a dummy dataloader for testing.
    In production, replace with actual dataset loading.
    """
    from torch.utils.data import DataLoader, Dataset

    class DummyDataset(Dataset):
        """Dummy dataset for testing."""

        def __init__(self, tokenizer, seq_len=512, num_samples=10000):
            self.tokenizer = tokenizer
            self.seq_len = seq_len
            self.num_samples = num_samples

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Generate random tokens for testing
            input_ids = torch.randint(0, self.tokenizer.vocab_size, (self.seq_len,))
            return {"input_ids": input_ids, "labels": input_ids.clone()}

    dataset = DummyDataset(
        tokenizer,
        seq_len=cfg.training.max_seq_length,
        num_samples=cfg.training.max_steps * cfg.training.batch_size,
    )

    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=0,
    )


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main training function."""
    # Setup
    setup_logging(cfg)
    set_seed(cfg.seed)

    logger.info("=" * 60)
    logger.info("Fairy2i Quantization-Aware Training")
    logger.info("=" * 60)
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Load model
    logger.info(f"Loading model: {cfg.model.pretrained}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.pretrained,
        torch_dtype=getattr(torch, cfg.model.dtype),
        trust_remote_code=cfg.model.trust_remote_code,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.pretrained,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Convert to Fairy2 format
    num_stages = cfg.training.quantization.num_stages
    logger.info(f"Converting to Fairy2 format (W{num_stages} mode)")
    model = convert_to_fairy2(model, num_stages=num_stages)

    # Count layers
    layer_counts = count_fairy2_layers(model)
    logger.info(f"Layer counts: {layer_counts}")

    # Create dataloader
    logger.info("Creating dataloader")
    dataloader = create_dataloader(cfg, tokenizer)

    # Create trainer and train
    logger.info("Starting training")
    trainer = Fairy2Trainer(model, dataloader, cfg)
    results = trainer.train()

    logger.info(f"Training complete! Results: {results}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
