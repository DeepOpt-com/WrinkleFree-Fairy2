#!/usr/bin/env python3
"""Main training script for Fairy2i QAT with influence-based data selection.

This script provides the main entry point for Fairy2i quantization-aware
training. It uses Hydra for configuration management and CheaperTraining
for data loading with influence-based remixing.

Usage:
    # Basic training with SmolLM2-135M
    uv run python scripts/train.py model=smollm2_135m training=fairy2_w2

    # With W1 (1-bit) mode
    uv run python scripts/train.py model=smollm2_135m training=fairy2_w1

    # Limit training steps (for testing)
    uv run python scripts/train.py training.max_steps=100

    # Disable wandb
    uv run python scripts/train.py training.logging.wandb.enabled=false

    # Disable influence-based training
    uv run python scripts/train.py training.influence.enabled=false
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
from fairy2.data import (
    create_pretraining_dataloader,
    CHEAPERTRAINING_AVAILABLE,
)

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


def create_dataloaders(cfg: DictConfig, tokenizer):
    """Create training and probe dataloaders using CheaperTraining.

    Uses the high-level create_pretraining_dataloader() which loads the
    data config from CheaperTraining's YAML files automatically.

    Returns:
        tuple: (train_dataloader, mixed_dataset, probe_dataloaders)
            - train_dataloader: DataLoader for training
            - mixed_dataset: MixedDataset for influence weight updates (or None)
            - probe_dataloaders: dict of probe DataLoaders for influence (or None)
    """
    if not CHEAPERTRAINING_AVAILABLE:
        raise ImportError(
            "CheaperTraining is required for Fairy2 training. "
            "Install with: pip install -e ../WrinkleFree-CheaperTraining"
        )

    # Use high-level convenience function - no data config needed here!
    # Config lives in CheaperTraining's configs/data/mixed_pretrain.yaml
    logger.info("Creating training dataloader from CheaperTraining (mixed_pretrain)")
    train_dataloader, mixed_dataset, probe_dataloaders = create_pretraining_dataloader(
        tokenizer=tokenizer,
        batch_size=cfg.training.batch_size,
        max_length=cfg.training.max_seq_length,
        config_name="mixed_pretrain",  # Loads from CheaperTraining's configs
        with_probes=True,
        seed=cfg.seed,
    )

    if probe_dataloaders:
        logger.info(f"Created {len(probe_dataloaders)} probe dataloaders: {list(probe_dataloaders.keys())}")

    return train_dataloader, mixed_dataset, probe_dataloaders


def setup_influence(cfg: DictConfig, model, mixed_dataset, probe_dataloaders, optimizer):
    """Setup influence-based data selection if enabled.

    Returns optimizer (wrapped with InfluenceAwareOptimizer if enabled).
    """
    influence_config = getattr(cfg.training, "influence", None)
    if influence_config is None or not influence_config.get("enabled", False):
        logger.info("Influence-based training: disabled")
        return optimizer

    if mixed_dataset is None:
        logger.warning(
            "Influence enabled but no mixed_dataset available. "
            "Use data=mixed_pretrain for influence-based training."
        )
        return optimizer

    if probe_dataloaders is None:
        logger.warning(
            "Influence enabled but no probe_dataloaders available. "
            "Check data.probe config."
        )
        return optimizer

    # Import influence components
    try:
        from cheapertraining import (
            MixtureWeightCalculator,
            InfluenceAwareOptimizer,
            InfluenceConfig,
        )
    except ImportError:
        logger.error(
            "CheaperTraining influence module not found. "
            "Install with: pip install -e ../WrinkleFree-CheaperTraining"
        )
        raise

    logger.info("Setting up influence-based data selection")

    # Create influence config
    inf_config = InfluenceConfig(
        lambda_reg=influence_config.config.get("lambda_val", 1e-4),
    )

    # Get first probe dataloader for mixture calculation
    # (MixtureWeightCalculator expects single dataloader)
    probe_dataloader = next(iter(probe_dataloaders.values()))

    # Create mixture calculator
    mixture_calc = MixtureWeightCalculator(
        model=model,
        probe_dataloader=probe_dataloader,
        influence_config=inf_config,
    )

    # Wrap optimizer with influence-aware optimizer
    optimizer = InfluenceAwareOptimizer(
        optimizer=optimizer,
        mixture_calculator=mixture_calc,
        mixed_dataset=mixed_dataset,
        update_interval=influence_config.get("update_interval", 1000),
        learning_rate=influence_config.get("learning_rate", 0.2),
        rank=0,  # Single GPU for now
    )

    logger.info("Optimizer wrapped with InfluenceAwareOptimizer")
    logger.info(f"  - Update interval: {influence_config.get('update_interval', 1000)} steps")
    logger.info(f"  - Learning rate: {influence_config.get('learning_rate', 0.2)}")

    return optimizer


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

    # Create dataloaders
    train_dataloader, mixed_dataset, probe_dataloaders = create_dataloaders(cfg, tokenizer)

    # Log dataset info
    if mixed_dataset is not None:
        weights = mixed_dataset.get_current_weights()
        logger.info(f"Initial dataset weights: {weights}")

    # Create trainer and train
    # Note: Trainer creates optimizer internally, so we pass influence config
    logger.info("Starting training")
    trainer = Fairy2Trainer(
        model=model,
        dataloader=train_dataloader,
        config=cfg,
        mixed_dataset=mixed_dataset,
        probe_dataloaders=probe_dataloaders,
    )
    results = trainer.train()

    logger.info(f"Training complete! Results: {results}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
