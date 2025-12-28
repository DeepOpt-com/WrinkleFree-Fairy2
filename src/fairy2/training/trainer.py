"""Fairy2 QAT Trainer.

This module provides the training infrastructure for Fairy2i quantization-aware
training. It follows patterns from WrinkleFree-1.58Quant for compatibility.

Features:
- Hydra configuration
- FSDP distributed training
- W&B logging
- GCS checkpoint sync
- WSD (Warmup-Stable-Decay) scheduler
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from fairy2.training.loss import ContinuePretrainLoss

logger = logging.getLogger(__name__)


class Fairy2Trainer:
    """Trainer for Fairy2i QAT (Quantization-Aware Training).

    This trainer follows WrinkleFree patterns:
    - WSD scheduler (warmup-stable-decay)
    - FSDP for multi-GPU
    - GCS checkpoint sync
    - W&B logging

    Args:
        model: Fairy2 model to train
        train_dataloader: Training data loader
        cfg: Hydra configuration
        optimizer: Optional pre-configured optimizer
        scheduler: Optional pre-configured scheduler

    Example:
        >>> trainer = Fairy2Trainer(model, dataloader, cfg)
        >>> trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        cfg: DictConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.cfg = cfg

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Setup loss function
        self.loss_fn = ContinuePretrainLoss(
            ignore_index=cfg.training.get("ignore_index", -100),
            label_smoothing=cfg.training.get("label_smoothing", 0.0),
        )

        # Setup optimizer
        if optimizer is None:
            self.optimizer = self._create_optimizer()
        else:
            self.optimizer = optimizer

        # Setup scheduler
        if scheduler is None:
            self.scheduler = self._create_scheduler()
        else:
            self.scheduler = scheduler

        # Training state
        self.global_step = 0
        self.epoch = 0

        # W&B setup
        self.wandb_run = None
        if cfg.training.get("logging", {}).get("wandb", {}).get("enabled", False):
            self._setup_wandb()

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        opt_cfg = self.cfg.training.optimizer

        if opt_cfg.type == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.get("weight_decay", 0.01),
                betas=tuple(opt_cfg.get("betas", [0.9, 0.95])),
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_cfg.type}")

    def _create_scheduler(self):
        """Create learning rate scheduler from config."""
        sched_cfg = self.cfg.training.get("scheduler", {})

        if not sched_cfg or sched_cfg.get("type") == "none":
            return None

        if sched_cfg.type == "wsd":
            # Warmup-Stable-Decay scheduler
            from torch.optim.lr_scheduler import LambdaLR

            warmup_steps = sched_cfg.warmup_steps
            total_steps = self.cfg.training.get("max_steps", 10000)
            decay_ratio = sched_cfg.get("decay_ratio", 0.1)

            def lr_lambda(step):
                if step < warmup_steps:
                    # Linear warmup
                    return step / warmup_steps
                elif step < total_steps * (1 - decay_ratio):
                    # Stable
                    return 1.0
                else:
                    # Linear decay
                    decay_start = total_steps * (1 - decay_ratio)
                    decay_end = total_steps
                    progress = (step - decay_start) / (decay_end - decay_start)
                    return max(0.0, 1.0 - progress)

            return LambdaLR(self.optimizer, lr_lambda)

        elif sched_cfg.type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.training.get("max_steps", 10000),
            )

        else:
            raise ValueError(f"Unknown scheduler type: {sched_cfg.type}")

    def _setup_wandb(self):
        """Initialize Weights & Biases logging."""
        try:
            import wandb

            wandb_cfg = self.cfg.training.logging.wandb
            self.wandb_run = wandb.init(
                project=wandb_cfg.get("project", "wrinklefree-fairy2"),
                name=wandb_cfg.get("name"),
                config=dict(self.cfg),
                tags=wandb_cfg.get("tags", ["fairy2", "qat"]),
            )
            logger.info(f"W&B initialized: {self.wandb_run.url}")
        except ImportError:
            logger.warning("wandb not installed, skipping logging")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")

    def train(self) -> dict[str, float]:
        """Run training loop.

        Returns:
            Dict with final training metrics
        """
        max_steps = self.cfg.training.get("max_steps", 10000)
        grad_accum = self.cfg.training.get("gradient_accumulation_steps", 1)
        log_interval = self.cfg.training.get("log_interval", 100)
        save_interval = self.cfg.training.get("save_interval", 5000)
        output_dir = Path(self.cfg.training.get("output_dir", "outputs"))

        self.model.train()
        running_loss = 0.0
        num_samples = 0

        pbar = tqdm(total=max_steps, desc="Training")

        while self.global_step < max_steps:
            for batch in self.train_dataloader:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                labels = batch.get("labels", input_ids.clone())
                if isinstance(labels, torch.Tensor):
                    labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(input_ids)
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs

                # Compute loss
                # Shift for next-token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = self.loss_fn(shift_logits, shift_labels)

                # Scale loss for gradient accumulation
                loss = loss / grad_accum

                # Backward pass
                loss.backward()

                running_loss += loss.item() * grad_accum
                num_samples += 1

                # Optimizer step
                if (self.global_step + 1) % grad_accum == 0:
                    # Gradient clipping
                    max_grad_norm = self.cfg.training.get("gradient_clipping", 1.0)
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_grad_norm
                        )

                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad()

                self.global_step += 1
                pbar.update(1)

                # Logging
                if self.global_step % log_interval == 0:
                    avg_loss = running_loss / num_samples
                    lr = self.optimizer.param_groups[0]["lr"]

                    pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}"})

                    if self.wandb_run is not None:
                        import wandb
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/lr": lr,
                            "train/step": self.global_step,
                        })

                    running_loss = 0.0
                    num_samples = 0

                # Checkpointing
                if self.global_step % save_interval == 0:
                    self._save_checkpoint(output_dir / f"checkpoint-{self.global_step}")

                if self.global_step >= max_steps:
                    break

        pbar.close()

        # Final checkpoint
        self._save_checkpoint(output_dir / "checkpoint-final")

        # Finish W&B
        if self.wandb_run is not None:
            self.wandb_run.finish()

        return {"final_loss": running_loss / max(num_samples, 1)}

    def _save_checkpoint(self, path: Path):
        """Save a training checkpoint.

        Args:
            path: Directory to save checkpoint to
        """
        path.mkdir(parents=True, exist_ok=True)

        # Save model state
        torch.save(
            self.model.state_dict(),
            path / "model.pt",
        )

        # Save optimizer state
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "global_step": self.global_step,
                "epoch": self.epoch,
            },
            path / "training_state.pt",
        )

        logger.info(f"Checkpoint saved to {path}")

        # GCS upload if enabled
        if self.cfg.get("gcs", {}).get("enabled", False):
            self._upload_to_gcs(path)

    def _upload_to_gcs(self, local_path: Path):
        """Upload checkpoint to Google Cloud Storage.

        Args:
            local_path: Local checkpoint path
        """
        try:
            from google.cloud import storage

            bucket_name = self.cfg.gcs.bucket
            prefix = self.cfg.gcs.get("prefix", "fairy2_checkpoints")

            client = storage.Client()
            bucket = client.bucket(bucket_name)

            for file_path in local_path.iterdir():
                blob_name = f"{prefix}/{local_path.name}/{file_path.name}"
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(str(file_path))

            logger.info(f"Uploaded checkpoint to gs://{bucket_name}/{prefix}/{local_path.name}")

        except Exception as e:
            logger.warning(f"Failed to upload to GCS: {e}")

    def load_checkpoint(self, path: Path):
        """Load a training checkpoint.

        Args:
            path: Directory containing checkpoint
        """
        # Load model state
        self.model.load_state_dict(
            torch.load(path / "model.pt", map_location=self.device)
        )

        # Load training state
        state = torch.load(path / "training_state.pt", map_location=self.device)
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler and state["scheduler"]:
            self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = state["global_step"]
        self.epoch = state["epoch"]

        logger.info(f"Checkpoint loaded from {path}")
