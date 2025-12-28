"""Fairy2 training components.

This module provides training infrastructure for Fairy2i QAT:
- Fairy2Trainer: Main trainer with FSDP, W&B, and GCS support
- ContinuePretrainLoss: Standard next-token prediction loss
"""

from fairy2.training.loss import ContinuePretrainLoss
from fairy2.training.trainer import Fairy2Trainer

__all__ = [
    "Fairy2Trainer",
    "ContinuePretrainLoss",
]
