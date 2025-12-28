"""Loss functions for Fairy2i training.

This module provides loss functions for Fairy2i quantization-aware training.
The primary loss is standard cross-entropy for next-token prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuePretrainLoss(nn.Module):
    """Standard cross-entropy loss for continue pretraining.

    This is the same loss used in Stage 2 of BitDistill training.
    It computes next-token prediction loss without any distillation.

    Args:
        ignore_index: Token index to ignore in loss computation (default: -100)
        label_smoothing: Label smoothing factor (default: 0.0)

    Example:
        >>> loss_fn = ContinuePretrainLoss()
        >>> logits = torch.randn(2, 10, 50000)  # (batch, seq_len, vocab)
        >>> labels = torch.randint(0, 50000, (2, 10))
        >>> loss = loss_fn(logits, labels)
    """

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy loss.

        Args:
            logits: Model output logits, shape (batch, seq_len, vocab_size)
            labels: Target labels, shape (batch, seq_len)
                    Use -100 for positions to ignore.

        Returns:
            Scalar loss tensor
        """
        # Flatten for cross entropy
        # (batch * seq_len, vocab_size)
        logits_flat = logits.view(-1, logits.size(-1))
        # (batch * seq_len,)
        labels_flat = labels.view(-1)

        loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )

        return loss


class Fairy2QATLoss(nn.Module):
    """Loss function for Fairy2i QAT with optional regularization.

    Combines:
    - Cross-entropy loss for next-token prediction
    - Optional L2 regularization on quantization error

    Args:
        ignore_index: Token index to ignore in loss computation
        label_smoothing: Label smoothing factor
        quant_error_weight: Weight for quantization error penalty (default: 0.0)
    """

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        quant_error_weight: float = 0.0,
    ):
        super().__init__()
        self.ce_loss = ContinuePretrainLoss(ignore_index, label_smoothing)
        self.quant_error_weight = quant_error_weight

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        model: torch.nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute QAT loss.

        Args:
            logits: Model output logits
            labels: Target labels
            model: Model (for quantization error, optional)

        Returns:
            Dict with "loss" (total) and component losses
        """
        ce_loss = self.ce_loss(logits, labels)

        result = {
            "loss": ce_loss,
            "ce_loss": ce_loss,
        }

        # Add quantization error penalty if requested
        if self.quant_error_weight > 0 and model is not None:
            quant_error = self._compute_quant_error(model)
            result["quant_error"] = quant_error
            result["loss"] = ce_loss + self.quant_error_weight * quant_error

        return result

    def _compute_quant_error(self, model: torch.nn.Module) -> torch.Tensor:
        """Compute total quantization error across all Fairy2Linear layers."""
        from fairy2.models.fairy2_linear import Fairy2Linear

        total_error = 0.0
        count = 0

        for module in model.modules():
            if isinstance(module, Fairy2Linear):
                errors = module.quantization_error()
                total_error += errors["U_mse"] + errors["W_mse"]
                count += 1

        if count > 0:
            return torch.tensor(total_error / count, device=next(model.parameters()).device)
        return torch.tensor(0.0)
