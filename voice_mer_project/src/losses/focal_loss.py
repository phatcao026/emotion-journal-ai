"""
focal_loss.py

Multi-Class Focal Loss for imbalanced emotion classification.

Focal Loss (Lin et al., 2017) reduces the contribution of well-classified
examples, focusing training on hard negatives:

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

With gamma=0 this reduces to standard Cross-Entropy.  With gamma=2.0 (our
default), examples with p_t > 0.5 are significantly down-weighted.

Reference:
    https://arxiv.org/abs/1708.02002
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MultiClassFocalLoss(nn.Module):
    """Multi-class Focal Loss with optional per-class weighting.

    Args:
        gamma: Focusing parameter (>=0). Default 2.0.
        alpha: Per-class weights, shape (C,). None means uniform.
        reduction: ``"mean"`` | ``"sum"`` | ``"none"``.
        label_smoothing: Label-smoothing coefficient in [0, 1). Default 0.0.

    Shapes:
        - logits: ``(B, C)`` raw scores (pre-softmax).
        - targets: ``(B,)`` class indices in ``[0, C)``.
        - output: scalar when reduction is ``"mean"``/``"sum"``,
          ``(B,)`` when ``"none"``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'")
        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")

        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

        logger.info(
            "MultiClassFocalLoss: gamma=%.1f, reduction=%s, smoothing=%.2f",
            gamma, reduction, label_smoothing,
        )

    # ------------------------------------------------------------------

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Pre-softmax scores, shape ``(B, C)``.
            targets: Ground-truth class indices, shape ``(B,)``.

        Returns:
            Loss value (scalar or per-sample depending on *reduction*).
        """
        num_classes = logits.size(1)

        # Compute log-softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=1)           # (B, C)
        probs = log_probs.exp()                             # (B, C)

        # Gather the probability of the true class
        targets_one_hot = F.one_hot(targets, num_classes).float()  # (B, C)

        # Optional label smoothing
        if self.label_smoothing > 0:
            targets_one_hot = (
                (1.0 - self.label_smoothing) * targets_one_hot
                + self.label_smoothing / num_classes
            )

        # p_t for each class position
        p_t = (probs * targets_one_hot).sum(dim=1)          # (B,)

        # Focal modulation factor
        focal_weight = (1.0 - p_t) ** self.gamma            # (B,)

        # Per-sample cross-entropy (using log_probs for stability)
        ce = -(targets_one_hot * log_probs).sum(dim=1)      # (B,)

        # Per-class alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]  # (B,)
            ce = alpha_t * ce

        loss = focal_weight * ce                             # (B,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def extra_repr(self) -> str:
        return (
            f"gamma={self.gamma}, "
            f"alpha={'custom' if self.alpha is not None else 'none'}, "
            f"reduction='{self.reduction}', "
            f"label_smoothing={self.label_smoothing}"
        )
