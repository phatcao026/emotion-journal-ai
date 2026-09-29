"""
ldam_loss.py

Label-Distribution-Aware Margin (LDAM) Loss with Deferred Re-Weighting (DRW).
Designed to solve severe class imbalance in Speech & Text Emotion Recognition.

Theoretical grounding:
    - Cao, K. et al. (2019). Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss. NeurIPS 2019.
    - Huynh Thi, N. T. et al. (2024). Vietnamese Emotion Recognition from Voice and Text: A Confidence-Based Approach.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin (LDAM) Loss.

    Enforces a class-dependent margin Delta_j = C / (N_j ** 0.25) so that rare
    classes (like Anxiety or Guilt/Shame) are optimized with wider margins.

    Args:
        cls_num_list: List of sample counts per class [N_0, N_1, ..., N_C-1].
        max_m: Maximum margin factor (C). Default: 0.5.
        scale: Scaling parameter s for logits. Default: 30.0.
        weight: Optional class weights tensor.
    """

    def __init__(
        self,
        cls_num_list: Optional[Union[List[int], np.ndarray, torch.Tensor]] = None,
        max_m: float = 0.5,
        scale: float = 30.0,
        weight: Optional[torch.Tensor] = None,
        class_counts: Optional[Union[List[int], np.ndarray, torch.Tensor]] = None,
        s: Optional[float] = None,
    ) -> None:
        super().__init__()
        counts = class_counts if class_counts is not None else cls_num_list
        if counts is None:
            raise ValueError("Either cls_num_list or class_counts must be provided.")
        cls_num_arr = np.array(counts, dtype=np.float32)
        if (cls_num_arr <= 0).any():
            raise ValueError(f"All class counts must be positive, got {counts}")
        self.cls_num_arr = cls_num_arr
        self.scale = s if s is not None else scale

        # Margin: Delta_j = C / (N_j ** 0.25)
        m_list = 1.0 / np.power(cls_num_arr, 0.25)
        m_list = m_list * (max_m / np.max(m_list))
        self.register_buffer("m_list", torch.tensor(m_list, dtype=torch.float32))
        self.register_buffer("weight", weight if weight is not None else None)

    def drw_update(self, current_epoch: int, drw_start_epoch: int, beta: float = 0.9999) -> None:
        """Activate Deferred Re-Weighting (DRW) after drw_start_epoch."""
        if current_epoch >= drw_start_epoch:
            device = self.m_list.device
            drw_w = get_drw_weights(self.cls_num_arr, beta=beta, device=device)
            self.weight = drw_w
            logger.info("LDAM Deferred Re-Weighting (DRW) activated at epoch %d.", current_epoch)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute LDAM loss.

        Args:
            logits: (B, C) unnormalized class predictions.
            targets: (B,) class indices or (B, C) one-hot vectors.

        Returns:
            Scalar loss tensor.
        """
        if targets.ndim == 2:
            targets = targets.argmax(dim=-1)

        batch_size, num_classes = logits.shape

        # Select margin for ground-truth class
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, targets.unsqueeze(1), 1)

        margins = self.m_list.unsqueeze(0).expand(batch_size, -1)
        target_margins = margins[index].view(batch_size, 1)

        # Apply margin subtraction only to target class logit
        diff = torch.zeros_like(logits)
        diff.scatter_(1, targets.unsqueeze(1), target_margins)
        output_logits = self.scale * (logits - diff)

        return F.cross_entropy(output_logits, targets, weight=self.weight)


def get_drw_weights(
    cls_num_list: List[int],
    beta: float = 0.9999,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute Deferred Re-Weighting (DRW) class weights based on Effective Number of Samples.

    Weight: w_i = (1 - beta) / (1 - beta ** N_i), normalized to mean = 1.0.

    Args:
        cls_num_list: Sample counts per class.
        beta: Hyperparameter in (0, 1). Default: 0.9999.
        device: Target torch device.

    Returns:
        Tensor of shape (C,) with normalized class weights.
    """
    cls_nums = np.array(cls_num_list, dtype=np.float32)
    effective_num = 1.0 - np.power(beta, cls_nums)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.mean(weights)

    tensor_w = torch.tensor(weights, dtype=torch.float32)
    if device is not None:
        tensor_w = tensor_w.to(device)
    return tensor_w
