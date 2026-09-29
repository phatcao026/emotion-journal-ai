"""
uncertainty_loss.py

Multi-Task Homoscedastic Uncertainty Loss for Hierarchical Multimodal Emotion Recognition.
Learns adaptive task weights dynamically to balance Primary Head loss and Sub-emotion Head loss.

Theoretical grounding:
    - Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. CVPR 2018.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MultiTaskUncertaintyLoss(nn.Module):
    """Automatically balances multiple classification losses using task uncertainty.

    Math:
        L_total = 0.5 * exp(-s_primary) * L_primary + 0.5 * exp(-s_sub) * L_sub + 0.5 * (s_primary + s_sub)
        where s_i = log(sigma_i^2) is a learnable parameter.

    Args:
        num_tasks: Number of tasks to balance. Default: 2 (Primary + Sub-emotions).
    """

    def __init__(self, num_tasks: int = 2) -> None:
        super().__init__()
        self.num_tasks = num_tasks
        # Initialize log(sigma^2) = 0 so that exp(-s) = 1.0 initially
        self.log_vars = nn.Parameter(torch.zeros(num_tasks, dtype=torch.float32))

    def forward(
        self,
        *args: Union[torch.Tensor, List[torch.Tensor]],
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """Compute weighted multi-task loss.

        Args:
            *args: Either multiple scalar loss tensors, or a single list of scalar tensors.
            return_dict: If True, return (total_loss, metrics_dict).

        Returns:
            total_loss (torch.Tensor) or (total_loss, metrics)
        """
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            losses = list(args[0])
        else:
            losses = list(args)

        if len(losses) != self.num_tasks:
            raise ValueError(f"Expected {self.num_tasks} losses, got {len(losses)}")

        total_loss = torch.tensor(0.0, device=losses[0].device, dtype=torch.float32)
        metrics: Dict[str, float] = {}

        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            weighted_task_loss = 0.5 * precision * loss + 0.5 * self.log_vars[i]
            total_loss = total_loss + weighted_task_loss

            metrics[f"task_{i}_weight"] = float(precision.detach().item())
            metrics[f"task_{i}_raw_loss"] = float(loss.detach().item())

        if return_dict:
            return total_loss, metrics
        return total_loss
