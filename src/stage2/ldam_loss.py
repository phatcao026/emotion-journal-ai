"""
src/stage2/ldam_loss.py

Label-Distribution-Aware Margin (LDAM) Loss with Deferred Re-Weighting (DRW)
and L2-Normalized Cosine Classifier.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CosineLinear(nn.Module):
    """L2-normalized Cosine Linear layer required for bounded LDAM margins in [-1, 1].
    Retains an optional non-trainable bias buffer for 100% state_dict compatibility with nn.Linear.
    """

    def __init__(self, in_features: int, out_features: int, use_bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, p=2, dim=-1, eps=1e-7)
        w_norm = F.normalize(self.weight, p=2, dim=-1, eps=1e-7)
        return F.linear(x_norm, w_norm)


class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin (LDAM) Loss with ignore_index support."""

    def __init__(
        self,
        cls_num_list: Optional[Union[List[int], np.ndarray, torch.Tensor]] = None,
        max_m: float = 0.5,
        scale: float = 30.0,
        weight: Optional[torch.Tensor] = None,
        class_counts: Optional[Union[List[int], np.ndarray, torch.Tensor]] = None,
        s: Optional[float] = None,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        counts = class_counts if class_counts is not None else cls_num_list
        if counts is None:
            raise ValueError("Either cls_num_list or class_counts must be provided.")
        # Clamp minimum count to 1.0 to avoid division by zero on rare/unseen sub-emotions
        cls_num_arr = np.maximum(np.array(counts, dtype=np.float32), 1.0)
        self.cls_num_arr = cls_num_arr
        self.scale = s if s is not None else scale
        self.ignore_index = ignore_index

        m_list = 1.0 / np.power(cls_num_arr, 0.25)
        m_list = m_list * (max_m / np.max(m_list))
        self.register_buffer("m_list", torch.tensor(m_list, dtype=torch.float32))
        self.register_buffer("weight", weight if weight is not None else None)

    def drw_update(self, current_epoch: int, drw_start_epoch: int, beta: float = 0.9999) -> None:
        """Activate Deferred Re-Weighting (DRW) after drw_start_epoch."""
        if current_epoch >= drw_start_epoch:
            device = self.m_list.device
            drw_w = get_drw_weights(self.cls_num_arr.tolist(), beta=beta, device=device)
            self.weight = drw_w
            logger.info("LDAM Deferred Re-Weighting (DRW) activated at epoch %d.", current_epoch)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 2:
            targets = targets.argmax(dim=-1)

        valid_mask = (targets >= 0) & (targets != self.ignore_index)
        if not valid_mask.any():
            return logits.sum() * 0.0

        valid_logits = logits[valid_mask]
        valid_targets = targets[valid_mask]
        batch_size = valid_logits.size(0)

        index = torch.zeros_like(valid_logits, dtype=torch.bool)
        index.scatter_(1, valid_targets.unsqueeze(1), True)

        margins = self.m_list.to(valid_logits.device).unsqueeze(0).expand(batch_size, -1)
        target_margins = margins[index].view(batch_size, 1)

        diff = torch.zeros_like(valid_logits)
        diff.scatter_(1, valid_targets.unsqueeze(1), target_margins)
        output_logits = self.scale * (valid_logits - diff)

        weight = self.weight.to(valid_logits.device) if self.weight is not None else None
        return F.cross_entropy(output_logits, valid_targets, weight=weight)


def get_drw_weights(
    cls_num_list: List[float],
    beta: float = 0.9999,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    cls_nums = np.maximum(np.array(cls_num_list, dtype=np.float32), 1.0)
    effective_num = 1.0 - np.power(beta, cls_nums)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.mean(weights)

    tensor_w = torch.tensor(weights, dtype=torch.float32)
    if device is not None:
        tensor_w = tensor_w.to(device)
    return tensor_w


# ============================================================================
# SIBLING MASKING FOR 5 PRIMARY -> 11 SUB TAXONOMY
# ============================================================================
# 5 coarse sub-emotions with true samples in ViSEC + RAVDESS:
ACTIVE_5_SUB_INDICES: List[int] = [0, 1, 4, 6, 7]
# 6 fine-grained sub-emotions with zero support in ViSEC + RAVDESS:
ZERO_SUPPORT_SPEECH_SUB_INDICES: List[int] = [2, 3, 5, 8, 9, 10]

# Mapping from coarse sub-emotion (ViSEC) to its fine-grained siblings to mask out:
COARSE_SUB_TO_SIBLINGS: Dict[int, List[int]] = {
    0: [2, 3],  # JOY(0) -> siblings HOPE(2), CONNECTION(3)
    1: [],      # CALM(1)
    4: [8, 9],  # SADNESS(4) -> siblings GUILT_SHAME(8), LONELINESS(9)
    6: [5],     # FEAR(6) -> sibling ANXIETY(5)
    7: [10],    # ANGER(7) -> sibling DISGUST(10)
}


class SiblingMaskedLDAMLoss(LDAMLoss):
    """
    LDAM Loss with Sibling Negative Masking support for hierarchical coarse labels.
    Prevents coarse speech batches from penalizing fine-grained sibling predictions.
    """
    def __init__(
        self,
        cls_num_list: Optional[Union[List[int], np.ndarray, torch.Tensor]] = None,
        max_m: float = 0.5,
        scale: float = 30.0,
        weight: Optional[torch.Tensor] = None,
        coarse_to_siblings: Optional[Dict[int, List[int]]] = None,
        ignore_index: int = -100,
        num_classes: int = 11,
        use_sibling_masking: bool = True,
    ) -> None:
        super().__init__(
            cls_num_list=cls_num_list,
            max_m=max_m,
            scale=scale,
            weight=weight,
            ignore_index=ignore_index,
        )
        self.use_sibling_masking = use_sibling_masking
        mapping = coarse_to_siblings if coarse_to_siblings is not None else COARSE_SUB_TO_SIBLINGS
        mask_table = torch.zeros((num_classes, num_classes), dtype=torch.bool)
        for coarse_idx, siblings in mapping.items():
            for sib_idx in siblings:
                if coarse_idx < num_classes and sib_idx < num_classes:
                    mask_table[coarse_idx, sib_idx] = True
        self.register_buffer("sibling_mask_table", mask_table)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        is_coarse_speech: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if targets.ndim == 2:
            targets = targets.argmax(dim=-1)

        valid_mask = (targets >= 0) & (targets != self.ignore_index)
        if not valid_mask.any():
            return logits.sum() * 0.0

        valid_logits = logits[valid_mask]
        valid_targets = targets[valid_mask]
        batch_size = valid_logits.size(0)

        index = torch.zeros_like(valid_logits, dtype=torch.bool)
        index.scatter_(1, valid_targets.unsqueeze(1), True)

        margins = self.m_list.to(valid_logits.device).unsqueeze(0).expand(batch_size, -1)
        target_margins = margins[index].view(batch_size, 1)

        diff = torch.zeros_like(valid_logits)
        diff.scatter_(1, valid_targets.unsqueeze(1), target_margins)
        output_logits = self.scale * (valid_logits - diff)

        # Sibling Masking for coarse speech samples
        if self.use_sibling_masking and is_coarse_speech is not None:
            valid_coarse = is_coarse_speech[valid_mask].bool().to(valid_logits.device)
            if valid_coarse.any():
                sib_mask = self.sibling_mask_table[valid_targets] & valid_coarse.unsqueeze(1)
                output_logits = output_logits.masked_fill(sib_mask, -1e4)

        weight = self.weight.to(valid_logits.device) if self.weight is not None else None
        return F.cross_entropy(output_logits, valid_targets, weight=weight)

