"""
attention_pooling.py

Temporal Attention Pooling: aggregates frame-level features into a single
utterance-level vector via learned self-attention over the time axis.

Math:
    e_n  = tanh(W_1 x_n + b_1)                      (B, N, d_hidden)
    a_n  = softmax_N(W_2 e_n + b_2)                  (B, N, 1)  → squeeze → alpha (B, N)
    Z    = sum_n( alpha_n * x_n )                     (B, d_out)

Exported:
    TemporalAttentionPooling   (B, N, 768) → Z_audio (B, 768) + alpha (B, N)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TemporalAttentionPooling(nn.Module):
    """Self-attention pooling over the temporal axis.

    Args:
        input_dim:  Frame feature dimensionality (d_in).  Default 768.
        hidden_dim: Attention scorer hidden dim.  Default 256.
        output_dim: Output dimensionality (d_out).  Default 768.
                    A linear projection is added when ``output_dim != input_dim``.
        dropout:    Dropout applied to attention energies.  Default 0.1.

    Shapes:
        - x:     ``(B, N, input_dim)``
        - mask:  ``(B, N)`` bool — True = valid, False = padding.
        - Returns ``(z_audio, alpha)`` with shapes ``(B, output_dim)`` and ``(B, N)``.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        output_dim: int = 768,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be > 0")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Attention energy network
        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, 1)
        self.drop = nn.Dropout(dropout)

        # Optional output projection
        self.projection: Optional[nn.Linear] = None
        if output_dim != input_dim:
            self.projection = nn.Linear(input_dim, output_dim)

        logger.info(
            "TemporalAttentionPooling: d_in=%d, d_hid=%d, d_out=%d",
            input_dim, hidden_dim, output_dim,
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x:    Frame features ``(B, N, input_dim)``.
            mask: Padding mask  ``(B, N)`` — True at valid positions.

        Returns:
            ``(z_audio, alpha)`` with shapes ``(B, output_dim)`` and ``(B, N)``.
            ``alpha`` sums to 1 along dim-1.
        """
        # Attention energies
        e = torch.tanh(self.W1(x))          # (B, N, hidden_dim)
        e = self.drop(e)
        scores = self.W2(e).squeeze(-1)     # (B, N)

        # Mask padding positions to -inf before softmax
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        alpha = F.softmax(scores, dim=1)    # (B, N)  — sums to 1

        # Weighted sum
        z = torch.bmm(alpha.unsqueeze(1), x).squeeze(1)   # (B, input_dim)

        # Optional projection
        if self.projection is not None:
            z = self.projection(z)           # (B, output_dim)

        return z, alpha

    # ------------------------------------------------------------------
    def compute_attention_weights(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return attention weights without pooling (useful for visualization)."""
        e = torch.tanh(self.W1(x))
        scores = self.W2(e).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        return F.softmax(scores, dim=1)

    def get_top_k_frames(
        self, alpha: torch.Tensor, k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-k most-attended frame indices."""
        return torch.topk(alpha, k=min(k, alpha.size(1)), dim=1)

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, output_dim={self.output_dim}"
