"""
gated_fusion.py

Cross-Modal Gated Fusion: combines features from three information streams
into a unified vector Z_fused in R^(B x 896).

Input streams:
    - Z_audio    in R^(B x 768): Acoustic features from TemporalAttentionPooling
    - Z_semantic in R^(B x 768): Semantic features from PhoBERT
    - P_pause    in R^(B x 4):   Pause features from EmotionalPauseAnalyzer

Gated Fusion mechanism (per VOICE_PIPELINE_SPEC.md):
    1. Z_pause_proj = Dropout(ReLU(Linear(4 -> 128)))(P_pause) in R^(B x 128)
    2. Gating vector: g = sigmoid(W_g [Z_audio; Z_semantic] + b_g) in R^(B x 768)
    3. Z_bimodal = g * Z_audio + (1 - g) * Z_semantic             in R^(B x 768)
    4. Z_fused   = LayerNorm([Z_bimodal ; Z_pause_proj])          in R^(B x 896)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CrossModalGatedFusion(nn.Module):
    """Gated fusion combining acoustic, semantic, and pause features into Z_fused.

    Args:
        dim_audio (int): Dimensionality of Z_audio. Default: 768.
        dim_semantic (int): Dimensionality of Z_semantic. Default: 768.
        dim_pause (int): Dimensionality of projected pause features. Default: 128.
        pause_raw_dim (int): Dimensionality of input pause vector. Default: 4.
        output_dim (int): Dimensionality of Z_fused. Default: 896 (= 768 + 128).
        dropout (float): Dropout probability. Default: 0.2.

    Inputs:
        z_audio: (B, 768) acoustic vector.
        z_semantic: (B, 768) semantic vector.
        p_pause: (B, 4) paralinguistic pause vector.

    Outputs:
        Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            - z_fused: (B, 896) fused multimodal vector.
            - (gate_audio, gate_text): Gate tensors (each in [0, 1] with shape (B, 768)).
    """

    OUTPUT_DIM: int = 896

    def __init__(
        self,
        dim_audio: int = 768,
        dim_semantic: int = 768,
        dim_pause: int = 128,
        pause_raw_dim: int = 4,
        output_dim: int = 896,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.dim_audio = dim_audio
        self.dim_semantic = dim_semantic
        self.dim_pause = dim_pause
        self.pause_raw_dim = pause_raw_dim
        self.output_dim = output_dim

        expected_output = dim_audio + dim_pause
        if output_dim != expected_output:
            raise ValueError(
                f"output_dim={output_dim} must equal dim_audio + dim_pause = "
                f"{dim_audio} + {dim_pause} = {expected_output}"
            )

        # 1. Paralinguistic projection: R^4 -> R^128
        self.pause_projection = nn.Sequential(
            nn.Linear(pause_raw_dim, dim_pause),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 2. Gate network: [Z_audio; Z_semantic] -> g in R^768
        gate_in_dim = dim_audio + dim_semantic
        self.gate_layer = nn.Sequential(
            nn.Linear(gate_in_dim, dim_audio),
            nn.Sigmoid(),
        )

        # 3. Output normalization & dropout
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        logger.info(
            "CrossModalGatedFusion initialized: audio=%d, semantic=%d, pause=%d->%d, out=%d",
            dim_audio,
            dim_semantic,
            pause_raw_dim,
            dim_pause,
            output_dim,
        )

    def forward(
        self,
        z_audio: torch.Tensor,
        z_semantic: torch.Tensor,
        p_pause: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass combining acoustic, semantic, and pause streams.

        Args:
            z_audio (torch.Tensor): shape (B, 768).
            z_semantic (torch.Tensor): shape (B, 768).
            p_pause (torch.Tensor): shape (B, 4).

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - z_fused: shape (B, 896).
                - (gate_audio, gate_text): each shape (B, 768).
        """
        # 1. Project pause features
        z_pause_proj = self.pause_projection(p_pause)  # (B, 128)

        # 2. Concatenate audio and text
        concat_bimodal = torch.cat([z_audio, z_semantic], dim=-1)  # (B, 1536)

        # 3. Compute gating vector g in [0, 1]^(B x 768)
        gate_audio = self.gate_layer(concat_bimodal)  # g
        gate_text = 1.0 - gate_audio                  # 1 - g

        # 4. Gated combination
        z_bimodal = gate_audio * z_audio + gate_text * z_semantic  # (B, 768)

        # 5. Concatenate with projected pause representation
        z_fused_raw = torch.cat([z_bimodal, z_pause_proj], dim=-1)  # (B, 896)

        # 6. Normalization and dropout
        z_fused = self.dropout(self.layer_norm(z_fused_raw))

        return z_fused, (gate_audio, gate_text)

    def get_modality_importance(
        self,
        z_audio: torch.Tensor,
        z_semantic: torch.Tensor,
        p_pause: torch.Tensor,
    ) -> Dict[str, Union[float, torch.Tensor]]:
        """Compute relative contribution metrics for audio vs. text modalities.

        Args:
            z_audio: shape (B, 768).
            z_semantic: shape (B, 768).
            p_pause: shape (B, 4).

        Returns:
            dict containing mean gate values and full gate tensors.
        """
        with torch.no_grad():
            _, (gate_audio, gate_text) = self.forward(z_audio, z_semantic, p_pause)
            return {
                "audio_importance": float(gate_audio.mean().item()),
                "text_importance": float(gate_text.mean().item()),
                "audio_gate": gate_audio,
                "text_gate": gate_text,
            }

    def extra_repr(self) -> str:
        """String representation of module configuration."""
        return (
            f"d_audio={self.dim_audio}, d_semantic={self.dim_semantic}, "
            f"d_pause={self.pause_raw_dim}->{self.dim_pause}, "
            f"d_out={self.output_dim}"
        )
