"""
src/stage2/gated_fusion_stage2.py
---------------------------------
Stage 2 SOTA Cross-Modal Gated Fusion Module.
Fuses acoustic SSL (emotion2vec+), semantic text (PhoBERT), paralinguistic pauses,
and explicit acoustic melody biomarkers (F0, jitter, shimmer, HNR).

Mathematical formulation:
    - Gating: g_a = sigma(W_g [z_audio; z_semantic] + b_g)
              g_t = 1 - g_a
              z_bimodal = g_a * z_audio + g_t * z_semantic
    - Projections: z_pause_proj = Linear(p_pause) in R^128
                   z_melody_proj = Linear(p_melody) in R^dim_melody
    - Tri-modal Fusion: z_fused = LayerNorm([z_bimodal; z_pause_proj; z_melody_proj]) in R^(768 + 128 + dim_melody)
"""

import logging
from typing import Optional, Tuple
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Stage2GatedFusion(nn.Module):
    """Stage 2 Multi-Modal Gated Fusion Layer with Acoustic Biomarkers.

    Args:
        dim_audio: Dimension of acoustic SSL embedding (default: 768).
        dim_semantic: Dimension of PhoBERT text embedding (default: 768).
        dim_pause: Dimension of projected pause feature (default: 128).
        pause_raw_dim: Raw pause features count (default: 4).
        dim_melody: Dimension of projected melody biomarkers (default: 32).
        melody_raw_dim: Raw physical melody features count (default: 32).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        dim_audio: int = 768,
        dim_semantic: int = 768,
        dim_pause: int = 128,
        pause_raw_dim: int = 4,
        dim_melody: int = 32,
        melody_raw_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.dim_audio = dim_audio
        self.dim_semantic = dim_semantic
        self.dim_pause = dim_pause
        self.pause_raw_dim = pause_raw_dim
        self.dim_melody = dim_melody
        self.melody_raw_dim = melody_raw_dim
        self.output_dim = dim_audio + dim_pause + (dim_melody if dim_melody > 0 else 0)

        # 1. Paralinguistic pause projection: R^4 -> R^128
        self.pause_projection = nn.Sequential(
            nn.Linear(pause_raw_dim, dim_pause),
            nn.LayerNorm(dim_pause),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 2. Melody physical biomarker projection: R^32 -> R^dim_melody
        if dim_melody > 0:
            self.melody_projection: Optional[nn.Module] = nn.Sequential(
                nn.Linear(melody_raw_dim, dim_melody),
                nn.LayerNorm(dim_melody),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.melody_projection = None

        # 3. Gating network: [Z_audio; Z_semantic] -> g in R^dim_audio
        gate_in_dim = dim_audio + dim_semantic
        self.gate_layer = nn.Sequential(
            nn.Linear(gate_in_dim, dim_audio),
            nn.Sigmoid(),
        )

        # 4. Normalization and dropout
        self.layer_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        z_audio: torch.Tensor,
        z_semantic: torch.Tensor,
        p_pause: torch.Tensor,
        p_melody: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Perform multimodal fusion.

        Args:
            z_audio: (B, dim_audio)
            z_semantic: (B, dim_semantic)
            p_pause: (B, pause_raw_dim)
            p_melody: Optional (B, melody_raw_dim)

        Returns:
            Tuple of:
                z_fused: (B, output_dim)
                (gate_audio, gate_text): each (B, dim_audio)
        """
        # 1. Project pause features
        z_pause_proj = self.pause_projection(p_pause)  # (B, 128)

        # 2. Calculate dynamic modal gates
        concat_modalities = torch.cat([z_audio, z_semantic], dim=-1)  # (B, 1536)
        gate_audio = self.gate_layer(concat_modalities)  # (B, 768)
        gate_text = 1.0 - gate_audio

        # 3. Modality blending
        z_bimodal = gate_audio * z_audio + gate_text * z_semantic  # (B, 768)

        # 4. Concatenate projected streams
        streams = [z_bimodal, z_pause_proj]
        if p_melody is not None and self.melody_projection is not None:
            z_melody_proj = self.melody_projection(p_melody)  # (B, dim_melody)
            streams.append(z_melody_proj)

        z_fused_raw = torch.cat(streams, dim=-1)  # (B, output_dim)
        z_fused = self.dropout(self.layer_norm(z_fused_raw))

        return z_fused, (gate_audio, gate_text)
