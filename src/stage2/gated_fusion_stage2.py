"""
src/stage2/gated_fusion_stage2.py
---------------------------------
Stage 2 SOTA Cross-Modal Gated Fusion Module.
Fuses acoustic SSL (emotion2vec+), semantic text (PhoBERT), paralinguistic pauses,
and explicit acoustic melody biomarkers (F0, jitter, shimmer, HNR).

Features:
    - Dynamic Modality Gating with Temperature Sharpening
    - Modality Dropout (Audio/Text DropPath) to prevent semantic dominance
    - Adaptive handling for missing / empty modalities
    - Full backward compatibility with pretrained Stage 2 checkpoints
"""

import logging
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        modality_dropout: Probability of randomly dropping a modality during training (default: 0.15).
        gate_temperature: Temperature for gate sharpness (default: 1.0; < 1.0 sharpens selection).
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
        modality_dropout: float = 0.15,
        gate_temperature: float = 0.8,
    ) -> None:
        super().__init__()
        self.dim_audio = dim_audio
        self.dim_semantic = dim_semantic
        self.dim_pause = dim_pause
        self.pause_raw_dim = pause_raw_dim
        self.dim_melody = dim_melody
        self.melody_raw_dim = melody_raw_dim
        self.modality_dropout = modality_dropout
        self.gate_temperature = max(1e-3, gate_temperature)
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
        # Preserves exact parameter naming (gate_layer.0) for checkpoint compatibility
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
        text_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Perform multimodal fusion.

        Args:
            z_audio: (B, dim_audio)
            z_semantic: (B, dim_semantic)
            p_pause: (B, pause_raw_dim)
            p_melody: Optional (B, melody_raw_dim)
            text_mask: Optional bool tensor (B,) indicating valid non-empty text
            audio_mask: Optional bool tensor (B,) indicating valid audio

        Returns:
            Tuple of:
                z_fused: (B, output_dim)
                (gate_audio, gate_text): each (B, dim_audio)
        """
        batch_size = z_audio.shape[0]
        device = z_audio.device

        # 1. Modality Dropout (Audio-Text DropPath during training to combat PhoBERT dominance)
        effective_z_audio = z_audio
        effective_z_semantic = z_semantic

        if self.training and self.modality_dropout > 0.0:
            # Randomly decide whether to drop text (p=modality_dropout) or audio (p=0.5*modality_dropout)
            rand_val = torch.rand(batch_size, 1, device=device)
            # Drop text branch: force audio encoder to perform independently
            drop_text_mask = (rand_val < self.modality_dropout).float()
            # Drop audio branch: force text encoder to perform independently
            drop_audio_mask = ((rand_val >= self.modality_dropout) & 
                               (rand_val < 1.5 * self.modality_dropout)).float()

            effective_z_semantic = effective_z_semantic * (1.0 - drop_text_mask)
            effective_z_audio = effective_z_audio * (1.0 - drop_audio_mask)

        # 2. Project pause features
        z_pause_proj = self.pause_projection(p_pause)  # (B, 128)

        # 3. Calculate dynamic modal gates with temperature scaling
        concat_modalities = torch.cat([effective_z_audio, effective_z_semantic], dim=-1)  # (B, 1536)
        
        # Access linear layer directly for temperature sharpening
        linear_module = self.gate_layer[0]
        gate_logits = linear_module(concat_modalities) / self.gate_temperature
        gate_audio = torch.sigmoid(gate_logits)  # (B, 768)
        gate_text = 1.0 - gate_audio

        # 4. Handle explicitly masked or missing modalities
        if text_mask is not None:
            # If text is missing/empty, force audio gate -> 1.0
            tm = text_mask.view(-1, 1).float()
            gate_audio = torch.where(tm > 0.5, gate_audio, torch.ones_like(gate_audio))
            gate_text = 1.0 - gate_audio

        if audio_mask is not None:
            am = audio_mask.view(-1, 1).float()
            gate_text = torch.where(am > 0.5, gate_text, torch.ones_like(gate_text))
            gate_audio = 1.0 - gate_text

        # 5. Modality blending
        z_bimodal = gate_audio * effective_z_audio + gate_text * effective_z_semantic  # (B, 768)

        # 6. Concatenate projected streams
        streams = [z_bimodal, z_pause_proj]
        if p_melody is not None and self.melody_projection is not None:
            z_melody_proj = self.melody_projection(p_melody)  # (B, dim_melody)
            streams.append(z_melody_proj)

        z_fused_raw = torch.cat(streams, dim=-1)  # (B, output_dim)
        z_fused = self.dropout(self.layer_norm(z_fused_raw))

        return z_fused, (gate_audio, gate_text)
