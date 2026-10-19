"""
src/stage2/gated_fusion_stage2.py
---------------------------------
Stage 2 SOTA Cross-Modal Gated Fusion Module.
"""

import logging
from typing import Optional, Tuple
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Stage2GatedFusion(nn.Module):
    """Stage 2 Multi-Modal Gated Fusion Layer with Acoustic Biomarkers."""

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
        gate_in_dim = dim_audio + dim_semantic
        self.gate_layer = nn.Sequential(
            nn.Linear(gate_in_dim, dim_audio),
            nn.Sigmoid(),
        )

        # 4. Normalization and dropout
        self.layer_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(dropout)

        # 5. Decoupled routing fallback for text-only samples (e_no_prosody: 128 + 32 = 160D)
        prosody_dim = dim_pause + (dim_melody if dim_melody > 0 else 0)
        self.e_no_prosody = nn.Parameter(torch.zeros(1, prosody_dim))
        nn.init.normal_(self.e_no_prosody, mean=0.0, std=0.02)
        self.use_decoupled_routing = False

    def forward(
        self,
        z_audio: torch.Tensor,
        z_semantic: torch.Tensor,
        p_pause: torch.Tensor,
        p_melody: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = z_audio.shape[0]
        device = z_audio.device

        effective_z_audio = z_audio
        effective_z_semantic = z_semantic
        audio_keep_factor = torch.ones(batch_size, 1, device=device)

        # 1. Safe Modality Dropout (only drops a modality if the other modality is present)
        if self.training and self.modality_dropout > 0.0:
            rand_val = torch.rand(batch_size, 1, device=device)
            drop_text_mask = (rand_val < self.modality_dropout).float()
            drop_audio_mask = (
                (rand_val >= self.modality_dropout)
                & (rand_val < 1.5 * self.modality_dropout)
            ).float()

            if audio_mask is not None:
                drop_text_mask = drop_text_mask * audio_mask.view(-1, 1).float()
            if text_mask is not None:
                drop_audio_mask = drop_audio_mask * text_mask.view(-1, 1).float()

            effective_z_semantic = effective_z_semantic * (1.0 - drop_text_mask)
            audio_keep_factor = 1.0 - drop_audio_mask
            effective_z_audio = effective_z_audio * audio_keep_factor

        if audio_mask is not None:
            audio_keep_factor = audio_keep_factor * audio_mask.view(-1, 1).float()
            effective_z_audio = effective_z_audio * audio_mask.view(-1, 1).float()

        if text_mask is not None:
            effective_z_semantic = effective_z_semantic * text_mask.view(-1, 1).float()

        # 2. Project pause features and mask when audio is absent
        z_pause_proj = self.pause_projection(p_pause) * audio_keep_factor

        # 3. Calculate dynamic modal gates with temperature scaling
        concat_modalities = torch.cat([effective_z_audio, effective_z_semantic], dim=-1)
        linear_module = self.gate_layer[0]
        gate_logits = linear_module(concat_modalities) / self.gate_temperature
        gate_audio = torch.sigmoid(gate_logits)
        gate_text = 1.0 - gate_audio

        # 4. Handle explicitly masked or missing modalities
        if text_mask is not None or audio_mask is not None:
            tm = text_mask.view(-1, 1).float() if text_mask is not None else torch.ones(batch_size, 1, device=device)
            am = audio_mask.view(-1, 1).float() if audio_mask is not None else torch.ones(batch_size, 1, device=device)

            only_text = (am <= 0.5) & (tm > 0.5)
            only_audio = (tm <= 0.5)

            gate_audio = torch.where(only_audio, torch.ones_like(gate_audio), gate_audio)
            gate_audio = torch.where(only_text, torch.zeros_like(gate_audio), gate_audio)
            gate_text = 1.0 - gate_audio

        # 5. Modality blending
        z_bimodal = gate_audio * effective_z_audio + gate_text * effective_z_semantic

        # 6. Concatenate projected streams (with masked melody when audio is absent)
        if self.use_decoupled_routing and audio_mask is not None:
            am = audio_mask.view(-1, 1).float()
            is_text_only = (am <= 0.5)
            z_mel = self.melody_projection(p_melody) if (p_melody is not None and self.melody_projection is not None) else torch.zeros(batch_size, self.dim_melody, device=device)
            prosody_real = torch.cat([self.pause_projection(p_pause), z_mel], dim=-1)
            prosody_fallback = self.e_no_prosody.expand(batch_size, -1)
            prosody_fused = torch.where(is_text_only, prosody_fallback, prosody_real)
            streams = [z_bimodal, prosody_fused]
        else:
            streams = [z_bimodal, z_pause_proj]
            if p_melody is not None and self.melody_projection is not None:
                z_melody_proj = self.melody_projection(p_melody) * audio_keep_factor
                streams.append(z_melody_proj)

        z_fused_raw = torch.cat(streams, dim=-1)
        z_fused = self.dropout(self.layer_norm(z_fused_raw))

        return z_fused, (gate_audio, gate_text)
