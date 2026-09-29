"""
src/stage2/hierarchical_mer_stage2.py
-------------------------------------
Stage 2 SOTA Hierarchical Multimodal Emotion Recognition Model.
Features:
    1. Hybrid Acoustic Encoder: emotion2vec+ (768) + ExplicitMelodyExtractor (32) + TemporalAttentionPooling.
    2. Multilingual Semantic Encoder: PhoBERT v2 (768).
    3. Multimodal Gated Fusion: Stage2GatedFusion.
    4. Hierarchical Inference Mask: Conditionally masks invalid sub-emotion logits with -inf,
       guaranteeing mathematically 0% hierarchical constraint violations.
    5. Clean separation from Stage 1.5 baseline models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.audio.asr_transcriber import PhoWhisperTranscriber
from src.audio.emotion2vec_lora import LoRAEmotion2Vec, LoRAConfig
from src.audio.attention_pooling import TemporalAttentionPooling
from src.audio.vad_preprocessor import VADPreprocessor
from src.text.text_encoder import PhoBERTEncoder
from src.stage2.gated_fusion_stage2 import Stage2GatedFusion
from src.stage2.melody_extractor import ExplicitMelodyExtractor

logger = logging.getLogger(__name__)

# Canonical 5-class Primary -> 11-class Sub Emotion mapping from data/taxonomy.json
TAXONOMY_MAP: Dict[int, List[int]] = {
    0: [0, 1, 2],       # JOY -> HAPPINESS, PRIDE, HOPE/GRATITUDE (CALM, HOPE, CONNECTION)
    1: [4, 8, 9],       # SADNESS -> SADNESS, GUILT_SHAME, LONELINESS
    2: [5, 6],          # ANXIETY -> ANXIETY, FEAR
    3: [7, 10],         # ANGER -> ANGER, DISGUST
    4: [1],             # NEUTRAL -> CALM
}

def build_taxonomy_mask_matrix(num_primary: int = 5, num_sub: int = 11) -> torch.Tensor:
    """Build a (num_primary, num_sub) binary mask matrix where 1 = permitted, 0 = forbidden."""
    mask = torch.zeros(num_primary, num_sub, dtype=torch.float32)
    for p_idx, s_indices in TAXONOMY_MAP.items():
        if p_idx < num_primary:
            for s_idx in s_indices:
                if s_idx < num_sub:
                    mask[p_idx, s_idx] = 1.0
    return mask

TAXONOMY_MASK_MATRIX = build_taxonomy_mask_matrix(5, 11)


def apply_hierarchical_mask(
    primary_logits: torch.Tensor,
    sub_logits: torch.Tensor,
    mask_matrix: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply conditional logit mask on sub_logits based on predicted primary class."""
    if mask_matrix is None:
        mask_matrix = TAXONOMY_MASK_MATRIX
    device = sub_logits.device
    mask_matrix = mask_matrix.to(device)

    # Predicted primary classes: (B,)
    primary_preds = torch.argmax(primary_logits, dim=-1)
    batch_mask = mask_matrix[primary_preds]  # (B, num_sub)

    # Set forbidden classes to -inf
    neg_inf = torch.tensor(float("-inf"), device=device, dtype=sub_logits.dtype)
    masked_sub_logits = torch.where(batch_mask > 0.5, sub_logits, neg_inf)
    return masked_sub_logits


@dataclass
class Stage2ModelOutput:
    """Output container for Stage2HierarchicalMERModel."""
    primary_logits: torch.Tensor        # (B, 5)
    sub_logits: torch.Tensor            # (B, 11) raw unconstrained
    primary_probs: torch.Tensor         # (B, 5)
    sub_probs: torch.Tensor             # (B, 11) raw unconstrained
    masked_sub_logits: torch.Tensor     # (B, 11) with -inf mask applied
    masked_sub_probs: torch.Tensor      # (B, 11) normalized over valid sub-emotions
    z_audio: torch.Tensor               # (B, 768)
    z_semantic: torch.Tensor            # (B, 768)
    z_fused: torch.Tensor               # (B, output_dim)
    p_melody: Optional[torch.Tensor]    # (B, 32)
    attention_weights: Optional[torch.Tensor] = None
    gate_audio: Optional[torch.Tensor] = None
    gate_text: Optional[torch.Tensor] = None

    def primary_predictions(self) -> torch.Tensor:
        """Returns integer class predictions for primary emotions."""
        return torch.argmax(self.primary_logits, dim=-1)

    def sub_predictions(self, enforce_taxonomy: bool = True) -> torch.Tensor:
        """Returns integer class predictions for sub emotions."""
        if enforce_taxonomy:
            return torch.argmax(self.masked_sub_logits, dim=-1)
        return torch.argmax(self.sub_logits, dim=-1)


class Stage2HierarchicalMERModel(nn.Module):
    """Stage 2 SOTA Multimodal Emotion Recognition Model."""

    PRIMARY_LABELS = ["Joy", "Sadness", "Anxiety", "Anger", "Neutral"]
    SUB_LABELS = [
        "JOY", "CALM", "HOPE", "CONNECTION", "SADNESS",
        "ANXIETY", "FEAR", "ANGER", "GUILT_SHAME", "LONELINESS", "DISGUST",
    ]

    def __init__(
        self,
        lora_config: Optional[LoRAConfig] = None,
        num_primary_classes: int = 5,
        num_sub_classes: int = 11,
        fusion_dropout: float = 0.2,
        head_dropout: float = 0.3,
        use_melody: bool = True,
        dim_melody: int = 32,
        enforce_taxonomy_mask: bool = True,
        asr_model_id: str = "vinai/phowhisper-base",
        text_model_id: str = "vinai/phobert-base-v2",
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.num_primary_classes = num_primary_classes
        self.num_sub_classes = num_sub_classes
        self.use_melody = use_melody
        self.dim_melody = dim_melody if use_melody else 0
        self.enforce_taxonomy_mask = enforce_taxonomy_mask

        # 1. Acoustic Stream: SSL + Pooling
        self.audio_encoder = LoRAEmotion2Vec(
            lora_config=lora_config or LoRAConfig(r=8, lora_alpha=16),
            device=device,
        )
        self.pooling = TemporalAttentionPooling(input_dim=768)

        # 2. Acoustic Biomarker Stream: Explicit Melody (F0, Jitter, Shimmer, HNR)
        if self.use_melody:
            self.melody_extractor = ExplicitMelodyExtractor(sample_rate=16000)
        else:
            self.melody_extractor = None

        # 3. Speech-to-Text & Semantic Encoder
        self.asr = PhoWhisperTranscriber(
            model_id=asr_model_id,
            device=device,
            max_new_tokens=256,
        )
        self.text_encoder = PhoBERTEncoder(
            model_id=text_model_id,
            device=device,
        )

        # 4. Gated Fusion
        self.fusion = Stage2GatedFusion(
            dim_audio=768,
            dim_semantic=768,
            dim_pause=128,
            pause_raw_dim=4,
            dim_melody=self.dim_melody,
            melody_raw_dim=32,
            dropout=fusion_dropout,
        )

        # 5. Dual Classification Heads
        # Primary Head input: z_fused in R^(768 + 128 + dim_melody)
        fused_dim = 768 + 128 + self.dim_melody
        self.primary_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(256, num_primary_classes),
        )

        # Sub Head input: semantic + auxiliary acoustic context
        self.sub_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(256, num_sub_classes),
        )

        self.register_buffer("taxonomy_mask", TAXONOMY_MASK_MATRIX)

    def load_pretrained(self) -> None:
        """Download and load pretrained weights for underlying backbones."""
        self.audio_encoder.load_pretrained()
        self.asr.load_pretrained()
        self.text_encoder.load_pretrained()
        logger.info("Stage 2 MER backbones successfully loaded.")

    def forward(
        self,
        waveforms: torch.Tensor,
        speech_timestamps: Optional[List[List[dict]]] = None,
        texts: Optional[List[str]] = None,
        use_asr: bool = False,
    ) -> Stage2ModelOutput:
        """Full forward pass of the Stage 2 model."""
        device = waveforms.device

        # 1. Acoustic branch
        frame_feats = self.audio_encoder(waveforms)
        z_audio, alpha = self.pooling(frame_feats)

        # 2. Melody extraction
        if self.use_melody and self.melody_extractor is not None:
            p_melody = self.melody_extractor(waveforms).to(device)
        else:
            p_melody = None

        # 3. Speech pause features
        p_pause = self._compute_pause_features(speech_timestamps, waveforms, device)

        # 4. Semantic text branch
        if texts is None and use_asr:
            texts = [self.asr.transcribe(waveforms[i], sample_rate=16000) for i in range(waveforms.size(0))]
        elif texts is None:
            texts = [""] * waveforms.size(0)

        z_semantic = self.text_encoder(texts).to(device)

        # 5. Multimodal fusion
        z_fused, (gate_audio, gate_text) = self.fusion(
            z_audio=z_audio,
            z_semantic=z_semantic,
            p_pause=p_pause,
            p_melody=p_melody,
        )

        # 6. Classification logits
        primary_logits = self.primary_head(z_fused)
        sub_logits = self.sub_head(z_semantic)

        # 7. Hierarchical Masking
        masked_sub_logits = apply_hierarchical_mask(
            primary_logits=primary_logits,
            sub_logits=sub_logits,
            mask_matrix=self.taxonomy_mask,
        )

        primary_probs = F.softmax(primary_logits, dim=-1)
        sub_probs = F.softmax(sub_logits, dim=-1)
        masked_sub_probs = F.softmax(masked_sub_logits, dim=-1)

        return Stage2ModelOutput(
            primary_logits=primary_logits,
            sub_logits=sub_logits,
            primary_probs=primary_probs,
            sub_probs=sub_probs,
            masked_sub_logits=masked_sub_logits,
            masked_sub_probs=masked_sub_probs,
            z_audio=z_audio,
            z_semantic=z_semantic,
            z_fused=z_fused,
            p_melody=p_melody,
            attention_weights=alpha,
            gate_audio=gate_audio,
            gate_text=gate_text,
        )

    def _compute_pause_features(
        self,
        speech_timestamps: Optional[List[List[dict]]],
        waveforms: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute 4-dimensional pause descriptors: [ratio, count_per_min, max_dur, mean_dur]."""
        batch_size = waveforms.size(0)
        feats = torch.zeros(batch_size, 4, device=device)
        total_samples = waveforms.size(1)
        total_sec = max(total_samples / 16000.0, 1.0)

        for b in range(batch_size):
            if not speech_timestamps or not speech_timestamps[b]:
                continue
            ts = speech_timestamps[b]
            speech_samples = sum(seg["end"] - seg["start"] for seg in ts)
            pause_samples = max(0, total_samples - speech_samples)
            pause_ratio = pause_samples / total_samples

            pause_durations = []
            for i in range(len(ts) - 1):
                p_len = (ts[i + 1]["start"] - ts[i]["end"]) / 16000.0
                if p_len > 0.05:
                    pause_durations.append(p_len)

            count_per_min = (len(pause_durations) / total_sec) * 60.0
            max_dur = max(pause_durations) if pause_durations else 0.0
            mean_dur = sum(pause_durations) / len(pause_durations) if pause_durations else 0.0

            feats[b] = torch.tensor([pause_ratio, count_per_min, max_dur, mean_dur], device=device)

        return feats
