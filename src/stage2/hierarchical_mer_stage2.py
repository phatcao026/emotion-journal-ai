"""
src/stage2/hierarchical_mer_stage2.py
-------------------------------------
Stage 2 SOTA Hierarchical Multimodal Emotion Recognition Model.
Features:
    1. Hybrid Acoustic Encoder: emotion2vec+ (768) + ExplicitMelodyExtractor (32) + TemporalAttentionPooling.
    2. Multilingual Semantic Encoder: PhoBERT v2 (768).
    3. Multimodal Gated Fusion: Stage2GatedFusion with Modality Dropout.
    4. Hierarchical Dual-Head:
       - Primary Head takes fused representation (928D).
       - Sub-Emotion Head takes full multimodal representation (928D) with backward compatibility.
    5. Soft Bayesian Probabilistic Routing & Joint MAP Inference:
       - Mitigates cascading error when primary head is ambiguous.
       - Guarantees 0.00% taxonomy constraint violation under joint inference.
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
from src.stage2.ldam_loss import CosineLinear

logger = logging.getLogger(__name__)

# Canonical 5-class Primary -> 11-class Sub Emotion mapping from data/taxonomy.json
TAXONOMY_MAP: Dict[int, List[int]] = {
    0: [0, 1, 2, 3],    # JOY -> JOY (0), CALM (1), HOPE (2), CONNECTION (3)
    1: [4, 8, 9],       # SADNESS -> SADNESS (4), GUILT_SHAME (8), LONELINESS (9)
    2: [5, 6],          # ANXIETY -> ANXIETY (5), FEAR (6)
    3: [7, 10],         # ANGER -> ANGER (7), DISGUST (10)
    4: [1],             # NEUTRAL -> CALM (1)
}

# First-person Vietnamese psychological descriptions for anchoring 11 Sub-Emotion prototypes
VIETNAMESE_SUB_PROTOTYPE_PROMPTS: Dict[int, str] = {
    0: "Tôi cảm thấy rất vui vẻ, hạnh phúc, hào hứng và tràn đầy năng lượng tích cực.",
    1: "Tâm trạng tôi hôm nay bình yên, thư thái, nhẹ nhàng, tĩnh lặng và cân bằng.",
    2: "Tôi tràn đầy hy vọng, lạc quan, mong chờ và tin tưởng vào những điều tốt đẹp phía trước.",
    3: "Tôi cảm thấy gắn kết, yêu thương, biết ơn, được sẻ chia và thấu hiểu sâu sắc.",
    4: "Tôi cảm thấy buồn bã, thất vọng, chán nản và lòng nặng trĩu nỗi buồn.",
    5: "Tôi cảm thấy lo âu, bồn chồn, căng thẳng, áp lực và suy nghĩ không ngừng.",
    6: "Tôi cảm thấy sợ hãi, hoảng loạn, bất an và lo sợ hiểm nguy sắp xảy ra.",
    7: "Tôi cảm thấy tức giận, bực bội, phẫn nộ và bức xúc.",
    8: "Tôi cảm thấy tội lỗi, xấu hổ, hối hận, day dứt và tự trách bản thân mình.",
    9: "Tôi cảm thấy cô đơn, trống trải, lẻ loi một mình và không có ai bên cạnh thấu hiểu.",
    10: "Tôi cảm thấy chán ghét, ác cảm, khó chịu và muốn khước từ mọi thứ.",
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


def compute_soft_hierarchical_probs(
    primary_logits: torch.Tensor,
    sub_logits: torch.Tensor,
    mask_matrix: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute soft Bayesian hierarchical distribution over sub-emotions:
    P(sub_s | X) = sum_p P(primary_p | X) * P(sub_s | primary_p, X)
    Prevents single-label cascading error when primary class is ambiguous.
    """
    if mask_matrix is None:
        mask_matrix = TAXONOMY_MASK_MATRIX
    device = sub_logits.device
    mask_matrix = mask_matrix.to(device)

    primary_probs = F.softmax(primary_logits, dim=-1)  # (B, num_primary)
    batch_size, num_sub = sub_logits.shape
    num_primary = primary_probs.shape[1]

    # Masked softmax over allowed sub classes per primary class
    large_neg = torch.tensor(-1e4, device=device, dtype=sub_logits.dtype)
    sub_exp = sub_logits.unsqueeze(1).expand(batch_size, num_primary, num_sub)
    mask_exp = mask_matrix.unsqueeze(0).expand(batch_size, num_primary, num_sub)

    masked_sub_per_p = torch.where(mask_exp > 0.5, sub_exp, large_neg)
    probs_sub_given_p = F.softmax(masked_sub_per_p, dim=-1)  # (B, num_primary, num_sub)

    # Marginalize over primary classes: (B, num_sub)
    soft_sub_probs = torch.sum(primary_probs.unsqueeze(-1) * probs_sub_given_p, dim=1)
    return soft_sub_probs


def joint_hierarchical_prediction(
    primary_logits: torch.Tensor,
    sub_logits: torch.Tensor,
    mask_matrix: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Joint MAP (p*, s*) = argmax_{(p, s): mask[p, s]=1} P(p) * P(s|p).
    Guarantees 0.00% taxonomy constraint violation while allowing sub evidence to inform primary.
    """
    if mask_matrix is None:
        mask_matrix = TAXONOMY_MASK_MATRIX
    device = sub_logits.device
    mask_matrix = mask_matrix.to(device)

    primary_probs = F.softmax(primary_logits, dim=-1)  # (B, P)
    batch_size, num_sub = sub_logits.shape
    num_primary = primary_probs.shape[1]

    large_neg = torch.tensor(-1e4, device=device, dtype=sub_logits.dtype)
    sub_exp = sub_logits.unsqueeze(1).expand(batch_size, num_primary, num_sub)
    mask_exp = mask_matrix.unsqueeze(0).expand(batch_size, num_primary, num_sub)

    masked_sub = torch.where(mask_exp > 0.5, sub_exp, large_neg)
    probs_sub_given_p = F.softmax(masked_sub, dim=-1)  # (B, P, S)

    # Joint probability matrix: P(p, s) = P(p) * P(s|p) of shape (B, P, S)
    joint_probs = primary_probs.unsqueeze(-1) * probs_sub_given_p * mask_exp

    # Flatten (P, S) -> (P * S) to find global argmax per sample
    flat_joint = joint_probs.view(batch_size, -1)
    best_flat_idx = torch.argmax(flat_joint, dim=-1)

    p_star = best_flat_idx // num_sub
    s_star = best_flat_idx % num_sub
    return p_star, s_star


@dataclass
class Stage2ModelOutput:
    """Output container for Stage2HierarchicalMERModel."""
    primary_logits: torch.Tensor        # (B, 5)
    sub_logits: torch.Tensor            # (B, 11) raw unconstrained
    primary_probs: torch.Tensor         # (B, 5)
    sub_probs: torch.Tensor             # (B, 11) raw unconstrained
    masked_sub_logits: torch.Tensor     # (B, 11) with -inf mask applied
    masked_sub_probs: torch.Tensor      # (B, 11) normalized over valid sub-emotions
    soft_sub_probs: torch.Tensor        # (B, 11) Bayesian soft hierarchical distribution
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

    def sub_predictions(self, enforce_taxonomy: bool = True, mode: str = "joint") -> torch.Tensor:
        """Returns integer class predictions for sub emotions.
        Args:
            enforce_taxonomy: Whether to enforce taxonomy constraint.
            mode: 'joint' (Joint MAP with 0% violation), 'soft' (Soft Bayesian MAP), or 'hard' (-inf mask).
        """
        if not enforce_taxonomy:
            return torch.argmax(self.sub_logits, dim=-1)

        if mode == "joint":
            _, s_pred = joint_hierarchical_prediction(self.primary_logits, self.sub_logits)
            return s_pred
        elif mode == "soft":
            return torch.argmax(self.soft_sub_probs, dim=-1)
        else:
            return torch.argmax(self.masked_sub_logits, dim=-1)

    def joint_predictions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns mutually consistent (primary, sub) class predictions."""
        return joint_hierarchical_prediction(self.primary_logits, self.sub_logits)


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
        sub_head_input_dim: Optional[int] = None,
        asr_model_id: str = "vinai/phowhisper-base",
        text_model_id: str = "vinai/phobert-base-v2",
        modality_dropout: float = 0.15,
        gate_temperature: float = 0.8,
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
            modality_dropout=modality_dropout,
            gate_temperature=gate_temperature,
        )

        # 5. Dual Classification Heads
        # Primary Head input: z_fused in R^(768 + 128 + dim_melody) = 928D
        fused_dim = 768 + 128 + self.dim_melody
        self.fused_dim = fused_dim
        self.primary_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            CosineLinear(256, num_primary_classes),
        )

        # Sub Head: by default connects to fused_dim (928D) so it receives both acoustic and semantic context.
        # Can be set to 768 for legacy checkpoints.
        self.sub_head_input_dim = sub_head_input_dim if sub_head_input_dim is not None else fused_dim
        self.sub_head = nn.Sequential(
            nn.Linear(self.sub_head_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            CosineLinear(256, num_sub_classes),
        )

        self.register_buffer("taxonomy_mask", TAXONOMY_MASK_MATRIX)

    def load_pretrained(self) -> None:
        """Download and load pretrained weights for underlying backbones."""
        if hasattr(self.audio_encoder, "load_pretrained"):
            self.audio_encoder.load_pretrained()
        if hasattr(self.asr, "load_pretrained"):
            self.asr.load_pretrained()
        elif hasattr(self.asr, "load_model"):
            self.asr.load_model()
        if hasattr(self.text_encoder, "load_pretrained"):
            self.text_encoder.load_pretrained()
        logger.info("Stage 2 MER backbones successfully loaded.")

    def load_checkpoint_compatible(
        self,
        checkpoint_path: Union[str, Path],
        map_location: Union[str, torch.device] = "cpu",
    ) -> Tuple[List[str], List[str]]:
        """Load checkpoint with automatic shape adaptation (e.g., expanding sub_head from 768 -> 928)."""
        ckpt = torch.load(str(checkpoint_path), map_location=map_location, weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        model_dict = self.state_dict()

        adapted_dict = {}
        for k, v in state_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    adapted_dict[k] = v
                elif k == "sub_head.0.weight" and v.shape == torch.Size([256, 768]) and model_dict[k].shape == torch.Size([256, 928]):
                    # Expand weights: preserve 768D weights, initialize remaining 160D with small random values
                    expanded_w = model_dict[k].clone()
                    expanded_w[:, :768] = v
                    expanded_w[:, 768:] = 0.01 * torch.randn_like(expanded_w[:, 768:])
                    adapted_dict[k] = expanded_w
                    logger.info("Adapted sub_head.0.weight from [256, 768] to [256, 928] with zero-padding.")
                else:
                    logger.warning("Skipping incompatible tensor %s: ckpt %s vs model %s", k, v.shape, model_dict[k].shape)

        missing, unexpected = self.load_state_dict(adapted_dict, strict=False)
        logger.info("Checkpoint loaded: %d tensors loaded, missing=%d, unexpected=%d", len(adapted_dict), len(missing), len(unexpected))
        return missing, unexpected

    @torch.no_grad()
    def init_semantic_sub_prototypes(self) -> None:
        """Anchor sub_head CosineLinear weights using PhoBERT embeddings of Vietnamese emotion descriptions."""
        prompts = [VIETNAMESE_SUB_PROTOTYPE_PROMPTS[i] for i in range(self.num_sub_classes)]
        z_sem = self.text_encoder(prompts).to(self.device)  # (11, 768)
        if self.sub_head_input_dim > 768:
            pad = torch.zeros(self.num_sub_classes, self.sub_head_input_dim - 768, device=self.device)
            sub_in = torch.cat([z_sem, pad], dim=-1)
        else:
            sub_in = z_sem[:, :self.sub_head_input_dim]

        hidden = self.sub_head[2](self.sub_head[1](self.sub_head[0](sub_in)))  # (11, 256)
        self.sub_head[4].weight.data.copy_(F.normalize(hidden, p=2, dim=-1))
        logger.info("Initialized 11 Sub-Emotion Cosine prototypes from Vietnamese PhoBERT descriptions.")

    def set_audio_trainable(self, trainable: bool = True) -> None:
        """Control trainability of audio branch, ensuring foundation backbone stays strictly frozen."""
        if hasattr(self.audio_encoder, "set_lora_trainable"):
            self.audio_encoder.set_lora_trainable(trainable)
        else:
            for name, p in self.audio_encoder.named_parameters():
                if "lora_" in name.lower():
                    p.requires_grad = trainable
                else:
                    p.requires_grad = False
        logger.info("Stage2HierarchicalMERModel: set_audio_trainable(%s)", trainable)

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
        audio_mask = (waveforms.abs().amax(dim=-1) > 1e-5)
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
        text_mask = None
        if texts is None and use_asr:
            texts = [self.asr.transcribe(waveforms[i], sample_rate=16000) for i in range(waveforms.size(0))]
        elif texts is None:
            texts = [""] * waveforms.size(0)

        # Build text_mask indicating whether input text is non-empty
        text_mask = torch.tensor([bool(t and t.strip()) for t in texts], device=device, dtype=torch.bool)
        z_semantic = self.text_encoder(texts).to(device)

        # 5. Multimodal fusion with text_mask and audio_mask
        z_fused, (gate_audio, gate_text) = self.fusion(
            z_audio=z_audio,
            z_semantic=z_semantic,
            p_pause=p_pause,
            p_melody=p_melody,
            text_mask=text_mask,
            audio_mask=audio_mask,
        )

        # 6. Classification logits
        primary_logits = self.primary_head(z_fused)

        # Sub-head receives full fused representation if configured (928D), else semantic (768D)
        sub_input = z_fused if self.sub_head_input_dim == self.fused_dim else z_semantic
        sub_logits = self.sub_head(sub_input)

        # 7. Hierarchical Masking & Soft Probabilistic Routing
        masked_sub_logits = apply_hierarchical_mask(
            primary_logits=primary_logits,
            sub_logits=sub_logits,
            mask_matrix=self.taxonomy_mask,
        )
        soft_sub_probs = compute_soft_hierarchical_probs(
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
            soft_sub_probs=soft_sub_probs,
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
