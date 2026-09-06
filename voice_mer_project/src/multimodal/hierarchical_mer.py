"""
hierarchical_mer.py

Hierarchical Multimodal Emotion Recognition Model for Vietnamese Speech.
Integrates acoustic, semantic, and paralinguistic streams with Dual Classification Heads.

Overall architecture (per VOICE_PIPELINE_SPEC.md):
    [Raw Audio Input: (B, T), 16kHz]
            │
            ├───────────────────────────────┬───────────────────────────────┐
            ▼                               ▼                               ▼  
     [Stream 1: Acoustic]          [Stream 2: Semantic]        [Stream 3: Paralinguistic]
     Silero-VAD + Chunking         PhoWhisper-base             Pause Analyzer
     (B, N, 6s*16000)              Text Transcripts (B)        P_pause: (B, 4)
            │                               │                               │
     emotion2vec-base + LoRA       PhoBERT-base Encoder        MLP Projector
     z_i: (B, N, 768)              Z_semantic: (B, 768)        Z_pause: (B, 128)
            │                               │                               │
     Temporal Attention Pooling             │                               │
     Z_audio: (B, 768)                      │                               │
            │                               │                               │
            └───────────────────────────────┼───────────────────────────────┘
                                            ▼
                            [Cross-Modal Gated Fusion]
                  Z_fused = [ g ⊙ Z_audio + (1-g) ⊙ Z_semantic ; Z_pause ]
                                    (B, 896)
                                            │
                        ┌───────────────────┴───────────────────┐
                        ▼                                       ▼
           [Head 1: Primary 5-Class]               [Head 2: Fine-grained Sub-class]
           Softmax + Focal Loss (γ=2.0)            Conditioned on Z_semantic (10 classes)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..audio.attention_pooling import TemporalAttentionPooling
from ..audio.emotion2vec_lora import LoRAConfig, LoRAEmotion2Vec
from ..audio.pause_analyzer import EmotionalPauseAnalyzer
from ..multimodal.asr_transcriber import PhoWhisperTranscriber
from ..multimodal.gated_fusion import CrossModalGatedFusion
from ..text.text_encoder import PhoBERTEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ModelOutput:
    """Output container for HierarchicalMERModel and VoiceOnlyMERModel.

    Attributes:
        primary_logits (torch.Tensor): Logits from primary head (B, 5).
        sub_logits (Optional[torch.Tensor]): Logits from sub head (B, 10).
        primary_probs (Optional[torch.Tensor]): Softmax probabilities (B, 5).
        sub_probs (Optional[torch.Tensor]): Softmax probabilities (B, 10).
        z_audio (Optional[torch.Tensor]): Pooled acoustic representation (B, 768).
        z_semantic (Optional[torch.Tensor]): Semantic representation (B, 768).
        z_fused (Optional[torch.Tensor]): Multimodal representation (B, 896).
        attention_weights (Optional[torch.Tensor]): Attention weights (B, N).
        gate_audio (Optional[torch.Tensor]): Gating values for audio (B, 768).
        gate_text (Optional[torch.Tensor]): Gating values for text (B, 768).
    """

    primary_logits: torch.Tensor
    sub_logits: Optional[torch.Tensor] = None
    primary_probs: Optional[torch.Tensor] = None
    sub_probs: Optional[torch.Tensor] = None
    z_audio: Optional[torch.Tensor] = None
    z_semantic: Optional[torch.Tensor] = None
    z_fused: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None
    gate_audio: Optional[torch.Tensor] = None
    gate_text: Optional[torch.Tensor] = None

    def primary_predictions(self) -> torch.Tensor:
        """Return argmax indices for primary emotions."""
        return self.primary_logits.argmax(dim=-1)

    def sub_predictions(self) -> Optional[torch.Tensor]:
        """Return argmax indices for sub emotions if available."""
        if self.sub_logits is None:
            return None
        return self.sub_logits.argmax(dim=-1)


# ---------------------------------------------------------------------------
# Voice-Only Model
# ---------------------------------------------------------------------------

class VoiceOnlyMERModel(nn.Module):
    """Emotion recognition model using only acoustic and pause features."""

    def __init__(
        self,
        lora_config: Optional[LoRAConfig] = None,
        num_primary_classes: int = 5,
        attention_hidden_dim: int = 256,
        dropout: float = 0.3,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device_str = device
        self.num_primary_classes = num_primary_classes

        self.audio_encoder = LoRAEmotion2Vec(
            lora_config=lora_config or LoRAConfig(),
            device=device,
        )

        self.attention_pooling = TemporalAttentionPooling(
            input_dim=768,
            hidden_dim=attention_hidden_dim,
            output_dim=768,
            dropout=dropout,
        )

        self.pause_analyzer = EmotionalPauseAnalyzer()

        self.pause_projection = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.primary_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768 + 128, num_primary_classes),
        )

        logger.info("VoiceOnlyMERModel initialized: num_primary=%d", num_primary_classes)

    def load_pretrained(self) -> None:
        """Load pretrained weights for acoustic encoder."""
        self.audio_encoder.load_pretrained()

    def forward(
        self,
        waveforms: torch.Tensor,
        speech_timestamps: Optional[List[List[dict]]] = None,
        waveform_lengths: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """Forward pass for voice-only branch.

        Args:
            waveforms (torch.Tensor): (B, T) or (B, 1, T).
            speech_timestamps (Optional[List[List[dict]]]): VAD timestamps.
            waveform_lengths (Optional[torch.Tensor]): Actual sample lengths.

        Returns:
            ModelOutput: output dataclass.
        """
        device = next(self.parameters()).device
        waveforms = waveforms.to(device)
        b = waveforms.shape[0]

        # 1. Acoustic encoding
        frame_features = self.audio_encoder(waveforms)  # (B, N, 768)

        # 2. Attention pooling
        z_audio, alpha = self.attention_pooling(frame_features)  # (B, 768), (B, N)

        # 3. Paralinguistic pause vector
        p_pause_tensors = []
        for i in range(b):
            ts = speech_timestamps[i] if speech_timestamps and i < len(speech_timestamps) else []
            feats = self.pause_analyzer.analyze_from_timestamps(ts)
            p_pause_tensors.append(feats.to_tensor())
        p_pause = torch.stack(p_pause_tensors).to(device)  # (B, 4)

        z_pause_proj = self.pause_projection(p_pause)  # (B, 128)

        # 4. Fused voice representation
        z_voice = torch.cat([z_audio, z_pause_proj], dim=-1)  # (B, 896)

        # 5. Primary head
        primary_logits = self.primary_head(z_voice)  # (B, 5)
        primary_probs = F.softmax(primary_logits, dim=-1)

        return ModelOutput(
            primary_logits=primary_logits,
            primary_probs=primary_probs,
            z_audio=z_audio,
            z_fused=z_voice,
            attention_weights=alpha,
        )


# ---------------------------------------------------------------------------
# Hierarchical Multimodal Model
# ---------------------------------------------------------------------------

class HierarchicalMERModel(nn.Module):
    """Hierarchical multimodal emotion recognition model (Acoustic + Text + Pause).

    Primary Head (5 classes per spec):
        Joy, Sadness, Anxiety, Anger, Neutral
    Sub Head (10 classes per spec):
        Gratitude, Pride, Relief, Disappointment, Remorse,
        Loneliness, Nervousness, Fear, Annoyance, Realization
    """

    PRIMARY_LABELS: List[str] = ["Joy", "Sadness", "Anxiety", "Anger", "Neutral"]
    SUB_LABELS: List[str] = [
        "Gratitude", "Pride", "Relief", "Disappointment", "Remorse",
        "Loneliness", "Nervousness", "Fear", "Annoyance", "Realization",
    ]

    def __init__(
        self,
        lora_config: Optional[LoRAConfig] = None,
        num_primary_classes: int = 5,
        num_sub_classes: int = 10,
        attention_hidden_dim: int = 256,
        fusion_dropout: float = 0.2,
        head_dropout: float = 0.3,
        asr_model_id: str = "vinai/phowhisper-base",
        text_model_id: str = "vinai/phobert-base-v2",
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device_str = device
        self.num_primary_classes = num_primary_classes
        self.num_sub_classes = num_sub_classes

        # ── Acoustic Branch ──────────────────────────────────────────
        self.audio_encoder = LoRAEmotion2Vec(
            lora_config=lora_config or LoRAConfig(),
            device=device,
        )
        self.attention_pooling = TemporalAttentionPooling(
            input_dim=768,
            hidden_dim=attention_hidden_dim,
            output_dim=768,
            dropout=fusion_dropout,
        )

        # ── ASR Branch ───────────────────────────────────────────────
        self.asr = PhoWhisperTranscriber(
            model_id=asr_model_id,
            device=device,
        )

        # ── Text Branch ──────────────────────────────────────────────
        self.text_encoder = PhoBERTEncoder(
            model_id=text_model_id,
            device=device,
        )

        # ── Pause Branch ─────────────────────────────────────────────
        self.pause_analyzer = EmotionalPauseAnalyzer()

        # ── Cross-Modal Gated Fusion ─────────────────────────────────
        self.fusion = CrossModalGatedFusion(
            dim_audio=768,
            dim_semantic=768,
            dim_pause=128,
            pause_raw_dim=4,
            output_dim=896,
            dropout=fusion_dropout,
        )

        # ── Dual Classification Heads ────────────────────────────────
        # Primary head: Z_fused (896) -> 5 classes
        self.primary_head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(896, num_primary_classes),
        )

        # Sub head: Z_semantic (768) -> 10 fine-grained classes
        self.sub_head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(768, num_sub_classes),
        )

        logger.info(
            "HierarchicalMERModel initialized: primary=%d, sub=%d",
            num_primary_classes,
            num_sub_classes,
        )

    def load_pretrained(self) -> None:
        """Load pretrained components (emotion2vec, PhoWhisper, PhoBERT)."""
        logger.info("Loading pretrained weights for all components...")
        self.audio_encoder.load_pretrained()
        self.text_encoder.load_pretrained()
        self.asr.load_model()
        logger.info("Pretrained loading complete.")

    def forward(
        self,
        waveforms: torch.Tensor,
        speech_timestamps: Optional[List[List[dict]]] = None,
        texts: Optional[List[str]] = None,
        waveform_lengths: Optional[torch.Tensor] = None,
        use_asr: bool = True,
    ) -> ModelOutput:
        """Full hierarchical multimodal forward pass.

        Args:
            waveforms (torch.Tensor): Audio waveform tensor of shape (B, T) or (B, 1, T).
            speech_timestamps (Optional[List[List[dict]]]): VAD timestamps per sample.
            texts (Optional[List[str]]): Ground-truth or pre-extracted transcripts.
            waveform_lengths (Optional[torch.Tensor]): Lengths of audio recordings.
            use_asr (bool): If True and texts is None, transcribe via ASR.

        Returns:
            ModelOutput: Contains primary_logits, sub_logits, probabilities, and gate vectors.
        """
        device = next(self.parameters()).device
        waveforms = waveforms.to(device)
        b = waveforms.shape[0]

        # 1. Acoustic encoding & attention pooling
        frame_features = self.audio_encoder(waveforms)          # (B, N, 768)
        z_audio, alpha = self.attention_pooling(frame_features)  # (B, 768), (B, N)

        # 2. Text extraction & encoding
        if texts is None:
            if use_asr:
                audio_list = [waveforms[i].squeeze().detach().cpu() for i in range(b)]
                asr_results = self.asr.transcribe_batch(audio_list, sample_rates=[16000] * b)
                texts = [res.text for res in asr_results]
            else:
                texts = ["Nhật ký cảm xúc hôm nay."] * b

        z_semantic = self.text_encoder(texts=texts)  # (B, 768)

        # 3. Paralinguistic pause vector
        p_pause_tensors = []
        for i in range(b):
            ts = speech_timestamps[i] if speech_timestamps and i < len(speech_timestamps) else []
            feats = self.pause_analyzer.analyze_from_timestamps(ts)
            p_pause_tensors.append(feats.to_tensor())
        p_pause = torch.stack(p_pause_tensors).to(device)  # (B, 4)

        # 4. Cross-modal gated fusion
        z_fused, (gate_audio, gate_text) = self.fusion(z_audio, z_semantic, p_pause)

        # 5. Dual classification heads
        primary_logits = self.primary_head(z_fused)     # (B, 5)
        sub_logits = self.sub_head(z_semantic)          # (B, 10)

        primary_probs = F.softmax(primary_logits, dim=-1)
        sub_probs = F.softmax(sub_logits, dim=-1)

        return ModelOutput(
            primary_logits=primary_logits,
            sub_logits=sub_logits,
            primary_probs=primary_probs,
            sub_probs=sub_probs,
            z_audio=z_audio,
            z_semantic=z_semantic,
            z_fused=z_fused,
            attention_weights=alpha,
            gate_audio=gate_audio,
            gate_text=gate_text,
        )

    def predict(
        self,
        waveforms: torch.Tensor,
        speech_timestamps: Optional[List[List[dict]]] = None,
        texts: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        """Perform end-to-end inference and return predicted emotion names."""
        self.eval()
        with torch.no_grad():
            output = self.forward(waveforms, speech_timestamps=speech_timestamps, texts=texts)
            p_preds = output.primary_predictions().cpu().tolist()
            s_preds = output.sub_predictions().cpu().tolist()

            p_labels = [self.PRIMARY_LABELS[idx] for idx in p_preds]
            s_labels = [self.SUB_LABELS[idx] for idx in s_preds]

            return p_labels, s_labels

    def get_trainable_components(self) -> Dict[str, List[str]]:
        """Return names of trainable parameters grouped by component."""
        groups: Dict[str, List[str]] = {
            "audio_encoder": [],
            "attention_pooling": [],
            "text_encoder": [],
            "fusion": [],
            "primary_head": [],
            "sub_head": [],
        }
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "audio_encoder" in name:
                groups["audio_encoder"].append(name)
            elif "attention_pooling" in name:
                groups["attention_pooling"].append(name)
            elif "text_encoder" in name:
                groups["text_encoder"].append(name)
            elif "fusion" in name:
                groups["fusion"].append(name)
            elif "primary_head" in name:
                groups["primary_head"].append(name)
            elif "sub_head" in name:
                groups["sub_head"].append(name)
        return groups

    def count_parameters(self) -> Dict[str, int]:
        """Count total and trainable parameters per component."""
        counts = {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "audio_encoder": sum(p.numel() for p in self.audio_encoder.parameters() if p.requires_grad),
            "text_encoder": sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad),
            "fusion": sum(p.numel() for p in self.fusion.parameters() if p.requires_grad),
            "primary_head": sum(p.numel() for p in self.primary_head.parameters() if p.requires_grad),
            "sub_head": sum(p.numel() for p in self.sub_head.parameters() if p.requires_grad),
        }
        return counts

    def save_checkpoint(
        self,
        path: Union[str, Path],
        epoch: int,
        optimizer_state: Optional[dict] = None,
        metrics: Optional[dict] = None,
    ) -> None:
        """Save model checkpoint to disk."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "epoch": epoch,
            "state_dict": self.state_dict(),
            "optimizer_state": optimizer_state,
            "metrics": metrics or {},
            "num_primary_classes": self.num_primary_classes,
            "num_sub_classes": self.num_sub_classes,
        }
        torch.save(payload, str(target_path))
        logger.info("Saved checkpoint to %s (epoch %d)", target_path, epoch)

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: Union[str, Path], device: str = "cpu"
    ) -> "HierarchicalMERModel":
        """Load model instance from saved checkpoint."""
        p = Path(checkpoint_path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {p}")

        checkpoint = torch.load(str(p), map_location=device, weights_only=False)
        model = cls(
            num_primary_classes=checkpoint.get("num_primary_classes", 5),
            num_sub_classes=checkpoint.get("num_sub_classes", 10),
            device=device,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(torch.device(device))
        logger.info("Loaded HierarchicalMERModel from %s (epoch %d)", p, checkpoint.get("epoch", -1))
        return model
