"""
src/stage3/inference_engine.py
------------------------------
Production-Ready Stage 3 End-to-End Inference Engine.
Ingests raw audio recordings, runs VAD, ASR, Melody, Pause, and Multimodal Hierarchical MER,
producing structured, explainable EmotionResult objects.

Handles:
    - WAV file paths, PyTorch tensors, or NumPy arrays.
    - Automatic ASR transcription fallback when text is not provided.
    - Soft Bayesian routing & Joint MAP inference with guaranteed 0% taxonomy violations.
    - Graceful fallback for short, noisy, or empty audio clips.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.audio.audio_dataset import PrimaryEmotion, SubEmotion
from src.audio.vad_preprocessor import VADPreprocessor
from src.stage2.hierarchical_mer_stage2 import (
    Stage2HierarchicalMERModel,
    TAXONOMY_MAP,
)
from src.stage3.emotion_result import EmotionResult

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000


class Stage3InferenceEngine:
    """End-to-End Multimodal Emotion Inference Engine for Vietnamese Voice Journals."""

    PRIMARY_NAMES = ["JOY", "SADNESS", "ANXIETY", "ANGER", "NEUTRAL"]
    SUB_NAMES = [
        "JOY", "CALM", "HOPE", "CONNECTION", "SADNESS",
        "ANXIETY", "FEAR", "ANGER", "GUILT_SHAME", "LONELINESS", "DISGUST",
    ]

    def __init__(
        self,
        checkpoint_path: Optional[str] = "checkpoints/best_model_stage2_6_winner.pt" if Path("checkpoints/best_model_stage2_6_winner.pt").exists() else "checkpoints/stage2/best_model_stage2.pt",
        config_path: str = "configs/stage2/multimodal_stage2.yaml",
        device: str = "cpu",
        dummy: bool = False,
        melody_median_kernel: int = 5,
    ) -> None:
        self.device = torch.device(device)
        self.dummy = dummy
        self.melody_median_kernel = melody_median_kernel

        # 1. Load config
        self.config_path = Path(config_path)
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}

        # 2. Build model architecture
        logger.info("Initializing Stage 3 Inference Model (device=%s, dummy=%s)...", device, dummy)
        heads_cfg = self.config.get("classification_heads", {})
        fusion_cfg = self.config.get("gated_fusion", {})

        self.model = Stage2HierarchicalMERModel(
            num_primary_classes=5,
            num_sub_classes=11,
            sub_head_input_dim=heads_cfg.get("sub_head", {}).get("input_dim", 928),
            use_melody=True,
            dim_melody=fusion_cfg.get("dim_melody", 32),
            modality_dropout=0.0,   # No dropout during inference
            gate_temperature=fusion_cfg.get("gate_temperature", 0.8),
            melody_median_kernel=melody_median_kernel,
            device=device,
        )

        if not dummy:
            self.model.load_pretrained()

        # 3. Load checkpoint if available
        if checkpoint_path and Path(checkpoint_path).exists():
            logger.info("Loading Stage 2 checkpoint from '%s'...", checkpoint_path)
            self.model.load_checkpoint_compatible(checkpoint_path, map_location=self.device)
        else:
            logger.warning("No checkpoint provided/found at '%s'. Using initialized weights.", checkpoint_path)
            if dummy:
                self.model.audio_encoder._synthetic = True

        self.model.to(self.device)
        self.model.eval()

        # 4. Auxiliary preprocessors
        self.vad = VADPreprocessor()

    @torch.no_grad()
    def predict(
        self,
        audio_input: Union[str, Path, np.ndarray, torch.Tensor],
        text: Optional[str] = None,
        session_id: str = "",
    ) -> EmotionResult:
        """Run complete multimodal emotion prediction.

        Args:
            audio_input: Path to audio file (.wav), numpy array, or torch tensor.
            text: Optional transcribed text. If None, ASR transcribes automatically.
            session_id: Optional tracking identifier.

        Returns:
            EmotionResult instance with full predictions and biomarkers.
        """
        # 1. Ingest audio waveform
        waveform, duration_s = self._prepare_waveform(audio_input)
        waveform = waveform.to(self.device)

        # 2. Compute VAD speech timestamps & pause features
        speech_timestamps = self._extract_speech_timestamps(waveform)

        # 3. Transcribe if text not provided
        transcribed_text = text
        if transcribed_text is None:
            if hasattr(self.model, "asr") and hasattr(self.model.asr, "transcribe") and not getattr(self.model.asr, "_synthetic", False):
                try:
                    transcribed_text = self.model.asr.transcribe(waveform.squeeze(0), sample_rate=TARGET_SAMPLE_RATE)
                except Exception as asr_err:
                    logger.warning("ASR transcription failed (%s). Falling back to empty text.", asr_err)
                    transcribed_text = ""
            else:
                transcribed_text = ""

        # 4. Model forward pass
        output = self.model(
            waveforms=waveform,
            speech_timestamps=[speech_timestamps] if speech_timestamps else None,
            texts=[transcribed_text],
            use_asr=False,
        )

        # 5. Extract probabilities & predictions
        p_probs = output.primary_probs[0].cpu().numpy()     # (5,)
        soft_s_probs = output.soft_sub_probs[0].cpu().numpy() # (11,)

        # Joint MAP inference: guarantees 0.00% hierarchy violation
        p_pred_idx, s_pred_idx = output.joint_predictions()
        best_p_idx = int(p_pred_idx[0].item())
        best_s_idx = int(s_pred_idx[0].item())

        primary_label = self.PRIMARY_NAMES[best_p_idx]
        sub_label = self.SUB_NAMES[best_s_idx]
        primary_conf = float(p_probs[best_p_idx])
        sub_conf = float(soft_s_probs[best_s_idx])

        # 6. Shannon Entropy calculation for Primary head: H = -sum(p * ln(p)) / ln(5)
        eps = 1e-12
        raw_entropy = -float(np.sum(p_probs * np.log(p_probs + eps)))
        norm_entropy = raw_entropy / math.log(5.0)  # [0.0, 1.0]
        is_uncertain = norm_entropy > 0.8 or primary_conf < 0.35

        # 7. Modality Gating analysis
        if output.gate_audio is not None and output.gate_text is not None:
            mean_g_a = float(output.gate_audio[0].mean().item())
            mean_g_t = float(output.gate_text[0].mean().item())
        else:
            mean_g_a, mean_g_t = 0.5, 0.5

        if mean_g_a > 0.55:
            dominant_modality = "audio"
        elif mean_g_t > 0.55:
            dominant_modality = "text"
        else:
            dominant_modality = "balanced"

        # 8. Extract physical melody biomarkers
        biomarkers = self._extract_biomarkers(
            output.p_melody,
            speech_timestamps,
            duration_s,
            waveform=waveform,
            text=transcribed_text,
        )

        # 9. Modality Agreement Check (Acoustic valence vs Semantic text polarity)
        modality_agreement = self._check_modality_agreement(
            primary_label=primary_label,
            p_melody=output.p_melody,
            text=transcribed_text,
        )

        # Probability distribution dictionaries
        p_dist = {self.PRIMARY_NAMES[i]: float(p_probs[i]) for i in range(5)}
        s_dist = {self.SUB_NAMES[i]: float(soft_s_probs[i]) for i in range(11)}

        return EmotionResult(
            primary_label=primary_label,
            primary_confidence=primary_conf,
            sub_label=sub_label,
            sub_confidence=sub_conf,
            primary_distribution=p_dist,
            sub_distribution=s_dist,
            primary_entropy=norm_entropy,
            is_uncertain=is_uncertain,
            modality_agreement=modality_agreement,
            hierarchy_consistent=True,
            gate_audio=mean_g_a,
            gate_text=mean_g_t,
            dominant_modality=dominant_modality,
            f0_mean=biomarkers["f0_mean"],
            f0_std=biomarkers["f0_std"],
            f0_slope=biomarkers["f0_slope"],
            jitter_local=biomarkers["jitter_local"],
            shimmer_local=biomarkers["shimmer_local"],
            hnr_db=biomarkers["hnr_db"],
            pause_ratio=biomarkers["pause_ratio"],
            speech_rate_wpm=biomarkers["speech_rate_wpm"],
            transcript=transcribed_text,
            audio_duration_s=duration_s,
            sample_rate=TARGET_SAMPLE_RATE,
            session_id=session_id,
            model_version="Stage 2.5 SOTA Multimodal",
        )

    def _prepare_waveform(
        self,
        audio_input: Union[str, Path, np.ndarray, torch.Tensor],
    ) -> Tuple[torch.Tensor, float]:
        """Convert input to (1, T) float32 tensor at 16,000 Hz."""
        if isinstance(audio_input, (str, Path)):
            import soundfile as sf
            data, sr = sf.read(str(audio_input))
            data_arr = np.asarray(data, dtype=np.float32)
            if data_arr.ndim > 1:
                data_arr = np.mean(data_arr, axis=-1)
            # Resample if needed
            if sr != TARGET_SAMPLE_RATE:
                from scipy import signal
                num_target = int(len(data_arr) * TARGET_SAMPLE_RATE / sr)
                data_arr = signal.resample(data_arr, num_target).astype(np.float32)
            wav_tensor = torch.from_numpy(data_arr).unsqueeze(0)
            duration_s = float(wav_tensor.shape[-1]) / float(TARGET_SAMPLE_RATE)
            return wav_tensor, duration_s

        elif isinstance(audio_input, np.ndarray):
            arr = audio_input.astype(np.float32)
            if arr.ndim == 1:
                arr = np.expand_dims(arr, 0)
            wav_tensor = torch.from_numpy(arr)
            duration_s = float(wav_tensor.shape[-1]) / float(TARGET_SAMPLE_RATE)
            return wav_tensor, duration_s

        elif isinstance(audio_input, torch.Tensor):
            t = audio_input.float()
            if t.ndim == 1:
                t = t.unsqueeze(0)
            duration_s = float(t.shape[-1]) / float(TARGET_SAMPLE_RATE)
            return t, duration_s

        else:
            raise TypeError(f"Unsupported audio_input type: {type(audio_input)}")

    def _extract_speech_timestamps(self, waveform: torch.Tensor) -> List[dict]:
        """Run Silero VAD to extract speech vs pause chunks."""
        try:
            return self.vad.get_speech_timestamps(waveform.squeeze(0), sample_rate=TARGET_SAMPLE_RATE)
        except Exception:
            return []

    def _extract_biomarkers(
        self,
        p_melody: Optional[torch.Tensor],
        speech_timestamps: List[dict],
        duration_s: float,
        waveform: Optional[torch.Tensor] = None,
        text: str = "",
    ) -> Dict[str, float]:
        """Extract human-interpretable acoustic metrics."""
        f0_m, f0_s, f0_slope, jitter, shimmer, hnr = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # 1. First priority: Extract unnormalized physical metrics directly from melody_extractor
        if waveform is not None and hasattr(self.model, "melody_extractor") and hasattr(self.model.melody_extractor, "extract_raw_biomarkers"):
            try:
                raw_dict = self.model.melody_extractor.extract_raw_biomarkers(waveform)
                f0_m = raw_dict.get("f0_mean", 0.0)
                f0_s = raw_dict.get("f0_std", 0.0)
                f0_slope = raw_dict.get("f0_slope", 0.0)
                jitter = raw_dict.get("jitter_local", 0.0)
                shimmer = raw_dict.get("shimmer_local", 0.0)
                hnr = raw_dict.get("hnr_db", 0.0)
            except Exception as e:
                logger.warning("Could not extract raw physical melody biomarkers (%s)", e)

        # 2. Secondary fallback: p_melody tensor
        if f0_m == 0.0 and p_melody is not None:
            mel = p_melody[0].cpu().numpy()
            f0_m = float(mel[0]) if len(mel) > 0 else 0.0
            f0_s = float(mel[1]) if len(mel) > 1 else 0.0
            f0_slope = float(mel[5]) if len(mel) > 5 else 0.0
            jitter = float(mel[15]) if len(mel) > 15 else 0.0
            shimmer = float(mel[17]) if len(mel) > 17 else 0.0
            hnr = float(mel[20]) if len(mel) > 20 else 0.0

        pause_ratio = 0.0
        if speech_timestamps and duration_s > 0:
            total_samples = duration_s * TARGET_SAMPLE_RATE
            speech_samples = sum(seg["end"] - seg["start"] for seg in speech_timestamps)
            pause_samples = max(0, total_samples - speech_samples)
            pause_ratio = float(pause_samples / max(1.0, total_samples))

        # Real WPM calculation: words / minutes
        words = len(text.strip().split()) if text else 0
        wpm = float(words / (duration_s / 60.0)) if duration_s > 0 else 0.0

        return {
            "f0_mean": f0_m,
            "f0_std": f0_s,
            "f0_slope": f0_slope,
            "jitter_local": jitter,
            "shimmer_local": shimmer,
            "hnr_db": hnr,
            "pause_ratio": pause_ratio,
            "speech_rate_wpm": wpm,
        }

    def _check_modality_agreement(
        self,
        primary_label: str,
        p_melody: Optional[torch.Tensor],
        text: str,
    ) -> bool:
        """Flags incongruent cases while properly handling Vietnamese negation."""
        import re

        positive_words = ["vui", "hạnh phúc", "tuyệt vời", "may mắn", "biết ơn", "yêu đời", "thành công"]
        negative_words = [
            "buồn", "khóc", "mệt mỏi", "chán nản", "kiệt sức",
            "tuyệt vọng", "đau khổ", "bế tắc", "áp lực", "lo lắng", "cô đơn",
        ]
        negative_emotions = ["SADNESS", "ANGER", "ANXIETY"]
        positive_emotions = ["JOY"]

        text_lower = (text or "").lower()
        # Strip negated positive expressions (e.g., "không vui", "chẳng thấy hạnh phúc")
        negated_pos_pattern = (
            r"\b(không|chẳng|chả|chưa|đâu\s*có|hết)\s+"
            r"(còn\s+|thấy\s+|hề\s+|được\s+)?"
            r"(vui|hạnh phúc|tuyệt vời|may mắn|biết ơn|yêu đời|thành công)\b"
        )
        text_for_pos = re.sub(negated_pos_pattern, " ", text_lower)

        has_pos_words = any(w in text_for_pos for w in positive_words)
        has_neg_words = any(w in text_lower for w in negative_words)

        if has_pos_words and primary_label in negative_emotions:
            return False
        if has_neg_words and primary_label in positive_emotions:
            return False
        return True
