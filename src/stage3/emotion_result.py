"""
src/stage3/emotion_result.py
----------------------------
Unified Data Structure for Stage 3 Emotion Recognition & Journaling Analytics.
Encapsulates:
    - Hierarchical predictions (Primary + Sub-emotions)
    - Bayesian uncertainty & entropy
    - Modality gating dynamics (audio vs. text dominance)
    - Interpretable acoustic biomarkers (F0, Jitter, Shimmer, HNR, Pause)
    - Serialization & Clinical summary formatters
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class EmotionResult:
    """Production-ready structured emotion representation for Voice Journals."""

    # 1. Hierarchical Emotion Classifications
    primary_label: str                          # e.g., "SADNESS", "JOY", "ANXIETY", "ANGER", "NEUTRAL"
    primary_confidence: float                   # [0.0, 1.0]
    sub_label: str                              # e.g., "LONELINESS", "CALM", "FEAR", "HOPE"
    sub_confidence: float                       # [0.0, 1.0]

    # Full probability distributions
    primary_distribution: Dict[str, float] = field(default_factory=dict)
    sub_distribution: Dict[str, float] = field(default_factory=dict)

    # 2. Uncertainty & Hierarchical Consistency
    primary_entropy: float = 0.0                # Shannon Entropy H(P_primary)
    is_uncertain: bool = False                  # True if entropy > uncertainty threshold (default: 0.8)
    modality_agreement: bool = True             # True if acoustic prior aligns with text semantics
    hierarchy_consistent: bool = True           # Guaranteed True under Stage 2 SOTA logit masking

    # 3. Modality Gating Dynamics
    gate_audio: float = 0.5                     # Fraction of decision driven by acoustic channel [0, 1]
    gate_text: float = 0.5                      # Fraction of decision driven by text semantics [0, 1]
    dominant_modality: str = "balanced"         # "audio" | "text" | "balanced"

    # 4. Interpretable Acoustic & Prosodic Biomarkers
    f0_mean: float = 0.0                        # Mean Fundamental Frequency (Hz)
    f0_std: float = 0.0                         # Pitch standard deviation (Hz)
    f0_slope: float = 0.0                       # Pitch contour slope (negative for choking / weeping)
    jitter_local: float = 0.0                   # Micro-frequency perturbation (voice instability)
    shimmer_local: float = 0.0                  # Micro-amplitude perturbation
    hnr_db: float = 0.0                         # Harmonics-to-Noise Ratio (breathiness / roughness)
    pause_ratio: float = 0.0                    # Fraction of silent duration [0.0, 1.0]
    speech_rate_wpm: float = 0.0                # Words per minute estimated

    # 5. Semantic Transcript & Audio Metadata
    transcript: str = ""                        # ASR transcription
    audio_duration_s: float = 0.0               # Total audio duration in seconds
    sample_rate: int = 16000

    # 6. Session & Model Tracing
    session_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    model_version: str = "Stage 2.5 SOTA Multimodal"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EmotionResult:
        """Instantiate from dictionary."""
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def summary_card(self) -> str:
        """Generate human-readable Markdown summary card for journaling UI."""
        conf_p_pct = f"{self.primary_confidence * 100:.1f}%"
        conf_s_pct = f"{self.sub_confidence * 100:.1f}%"
        agreement_str = " Đồng thuận" if self.modality_agreement else "⚠️ Bất đồng phương thức (Lời nói vs Giọng điệu)"
        uncertainty_str = "⚠️ Cần xác nhận lại" if self.is_uncertain else " Độ tin cậy cao"

        lines = [
            f"###  Nhật Ký Cảm Xúc [{self.timestamp[:10]}]",
            f"- **Cảm xúc chủ đạo:** `{self.primary_label}` ({conf_p_pct})",
            f"- **Sắc thái chi tiết:** `{self.sub_label}` ({conf_s_pct})",
            f"- **Trạng thái phân tích:** {uncertainty_str} | {agreement_str}",
            f"- **Kênh dẫn dắt:** `{self.dominant_modality.upper()}` (Audio: {self.gate_audio*100:.1f}% | Text: {self.gate_text*100:.1f}%)",
            f"- **Chỉ số âm học:** F0 = {self.f0_mean:.1f}Hz | HNR = {self.hnr_db:.1f}dB | Tạm dừng = {self.pause_ratio*100:.1f}%",
        ]
        if self.transcript:
            lines.append(f"- **Nội dung:** *\"{self.transcript}\"*")
        return "\n".join(lines)
