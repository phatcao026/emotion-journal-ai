"""
dynamics.py

Document-Level Emotion Dynamics & Valence-Arousal Circumplex Modeling for Diaries.

Grounding:
    1. Russell's Circumplex Model (Russell, 1980):
       Every discrete emotion is mapped to 2D continuous space:
       - Valence (Pleasure vs. Displeasure) in [-1.0, +1.0]
       - Arousal (High Activation vs. Deactivation) in [-1.0, +1.0]
    2. Continuous Expectation Formulation (Buechel & Hahn, 2017):
       Computes expected (V, A) for sentence s by taking the probability-weighted
       expectation over the 11-emotion taxonomy:
           V(s) = sum_e p(e|s) * V_e / sum_e p(e|s)
           A(s) = sum_e p(e|s) * A_e / sum_e p(e|s)
    3. Peak-End Rule (Kahneman, 2000):
       Synthesizes the retrospective emotional memory of a journal entry
       from the most intense sentence (Peak) and the final sentence (End).
    4. Trajectory Pattern Matching:
       Identifies emotional shifts: 'Rebound', 'Downturn', 'Stressful', 'Cathartic', 'Stable'.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.text.text_preprocessor import VietnameseTextPreprocessor

logger = logging.getLogger(__name__)

# Empirical Russell Coordinates for the 11-emotion taxonomy in [-1.0, +1.0]
RUSSELL_COORDINATES: Dict[str, Tuple[float, float]] = {
    # Positive Valence (Pleasure)
    "JOY": (+0.85, +0.60),          # High valence, high arousal
    "CALM": (+0.75, -0.65),         # High valence, low arousal (serenity)
    "HOPE": (+0.70, +0.40),         # Positive valence, moderate arousal
    "CONNECTION": (+0.80, +0.25),   # High valence, mild arousal (belonging)

    # Negative Valence (Displeasure)
    "SADNESS": (-0.80, -0.55),      # Low valence, low arousal (dejected)
    "ANXIETY": (-0.65, +0.70),      # Low valence, high arousal (apprehension)
    "FEAR": (-0.75, +0.80),         # Very low valence, very high arousal
    "ANGER": (-0.70, +0.75),        # Low valence, high arousal (fury)
    "GUILT_SHAME": (-0.75, +0.20),   # Very low valence, moderate arousal
    "LONELINESS": (-0.70, -0.45),   # Low valence, low arousal (isolation)
    "DISGUST": (-0.60, +0.35),      # Moderate negative valence, moderate arousal
}


@dataclass
class SentenceEmotionState:
    """Emotion state of a single sentence in the journal."""
    index: int
    text: str
    probabilities: Dict[str, float]
    dominant_emotions: List[str]
    valence: float
    arousal: float


@dataclass
class JournalDynamicsReport:
    """Full emotional trajectory and synthesis report for an entire journal entry."""
    total_sentences: int
    sentences: List[SentenceEmotionState]
    valence_trajectory: List[float]
    arousal_trajectory: List[float]
    average_valence: float
    average_arousal: float
    peak_sentence: SentenceEmotionState
    end_sentence: SentenceEmotionState
    peak_end_valence: float
    peak_end_arousal: float
    trajectory_pattern: str
    empathic_reflection: str


class JournalEmotionDynamicsAnalyzer:
    """Analyzes sentence-by-sentence emotion dynamics and calculates Valence/Arousal trajectory."""

    def __init__(
        self,
        classifier: Optional[Any] = None,
        preprocessor: Optional[VietnameseTextPreprocessor] = None,
        threshold: float = 0.4,
    ) -> None:
        self.classifier = classifier
        self.preprocessor = preprocessor or VietnameseTextPreprocessor(normalize_teencode=True)
        self.threshold = threshold

    def compute_sentence_va(self, probs: Dict[str, float]) -> Tuple[float, float]:
        """Compute expected Valence & Arousal given emotion probability distribution."""
        v_sum = 0.0
        a_sum = 0.0
        weight_sum = 0.0

        for emo, p in probs.items():
            if emo in RUSSELL_COORDINATES:
                ve, ae = RUSSELL_COORDINATES[emo]
                # Soft non-linear emphasis on prominent emotions
                weight = p ** 1.5
                v_sum += weight * ve
                a_sum += weight * ae
                weight_sum += weight

        if weight_sum > 1e-6:
            return round(v_sum / weight_sum, 3), round(a_sum / weight_sum, 3)
        return 0.0, 0.0

    def classify_trajectory_pattern(self, valences: List[float], arousals: List[float]) -> str:
        """Classify emotional arc across sentences."""
        if not valences:
            return "Neutral"
        if len(valences) == 1:
            return "Positive" if valences[0] > 0.2 else ("Negative" if valences[0] < -0.2 else "Stable")

        v_start = float(np.mean(valences[: max(1, len(valences) // 3)]))
        v_end = float(np.mean(valences[-max(1, len(valences) // 3):]))
        mean_v = float(np.mean(valences))
        mean_a = float(np.mean(arousals))

        v_diff = v_end - v_start

        if v_start < -0.2 and v_end > 0.15:
            return "Rebound (Phục hồi tích cực)"
        elif v_start > 0.2 and v_end < -0.15:
            return "Downturn (Trùng xuống / Bất an dần)"
        elif mean_v < -0.3 and mean_a > 0.3:
            return "Stressful (Căng thẳng / Dồn nén)"
        elif v_start < -0.2 and arousals[-1] < -0.2:
            return "Cathartic (Giải tỏa / Nhẹ nhõm dần)"
        elif mean_v > 0.3:
            return "Positive (Lạc quan / Tươi sáng)"
        elif mean_v < -0.3:
            return "Gloomy (U buồn / Chùng lắng)"
        else:
            return "Equilibrium (Cân bằng / Tĩnh lặng)"

    def generate_empathic_reflection(self, pattern: str, peak: SentenceEmotionState, end: SentenceEmotionState) -> str:
        """Synthesize an empathic reflection based on psychological findings."""
        if "Rebound" in pattern:
            return (
                "Bài nhật ký ghi nhận một diễn biến tâm lý rất đáng mừng: dù bắt đầu với những trăn trở, "
                "bạn đã tự tìm lại được sự nhẹ nhõm và hy vọng ở phần kết. Đó là một sức mạnh phục hồi nội tâm tuyệt vời."
            )
        elif "Downturn" in pattern:
            return (
                "Dường như có một điều gì đó về cuối ngày đang làm bạn chùng xuống hoặc lo nghĩ nhiều hơn. "
                "Hãy cho phép bản thân nghỉ ngơi tối nay, đừng quá khắt khe với chính mình nhé."
            )
        elif "Stressful" in pattern:
            return (
                "Dòng cảm xúc thể hiện mức độ kích hoạt và căng thẳng tương đối cao. "
                "Bạn đã phải gánh vác nhiều năng lượng dồn nén hôm nay; việc trút ra trang nhật ký là bước đầu tiên rất tốt để hạ nhiệt tâm trí."
            )
        elif "Positive" in pattern:
            return (
                "Một trang nhật ký ngập tràn nguồn năng lượng tích cực và sự kết nối! "
                "Hy vọng bạn sẽ lưu giữ trọn vẹn những cảm xúc ấm áp này cho những ngày tiếp theo."
            )
        else:
            return (
                "Nhật ký phản ánh một khoảng lặng chiêm nghiệm. Cảm xúc đã được bày tỏ một cách chân thật và bình yên."
            )

    def analyze_document(
        self,
        text: str,
        sentence_probabilities: Optional[List[Dict[str, float]]] = None,
    ) -> JournalDynamicsReport:
        """Run full document-level emotion dynamics pipeline."""
        sentences = self.preprocessor.segment_sentences(text)
        if not sentences:
            sentences = [text]

        states: List[SentenceEmotionState] = []
        valences: List[float] = []
        arousals: List[float] = []

        for idx, sent in enumerate(sentences):
            # If probabilities supplied externally (from model inference)
            if sentence_probabilities and idx < len(sentence_probabilities):
                probs = sentence_probabilities[idx]
            else:
                # Default heuristic or mock baseline
                probs = {"CALM": 0.5, "JOY": 0.3}

            dom_emotions = [e for e, p in probs.items() if p >= self.threshold]
            v, a = self.compute_sentence_va(probs)
            valences.append(v)
            arousals.append(a)

            states.append(
                SentenceEmotionState(
                    index=idx,
                    text=sent,
                    probabilities=probs,
                    dominant_emotions=dom_emotions,
                    valence=v,
                    arousal=a,
                )
            )

        # Peak calculation: maximum absolute valence displacement from neutral
        peak_idx = int(np.argmax([abs(s.valence) + 0.3 * abs(s.arousal) for s in states]))
        peak_sent = states[peak_idx]
        end_sent = states[-1]

        # Kahneman Peak-End Rule
        peak_end_v = round(0.5 * peak_sent.valence + 0.5 * end_sent.valence, 3)
        peak_end_a = round(0.5 * peak_sent.arousal + 0.5 * end_sent.arousal, 3)

        pattern = self.classify_trajectory_pattern(valences, arousals)
        reflection = self.generate_empathic_reflection(pattern, peak_sent, end_sent)

        return JournalDynamicsReport(
            total_sentences=len(states),
            sentences=states,
            valence_trajectory=valences,
            arousal_trajectory=arousals,
            average_valence=round(float(np.mean(valences)), 3),
            average_arousal=round(float(np.mean(arousals)), 3),
            peak_sentence=peak_sent,
            end_sentence=end_sent,
            peak_end_valence=peak_end_v,
            peak_end_arousal=peak_end_a,
            trajectory_pattern=pattern,
            empathic_reflection=reflection,
        )
