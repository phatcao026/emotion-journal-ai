"""
src/stage3/crisis_detector.py
-----------------------------
Multi-Tier Mental Health Crisis & Safety Net Detector for Emotion Journal AI.

Safety Net Architecture:
    - Tier 1: High-Risk Lexical Safety Filter (Suicide, Self-Harm, Life Termination expressions in Vietnamese)
    - Tier 2: Acute Acoustic Distress Detection (Extreme vocal instability, choking/crying pitch slopes)
    - Tier 3: Longitudinal Depression / Burnout Streak Alarm (from LongitudinalTracker)

Ethics & Privacy Notice:
    - Strictly non-punitive, supportive intervention.
    - Provides immediate access to Vietnamese mental health support hotlines.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from src.stage3.emotion_result import EmotionResult

logger = logging.getLogger(__name__)

# Vietnamese High-Risk Keywords for self-harm and suicide
EMERGENCY_LEXICAL_PATTERNS = [
    r"tự\s*tử",
    r"tự\s*sát",
    r"muốn\s*chết",
    r"chết\s*đi\s*cho\s*rồi",
    r"chết\s*cho\s*xong",
    r"không\s*muốn\s*sống\s*nữa",
    r"chấm\s*dứt\s*cuộc\s*(đời|sống)",
    r"kết\s*thúc\s*cuộc\s*(đời|sống)",
    r"rạch\s*tay",
    r"uống\s*thuốc\s*(ngủ|độc|tự\s*vẫn)",
    r"nhảy\s*(cầu|lầu)",
    r"treo\s*cổ",
]

# Warning patterns: profound hopelessness / isolation
DISTRESS_LEXICAL_PATTERNS = [
    r"không\s*còn\s*lối\s*thoát",
    r"bế\s*tắc\s*hoàn\s*toàn",
    r"tuyệt\s*vọng",
    r"mất\s*hết\s*hy\s*vọng",
    r"không\s*ai\s*cần\s*mình",
    r"gánh\s*nặng\s*cho\s*gia\s*đình",
    r"kiệt\s*quệ\s*quá\s*rồi",
]

VIETNAM_SUPPORT_HOTLINES = """
- Đường dây nóng Ngày Mai (Hỗ trợ người trầm cảm): 096 306 1414 (13h00 - 20h30 hàng ngày)
- Tổng đài Quốc gia Bảo vệ Trẻ em & Tâm lý vị thành niên: 111 (24/7, Miễn phí)
- Viện Sức khỏe Tâm thần - Bệnh viện Bạch Mai: 024 3869 3731
- Cấp cứu Y tế Khẩn cấp: 115
"""


@dataclass
class CrisisAlert:
    """Safety alert descriptor returned by CrisisDetector."""
    is_crisis: bool = False
    severity_level: str = "none"               # "none" | "mild" | "high" | "emergency"
    triggers: List[str] = field(default_factory=list)
    support_hotlines: str = VIETNAM_SUPPORT_HOTLINES
    intervention_prompt: str = ""

    def summary(self) -> str:
        if not self.is_crisis:
            return "✅ Trạng thái an toàn, không phát hiện dấu hiệu khủng hoảng cấp tính."
        triggers_str = "; ".join(self.triggers)
        return (
            f"🚨 CẢNH BÁO TÂM LÝ [{self.severity_level.upper()}]:\n"
            f"- Nguyên nhân kích hoạt: {triggers_str}\n"
            f"- Khuyến nghị: {self.intervention_prompt}\n"
            f"- Hỗ trợ khẩn cấp:\n{self.support_hotlines}"
        )


class CrisisDetector:
    """Evaluates multi-modal journal entries for mental health crisis triggers."""

    def __init__(
        self,
        f0_slope_threshold: float = -8.0,      # Giảm > 8 Hz/std-time (ngữ điệu sụt dốc / nghẹn ngào)
        hnr_distress_threshold: float = 4.5,   # HNR vật lý < 4.5 dB (hụt hơi / khàn đặc / nấc nghẹn)
        consecutive_negative_alarm: int = 4,
    ) -> None:
        self.f0_slope_threshold = f0_slope_threshold
        self.hnr_distress_threshold = hnr_distress_threshold
        self.consecutive_negative_alarm = consecutive_negative_alarm

    def evaluate(
        self,
        result: EmotionResult,
        consecutive_negative_streak: int = 0,
    ) -> CrisisAlert:
        """Scan text, acoustics, and streak for crisis patterns."""
        triggers: List[str] = []
        severity = "none"
        text = (result.transcript or "").lower()

        # 1. Tier 1: Emergency Lexical Filter
        for pattern in EMERGENCY_LEXICAL_PATTERNS:
            if re.search(pattern, text):
                triggers.append(f"Phát hiện biểu hiện nguy cơ cao ('{pattern}')")
                severity = "emergency"
                break

        # 2. Tier 1.5: Deep Distress Lexical Filter
        if severity != "emergency":
            for pattern in DISTRESS_LEXICAL_PATTERNS:
                if re.search(pattern, text):
                    triggers.append(f"Phát hiện trạng thái kiệt quệ cảm xúc ('{pattern}')")
                    severity = "high"
                    break

        # 3. Tier 2: Acute Acoustic Distress (Choking / sobbing voice signals)
        if result.f0_mean > 0:  # Valid audio present
            acoustic_flag = False
            if 0 < result.hnr_db < self.hnr_distress_threshold and result.primary_label in ("SADNESS", "ANXIETY"):
                triggers.append("Tín hiệu âm học cho thấy sự hụt hơi / nấc nghẹn nghiêm trọng")
                acoustic_flag = True
            if getattr(result, "f0_slope", 0.0) < self.f0_slope_threshold and result.primary_label in ("SADNESS", "ANXIETY"):
                triggers.append("Tín hiệu ngữ điệu sụt dốc mạnh (nức nở / nghẹn ngào)")
                acoustic_flag = True

            if acoustic_flag:
                if severity == "none":
                    severity = "mild"
                elif severity == "mild":
                    severity = "high"

        # 4. Tier 3: Longitudinal Depression Streak
        if consecutive_negative_streak >= self.consecutive_negative_alarm:
            triggers.append(f"Chuỗi cảm xúc tiêu cực kéo dài liên tiếp {consecutive_negative_streak} ngày")
            if severity == "none":
                severity = "mild"

        is_crisis = severity in ("high", "emergency")

        if severity == "emergency":
            intervention_prompt = (
                "Chúng mình rất lo lắng cho bạn và bạn không phải đối mặt với nỗi đau này một mình. "
                "Xin hãy dừng lại một nhịp và liên hệ ngay với người thân hoặc đường dây nóng hỗ trợ tâm lý bên dưới."
            )
        elif severity == "high":
            intervention_prompt = (
                "Bạn đang phải trải qua một giai đoạn rất nặng nề và mệt mỏi. "
                "Đừng ngần ngại chia sẻ gánh nặng này với chuyên gia tâm lý hoặc người bạn tin tưởng nhất."
            )
        elif severity == "mild":
            intervention_prompt = (
                "Cảm xúc của bạn những ngày gần đây đang có xu hướng chùng xuống. Hãy dành thời gian nghỉ ngơi và tự chăm sóc bản thân."
            )
        else:
            intervention_prompt = ""

        return CrisisAlert(
            is_crisis=is_crisis,
            severity_level=severity,
            triggers=triggers,
            intervention_prompt=intervention_prompt,
        )
