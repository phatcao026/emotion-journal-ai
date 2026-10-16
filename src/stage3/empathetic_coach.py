"""
src/stage3/empathetic_coach.py
------------------------------
Empathetic AI Journal Companion & CBT-Informed Micro-Coach.

Features:
    - Evidence-based CBT (Cognitive Behavioral Therapy) & Self-Compassion reflection.
    - Two-tier execution:
        1. Local Offline Engine: Deterministic, warm, psychologically validated templates (0ms latency, zero API costs).
        2. LLM Engine: Connects to Gemini or OpenAI when API keys are configured.
    - Context-aware: Integrates acoustic biomarkers (e.g. noticing breathlessness or pauses).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.stage3.crisis_detector import CrisisAlert
from src.stage3.emotion_result import EmotionResult

logger = logging.getLogger(__name__)


@dataclass
class EmpatheticResponse:
    """Structured empathetic feedback returned to user."""
    response_text: str
    reflection_question: str
    cbt_exercise: Optional[str] = None
    suggested_tags: List[str] = field(default_factory=list)
    engine_used: str = "offline_cbt"

    def format_display(self) -> str:
        out = [
            f"💡 **Phản hồi từ Trợ Lý Đồng Hành ({self.engine_used}):**\n",
            self.response_text,
            f"\n❓ **Câu hỏi phản chiếu:** *{self.reflection_question}*",
        ]
        if self.cbt_exercise:
            out.append(f"\n🧘 **Bài tập điều hòa gợi ý:**\n{self.cbt_exercise}")
        if self.suggested_tags:
            out.append(f"\n🏷️ **Thẻ cảm xúc:** `{'`, `'.join(self.suggested_tags)}`")
        return "\n".join(out)


# Offline CBT Guidelines & Prompts by Primary/Sub emotions
CBT_KNOWLEDGE_BASE = {
    "SADNESS": {
        "LONELINESS": {
            "validation": "Mình nghe thấy sự trống trải và cô đơn trong từng lời bạn chia sẻ hôm nay. Khi phải ôm một mình những trăn trở, không gian xung quanh dường như tĩnh lặng và nặng nề hơn rất nhiều.",
            "reflection": "Khoảnh khắc nào trong ngày hôm nay khiến bạn cảm thấy khoảng cách với mọi người rõ rệt nhất?",
            "cbt": "Kỹ thuật Self-Compassion (Tự trắc ẩn): Đặt một tay lên ngực, hít một hơi sâu và tự nhủ: 'Đây là một khoảnh khắc khó khăn. Cảm giác cô đơn là điều tự nhiên mà ai cũng từng trải qua. Mình cho phép mình được dịu dàng với chính mình lúc này.'",
            "tags": ["cô đơn", "tự trắc ẩn", "kết nối bản thân"],
        },
        "GUILT_SHAME": {
            "validation": "Cảm giác day dứt và tự trách luôn là một trong những gánh nặng tâm lý khó chịu nhất. Bạn dường như đang đặt lên vai mình những tiêu chuẩn rất khắt khe.",
            "reflection": "Nếu người bạn thân nhất của bạn rơi vào hoàn cảnh tương tự, bạn sẽ nói gì để an ủi họ?",
            "cbt": "Kỹ thuật CBT Tái đóng khung suy nghĩ: Viết ra lỗi lầm bạn đang tự dằn vặt, sau đó ghi chú lại 2 yếu tố khách quan nằm ngoài tầm kiểm soát của bạn khi chuyện đó xảy ra.",
            "tags": ["buông bỏ", "tự tha thứ", "giảm tự trách"],
        },
        "DEFAULT": {
            "validation": "Cảm ơn bạn đã can đảm trải lòng về những nỗi buồn này. Buồn bã là một tín hiệu tự nhiên cho thấy có điều gì đó thực sự có ý nghĩa đối với bạn vừa bị tổn thương.",
            "reflection": "Cơ thể bạn lúc này đang cảm thấy như thế nào, và bạn có đang cần một giấc ngủ hoặc một không gian yên tĩnh không?",
            "cbt": "Kỹ thuật Grounding 5-4-3-2-1: Quan sát 5 thứ bạn nhìn thấy, 4 thứ bạn có thể chạm vào, 3 âm thanh bạn nghe được, 2 mùi hương, và 1 điều bạn biết ơn.",
            "tags": ["lắng nghe", "xoa dịu", "buồn bã"],
        },
    },
    "ANXIETY": {
        "FEAR": {
            "validation": "Mình cảm nhận được nhịp thở gấp và nỗi sợ hãi đang dâng lên trong bạn. Cảm giác này giống như bạn đang phải đối diện với một hiểm nguy cận kề.",
            "reflection": "Nỗi sợ này đang muốn cảnh báo cho bạn về điều gì, và kịch bản tồi tệ nhất có thực sự chắc chắn sẽ xảy ra không?",
            "cbt": "Kỹ thuật Thở 4-7-8: Hít vào bằng mũi trong 4 giây, giữ hơi thở trong 7 giây, và thở ra từ từ bằng miệng trong 8 giây. Lặp lại 4 chu kỳ để kích hoạt hệ thần kinh phó giao cảm.",
            "tags": ["hạ nhiệt âu lo", "thở 4-7-8", "an toàn"],
        },
        "DEFAULT": {
            "validation": "Những suy nghĩ xoay vần và cảm giác bồn chồn đang khiến tâm trí bạn không thể nghỉ ngơi. Cảm giác lo âu rất mệt mỏi vì nó tiêu tốn rất nhiều năng lượng.",
            "reflection": "Trong tất cả những điều bạn đang lo, điều nào nằm trong tầm tay bạn có thể hành động ngay trong 10 phút tới?",
            "cbt": "Vòng tròn kiểm soát (Circle of Control): Phân loại các nỗi lo thành 2 cột: 'Những gì mình kiểm soát được' (hành động của mình) và 'Những gì mình không kiểm soát được' (suy nghĩ của người khác, tương lai).",
            "tags": ["lo âu", "vòng tròn kiểm soát", "bình tâm"],
        },
    },
    "ANGER": {
        "DISGUST": {
            "validation": "Sự phản kháng và khó chịu tột cùng đang xuất hiện. Khi những ranh giới cá nhân của bạn bị xâm phạm, cảm giác bực bội và bài xích là phản xạ tự nhiên.",
            "reflection": "Ranh giới cá nhân nào của bạn vừa bị chạm tới, và bạn muốn thiết lập lại khoảng cách an toàn như thế nào?",
            "cbt": "Kỹ thuật Thiết lập Ranh giới An toàn: Hít một hơi thật sâu, hình dung một vòng tròn bảo vệ xung quanh bạn nơi những hành vi tiêu cực của người khác không được phép bước vào.",
            "tags": ["ranh giới", "bảo vệ bản thân", "giải tỏa"],
        },
        "DEFAULT": {
            "validation": "Cơn giận của bạn là hoàn toàn có cơ sở khi mọi việc diễn ra không công bằng hoặc không như kỳ vọng. Giận dữ là năng lượng bảo vệ bản thân.",
            "reflection": "Đằng sau cơn giận dữ này, có nỗi thất vọng hay tổn thương nào đang ẩn giấu không?",
            "cbt": "Kỹ thuật Xả Năng lượng Thể chất: Đứng dậy, vươn vai thật căng, rửa mặt bằng nước lạnh hoặc siết chặt hai bàn tay trong 5 giây rồi thả lỏng hoàn toàn (PMR).",
            "tags": ["giải phóng giận dữ", "thả lỏng cơ", "cân bằng"],
        },
    },
    "JOY": {
        "HOPE": {
            "validation": "Thật tuyệt vời khi thấy nguồn năng lượng tích cực và niềm tin vào ngày mai đang trở lại với bạn!",
            "reflection": "Bước đi nhỏ đầu tiên mà bạn hào hứng muốn thực hiện vào ngày mai là gì?",
            "cbt": "Ghi nhận thành tựu nhỏ: Viết ra 1 điều bạn đã làm tốt hôm nay, dù là nhỏ nhất.",
            "tags": ["hy vọng", "tích cực", "năng lượng mới"],
        },
        "DEFAULT": {
            "validation": "Nghe giọng điệu tươi sáng của bạn hôm nay làm cả trang nhật ký bừng sáng. Thật quý giá khi được lưu giữ lại những khoảnh khắc trọn vẹn này.",
            "reflection": "Điều gì là xúc tác chính mang lại niềm vui này cho bạn hôm nay?",
            "cbt": "Kỹ thuật Savouring (Thưởng thức niềm vui): Dành 1 phút nhắm mắt lại và hồi tưởng cảm giác vui vẻ này để neo giữ cảm xúc vào tế bào thần kinh.",
            "tags": ["biết ơn", "hạnh phúc", "thưởng thức"],
        },
    },
    "NEUTRAL": {
        "CALM": {
            "validation": "Một ngày êm đềm và tĩnh lặng. Đôi khi không có biến động nào cũng chính là một món quà bình yên lớn nhất.",
            "reflection": "Bạn có đang tận hưởng được sự thảnh thơi này không, hay bạn đang chờ đợi một điều gì đó mới mẻ?",
            "cbt": "Thực hành Chánh niệm (Mindfulness): Thưởng thức trọn vẹn một ly nước ấm và cảm nhận từng ngụm nước đi vào cơ thể.",
            "tags": ["bình yên", "tĩnh lặng", "chánh niệm"],
        },
        "DEFAULT": {
            "validation": "Một ngày trôi qua với nhịp điệu bình thường, cân bằng.",
            "reflection": "Có điều gì nhỏ bé hôm nay đã khiến bạn khẽ mỉm cười mà bạn suýt quên không?",
            "cbt": "Nhìn lại ngày qua: Ghi lại một quan sát thú vị về thế giới xung quanh bạn hôm nay.",
            "tags": ["cân bằng", "nhật ký ngày", "bình thường"],
        },
    },
}


class EmpatheticCoach:
    """Provides psychologically sound empathetic coaching based on EmotionResult."""

    def __init__(self, use_llm: bool = False, model_name: str = "gemini-1.5-flash") -> None:
        self.use_llm = use_llm
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    def generate_response(
        self,
        emotion_result: EmotionResult,
        crisis_alert: Optional[CrisisAlert] = None,
    ) -> EmpatheticResponse:
        """Generate empathetic feedback."""
        # 1. If crisis alert triggered, return emergency supportive response immediately
        if crisis_alert and crisis_alert.is_crisis:
            return EmpatheticResponse(
                response_text=crisis_alert.summary(),
                reflection_question="Bạn có thể gọi ngay cho một người bạn thân hoặc đường dây nóng để được lắng nghe lúc này được không?",
                cbt_exercise="Xin hãy tạm dừng mọi suy nghĩ, uống một ngụm nước ấm và ở cạnh một người bạn tin tưởng.",
                suggested_tags=["hỗ trợ khẩn cấp", "an toàn"],
                engine_used="crisis_safety_override",
            )

        # 2. If LLM enabled and API key exists, attempt LLM call
        if self.use_llm and self.api_key:
            try:
                return self._generate_llm_response(emotion_result)
            except Exception as e:
                logger.warning("LLM generation failed (%s), falling back to offline CBT engine.", e)

        # 3. Fallback to robust offline CBT knowledge base
        return self._generate_offline_cbt_response(emotion_result)

    def _generate_offline_cbt_response(self, result: EmotionResult) -> EmpatheticResponse:
        """Deterministic, warm, psychologically validated template generator."""
        p_label = result.primary_label.upper()
        s_label = result.sub_label.upper()

        primary_kb = CBT_KNOWLEDGE_BASE.get(p_label, CBT_KNOWLEDGE_BASE["NEUTRAL"])
        entry = primary_kb.get(s_label, primary_kb.get("DEFAULT", CBT_KNOWLEDGE_BASE["NEUTRAL"]["DEFAULT"]))

        # Personalize with acoustic observation if incongruent
        validation = entry["validation"]
        if not result.modality_agreement:
            validation += " (Tuy lời nói có vẻ bình lặng, nhưng qua ngữ điệu và nhịp ngập ngừng, mình cảm thấy trong lòng bạn đang có những giằng xé khó nói thành lời)."

        return EmpatheticResponse(
            response_text=validation,
            reflection_question=entry["reflection"],
            cbt_exercise=entry["cbt"],
            suggested_tags=entry["tags"],
            engine_used="offline_cbt_engine",
        )

    def _generate_llm_response(self, result: EmotionResult) -> EmpatheticResponse:
        """Call Gemini API with structured prompt."""
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)

        prompt = f"""
Bạn là chuyên gia tư vấn tâm lý đồng hành bằng tiếng Việt cho ứng dụng Nhật Ký Cảm Xúc.
Thông tin nhật ký của người dùng:
- Lời nói: "{result.transcript}"
- Cảm xúc chính: {result.primary_label} (Độ tin cậy: {result.primary_confidence:.0%})
- Sắc thái phụ: {result.sub_label} (Độ tin cậy: {result.sub_confidence:.0%})
- Chỉ số âm học: F0={result.f0_mean:.1f}Hz, HNR={result.hnr_db:.1f}dB, Tạm dừng={result.pause_ratio:.0%}
- Đồng thuận phương thức: {'Đồng thuận' if result.modality_agreement else 'Bất đồng (ngữ điệu khác câu từ)'}

Yêu cầu phản hồi:
1. Xác nhận cảm xúc (validation) ấm áp, không phán xét, phản chiếu đúng cảm xúc ngầm ẩn (2-3 câu).
2. Một câu hỏi mở gợi mở nội tâm (reflection question).
3. Một bài tập điều hòa hoặc tái đóng khung nhận thức CBT ngắn gọn (1-2 câu).

Trả về định dạng JSON:
{{"validation": "...", "reflection": "...", "cbt": "..."}}
"""
        resp = model.generate_content(prompt)
        import json
        text = resp.text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        data = json.loads(text)

        return EmpatheticResponse(
            response_text=data.get("validation", ""),
            reflection_question=data.get("reflection", ""),
            cbt_exercise=data.get("cbt", ""),
            suggested_tags=[result.primary_label.lower(), result.sub_label.lower()],
            engine_used=f"gemini_llm ({self.model_name})",
        )
