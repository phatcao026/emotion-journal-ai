"""
journal_filter.py
-----------------
Bộ lọc & Phân tích chuyên sâu dữ liệu "Nhật ký & Tâm sự chứa đựng cảm xúc" (Vietnamese Journal & Confession Filter v2).

Nhiệm vụ:
1. Đồng bộ tiền xử lý với pipeline NLP dự án (`src/text/text_preprocessor.py`):
   - Chuẩn hóa Unicode NFC, xử lý bảng Teencode chuẩn, khử nhiễu ký tự.
2. Kiểm định Ngôi xưng tự sự & Chiều sâu nội tâm (First-Person Gatekeeper).
3. Loại bỏ nhiễu: quảng cáo, bán hàng, showbiz, tin tức xã hội (Blacklist).
4. Phát hiện và xử lý Nguy cơ Tự hại / Khủng hoảng tâm lý (Crisis & Self-Harm Quarantine):
   - Đánh dấu `is_crisis = True` để cách ly riêng vào `crisis_quarantine/`.
5. Nhận diện phạm vi Phủ định (Negation Scope Detection):
   - Đảo chiều cảm xúc khi từ khóa tích cực đi liền sau từ phủ định ("không vui", "chẳng an tâm").
6. Khử thông tin định danh cá nhân nhạy cảm (PII Redaction: số điện thoại, email, link).
7. Phân tầng cảm xúc theo `data/taxonomy.json` (5 Primary, 11 Sub).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from src.text.text_preprocessor import VietnameseTextPreprocessor
except Exception:
    VietnameseTextPreprocessor = None

# 1. Đại từ nhân xưng & chỉ dấu ngôi thứ nhất (First-person self-reference)
# Bao gồm cả đại từ xưng hô tự sự thân mật trên mạng xã hội (Threads, Confessions)
FIRST_PERSON_MARKERS = [
    "tôi", "mình", "tớ", "em", "anh", "ta", "bản thân", "lòng mình",
    "trong lòng", "bản thân tôi", "chính mình", "cuộc đời tôi", "ngày của tôi",
    "riêng tôi", "với tôi", "đối với tôi", "tự dặn lòng", "nội tâm",
    "tui", "toi", "mk", "mik", "mềnh", "tao", "t", "e"
]

# 2. Dấu hiệu văn phong tự sự, trải lòng (Introspective & Diary context)
JOURNAL_REFLECTIVE_MARKERS = [
    "hôm nay", "dạo này", "những ngày qua", "đêm muộn", "nhiều lúc", "đôi khi",
    "nghĩ lại", "chợt nhận ra", "cảm thấy", "thấy lòng", "ước gì", "giá như",
    "tự hỏi", "chông chênh", "lạc lõng", "áp lực", "mệt mỏi", "bế tắc", "bất lực",
    "tổn thương", "an yên", "nhẹ lòng", "quá tải", "cố gắng", "bỏ cuộc", "chìm sâu",
    "một mình", "không ai hiểu", "lặng lẽ", "kỷ niệm", "ký ức", "day dứt",
    "lần đầu", "muộn phiền", "đau khổ", "tan biến", "mệt lả", "tự trách", "hoang mang", "bật khóc"
]

# 3. Danh sách đen loại trừ (Blacklist: Quảng cáo, bán lẻ, tin tức, showbiz, PR)
COMMERCIAL_AND_NOISE_BLACKLIST = [
    "mua ngay", "giá chỉ", "inbox shop", "đặt hàng", "freeship", "voucher", "giảm giá",
    "sale off", "link ở bio", "link bio", "sản phẩm", "tài trợ", "brand", "nhãn hàng",
    "hợp tác", "giveaway", "minigame", "tuyển dụng", "tuyển sinh", "học phí", "khóa học",
    "liên hệ zalo", "booking", "quảng cáo", "quang cao", "quà tặng", "săn sale", "mã giảm",
    "cơ quan chức năng", "công an", "thủ tướng", "chính phủ", "chứng khoán", "bất động sản",
    "trailer", "tập phát sóng", "showbiz", "diễn viên", "ca sĩ vừa ra mắt", "review công nghệ"
]

# 3.1. Nhận diện hành vi phe vé, chợ đen, mua bán vé concert / sự kiện (Transactional Scalping)
TICKET_SCALPING_PATTERNS = [
    r"(?:pass|tìm|cần\s+tìm|cần\s+pass|mua|nhượng|trade|đổi)\s+(?:\d+\s+)?(?:vé|slot|tix|ticket|hạng|ghế|merch)",
    r"(?:vé\s+)?cần\s+pass|ai\s+pass|pass\s+nhanh|chốt\s+nhanh|inbox\s+mình|ib\s+mình|nhượng\s+lại",
    r"(?:giá\s+vé|wts|wtb|gdtt|gdtg|full\s+merch|fast\s+track|check\s+legit|sold\s+out|sold\s+hết|presale|general\s+sale|hồi\s+máu|camp\s+vé|tiền\s+camp)",
    r"(?:tiếp\s+tục\s+có\s+mặt\s+tại|mở\s+màn\s+concert|lineup|đêm\s+diễn\s+bùng\s+nổ)"
]

# 4. Danh sách từ khóa Nguy cấp / Khủng hoảng tâm lý (Crisis & Self-Harm)
CRISIS_KEYWORDS = [
    "tự tử", "muốn chết", "tự sát", "kết thúc cuộc đời", "giải thoát cho xong",
    "rạch tay", "không muốn sống nữa", "uống thuốc ngủ", "nhảy lầu", "chết đi cho rồi",
    "tự kết liễu", "tự hại", "tự huỷ", "kết liễu đời mình", "muốn biến mất khỏi thế giới"
]

# 5. Từ phủ định tiếng Việt
NEGATION_WORDS = [
    "không", "chẳng", "chưa", "đâu có", "k", "ko", "khum", "hông", "chả", "đếch"
]

# 6. Từ điển cảm xúc phân tầng theo taxonomy.json của dự án
TAXONOMY_LEXICON = {
    # Nhóm JOY
    "JOY": [
        "vui", "hạnh phúc", "cười", "tuyệt vời", "mừng", "yêu đời", "thích thú",
        "hào hứng", "phấn khởi", "tự hào", "sung sướng", "tươi vui", "rạng rỡ"
    ],
    "CALM": [
        "bình yên", "thanh thản", "nhẹ nhõm", "tĩnh lặng", "an nhiên", "thư thái",
        "an tâm", "lắng lòng", "nhẹ lòng", "yên ả", "dịu lại"
    ],
    "HOPE": [
        "hy vọng", "hi vọng", "tin tưởng", "lạc quan", "cố gắng", "tương lai",
        "vững tin", "ngày mai sẽ tốt", "vượt qua", "động lực"
    ],
    "CONNECTION": [
        "kết nối", "ấm áp", "thấu hiểu", "sẻ chia", "yêu thương", "đồng cảm",
        "gần gũi", "bên cạnh", "đồng hành", "tri kỷ", "tình cảm gia đình", "gia đình ấm áp", "yêu gia đình"
    ],

    # Nhóm SADNESS
    "SADNESS": [
        "buồn", "khóc", "đau lòng", "nước mắt", "tủi thân", "xót xa", "bất lực",
        "chán nản", "nỗi buồn", "đau đớn", "tan vỡ", "mất mát", "tuyệt vọng", "bế tắc"
    ],
    "LONELINESS": [
        "cô đơn", "lạc lõng", "một mình", "trống rỗng", "bơ vơ", "chơi vơi",
        "không có ai", "lẻ loi", "cô độc", "trơ trọi"
    ],
    "GUILT_SHAME": [
        "áy náy", "hối hận", "ân hận", "tự trách", "cắn rứt", "xấu hổ", "day dứt",
        "lỗi lầm", "dằn vặt", "tội lỗi"
    ],

    # Nhóm ANXIETY
    "ANXIETY": [
        "lo lắng", "lo âu", "bất an", "bồn chồn", "áp lực", "stress", "quá tải",
        "ngột ngạt", "chênh vênh", "rối bời", "hoang mang", "mơ hồ", "áp bách"
    ],
    "FEAR": [
        "sợ", "sợ hãi", "hoảng sợ", "hoảng loạn", "ám ảnh", "khiếp sợ", "kinh hoàng",
        "run rẩy", "giật mình", "kinh hãi", "sợ mất"
    ],

    # Nhóm ANGER
    "ANGER": [
        "tức", "bực", "cáu", "phẫn nộ", "ức chế", "điên tiết", "ghét", "cay cú",
        "bực mình", "bực bội", "tức giận", "cay đắng", "khó chịu", "nổi xung"
    ],
    "DISGUST": [
        "ghê tởm", "kinh tởm", "kinh sợ", "dị ứng", "chán ghét", "tởm lợm", "ghê sợ"
    ]
}

# Mapping từ Sub-emotion về Primary-emotion (chuẩn theo taxonomy.json)
SUB_TO_PRIMARY = {
    "JOY": "JOY",
    "CALM": "JOY",
    "HOPE": "JOY",
    "CONNECTION": "JOY",
    "SADNESS": "SADNESS",
    "LONELINESS": "SADNESS",
    "GUILT_SHAME": "SADNESS",
    "ANXIETY": "ANXIETY",
    "FEAR": "ANXIETY",
    "ANGER": "ANGER",
    "DISGUST": "ANGER"
}

# 7. Dấu hiệu văn phong thuyết giảng, hướng dẫn, dẫn chương trình podcast (Didactic & Presenter Markers)
DIDACTIC_PRESENTER_MARKERS = [
    "chào mừng các bạn", "xin chào các bạn", "chào quý vị", "thính giả thân mến", "kính thưa quý vị",
    "chương trình hôm nay", "tập podcast", "tập hôm nay", "video hôm nay", "chủ đề ngày hôm nay",
    "chúng ta cần", "chúng ta phải", "chúng ta nên", "chúng ta hãy", "chúng ta cùng",
    "bài học rút ra", "bài học hôm nay", "lời khuyên", "phương pháp", "kỹ năng sống", "thói quen tốt",
    "hãy nhớ rằng", "nhớ like", "đăng ký kênh", "chia sẻ video", "comment bên dưới", "bình luận bên dưới",
    "thí dụ như", "ví dụ như", "người thành công", "mỗi công dân", "cộng đồng mạng", "xã hội hiện nay"
]

# 8. Dấu hiệu trải lòng cảm xúc nội tâm sâu sắc (Deep Affective Predicates)
AFFECTIVE_PREDICATES = [
    "cảm thấy", "thấy lòng", "trong lòng", "lòng mình", "nghĩ lại", "nghẹt thở", "bật khóc", "rơi nước mắt",
    "áp lực", "cô đơn", "lạc lõng", "chông chênh", "bế tắc", "tuyệt vọng", "ấm áp", "nhẹ nhõm", "an yên",
    "hạnh phúc", "biết ơn", "tủi thân", "đau lòng", "bất lực", "tự trách", "day dứt", "vững lòng", "hy vọng",
    "mệt mỏi", "buồn phiền", "quá tải", "hoang mang", "lo sợ", "sợ hãi", "bồn chồn", "chơi vơi", "đau đớn",
    "chìm sâu", "vỡ vụn", "nước mắt rơi", "khóc một mình", "không ai hiểu", "tự dặn lòng"
]

POSITIVE_SUBS = {"JOY", "CALM", "HOPE", "CONNECTION"}


class AffectiveSelfDisclosureFilter:
    """
    Bộ kiểm định Chiều sâu Bộc bạch Cảm xúc (Affective Self-Disclosure Gatekeeper).
    Phân định chính xác giữa:
    - Tier 1: Deep Affective Journal / Confession ('deep_journal')
      -> Ngôi thứ nhất tự sự, cảm xúc chân thực, không thuyết giảng.
    - Tier 2: General Spoken Monologue MER ('general_monologue')
      -> Độc thoại sạch, podcast tản văn, đọc truyện/triết lý sống, có ích cho pretraining acoustic prosody.
    - Rejected Noise ('rejected')
      -> Lời chào đầu/cuối kênh, quảng cáo, câu ngắn vô nghĩa.
    """

    def __init__(self, min_word_count: int = 10):
        self.min_word_count = min_word_count

    def evaluate_affective_disclosure(self, text: str) -> Dict[str, Any]:
        if not text or len(text.strip().split()) < self.min_word_count:
            return {
                "data_tier": "rejected",
                "disclosure_score": 0.0,
                "first_person_count": 0,
                "affective_count": 0,
                "didactic_count": 0,
                "matched_didactic": [],
                "matched_affective": [],
                "matched_pronouns": [],
                "reason": "Text quá ngắn hoặc rỗng"
            }

        text_lower = text.lower()

        # 1. Quét dấu hiệu thuyết giảng / dẫn chương trình (Didactic)
        matched_didactic = []
        for kw in DIDACTIC_PRESENTER_MARKERS:
            pattern = r"(?<!\w)" + re.escape(kw) + r"(?!\w)"
            if re.search(pattern, text_lower):
                matched_didactic.append(kw)

        # 2. Quét đại từ nhân xưng ngôi thứ nhất (First-person pronouns)
        matched_pronouns = []
        for kw in FIRST_PERSON_MARKERS:
            if kw in ["anh", "chị"]:
                pattern = r"(?<!chúc mừng\s)(?<!\w)" + re.escape(kw) + r"(?!\s+[A-ZÀ-Ỹ])(?!\w)"
                target_str = text
            elif kw in ["t", "e"]:
                pattern = r"(?<!\w)" + re.escape(kw) + r"(?!\w)"
                target_str = text_lower
            else:
                pattern = r"(?<!\w)" + re.escape(kw) + r"(?!\w)"
                target_str = text_lower
            if re.search(pattern, target_str):
                matched_pronouns.append(kw)

        # 3. Quét vị ngữ trạng thái cảm xúc nội tâm (Affective Predicates)
        matched_affective = []
        for kw in AFFECTIVE_PREDICATES:
            pattern = r"(?<!\w)" + re.escape(kw) + r"(?!\w)"
            if re.search(pattern, text_lower):
                matched_affective.append(kw)

        fp_count = len(matched_pronouns)
        aff_count = len(matched_affective)
        didactic_count = len(matched_didactic)

        # Tính điểm bộc bạch cảm xúc cá nhân (0.0 - 1.0)
        raw_score = (fp_count * 0.3) + (aff_count * 0.35) - (didactic_count * 0.4)
        disclosure_score = max(0.0, min(1.0, round(raw_score, 2)))

        # Phân tầng
        if didactic_count >= 2 or (fp_count == 0 and didactic_count >= 1):
            tier = "general_monologue"
            reason = f"Độc thoại diễn thuyết / podcast chia sẻ (phát hiện {didactic_count} chỉ dấu thuyết giảng)"
        elif fp_count >= 1 and aff_count >= 1 and didactic_count == 0:
            tier = "deep_journal"
            reason = f"Nhật ký tự sự bộc bạch cảm xúc sâu sắc (Tier 1 Core Journal)"
        elif fp_count >= 1 or aff_count >= 1:
            tier = "deep_journal" if disclosure_score >= 0.35 else "general_monologue"
            reason = "Tự sự trải lòng đời thường" if tier == "deep_journal" else "Độc thoại tâm sự tổng quát"
        else:
            tier = "general_monologue"
            reason = "Độc thoại tự nhiên không bộc lộ ngôi xưng sâu sắc"

        return {
            "data_tier": tier,
            "disclosure_score": disclosure_score,
            "first_person_count": fp_count,
            "affective_count": aff_count,
            "didactic_count": didactic_count,
            "matched_didactic": matched_didactic,
            "matched_affective": matched_affective,
            "matched_pronouns": matched_pronouns,
            "reason": reason
        }


class JournalFilter:
    """Bộ lọc & Trích xuất cảm xúc chuẩn hóa cho bài viết Nhật ký / Tâm sự (Phiên bản v2)."""

    def __init__(self, min_word_count: int = 15, normalize_teencode: bool = True):
        self.min_word_count = min_word_count
        self.disclosure_filter = AffectiveSelfDisclosureFilter(min_word_count=min_word_count)
        if VietnameseTextPreprocessor is not None:
            self.preprocessor = VietnameseTextPreprocessor(
                lowercase=True,
                normalize_teencode=normalize_teencode
            )
        else:
            self.preprocessor = None

    def mask_pii(self, text: str) -> str:
        """Ẩn danh hóa thông tin cá nhân: số điện thoại, email, URL."""
        # Email
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL]', text)
        # Số điện thoại VN (10 số, bắt đầu 03, 05, 07, 08, 09 hoặc +84)
        text = re.sub(r'(?:\+?84|0)(?:3|5|7|8|9)\d{8}', '[PHONE]', text)
        # URL
        text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
        return text

    def check_crisis_indicators(self, text: str) -> Tuple[bool, List[str]]:
        """Phát hiện các từ khóa cảnh báo nguy cơ tự hại hoặc khủng hoảng tâm lý trầm trọng."""
        text_lower = text.lower()
        matched = []
        for kw in CRISIS_KEYWORDS:
            pattern = r"(?<!\w)" + re.escape(kw) + r"(?!\w)"
            if re.search(pattern, text_lower):
                matched.append(kw)
        return len(matched) > 0, matched

    def is_commercial_or_noise(self, text: str) -> Tuple[bool, Optional[str]]:
        """Kiểm tra bài viết có dính từ khóa quảng cáo, bán hàng, phe vé hoặc tin tức không."""
        text_lower = text.lower()
        # 1. Kiểm tra regex giao dịch thương mại / chợ vé / scalping chuyên biệt
        for pat in TICKET_SCALPING_PATTERNS:
            m = re.search(pat, text_lower)
            if m:
                return True, f"giao dịch/scalping ('{m.group(0)}')"

        # 2. Kiểm tra danh sách đen từ khóa chung (Blacklist)
        for kw in COMMERCIAL_AND_NOISE_BLACKLIST:
            pattern = r"(?<!\w)" + re.escape(kw) + r"(?!\w)"
            if re.search(pattern, text_lower):
                return True, kw
        return False, None

    def check_first_person_narrative(self, text: str) -> Dict[str, Any]:
        """Kiểm tra bài viết có ngôi xưng thứ nhất và phong cách tự sự nhật ký không."""
        text_lower = text.lower()
        matched_pronouns = []
        for kw in FIRST_PERSON_MARKERS:
            # Ngăn trường hợp đại từ 'anh/chị' đi kèm tên riêng viết hoa trong tin showbiz (vd: 'chúc mừng anh HIEUTHUHAI')
            if kw in ["anh", "chị"]:
                pattern = r"(?<!chúc mừng\s)(?<!\w)" + re.escape(kw) + r"(?!\s+[A-ZÀ-Ỹ])(?!\w)"
                target_str = text
            elif kw in ["t", "e"]:
                pattern = r"(?<!\w)" + re.escape(kw) + r"(?!\w)"
                target_str = text_lower
            else:
                pattern = r"(?<!\w)" + re.escape(kw) + r"(?!\w)"
                target_str = text_lower

            if re.search(pattern, target_str):
                matched_pronouns.append(kw)

        matched_reflective = []
        for kw in JOURNAL_REFLECTIVE_MARKERS:
            pattern = r"(?<!\w)" + re.escape(kw) + r"(?!\w)"
            if re.search(pattern, text_lower):
                matched_reflective.append(kw)

        is_narrative = (len(matched_pronouns) > 0) or (len(matched_reflective) >= 2)
        return {
            "is_narrative": is_narrative,
            "matched_pronouns": matched_pronouns,
            "matched_reflective": matched_reflective,
            "score": min(1.0, (len(matched_pronouns) * 0.4 + len(matched_reflective) * 0.2))
        }

    def _is_negated(self, text: str, match_start: int) -> bool:
        """Kiểm tra xem từ khóa cảm xúc có bị phủ định trong phạm vi 3 từ phía trước không."""
        prefix = text[:match_start].strip()
        words = prefix.split()
        if not words:
            return False
        # Kiểm tra 3 từ gần nhất phía trước
        preceding_tokens = [w.lower().strip(".,!?\"'") for w in words[-3:]]
        for nw in NEGATION_WORDS:
            if nw in preceding_tokens:
                return True
        return False

    def analyze_emotion(self, text: str) -> Dict[str, Any]:
        """
        Phân tích cảm xúc và ánh xạ theo 5 Primary & 11 Sub emotions.
        Tích hợp Negation Scope: Đảo chiều khi từ tích cực đi liền sau từ phủ định ("không vui").
        """
        text_lower = text.lower()
        sub_scores: Dict[str, float] = {}
        matched_words: Dict[str, List[str]] = {}

        for sub_label, keywords in TAXONOMY_LEXICON.items():
            hits = []
            for kw in keywords:
                pattern = r"(?<!\w)" + re.escape(kw) + r"(?!\w)"
                for m in re.finditer(pattern, text_lower):
                    is_neg = self._is_negated(text_lower, m.start())
                    if is_neg and sub_label in POSITIVE_SUBS:
                        # "không vui", "chẳng an tâm" -> chuyển sang tăng SADNESS hoặc ANXIETY
                        sub_scores["SADNESS"] = sub_scores.get("SADNESS", 0.0) + 1.2
                        matched_words.setdefault("SADNESS", []).append(f"không_{kw}")
                    else:
                        hits.append(kw)
                        sub_scores[sub_label] = sub_scores.get(sub_label, 0.0) + 1.0

            if hits:
                matched_words.setdefault(sub_label, []).extend(list(set(hits)))

        if not sub_scores:
            return {
                "primary_emotion": "NEUTRAL",
                "sub_emotion": "CALM",
                "emotion_score": 0.0,
                "matched_keywords": []
            }

        best_sub = max(sub_scores, key=sub_scores.get)
        primary_emotion = SUB_TO_PRIMARY.get(best_sub, "NEUTRAL")
        all_hits = [kw for kws in matched_words.values() for kw in kws]
        emotion_score = min(1.0, round(float(sub_scores[best_sub]) / 2.5, 2))

        return {
            "primary_emotion": primary_emotion,
            "sub_emotion": best_sub,
            "emotion_score": emotion_score,
            "matched_keywords": all_hits
        }

    def evaluate_entry(self, text: str) -> Dict[str, Any]:
        """
        Đánh giá toàn diện một mẫu văn bản nhật ký/tâm sự:
        - Khử bỏ tiền tố UI (ngày tháng, huy hiệu tác giả).
        - Chuẩn hóa text và khử teencode đồng bộ với pipeline NLP.
        - Phát hiện và cách ly dấu hiệu khủng hoảng (is_crisis).
        - Kiểm định chiều sâu tự sự ngôi thứ nhất.
        - Phân tích cảm xúc có xử lý phủ định.
        """
        if not text or not text.strip():
            return {
                "accepted": False,
                "reason": "Text rỗng",
                "word_count": 0,
                "is_crisis": False
            }

        # 0. Khử sạch các tiền tố giao diện UI (header ngày tháng, huy hiệu tác giả)
        cleaned_input = re.sub(
            r"^(?:\d{1,2}/\d{1,2}/\d{4}|\d+\s*(?:giờ|ngày|phút|tuần|tháng|năm|s|m|h|d|w)(?:\s*trước)?)\s*[·•\-–—]?\s*",
            "",
            text.strip(),
            flags=re.IGNORECASE
        )
        cleaned_input = re.sub(r"\s*[·•]?\s*tác giả\b", "", cleaned_input, flags=re.IGNORECASE).strip()
        if not cleaned_input:
            cleaned_input = text.strip()

        # 1. Tiền xử lý văn bản qua preprocessor dự án
        if self.preprocessor is not None:
            prep_res = self.preprocessor.process(cleaned_input)
            cleaned_text = prep_res.cleaned_text
        else:
            cleaned_text = re.sub(r"\s+", " ", cleaned_input).strip()

        # Ẩn danh hóa PII
        cleaned_text = self.mask_pii(cleaned_text)

        words = cleaned_text.split()
        word_count = len(words)

        default_result = {
            "accepted": False,
            "word_count": word_count,
            "is_crisis": False,
            "crisis_keywords": [],
            "cleaned_text": cleaned_text,
            "data_tier": "rejected",
            "disclosure_score": 0.0,
            "didactic_markers": [],
            "primary_emotion": "NEUTRAL",
            "sub_emotion": "CALM",
            "emotion_score": 0.0,
            "matched_keywords": []
        }

        # 2. Kiểm tra độ dài
        if word_count < self.min_word_count:
            return {
                **default_result,
                "reason": f"Text quá ngắn ({word_count} từ < tối thiểu {self.min_word_count})"
            }

        # 3. Kiểm tra blacklist quảng cáo / thương mại / tin tức
        is_spam, spam_kw = self.is_commercial_or_noise(cleaned_text)
        if is_spam:
            return {
                **default_result,
                "reason": f"Chứa từ khóa thương mại/quảng cáo/tin tức ('{spam_kw}')"
            }

        # 4. Kiểm tra dấu hiệu khủng hoảng / tự hại (Crisis Quarantine)
        has_crisis, crisis_kws = self.check_crisis_indicators(cleaned_text)

        # 5. Kiểm tra tính tự sự / nhật ký ngôi thứ nhất
        narrative_info = self.check_first_person_narrative(cleaned_text)
        if not narrative_info["is_narrative"]:
            return {
                **default_result,
                "is_crisis": has_crisis,
                "crisis_keywords": crisis_kws,
                "reason": "Thiếu phong cách nhật ký/tâm sự (không có ngôi xưng cá nhân hoặc từ trải lòng)"
            }

        # 6. Phân tích cảm xúc theo taxonomy
        emotion_info = self.analyze_emotion(cleaned_text)

        # 7. Kiểm định chiều sâu bộc bạch cảm xúc (Affective Self-Disclosure)
        disclosure_info = self.disclosure_filter.evaluate_affective_disclosure(cleaned_text)
        if has_crisis:
            data_tier = "crisis_quarantine"
        else:
            data_tier = disclosure_info["data_tier"]

        # Chấp nhận mẫu
        return {
            "accepted": True,
            "is_crisis": has_crisis,
            "crisis_keywords": crisis_kws,
            "data_tier": data_tier,
            "disclosure_score": disclosure_info["disclosure_score"],
            "didactic_markers": disclosure_info["matched_didactic"],
            "reason": "Mẫu khủng hoảng tâm lý (cần cách ly riêng)" if has_crisis else f"Mẫu {data_tier} hợp lệ ({disclosure_info['reason']})",
            "word_count": word_count,
            "cleaned_text": cleaned_text,
            "narrative_score": narrative_info["score"],
            "matched_pronouns": narrative_info["matched_pronouns"],
            "matched_reflective": narrative_info["matched_reflective"],
            "primary_emotion": emotion_info["primary_emotion"],
            "sub_emotion": emotion_info["sub_emotion"],
            "emotion_score": emotion_info["emotion_score"],
            "matched_keywords": emotion_info["matched_keywords"]
        }

    def evaluate_affective_disclosure(self, text: str) -> Dict[str, Any]:
        """Đánh giá trực tiếp chiều sâu bộc bạch tự sự & phân tầng dữ liệu."""
        return self.disclosure_filter.evaluate_affective_disclosure(text)
