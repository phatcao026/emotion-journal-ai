"""
test_crawler_pipeline.py
------------------------
Unit tests toàn diện cho hệ thống thu thập & lọc dữ liệu Nhật ký Cảm xúc (MER):
1. JournalFilter v2:
   - Ngôi xưng tự sự & loại bỏ quảng cáo / tin tức.
   - Nhận diện phạm vi phủ định (Negation Scope Detection: "không vui" -> SADNESS).
   - Phát hiện nguy cơ tự hại / khủng hoảng tâm lý (Crisis & Self-Harm Quarantine).
   - Tích hợp chuẩn hóa teencode và PII masking.
2. DatasetManager v2:
   - Lưu trữ đa phương thức (text, voice, text_voice).
   - Cách ly mẫu khủng hoảng vào crisis_quarantine/.
   - Băm danh tính tác giả (Author Hashing cho Participant-aware split).
   - Resumable deduplication (chống cào trùng lặp URL & sample_id).
   - Thống kê toàn diện (get_statistics).
3. VoiceProcessor:
   - Chuyển đổi và kiểm định âm học 16kHz Mono WAV, RMS, peak clipping.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.crawler.journal_filter import JournalFilter
from scripts.crawler.dataset_manager import CrawledDatasetManager
from scripts.crawler.voice_processor import VoiceProcessor


class TestCrawlerPipeline(unittest.TestCase):
    def setUp(self):
        self.filter = JournalFilter(min_word_count=15)
        self.temp_dir = tempfile.mkdtemp()
        self.dm = CrawledDatasetManager(base_dir=self.temp_dir)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_reject_short_text(self):
        short_text = "Hôm nay buồn quá."
        res = self.filter.evaluate_entry(short_text)
        self.assertFalse(res["accepted"])
        self.assertIn("quá ngắn", res["reason"])

    def test_reject_commercial_spam(self):
        spam_text = (
            "Hôm nay mình gửi tặng các bạn mã giảm giá cực hot, hãy nhấn link ở bio để mua ngay sản phẩm mới "
            "với giá ưu đãi nhất nhé mọi người ơi!"
        )
        res = self.filter.evaluate_entry(spam_text)
        self.assertFalse(res["accepted"])
        self.assertIn("thương mại", res["reason"])

    def test_reject_news_showbiz(self):
        news_text = (
            "Theo thông tin mới nhất từ cơ quan chức năng, vụ việc liên quan đến bộ phim và ca sĩ vừa ra mắt "
            "đang thu hút sự chú ý của đông đảo khán giả cả nước."
        )
        res = self.filter.evaluate_entry(news_text)
        self.assertFalse(res["accepted"])

    def test_accept_personal_sadness_confession(self):
        confession_text = (
            "Đêm muộn ngồi một mình trong căn phòng trọ nhỏ, dạo này áp lực công việc khiến mình thấy ngột ngạt "
            "và cô đơn vô cùng. Nhiều lúc muốn bỏ cuộc nhưng lại tự dặn lòng phải cố gắng tiếp tục."
        )
        res = self.filter.evaluate_entry(confession_text)
        self.assertTrue(res["accepted"])
        self.assertEqual(res["primary_emotion"], "SADNESS")
        self.assertEqual(res["sub_emotion"], "LONELINESS")
        self.assertIn("mình", res["matched_pronouns"])
        self.assertFalse(res.get("is_crisis", False))

    def test_accept_personal_joy_connection(self):
        diary_text = (
            "Hôm nay là một ngày thật bình yên và hạnh phúc của tôi. Buổi tối sum họp bên gia đình, cùng nhau "
            "nấu ăn và sẻ chia những câu chuyện đời thường khiến tôi thấy lòng ấm áp lạ thường."
        )
        res = self.filter.evaluate_entry(diary_text)
        self.assertTrue(res["accepted"])
        self.assertEqual(res["primary_emotion"], "JOY")
        self.assertIn(res["sub_emotion"], ["JOY", "CONNECTION", "CALM"])

    def test_negation_scope_handling(self):
        """Kiểm tra câu chứa từ phủ định + từ tích cực ('không vui') không bị gán nhầm thành JOY."""
        negated_text = (
            "Hôm nay tôi không thấy vui chút nào cả, trong lòng bế tắc và áp lực nặng nề kéo dài suốt cả tuần nay."
        )
        res = self.filter.evaluate_entry(negated_text)
        self.assertTrue(res["accepted"])
        # Không được ra JOY vì đã bị từ 'không' phủ định
        self.assertNotEqual(res["primary_emotion"], "JOY")
        self.assertIn(res["primary_emotion"], ["SADNESS", "ANXIETY"])

    def test_crisis_detection_and_quarantine(self):
        """Kiểm tra phát hiện từ khóa khủng hoảng/tự hại và lưu vào thư mục cách ly riêng."""
        crisis_text = (
            "Tôi quá mệt mỏi với cuộc sống này rồi, nhiều đêm ngồi một mình chỉ muốn giải thoát cho xong "
            "và kết thúc cuộc đời để không còn phải chịu đựng thêm bất cứ đau đớn nào nữa."
        )
        res = self.filter.evaluate_entry(crisis_text)
        self.assertTrue(res["is_crisis"])
        self.assertTrue(len(res["crisis_keywords"]) > 0)

        # Lưu vào khu vực cách ly
        sample_id = "test_crisis_001"
        crisis_path = self.dm.save_crisis_sample(
            sample_id=sample_id,
            text=res["cleaned_text"],
            crisis_keywords=res["crisis_keywords"],
            primary_label=res["primary_emotion"],
            sub_label=res["sub_emotion"]
        )
        self.assertTrue(crisis_path.exists())
        self.assertTrue(self.dm.crisis_dir.exists())
        self.assertIn("crisis_quarantine", str(crisis_path))

    def test_author_hash_and_anonymization(self):
        """Kiểm tra băm mã tác giả author_hash phục vụ Participant-aware split."""
        sample_id = "test_author_001"
        text = "Nhật ký của tôi: Ngày hôm nay trôi qua thật bình yên và nhẹ nhõm sau những ngày giông bão."
        self.dm.save_text_sample(
            sample_id=sample_id,
            text=text,
            primary_label="JOY",
            sub_label="CALM",
            emotion_score=0.8,
            meta={"author": "nguyenvana", "source_platform": "threads"}
        )

        stats = self.dm.get_statistics()
        self.assertEqual(stats["total_records"], 1)
        self.assertEqual(stats["unique_authors_count"], 1)

    def test_resumable_url_deduplication(self):
        """Kiểm tra cơ chế chống cào trùng lặp URL."""
        test_url = "https://vnexpress.net/test-tam-su-12345.html"
        self.assertFalse(self.dm.is_url_crawled(test_url))

        self.dm.save_text_sample(
            sample_id="test_url_001",
            text="Nhật ký hôm nay: Tôi thấy bình yên trong lòng sau một tuần nhiều áp lực công việc.",
            primary_label="JOY",
            sub_label="CALM",
            emotion_score=0.8,
            meta={"source_url": test_url}
        )

        # Sau khi lưu, URL phải được đánh dấu là đã cào
        self.assertTrue(self.dm.is_url_crawled(test_url))
        self.assertTrue(self.dm.is_sample_exists("test_url_001"))

    def test_pii_masking(self):
        """Kiểm tra che số điện thoại và email."""
        text_with_pii = (
            "Hôm nay mình rất buồn, nếu ai cần tâm sự hãy liên hệ email test@example.com "
            "hoặc gọi cho mình qua số 0912345678 nhé mọi người."
        )
        masked = self.filter.mask_pii(text_with_pii)
        self.assertNotIn("test@example.com", masked)
        self.assertNotIn("0912345678", masked)
        self.assertIn("[EMAIL]", masked)
        self.assertIn("[PHONE]", masked)

    def test_voice_processor_short_audio(self):
        """Kiểm tra VoiceProcessor chấp nhận các mẫu âm thanh chuẩn 16kHz."""
        import numpy as np
        import wave

        sr = 16000
        duration_s = 4.0
        t = np.linspace(0, duration_s, int(sr * duration_s), False)
        signal = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)

        wav_path = os.path.join(self.temp_dir, "test_4s.wav")
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(signal.tobytes())

        vp = VoiceProcessor(min_duration_s=3.0)
        audit = vp.inspect_and_audit(wav_path)
        self.assertTrue(audit["is_valid"])
        self.assertAlmostEqual(audit["duration_s"], 4.0, delta=0.1)

    def test_reject_ticket_scalping(self):
        """Kiểm tra loại bỏ 100% các bài viết phe vé/pass vé/chợ đen concert."""
        samples_scalping = [
            "Pass vé Cat 1 day 1 BIGBANG công khai 6tr5 + 1tr tiền camp tổng 7tr5 nhá. GDTT, legit tuyệt đối ạ mấy đứa scam né ra",
            "pass 1 vé Hoả Tâm 2 day 1 day 2 17/10 ạ giá gốc 3tr pass nhanh 2tr8 có fix nha mn",
            "Bạn nào có vé cần pass thì ib mình nha, giá hợp lý mình chốt nhanh ạ",
            "Do không thể đi concert và muốn hồi máu nhanh cho kế hoạch khác nên mình pass 1 Hắc Mã 1 day1 giá 2tr7 full merch có fast track",
            "Nhà em cần trade 1 vé Hắc Mã 1 day 1 sang 1 vé Hắc Mã 2 day 1 ạ, ai có nhu cầu trade thì inbox em với nha"
        ]
        for s in samples_scalping:
            res = self.filter.evaluate_entry(s)
            self.assertFalse(res["accepted"], f"Lẽ ra phải từ chối bài scalping: {s[:50]}...")
            self.assertIn("thương mại", res["reason"])

    def test_reject_showbiz_fanpage_news(self):
        """Kiểm tra loại bỏ tin tức fanpage PR showbiz và không nhận nhầm 'anh [Tên]' là tự sự."""
        pr_news = (
            "GIỮ CHUỖI TẠI DAY 3: NEGAV TIẾP TỤC CÓ MẶT TẠI MẮT NHẮM MẮT MỞ CONCERT. "
            "Chúc mừng anh HIEUTHUHAI mở concert Day 3 tại Hà Nội bùng nổ sân khấu."
        )
        res = self.filter.evaluate_entry(pr_news)
        self.assertFalse(res["accepted"])

    def test_accept_concert_emotional_diary(self):
        """Kiểm tra chấp nhận trải nghiệm đi concert mang tính tự sự cảm xúc chân thật."""
        concert_diary = (
            "Hôm nay đi concert một mình, lúc nghe bài hát quen thuộc cất lên mình đã bật khóc giữa biển người, "
            "nhớ thanh xuân của mình quá chừng."
        )
        res = self.filter.evaluate_entry(concert_diary)
        self.assertTrue(res["accepted"])
        self.assertEqual(res["primary_emotion"], "SADNESS")
        self.assertIn("mình", res["matched_pronouns"])

    def test_accept_gym_emotional_confession(self):
        """Kiểm tra chấp nhận tâm sự tập gym đời thường mang theo xung đột tâm lý/cảm xúc."""
        gym_confession = (
            "Có ai cãi nhau với ny vì thuê pt tập gym chưa ạ ? e thuê pt nữ vì nghĩ hướng dẫn tập sẽ tốt hơn "
            "nhưng ny e thì cứ ghen tuông này nọ, h còn bảo nếu e đi tập thì chia tay. mn bảo e phải làm tn đây ạ?"
        )
        res = self.filter.evaluate_entry(gym_confession)
        self.assertTrue(res["accepted"])
        self.assertIn("e", res["matched_pronouns"])

    def test_social_media_pronouns_and_markers(self):
        """Kiểm tra nhận diện đại từ tự xưng mạng xã hội (t, e, tui, toi, mk, mik)."""
        text_t = (
            "Ê bữa có bạn nào bảo nên chăm viết nhật ký biết ơn để manifest ý, kiểu t mới viết từ cuối tuần trước, "
            "nhưng trộm vía ngày nào cũng có tin vui ý và thấy yêu đời hẳn."
        )
        res = self.filter.evaluate_entry(text_t)
        self.assertTrue(res["accepted"])
        self.assertIn("t", res["matched_pronouns"])
        self.assertEqual(res["primary_emotion"], "JOY")

    def test_ui_artifact_stripping(self):
        """Kiểm tra tiền xử lý tự động cắt bỏ date header và author badge của Threads UI."""
        raw_ui = (
            "27/04/2026 · Tác giả Đêm muộn ngồi một mình trong căn phòng trọ nhỏ, "
            "dạo này áp lực công việc khiến mình thấy ngột ngạt và cô đơn vô cùng."
        )
        res = self.filter.evaluate_entry(raw_ui)
        self.assertTrue(res["accepted"])
        self.assertNotIn("27/04/2026", res["cleaned_text"])
        self.assertNotIn("tác giả", res["cleaned_text"].lower())


if __name__ == "__main__":
    unittest.main()
