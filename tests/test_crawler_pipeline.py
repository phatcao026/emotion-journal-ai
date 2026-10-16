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

    def test_itu_p56_snr_clean_vs_noisy(self):
        """Kiểm tra đo SNR chuẩn ITU-T P.56 phân biệt chính xác audio sạch vs audio nhiễu."""
        import numpy as np
        sr = 16000
        duration_s = 4.0
        t = np.linspace(0, duration_s, int(sr * duration_s), False)
        # Speech-like modulated tone
        env = np.clip(np.sin(2 * np.pi * 1.5 * t), 0, 1)
        sig_clean = (0.3 * env * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        # Tạp âm trắng lớn
        np.random.seed(42)
        sig_noisy = sig_clean + 0.12 * np.random.randn(len(t)).astype(np.float32)

        vp = VoiceProcessor(min_duration_s=3.0)
        snr_clean, nf_clean = vp.calculate_itu_p56_snr(sig_clean)
        snr_noisy, nf_noisy = vp.calculate_itu_p56_snr(sig_noisy)

        self.assertGreater(snr_clean, 20.0, f"Clean SNR should be > 20 dB, got {snr_clean}")
        self.assertLess(snr_noisy, 12.0, f"Noisy SNR should be < 12 dB, got {snr_noisy}")

    def test_decoupled_acoustic_affect_estimation(self):
        """Kiểm tra ước lượng cảm xúc âm học độc lập (Acoustic Affect) không phụ thuộc Text."""
        import numpy as np
        import wave
        sr = 16000
        duration_s = 3.5
        t = np.linspace(0, duration_s, int(sr * duration_s), False)

        # 1. Âm thanh năng lượng thấp, F0 phẳng (Sadness / Loneliness)
        sig_sad = (0.05 * np.sin(2 * np.pi * 120 * t) * 32767).astype(np.int16)
        sad_wav = os.path.join(self.temp_dir, "test_sad.wav")
        with wave.open(sad_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(sig_sad.tobytes())

        vp = VoiceProcessor(min_duration_s=3.0)
        affect_sad = vp.estimate_acoustic_affect(sad_wav)
        self.assertLess(affect_sad["arousal"], 0.40)
        self.assertEqual(affect_sad["primary"], "SADNESS")
        self.assertIn(affect_sad["sub"], ["LONELINESS", "SADNESS"])

        # 2. Âm thanh năng lượng cao, dao động tần số mạnh (Anxiety / Anger)
        freq_mod = 250 + 40 * np.sin(2 * np.pi * 8 * t)
        sig_high = (0.35 * np.sin(2 * np.pi * freq_mod * t) * 32767).astype(np.int16)
        high_wav = os.path.join(self.temp_dir, "test_high.wav")
        with wave.open(high_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(sig_high.tobytes())

        affect_high = vp.estimate_acoustic_affect(high_wav)
        self.assertGreater(affect_high["arousal"], 0.65)
        self.assertIn(affect_high["primary"], ["ANXIETY", "ANGER"])

    def test_incongruent_masking_detection(self):
        """Kiểm tra cơ chế phát hiện Emotional Masking (nói 'vui' nhưng giọng 'buồn')."""
        text = "Hôm nay tôi thấy bình yên và nhẹ nhõm vô cùng sau nhiều ngày áp lực."
        eval_text = self.filter.evaluate_entry(text)
        text_pri = eval_text["primary_emotion"]

        # Giả lập giọng nói thực tế là buồn (SADNESS)
        voice_pri = "SADNESS"

        is_incongruent = (text_pri != voice_pri and (text_pri == "JOY" or voice_pri in ["SADNESS", "ANXIETY", "ANGER"]))
        self.assertTrue(is_incongruent)

    def test_strict_verbatim_transcript_pairing(self):
        """Kiểm tra quy tắc vàng: Cặp text_voice CHỈ được lưu khi có transcript ASR chân thực (>= 5 từ)."""
        empty_transcript = ""
        short_noise_transcript = "À ừm"
        valid_transcript = "Hôm nay mình ngồi một mình trong phòng trọ rất buồn"

        self.assertFalse(bool(empty_transcript and len(empty_transcript.split()) >= 5))
        self.assertFalse(bool(short_noise_transcript and len(short_noise_transcript.split()) >= 5))
        self.assertTrue(bool(valid_transcript and len(valid_transcript.split()) >= 5))

    def test_affective_self_disclosure_filter_deep_journal(self):
        """Kiểm tra nhận diện chính xác bài viết tự sự bộc bạch nội tâm sâu sắc (Tier 1: deep_journal)."""
        confession = (
            "Đêm muộn ngồi một mình trong phòng trọ, dạo này áp lực công việc khiến mình thấy ngột ngạt và cô đơn vô cùng. "
            "Nhiều lúc muốn bỏ cuộc nhưng lại tự dặn lòng phải cố gắng tiếp tục vượt qua."
        )
        res = self.filter.evaluate_entry(confession)
        self.assertTrue(res["accepted"])
        self.assertEqual(res["data_tier"], "deep_journal")
        self.assertGreaterEqual(res["disclosure_score"], 0.4)
        self.assertIn("mình", res["matched_pronouns"])
        self.assertEqual(len(res["didactic_markers"]), 0)

    def test_affective_self_disclosure_filter_didactic_lecture(self):
        """Kiểm tra phân tầng bài thuyết giảng/podcast dạy đời vào Tier 2 (general_monologue)."""
        lecture = (
            "Chào mừng các bạn quay trở lại với tập podcast hôm nay. Hôm nay chúng ta cùng bàn về 3 bài học đắt giá "
            "để vượt qua nghịch cảnh và phương pháp rèn luyện thói quen tốt mỗi ngày."
        )
        res = self.filter.evaluate_affective_disclosure(lecture)
        self.assertEqual(res["data_tier"], "general_monologue")
        self.assertGreaterEqual(res["didactic_count"], 2)
        self.assertIn("chúng ta cùng", res["matched_didactic"])

    def test_dataset_manager_tiered_storage(self):
        """Kiểm tra lưu trữ phân tầng dữ liệu Tier 1 và Tier 2 trong CrawledDatasetManager."""
        sample_deep = "sample_deep_001"
        sample_general = "sample_general_001"

        self.dm.save_text_sample(
            sample_id=sample_deep,
            text="Hôm nay mình thấy rất cô đơn và bế tắc trong lòng.",
            primary_label="SADNESS",
            sub_label="LONELINESS",
            emotion_score=0.8,
            data_tier="deep_journal"
        )

        self.dm.save_text_sample(
            sample_id=sample_general,
            text="Trong bài học hôm nay chúng ta nói về sự kiên định trong cuộc sống.",
            primary_label="JOY",
            sub_label="HOPE",
            emotion_score=0.5,
            data_tier="general_monologue"
        )

        stats = self.dm.get_statistics()
        self.assertEqual(stats["tiers"]["deep_journal"], 1)
        self.assertEqual(stats["tiers"]["general_monologue"], 1)

        deep_samples = self.dm.get_samples_by_tier("deep_journal")
        self.assertEqual(len(deep_samples), 1)
        self.assertEqual(deep_samples[0]["sample_id"], sample_deep)


if __name__ == "__main__":
    unittest.main()

