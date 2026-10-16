"""
scripts/test_stage3_e2e.py
--------------------------
End-to-End Test Suite for Stage 3 Emotion Journal AI Pipeline.
Verifies:
    1. EmotionResult serialization & format integrity.
    2. Stage3InferenceEngine prediction with dummy synthetic audio.
    3. Joint MAP hierarchical consistency (0.00% violation).
    4. LongitudinalTracker SQLite logging & trend analysis.
    5. CrisisDetector lexical & acoustic trigger detection.
    6. EmpatheticCoach offline CBT response generation.
    7. Robustness on empty text and silent audio.
"""

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch

from src.stage2.hierarchical_mer_stage2 import TAXONOMY_MAP
from src.stage3 import (
    CrisisAlert,
    CrisisDetector,
    EmotionResult,
    EmpatheticCoach,
    LongitudinalTracker,
    Stage3InferenceEngine,
)


class TestStage3E2EPipeline(unittest.TestCase):

    def setUp(self):
        # 16kHz dummy waveform (1, 16000 * 3)
        self.sr = 16000
        self.duration_s = 3.0
        t = np.linspace(0, self.duration_s, int(self.duration_s * self.sr), endpoint=False)
        self.dummy_audio = (0.2 * np.sin(2 * np.pi * 220.0 * t) + 0.02 * np.random.randn(len(t))).astype(np.float32)

    def test_01_emotion_result_serialization(self):
        """Test EmotionResult dataclass to_dict, to_json, and from_dict."""
        res = EmotionResult(
            primary_label="SADNESS",
            primary_confidence=0.85,
            sub_label="LONELINESS",
            sub_confidence=0.72,
            f0_mean=185.4,
            transcript="Hôm nay tôi thấy cô độc",
        )
        d = res.to_dict()
        self.assertEqual(d["primary_label"], "SADNESS")
        self.assertEqual(d["sub_label"], "LONELINESS")

        json_str = res.to_json()
        self.assertIn("LONELINESS", json_str)

        restored = EmotionResult.from_dict(d)
        self.assertEqual(restored.primary_label, "SADNESS")
        self.assertEqual(restored.sub_confidence, 0.72)
        self.assertTrue(len(restored.summary_card()) > 20)

    def test_02_inference_engine_dummy_predict(self):
        """Test Stage3InferenceEngine predicts valid EmotionResult with dummy model."""
        engine = Stage3InferenceEngine(
            checkpoint_path=None,
            device="cpu",
            dummy=True,
        )
        res = engine.predict(
            audio_input=self.dummy_audio,
            text="Hôm nay mình rất vui vì hoàn thành xong dự án",
            session_id="test_sess_001",
        )
        self.assertIsInstance(res, EmotionResult)
        self.assertIn(res.primary_label, ["JOY", "SADNESS", "ANXIETY", "ANGER", "NEUTRAL"])
        self.assertIn(res.sub_label, engine.SUB_NAMES)
        self.assertTrue(0.0 <= res.primary_confidence <= 1.0)
        self.assertTrue(0.0 <= res.sub_confidence <= 1.0)
        self.assertTrue(0.0 <= res.gate_audio <= 1.0)
        self.assertTrue(0.0 <= res.gate_text <= 1.0)
        self.assertAlmostEqual(res.gate_audio + res.gate_text, 1.0, places=3)

    def test_03_hierarchy_consistency_zero_violation(self):
        """Verify that Joint MAP prediction guarantees strict taxonomy compliance."""
        engine = Stage3InferenceEngine(checkpoint_path=None, device="cpu", dummy=True)
        # Test across multiple random signals
        for i in range(5):
            noise = np.random.randn(self.sr * 2).astype(np.float32)
            res = engine.predict(audio_input=noise, text=f"Câu nhật ký thử nghiệm số {i}")
            p_idx = engine.PRIMARY_NAMES.index(res.primary_label)
            s_idx = engine.SUB_NAMES.index(res.sub_label)
            allowed_sub = TAXONOMY_MAP[p_idx]
            self.assertIn(s_idx, allowed_sub, f"Violation: sub {s_idx} ({res.sub_label}) not permitted under primary {p_idx} ({res.primary_label})")

    def test_04_longitudinal_tracker_sqlite(self):
        """Test inserting and querying mood tracking records in SQLite."""
        test_db = Path("data/test_journal_records/test_tracker.db")
        if test_db.exists():
            test_db.unlink()

        tracker = LongitudinalTracker(db_path=str(test_db))
        res1 = EmotionResult(
            primary_label="SADNESS",
            primary_confidence=0.8,
            sub_label="LONELINESS",
            sub_confidence=0.75,
            transcript="Thấy buồn và cô đơn",
        )
        tracker.log_session(res1)

        res2 = EmotionResult(
            primary_label="JOY",
            primary_confidence=0.9,
            sub_label="HOPE",
            sub_confidence=0.85,
            transcript="Hôm nay thấy có hy vọng hơn",
        )
        tracker.log_session(res2)

        history = tracker.get_history(days=7)
        self.assertEqual(len(history), 2)

        trends = tracker.compute_mood_trends(days=7)
        self.assertEqual(trends["total_sessions"], 2)
        self.assertIn("JOY", trends["primary_distribution"])
        self.assertIn("SADNESS", trends["primary_distribution"])

        # Clean up test database
        if test_db.exists():
            test_db.unlink()
            if test_db.parent.exists() and not any(test_db.parent.iterdir()):
                test_db.parent.rmdir()

    def test_05_crisis_detector(self):
        """Test crisis detector on emergency keywords and safe entries."""
        detector = CrisisDetector()

        # Emergency case
        suicidal_res = EmotionResult(
            primary_label="SADNESS",
            primary_confidence=0.95,
            sub_label="SADNESS",
            sub_confidence=0.9,
            transcript="Tôi cảm thấy kiệt quệ và muốn tự tử cho xong chuyện",
        )
        alert = detector.evaluate(suicidal_res)
        self.assertTrue(alert.is_crisis)
        self.assertEqual(alert.severity_level, "emergency")
        self.assertIn("111", alert.support_hotlines)

        # Safe case
        safe_res = EmotionResult(
            primary_label="JOY",
            primary_confidence=0.88,
            sub_label="JOY",
            sub_confidence=0.82,
            transcript="Hôm nay đi cà phê với bạn bè rất vui vẻ",
        )
        safe_alert = detector.evaluate(safe_res)
        self.assertFalse(safe_alert.is_crisis)
        self.assertEqual(safe_alert.severity_level, "none")

    def test_06_empathetic_coach_offline_cbt(self):
        """Test offline CBT empathetic coach response formatting across emotional states."""
        coach = EmpatheticCoach(use_llm=False)
        for primary in ["SADNESS", "ANXIETY", "ANGER", "JOY", "NEUTRAL"]:
            res = EmotionResult(
                primary_label=primary,
                primary_confidence=0.8,
                sub_label="DEFAULT",
                sub_confidence=0.7,
                transcript="Nhật ký thử nghiệm cho cảm xúc " + primary,
            )
            resp = coach.generate_response(res)
            self.assertTrue(len(resp.response_text) > 10)
            self.assertTrue(len(resp.reflection_question) > 5)
            self.assertIsNotNone(resp.cbt_exercise)
            self.assertIn("offline_cbt", resp.engine_used)

    def test_07_empty_text_and_silent_audio_robustness(self):
        """Ensure pipeline does not crash on empty text and silent audio."""
        engine = Stage3InferenceEngine(checkpoint_path=None, device="cpu", dummy=True)
        silent_audio = np.zeros(self.sr * 2, dtype=np.float32)

        res = engine.predict(audio_input=silent_audio, text="")
        self.assertIsInstance(res, EmotionResult)
        self.assertEqual(res.transcript, "")
        # Empty text should route gate towards audio
        self.assertTrue(res.gate_audio >= 0.5)

    def test_08_acoustic_distress_slope(self):
        """Verify that severe negative f0_slope triggers acoustic distress alert."""
        detector = CrisisDetector(f0_slope_threshold=-1.5)
        # Entry with weeping/choking pitch slope and sadness
        choking_res = EmotionResult(
            primary_label="SADNESS",
            primary_confidence=0.88,
            sub_label="SADNESS",
            sub_confidence=0.82,
            f0_mean=140.0,
            f0_slope=-2.4,  # Steep downward slope
            transcript="Hôm nay tôi mệt quá",
        )
        alert = detector.evaluate(choking_res)
        self.assertIn("sụt dốc mạnh", alert.triggers[0])
        self.assertIn(alert.severity_level, ["mild", "high"])

    def test_09_longitudinal_daily_streak_and_biomarkers(self):
        """Verify calendar-day grouping for streaks and storage of extended biomarkers."""
        test_db = Path("data/test_journal_records/test_tracker_daily.db")
        if test_db.exists():
            test_db.unlink()

        tracker = LongitudinalTracker(db_path=str(test_db))
        # Log 3 negative sessions on the SAME day (e.g. today)
        for i in range(3):
            res = EmotionResult(
                primary_label="SADNESS",
                primary_confidence=0.85,
                sub_label="LONELINESS",
                sub_confidence=0.75,
                f0_mean=160.0,
                f0_std=25.0,
                f0_slope=-1.8,
                jitter_local=0.03,
                shimmer_local=0.08,
                speech_rate_wpm=110.5,
                timestamp="2026-10-16T10:00:00",
                session_id=f"daily_test_sess_{i}",
            )
            tracker.log_session(res)

        trends = tracker.compute_mood_trends(days=7)
        # Must be 1 day streak, NOT 3 days
        self.assertEqual(trends["consecutive_negative_streak"], 1)

        # Check that extended biomarkers were properly saved
        history = tracker.get_history(days=7)
        self.assertEqual(len(history), 3)
        row = history[0]
        self.assertAlmostEqual(row["f0_slope"], -1.8, places=2)
        self.assertAlmostEqual(row["speech_rate_wpm"], 110.5, places=1)

        # Clean up
        if test_db.exists():
            test_db.unlink()
            if test_db.parent.exists() and not any(test_db.parent.iterdir()):
                test_db.parent.rmdir()


if __name__ == "__main__":
    unittest.main()
