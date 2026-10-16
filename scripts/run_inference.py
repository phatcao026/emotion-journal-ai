#!/usr/bin/env python3
"""
scripts/run_inference.py
------------------------
CLI Tool for Stage 3 SOTA Emotion Journal Inference & Empathetic Companion.

Usage:
    # Run with real audio file:
    python scripts/run_inference.py --audio path/to/journal.wav

    # Run with audio and known transcript:
    python scripts/run_inference.py --audio path/to/journal.wav --text "Hôm nay mình thấy rất mệt và buồn"

    # Fast offline dry-run test without downloads:
    python scripts/run_inference.py --dummy
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.stage3 import (
    CrisisDetector,
    EmpatheticCoach,
    LongitudinalTracker,
    Stage3InferenceEngine,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_inference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 3 Multimodal Emotion Journal Inference")
    parser.add_argument("--audio", type=str, default=None, help="Path to input audio file (.wav)")
    parser.add_argument("--text", type=str, default=None, help="Optional known text transcript")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/stage2/best_model_stage2.pt")
    parser.add_argument("--config", type=str, default="configs/stage2/multimodal_stage2.yaml")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dummy", action="store_true", help="Run with synthetic test audio")
    parser.add_argument("--no-db", action="store_true", help="Do not save session to SQLite tracker")
    parser.add_argument("--use-llm", action="store_true", help="Use Gemini/OpenAI API if available")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Handle dummy mode vs real audio
    if args.dummy or args.audio is None:
        logger.info("[*] Running in synthetic dummy mode with simulated audio signal...")
        duration_s = 5.0
        sr = 16000
        t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False)
        audio_input = (0.2 * np.sin(2 * np.pi * 220.0 * t) + 0.05 * np.random.randn(len(t))).astype(np.float32)
        sample_text = args.text or "Hôm nay tôi cảm thấy khá mệt mỏi và chỉ muốn ở một mình trong phòng."
    else:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            logger.error("Audio file not found: %s", audio_path)
            sys.exit(1)
        audio_input = audio_path
        sample_text = args.text

    # 2. Initialize Pipeline Modules
    engine = Stage3InferenceEngine(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        dummy=args.dummy,
    )
    tracker = LongitudinalTracker()
    crisis_detector = CrisisDetector()
    coach = EmpatheticCoach(use_llm=args.use_llm)

    # 3. Predict Emotion
    logger.info("Executing Multimodal Emotion Recognition...")
    result = engine.predict(audio_input=audio_input, text=sample_text)

    # 4. Check Mental Health Crisis
    streak = tracker.compute_mood_trends(days=7).get("consecutive_negative_streak", 0)
    crisis_alert = crisis_detector.evaluate(result, consecutive_negative_streak=streak)

    # 5. Generate Empathetic Coach Feedback
    response = coach.generate_response(result, crisis_alert=crisis_alert)

    # 6. Save to Tracker
    if not args.no_db:
        tracker.log_session(result)
        logger.info("Session persisted to SQLite tracker.")

    # 7. Print Beautiful Output Card
    print("\n" + "=" * 68)
    print("      STAGE 3 SOTA MULTIMODAL EMOTION JOURNAL ASSISTANT")
    print("=" * 68)
    print(result.summary_card())
    print("\n" + "-" * 68)
    if crisis_alert.is_crisis:
        print(crisis_alert.summary())
        print("-" * 68)
    print(response.format_display())
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
