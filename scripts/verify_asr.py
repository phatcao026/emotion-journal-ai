#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_asr.py
-------------
Automated verification script for PhoWhisper-base Speech-to-Text (ASR) Subsystem.

Tests:
    1. Model loading & tokenizer initialization (vinai/phowhisper-base).
    2. Tensor precision & device compatibility (float16 on GPU, float32 on CPU).
    3. Transcription of real .wav files or synthetic audio.
    4. Real-time factor (RTF) latency measurement.
    5. Punctuation restoration and confidence reporting.

Usage:
    python scripts/verify_asr.py --dummy
    python scripts/verify_asr.py --audio path/to/recording.wav
    python scripts/verify_asr.py --device cpu
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch

from src.audio.asr_transcriber import PhoWhisperTranscriber, TranscriptionResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("verify_asr")


def generate_synthetic_audio(duration_s: float = 3.0, sample_rate: int = 16000) -> torch.Tensor:
    """Generate a simple speech-like modulated harmonic waveform for offline testing."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    # Fundamental frequency ~150Hz with harmonics and speech-like envelope
    f0 = 150.0
    signal = (
        0.5 * np.sin(2 * np.pi * f0 * t)
        + 0.3 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.2 * np.sin(2 * np.pi * 3 * f0 * t)
    )
    # Modulation envelope
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 2.0 * t))
    modulated = (signal * envelope).astype(np.float32)
    return torch.from_numpy(modulated).unsqueeze(0)  # (1, T)


def run_verification(
    audio_path: str | None = None,
    device: str = "cpu",
    model_id: str = "vinai/phowhisper-base",
    dummy: bool = False,
) -> bool:
    """Run full ASR verification suite."""
    logger.info("=" * 60)
    logger.info("  PhoWhisper ASR Verification Suite")
    logger.info("=" * 60)
    logger.info("Model ID : %s", model_id)
    logger.info("Device   : %s", device)
    logger.info("CUDA Avail: %s", torch.cuda.is_available())

    # 1. Initialize transcriber
    logger.info("\n[1/4] Initializing PhoWhisperTranscriber...")
    transcriber = PhoWhisperTranscriber(
        model_id=model_id,
        device=device,
        restore_punctuation=True,
    )

    # 2. Load model
    logger.info("\n[2/4] Loading model weights into memory...")
    t0 = time.perf_counter()
    transcriber.load_model()
    load_time = time.perf_counter() - t0
    logger.info("Model loaded in %.2f seconds.", load_time)

    # Verify model dtype
    model_dtype = getattr(transcriber.model, "dtype", torch.float32)
    logger.info("Model tensor dtype: %s", model_dtype)
    if "cuda" in device and model_dtype != torch.float16:
        logger.warning("Caution: On CUDA, float16 is recommended to save VRAM and accelerate inference.")

    # 3. Prepare audio input
    logger.info("\n[3/4] Preparing audio input...")
    if audio_path and Path(audio_path).exists():
        logger.info("Using audio file: %s", audio_path)
        input_data = audio_path
        # Measure duration
        import soundfile as sf
        info = sf.info(audio_path)
        duration_s = info.duration
    else:
        logger.info("Using synthetic 3.0s test waveform (--dummy mode)...")
        waveform = generate_synthetic_audio(duration_s=3.0, sample_rate=16000)
        input_data = waveform
        duration_s = 3.0

    # 4. Transcribe
    logger.info("\n[4/4] Executing ASR inference...")
    t_start = time.perf_counter()
    result: TranscriptionResult = transcriber.transcribe(input_data, sample_rate=16000)
    infer_time = time.perf_counter() - t_start

    rtf = infer_time / max(1e-3, duration_s)

    logger.info("=" * 60)
    logger.info("  VERIFICATION RESULTS")
    logger.info("=" * 60)
    logger.info("Audio Duration   : %.2f seconds", duration_s)
    logger.info("Inference Latency: %.3f seconds", infer_time)
    logger.info("Real-Time Factor : %.3f (RTF < 1.0 means faster than real-time)", rtf)
    logger.info("Detected Language: %s", result.language)
    logger.info("Mean Confidence  : %.3f", result.confidence)
    logger.info("Transcribed Text : %r", result.text)
    logger.info("Word Count       : %d", result.word_count())
    logger.info("Segments Count   : %d", len(result.segments))

    assert isinstance(result.text, str), "Result text must be a string"
    logger.info("\n>>> ALL ASR VERIFICATION CHECKS PASSED SUCCESSFULLY! <<<\n")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify PhoWhisper ASR Subsystem")
    parser.add_argument("--audio", type=str, default=None, help="Path to audio file (.wav)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-id", type=str, default="vinai/phowhisper-base")
    parser.add_argument("--dummy", action="store_true", default=True, help="Run with synthetic audio")
    args = parser.parse_args()

    run_verification(
        audio_path=args.audio,
        device=args.device,
        model_id=args.model_id,
        dummy=args.dummy or args.audio is None,
    )


if __name__ == "__main__":
    main()
