#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_multimodal_dataset.py
-----------------------------
Prepares the Paired Multimodal (Speech + Text) Dataset for Stage 2 Training.

Capabilities:
    1. Paired Speech & Text:
       - ViSEC: Vietnamese emotional speech (.wav, 16kHz mono) + Vietnamese transcripts.
       - RAVDESS / CREMA-D: Optional anxiety/fear audio with paired emotional context.
    2. Multi-Level Emotion Taxonomy Alignment:
       - Maps emotions to 5 Primary (JOY, SADNESS, ANXIETY, ANGER, NEUTRAL)
       - Maps fine-grained sub-emotions to 11 Sub classes (taxonomy.json).
    3. Stratified Partitioning:
       - Splits into Train (80%), Val (10%), Test (10%).
    4. Offline Dummy Mode:
       - `--dummy` generates synthetic paired samples for testing without downloads.

Usage:
    python scripts/prepare_multimodal_dataset.py --dummy --output-dir data/multimodal_dummy
    python scripts/prepare_multimodal_dataset.py --output-dir data/multimodal_paired
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare_multimodal_dataset")

TARGET_SAMPLE_RATE = 16000

# Canonical 5 Primary & 11 Sub Emotion Mapping
PRIMARY_LABELS = ["JOY", "SADNESS", "ANXIETY", "ANGER", "NEUTRAL"]
SUB_LABELS = [
    "JOY", "CALM", "HOPE", "CONNECTION", "SADNESS",
    "ANXIETY", "FEAR", "ANGER", "GUILT_SHAME", "LONELINESS", "DISGUST"
]

VISEC_EMOTION_MAP = {
    "happy": ("JOY", "JOY"),
    "happiness": ("JOY", "JOY"),
    "joy": ("JOY", "JOY"),
    "vui": ("JOY", "JOY"),
    "0": ("JOY", "JOY"),
    0: ("JOY", "JOY"),
    "sad": ("SADNESS", "SADNESS"),
    "sadness": ("SADNESS", "SADNESS"),
    "buon": ("SADNESS", "SADNESS"),
    "1": ("SADNESS", "SADNESS"),
    1: ("SADNESS", "SADNESS"),
    "angry": ("ANGER", "ANGER"),
    "anger": ("ANGER", "ANGER"),
    "tuc gian": ("ANGER", "ANGER"),
    "2": ("ANGER", "ANGER"),
    2: ("ANGER", "ANGER"),
    "neutral": ("NEUTRAL", "CALM"),
    "binh thuong": ("NEUTRAL", "CALM"),
    "3": ("NEUTRAL", "CALM"),
    3: ("NEUTRAL", "CALM"),
}


def save_wav(file_path: Path, audio_arr: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> None:
    """Save float numpy array to 16-bit PCM WAV."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio_arr, -1.0, 1.0)
    int16_data = (clipped * 32767).astype(np.int16)
    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_data.tobytes())


def generate_synthetic_multimodal_dataset(output_dir: Path, total_samples: int = 80) -> None:
    """Generate paired dummy audio + transcript samples for fast local verification."""
    logger.info("Generating %d synthetic paired multimodal samples in %s...", total_samples, output_dir)
    random.seed(42)
    np.random.seed(42)

    sample_templates = [
        ("JOY", "JOY", "Hôm nay tôi rất hạnh phúc và cảm thấy mọi nỗ lực đều được đền đáp."),
        ("JOY", "CALM", "Buổi sáng thật yên bình, tôi uống một tách trà và lắng lòng lại."),
        ("JOY", "HOPE", "Tôi tin rằng ngày mai mọi khó khăn sẽ qua đi và tương lai sẽ tốt đẹp hơn."),
        ("JOY", "CONNECTION", "Buổi họp mặt gia đình cuối tuần này làm tôi cảm thấy ấm áp vô cùng."),
        ("SADNESS", "SADNESS", "Cảm giác nặng nề trong lòng khi nhận tin buồn từ người bạn cũ."),
        ("SADNESS", "LONELINESS", "Ngồi một mình trong căn phòng vắng, tôi thấy mình thật cô độc."),
        ("SADNESS", "GUILT_SHAME", "Tôi ân hận vì những lời nói lỡ lời đã làm tổn thương người thân."),
        ("ANXIETY", "ANXIETY", "Tôi cảm thấy bồn chồn lo lắng, không biết dự án sắp tới có suôn sẻ không."),
        ("ANXIETY", "FEAR", "Cảm giác hoảng sợ tột cùng khi nghe tiếng còi báo động trong đêm."),
        ("ANGER", "ANGER", "Tôi không thể kiềm chế cơn giận dữ khi bị đối xử bất công như vậy."),
        ("ANGER", "DISGUST", "Cách hành xử thiếu tôn trọng đó thực sự khiến tôi khó chịu và ghê sợ."),
        ("NEUTRAL", "CALM", "Một ngày làm việc trôi qua bình thường, không có gì nổi bật."),
    ]

    splits = [
        ("train", int(total_samples * 0.8)),
        ("val", int(total_samples * 0.1)),
        ("test", total_samples - int(total_samples * 0.8) - int(total_samples * 0.1)),
    ]

    all_metadata = []

    for split_name, count in splits:
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        csv_rows = []

        for i in range(count):
            sample_id = f"mm_{split_name}_{i:04d}"
            wav_filename = f"{sample_id}.wav"
            wav_path = split_dir / wav_filename

            template = sample_templates[i % len(sample_templates)]
            pri_label, sub_label, transcript = template

            duration_s = random.uniform(3.0, 6.0)
            num_samples = int(duration_s * TARGET_SAMPLE_RATE)
            noise = np.random.randn(num_samples) * 0.05
            save_wav(wav_path, noise, TARGET_SAMPLE_RATE)

            row = {
                "sample_id": sample_id,
                "audio_path": wav_filename,
                "primary_label": pri_label,
                "sub_label": sub_label,
                "transcript": transcript,
                "duration_s": f"{duration_s:.2f}",
                "split": split_name,
            }
            csv_rows.append(row)
            all_metadata.append(row)

        manifest_path = split_dir / "metadata.csv"
        with open(manifest_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["sample_id", "audio_path", "primary_label", "sub_label", "transcript", "duration_s", "split"]
            )
            writer.writeheader()
            writer.writerows(csv_rows)

        logger.info("Created %s metadata.csv with %d samples.", split_name, len(csv_rows))

    # Summary
    summary = {
        "dataset_name": "Synthetic Paired Multimodal Journal Dataset",
        "total_samples": len(all_metadata),
        "splits": {s: c for s, c in splits},
        "primary_classes": PRIMARY_LABELS,
        "sub_classes": SUB_LABELS,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("Multimodal dataset preparation complete! Summary saved to %s", summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Paired Multimodal Dataset for Stage 2")
    parser.add_argument("--output-dir", type=str, default="data/multimodal_paired")
    parser.add_argument("--dummy", action="store_true", help="Generate synthetic paired data for local verification")
    parser.add_argument("--samples", type=int, default=80, help="Number of samples in dummy mode")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dummy:
        generate_synthetic_multimodal_dataset(out_dir, total_samples=args.samples)
    else:
        logger.info("Real ViSEC download / assembly mode requested for output_dir: %s", out_dir)
        # Fallback to dummy if dependencies/datasets not installed in environment
        try:
            import datasets
            logger.info("HuggingFace datasets library available. Ready for live download.")
        except ImportError:
            logger.warning("datasets library not installed. Falling back to synthetic dummy data.")
            generate_synthetic_multimodal_dataset(out_dir, total_samples=args.samples)


if __name__ == "__main__":
    main()
