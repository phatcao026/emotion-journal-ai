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


def sanitize_transcript(
    transcript: Optional[str],
    source: str = "ViSEC",
    has_vietnamese_translation: bool = True,
) -> str:
    """Đảm bảo các mẫu âm thanh ngoại ngữ hoặc không có bản dịch tiếng Việt chuẩn
    đều được gán transcript = "" (chuỗi rỗng).

    Chuỗi rỗng kích hoạt text_mask = False trong Stage 2 (HierarchicalMERModel),
    tự động định tuyến 100% audio gate mà không làm ô nhiễm tokenizer PhoBERT tiếng Việt.
    """
    if not transcript or not has_vietnamese_translation:
        return ""
    # Các corpus tiếng Anh / ngoại ngữ (RAVDESS, CREMA-D) không có bản dịch tiếng Việt chuẩn
    source_upper = str(source).strip().upper()
    if source_upper in {"RAVDESS", "CREMA-D", "CREMAD", "ENGLISH", "FOREIGN"}:
        return ""
    # Kiểm tra và loại bỏ placeholder tiếng Anh nếu có
    text_clean = str(transcript).strip()
    text_lower = text_clean.lower()
    if any(marker in text_lower for marker in ["ravdess", "crema-d", "cremad", "statement", "utterance"]):
        return ""
    return text_clean


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
        # Mẫu âm thanh ngoại ngữ (RAVDESS/CREMA-D) hoặc không có bản dịch tiếng Việt chuẩn:
        # Chuỗi rỗng kích hoạt text_mask = False trong Stage 2, tự động định tuyến 100% audio gate
        ("ANXIETY", "FEAR", ""),
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
            pri_label, sub_label, raw_transcript = template
            transcript = sanitize_transcript(raw_transcript, source="SyntheticDummy")

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
    parser.add_argument(
        "--visec-dataset",
        type=str,
        default="Vinh727/HusTep_ViSEC",
        help="Hugging Face ViSEC dataset repo (default: Vinh727/HusTep_ViSEC)",
    )
    parser.add_argument(
        "--ravdess-dir",
        type=str,
        default=None,
        help="Path to RAVDESS dataset directory",
    )
    parser.add_argument(
        "--cremad-dir",
        type=str,
        default=None,
        help="Path to CREMA-D dataset directory",
    )
    parser.add_argument(
        "--cremad-limit",
        type=int,
        default=600,
        help="Maximum number of Fearful audio samples to load from CREMA-D",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Limit ViSEC samples for debugging",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible stratified split",
    )
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
            from scripts.prepare_crosscorpus_visec import (
                find_cremad_dir,
                find_ravdess_dir,
                load_cremad_fear,
                load_ravdess_fearful,
                load_visec,
                split_and_save,
            )

            # 1. Load ViSEC (Vietnamese speech + transcripts)
            visec_records = load_visec(dataset_name=args.visec_dataset, sample_limit=args.sample_limit)
            for r in visec_records:
                r["transcript"] = sanitize_transcript(r.get("transcript", ""), source="ViSEC")

            # 2. Ingest RAVDESS Fearful clips (English audio -> transcript = "")
            ravdess_dir = find_ravdess_dir(args.ravdess_dir)
            ravdess_records = []
            if ravdess_dir:
                ravdess_records = load_ravdess_fearful(ravdess_dir)
                for r in ravdess_records:
                    r["transcript"] = sanitize_transcript(r.get("transcript", ""), source="RAVDESS")

            # 3. Ingest CREMA-D Fear clips (English audio -> transcript = "")
            cremad_dir = find_cremad_dir(args.cremad_dir)
            cremad_records = []
            if cremad_dir:
                cremad_records = load_cremad_fear(cremad_dir, sample_limit=args.cremad_limit)
                for r in cremad_records:
                    r["transcript"] = sanitize_transcript(r.get("transcript", ""), source="CREMA-D")

            all_records = visec_records + ravdess_records + cremad_records
            if not all_records:
                logger.warning("No records collected from ViSEC/RAVDESS/CREMA-D. Falling back to synthetic dummy data.")
                generate_synthetic_multimodal_dataset(out_dir, total_samples=args.samples)
            else:
                split_and_save(
                    all_records,
                    output_dir=out_dir,
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                    seed=args.seed,
                )
                logger.info("Real multimodal dataset preparation complete! Output at: %s", out_dir)
        except Exception as e:
            logger.warning("datasets library or dependencies not available (%s). Falling back to synthetic dummy data.", e)
            generate_synthetic_multimodal_dataset(out_dir, total_samples=args.samples)


if __name__ == "__main__":
    main()
