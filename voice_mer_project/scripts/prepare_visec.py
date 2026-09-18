#!/usr/bin/env python3
"""
prepare_visec.py

Automated dataset preparation script for the ViSEC (Vietnamese Speech Emotion Corpus)
dataset hosted on Hugging Face (hustep-lab/ViSEC).

Capabilities:
    1. Downloads hustep-lab/ViSEC via Hugging Face `datasets` library.
    2. Dynamically detects audio, label, and transcript columns.
    3. Resamples audio to 16,000 Hz mono PCM WAV format.
    4. Maps ViSEC 4 emotions to the project's standard Primary/Sub ontology:
         - happy   -> JOY (Sub: JOY)
         - sad     -> SADNESS (Sub: SADNESS)
         - angry   -> ANGER (Sub: ANGER)
         - neutral -> NEUTRAL (Sub: CALM)
    5. Partitions samples into Train (80%), Val (10%), Test (10%) splits.
    6. Generates standard `metadata.csv` for each split compatible with `VoiceJournalDataset`.
    7. Supports `--dummy` mode to generate synthetic test data offline without HF connection.

Usage:
    python scripts/prepare_visec.py --output-dir data/visec
    python scripts/prepare_visec.py --output-dir data/visec --sample-limit 100
    python scripts/prepare_visec.py --output-dir data/visec --dummy
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("prepare_visec")

TARGET_SAMPLE_RATE = 16000

# Canonical label mapping from ViSEC -> Primary & Sub
LABEL_MAPPING = {
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
    "tuc_gian": ("ANGER", "ANGER"),
    "2": ("ANGER", "ANGER"),
    2: ("ANGER", "ANGER"),
    "neutral": ("NEUTRAL", "CALM"),
    "binh_thuong": ("NEUTRAL", "CALM"),
    "3": ("NEUTRAL", "CALM"),
    3: ("NEUTRAL", "CALM"),
    "fear": ("ANXIETY", "FEAR"),
    "fearful": ("ANXIETY", "FEAR"),
    "anxiety": ("ANXIETY", "ANXIETY"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare ViSEC dataset for Voice MER")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/visec",
        help="Directory where preprocessed splits and metadata.csv will be saved",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="hustep-lab/ViSEC",
        help="Hugging Face dataset identifier (default: hustep-lab/ViSEC)",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Limit total number of samples to process (for debugging/testing)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of data assigned to train split (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of data assigned to validation split (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/val/test splitting",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Generate synthetic samples offline instead of downloading from Hugging Face",
    )
    return parser.parse_args()


def resample_audio(audio_arr: np.ndarray, orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Resample audio array to target sample rate."""
    if orig_sr == target_sr:
        return audio_arr.astype(np.float32)

    try:
        from scipy import signal
        num_target_samples = int(len(audio_arr) * target_sr / orig_sr)
        return signal.resample(audio_arr, num_target_samples).astype(np.float32)
    except ImportError:
        # Fallback linear interpolation
        orig_indices = np.linspace(0, len(audio_arr) - 1, len(audio_arr))
        target_len = int(len(audio_arr) * target_sr / orig_sr)
        target_indices = np.linspace(0, len(audio_arr) - 1, target_len)
        return np.interp(target_indices, orig_indices, audio_arr).astype(np.float32)


def save_wav(file_path: Path, audio_arr: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> None:
    """Save floating-point numpy array as 16kHz mono PCM 16-bit WAV."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf
        sf.write(str(file_path), audio_arr, sample_rate, subtype="PCM_16")
    except ImportError:
        import wave
        # Normalize and convert float32 in [-1, 1] to int16
        clipped = np.clip(audio_arr, -1.0, 1.0)
        int16_data = (clipped * 32767).astype(np.int16)
        with wave.open(str(file_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(int16_data.tobytes())


def normalize_label(raw_label: Any, class_names: Optional[List[str]] = None) -> Tuple[str, str]:
    """Map raw label value/string to (primary_label, sub_label)."""
    if class_names is not None and isinstance(raw_label, int) and 0 <= raw_label < len(class_names):
        val_str = str(class_names[raw_label]).lower().strip()
    else:
        val_str = str(raw_label).lower().strip()

    if val_str in LABEL_MAPPING:
        return LABEL_MAPPING[val_str]

    # Substring heuristic
    for key, mapped in LABEL_MAPPING.items():
        if str(key) in val_str:
            return mapped

    logger.warning("Unrecognized label '%s'; defaulting to ('NEUTRAL', 'CALM')", raw_label)
    return ("NEUTRAL", "CALM")


def generate_dummy_data(output_dir: Path, total_samples: int = 60) -> None:
    """Generate offline synthetic dummy ViSEC dataset for testing."""
    logger.info("Generating %d synthetic dummy ViSEC samples in %s...", total_samples, output_dir)
    random.seed(42)
    np.random.seed(42)

    splits = [
        ("train", int(total_samples * 0.8)),
        ("val", int(total_samples * 0.1)),
        ("test", total_samples - int(total_samples * 0.8) - int(total_samples * 0.1)),
    ]

    emotions = [
        ("JOY", "JOY", "Tôi cảm thấy vô cùng hào hứng và phấn khởi hôm nay."),
        ("SADNESS", "SADNESS", "Thật sự rất buồn khi mọi chuyện diễn ra không như ý."),
        ("ANGER", "ANGER", "Tôi không thể chịu nổi sự bất công và vô lý này."),
        ("NEUTRAL", "CALM", "Mọi việc diễn ra bình thường, một ngày không có gì đặc biệt."),
    ]

    for split_name, count in splits:
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        csv_rows = []

        for i in range(count):
            sample_id = f"visec_{split_name}_{i:04d}"
            wav_filename = f"{sample_id}.wav"
            wav_path = split_dir / wav_filename

            pri_label, sub_label, transcript = emotions[i % len(emotions)]

            # Generate 3-5s of synthetic bandpassed noise
            duration_s = random.uniform(3.0, 5.0)
            num_samples = int(duration_s * TARGET_SAMPLE_RATE)
            noise = np.random.randn(num_samples) * 0.05
            save_wav(wav_path, noise, TARGET_SAMPLE_RATE)

            csv_rows.append({
                "sample_id": sample_id,
                "audio_path": wav_filename,
                "primary_label": pri_label,
                "sub_label": sub_label,
                "transcript": transcript,
            })

        # Write metadata.csv
        csv_path = split_dir / "metadata.csv"
        with open(csv_path, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "audio_path", "primary_label", "sub_label", "transcript"])
            writer.writeheader()
            writer.writerows(csv_rows)

        logger.info("Split [%s]: saved %d samples to %s", split_name.upper(), count, split_dir)


def extract_audio_from_record(rec: Dict[str, Any], sample_id: str) -> Tuple[Optional[np.ndarray], int]:
    """Robustly extract audio array and sample rate from a Hugging Face dataset record."""
    audio_obj = None

    # Priority 1: Check standard audio column names (ViSEC uses 'path' as Audio feature)
    for key in ["path", "audio", "wav", "speech", "sound", "file"]:
        if key in rec and rec[key] is not None:
            val = rec[key]
            # Unwrap list/tuple if returned as a list of items
            if isinstance(val, (list, tuple)) and len(val) > 0:
                val = val[0]

            if isinstance(val, dict) and ("array" in val or "bytes" in val or "path" in val or "src" in val):
                audio_obj = val
                break
            elif isinstance(val, (bytes, bytearray)):
                audio_obj = val
                break
            elif isinstance(val, np.ndarray):
                audio_obj = val
                break
            elif isinstance(val, (str, Path)) and (os.path.exists(str(val)) or str(val).startswith("http")):
                audio_obj = val
                break

    # Priority 2: Scan any field in rec that looks like a HuggingFace Audio dict
    if audio_obj is None:
        for val in rec.values():
            if isinstance(val, (list, tuple)) and len(val) > 0:
                val = val[0]
            if isinstance(val, dict) and ("array" in val or "bytes" in val or "src" in val):
                audio_obj = val
                break

    # Priority 3: Fallback to any string path or URL
    if audio_obj is None:
        for key in ["path", "file", "url"]:
            if key in rec:
                val = rec[key]
                if isinstance(val, (list, tuple)) and len(val) > 0:
                    val = val[0]
                if isinstance(val, (str, Path)):
                    audio_obj = val
                    break

    if audio_obj is None:
        return None, TARGET_SAMPLE_RATE

    # Unwrap list if audio_obj itself is still a list
    if isinstance(audio_obj, (list, tuple)) and len(audio_obj) > 0:
        audio_obj = audio_obj[0]

    # Parse audio_obj into numpy float32 array
    try:
        if isinstance(audio_obj, dict):
            if "array" in audio_obj and audio_obj["array"] is not None:
                arr = np.array(audio_obj["array"], dtype=np.float32)
                sr = int(audio_obj.get("sampling_rate", TARGET_SAMPLE_RATE) or TARGET_SAMPLE_RATE)
                return arr, sr
            elif "bytes" in audio_obj and audio_obj["bytes"] is not None:
                import io
                import soundfile as sf
                arr, sr = sf.read(io.BytesIO(audio_obj["bytes"]))
                return arr.astype(np.float32), sr
            elif "path" in audio_obj and audio_obj["path"] and os.path.exists(str(audio_obj["path"])):
                import soundfile as sf
                arr, sr = sf.read(str(audio_obj["path"]))
                return arr.astype(np.float32), sr
            elif "src" in audio_obj and isinstance(audio_obj["src"], str):
                import io
                import urllib.request
                import soundfile as sf
                with urllib.request.urlopen(audio_obj["src"]) as resp:
                    arr, sr = sf.read(io.BytesIO(resp.read()))
                return arr.astype(np.float32), sr

        elif isinstance(audio_obj, (bytes, bytearray)):
            import io
            import soundfile as sf
            arr, sr = sf.read(io.BytesIO(audio_obj))
            return arr.astype(np.float32), sr

        elif isinstance(audio_obj, np.ndarray):
            return audio_obj.astype(np.float32), TARGET_SAMPLE_RATE

        elif isinstance(audio_obj, (str, Path)):
            str_path = str(audio_obj)
            if os.path.exists(str_path):
                import soundfile as sf
                arr, sr = sf.read(str_path)
                return arr.astype(np.float32), sr
            elif str_path.startswith("http://") or str_path.startswith("https://"):
                import io
                import urllib.request
                import soundfile as sf
                with urllib.request.urlopen(str_path) as resp:
                    arr, sr = sf.read(io.BytesIO(resp.read()))
                return arr.astype(np.float32), sr

    except Exception as exc:
        logger.warning("Sample %s: exception while decoding audio (%s)", sample_id, exc)
        return None, TARGET_SAMPLE_RATE

    return None, TARGET_SAMPLE_RATE


def download_and_process_hf(
    dataset_name: str,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    sample_limit: Optional[int] = None,
    seed: int = 42,
) -> None:
    """Download ViSEC from Hugging Face, process waveforms, and output splits."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Hugging Face 'datasets' library is required. Run: pip install datasets soundfile")
        sys.exit(1)

    logger.info("Downloading dataset '%s' from Hugging Face Hub...", dataset_name)
    raw_dataset = load_dataset(dataset_name)

    # Consolidate into flat list of records
    all_records: List[Dict[str, Any]] = []
    class_names = None

    if isinstance(raw_dataset, dict):
        for split_key, ds in raw_dataset.items():
            if hasattr(ds, "features") and "label" in ds.features:
                feat = ds.features["label"]
                if hasattr(feat, "names"):
                    class_names = feat.names
            for row in ds:
                all_records.append(row)
    else:
        if hasattr(raw_dataset, "features") and "label" in raw_dataset.features:
            feat = raw_dataset.features["label"]
            if hasattr(feat, "names"):
                class_names = feat.names
        for row in raw_dataset:
            all_records.append(row)

    total_available = len(all_records)
    logger.info("Retrieved %d records from %s", total_available, dataset_name)

    if sample_limit and sample_limit < total_available:
        random.seed(seed)
        all_records = random.sample(all_records, sample_limit)
        logger.info("Limited dataset to %d samples as requested.", len(all_records))

    # Shuffle records deterministically
    random.seed(seed)
    random.shuffle(all_records)

    n_total = len(all_records)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    splits = {
        "train": all_records[:n_train],
        "val": all_records[n_train : n_train + n_val],
        "test": all_records[n_train + n_val :],
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, records in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        csv_rows = []

        logger.info("Processing split '%s' (%d samples)...", split_name, len(records))
        if records:
            first_rec = records[0]
            logger.info("Split '%s' sample 0 fields: %s", split_name, list(first_rec.keys()))
            for cand in ["path", "audio", "wav"]:
                if cand in first_rec:
                    v = first_rec[cand]
                    logger.info("Split '%s' sample 0 key '%s': type=%s preview=%s", split_name, cand, type(v), str(v)[:120])

        for idx, rec in enumerate(records):
            sample_id = f"visec_{split_name}_{idx:05d}"
            wav_filename = f"{sample_id}.wav"
            wav_dest_path = split_dir / wav_filename

            # 1. Extract audio robustly
            audio_arr, sr = extract_audio_from_record(rec, sample_id)

            if audio_arr is None or len(audio_arr) == 0:
                # If audio extraction fails, generate minimal silent clip to prevent crash
                logger.warning(
                    "Sample %s: could not read audio (keys=%s, path_type=%s, path_preview=%s)",
                    sample_id,
                    list(rec.keys()),
                    type(rec.get("path")),
                    str(rec.get("path"))[:60],
                )
                audio_arr = np.zeros(TARGET_SAMPLE_RATE * 3, dtype=np.float32)
                sr = TARGET_SAMPLE_RATE

            # Convert stereo to mono
            if audio_arr.ndim > 1:
                if audio_arr.shape[0] < audio_arr.shape[1]:
                    audio_arr = audio_arr.mean(axis=0)
                else:
                    audio_arr = audio_arr.mean(axis=-1)

            # Resample to 16kHz
            audio_resampled = resample_audio(audio_arr, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
            save_wav(wav_dest_path, audio_resampled, TARGET_SAMPLE_RATE)

            # 2. Extract label
            raw_lbl = rec.get("label", rec.get("emotion", rec.get("sentiment", "neutral")))
            pri_label, sub_label = normalize_label(raw_lbl, class_names=class_names)

            # 3. Extract transcript
            transcript = rec.get("transcription", rec.get("transcript", rec.get("text", rec.get("sentence", ""))))

            csv_rows.append({
                "sample_id": sample_id,
                "audio_path": wav_filename,
                "primary_label": pri_label,
                "sub_label": sub_label,
                "transcript": str(transcript or "").strip(),
            })

        # Save metadata.csv
        csv_path = split_dir / "metadata.csv"
        with open(csv_path, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "audio_path", "primary_label", "sub_label", "transcript"])
            writer.writeheader()
            writer.writerows(csv_rows)

        # Print distribution
        labels_dist = {}
        for row in csv_rows:
            labels_dist[row["primary_label"]] = labels_dist.get(row["primary_label"], 0) + 1

        logger.info("Split [%s] complete: %d samples | Distribution: %s", split_name.upper(), len(csv_rows), labels_dist)

    logger.info("ViSEC preprocessing successfully completed! Saved to: %s", output_dir)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)

    if args.dummy:
        generate_dummy_data(out_dir, total_samples=args.sample_limit or 60)
        return

    try:
        download_and_process_hf(
            dataset_name=args.dataset_name,
            output_dir=out_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            sample_limit=args.sample_limit,
            seed=args.seed,
        )
    except Exception as e:
        logger.error("Failed downloading '%s' from Hugging Face: %s", args.dataset_name, e)
        logger.info("You can run with '--dummy' for local offline testing if needed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
