#!/usr/bin/env python3
"""
prepare_crosscorpus_visec.py
----------------------------
Automated dataset preparation script for Balanced Cross-Corpus Speech Emotion Recognition.

Combines:
    1. ViSEC (Vietnamese Speech Emotion Corpus - hustep-lab/ViSEC):
       - Primary: JOY (happy), SADNESS (sad), ANGER (angry), NEUTRAL (neutral)
    2. RAVDESS (Emotional Speech Audio - uwrfkaggler/ravdess-emotional-speech-audio):
       - Primary: ANXIETY (fearful clips, emotion code 06)
       - Sub: FEAR
    3. CREMA-D (Optional, if mounted at /kaggle/input/cremad or specified):
       - Primary: ANXIETY (FEA clips)
       - Sub: FEAR

Pipeline:
    - Automatically discovers RAVDESS / CREMA-D from Kaggle inputs or local paths.
    - Decodes audio and resamples to 16,000 Hz mono PCM 16-bit WAV.
    - Applies Stratified Splitting (80% Train, 10% Val, 10% Test) across all 5 classes.
    - Outputs standard metadata.csv compatible with VoiceJournalDataset.
    - Supports --dummy mode for offline unit testing without downloads.

Usage:
    # On Kaggle with RAVDESS input mounted:
    python scripts/prepare_crosscorpus_visec.py \\
        --ravdess-dir /kaggle/input/ravdess-emotional-speech-audio \\
        --output-dir /kaggle/working/data/visec_crosscorpus

    # With both RAVDESS and CREMA-D:
    python scripts/prepare_crosscorpus_visec.py \\
        --ravdess-dir /kaggle/input/ravdess-emotional-speech-audio \\
        --cremad-dir /kaggle/input/cremad \\
        --output-dir /kaggle/working/data/visec_crosscorpus

    # Dummy test mode:
    python scripts/prepare_crosscorpus_visec.py --dummy --output-dir data/visec_crosscorpus
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
logger = logging.getLogger("prepare_crosscorpus_visec")

TARGET_SAMPLE_RATE = 16000

# Canonical label mapping from ViSEC -> Primary & Sub
VISEC_LABEL_MAPPING = {
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Cross-Corpus ViSEC + RAVDESS/CREMA-D dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/visec_crosscorpus",
        help="Target directory for processed audio and metadata.csv",
    )
    parser.add_argument(
        "--visec-dataset",
        type=str,
        default="hustep-lab/ViSEC",
        help="Hugging Face ViSEC dataset repo (default: hustep-lab/ViSEC)",
    )
    parser.add_argument(
        "--ravdess-dir",
        type=str,
        default=None,
        help="Path to RAVDESS dataset directory (checks /kaggle/input/ravdess-emotional-speech-audio by default)",
    )
    parser.add_argument(
        "--cremad-dir",
        type=str,
        default=None,
        help="Path to CREMA-D dataset directory (checks /kaggle/input/cremad by default)",
    )
    parser.add_argument(
        "--cremad-limit",
        type=int,
        default=600,
        help="Maximum number of Fearful audio samples to load from CREMA-D (default: 600)",
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
        "--sample-limit",
        type=int,
        default=None,
        help="Limit ViSEC samples for quick debugging",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible stratified split",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Generate synthetic samples offline without downloading",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Audio Processing Utilities
# ---------------------------------------------------------------------------

def resample_audio(audio_arr: np.ndarray, orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Resample 1D or 2D audio array to target sample rate."""
    if audio_arr.ndim > 1:
        if audio_arr.shape[0] > 1 and audio_arr.shape[1] > 1:
            audio_arr = np.mean(audio_arr, axis=1)
        else:
            audio_arr = audio_arr.squeeze()

    if orig_sr == target_sr:
        return audio_arr.astype(np.float32)

    try:
        from scipy import signal
        num_target_samples = int(len(audio_arr) * target_sr / orig_sr)
        return signal.resample(audio_arr, num_target_samples).astype(np.float32)
    except ImportError:
        orig_indices = np.linspace(0, len(audio_arr) - 1, len(audio_arr))
        target_len = int(len(audio_arr) * target_sr / orig_sr)
        target_indices = np.linspace(0, len(audio_arr) - 1, target_len)
        return np.interp(target_indices, orig_indices, audio_arr).astype(np.float32)


def save_wav(file_path: Path, audio_arr: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> None:
    """Save float32 array as 16kHz mono PCM 16-bit WAV."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio_arr, -1.0, 1.0)
    try:
        import soundfile as sf
        sf.write(str(file_path), clipped, sample_rate, subtype="PCM_16")
    except ImportError:
        import wave
        int16_data = (clipped * 32767).astype(np.int16)
        with wave.open(str(file_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(int16_data.tobytes())


def load_audio_file(file_path: Path) -> Tuple[Optional[np.ndarray], int]:
    """Load local audio file into float32 numpy array and sample rate."""
    try:
        import soundfile as sf
        data, sr = sf.read(str(file_path))
        if data.ndim > 1:
            data = data.mean(axis=-1)
        return data.astype(np.float32), sr
    except Exception as e:
        try:
            import torchaudio
            wav, sr = torchaudio.load(str(file_path))
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0)
            return wav.squeeze(0).numpy().astype(np.float32), sr
        except Exception as e2:
            logger.warning("Could not read audio file %s: %s | %s", file_path, e, e2)
            return None, TARGET_SAMPLE_RATE


# ---------------------------------------------------------------------------
# Hugging Face Audio Decoding (for ViSEC)
# ---------------------------------------------------------------------------

def extract_audio_from_hf_rec(rec: Dict[str, Any]) -> Tuple[Optional[np.ndarray], int]:
    """Robustly extract audio array and sample rate from Hugging Face record."""
    audio_obj = None
    for key in ["audio", "speech", "wav", "sound"]:
        if key in rec:
            val = rec[key]
            if isinstance(val, (list, tuple)) and len(val) > 0:
                val = val[0]
            audio_obj = val
            break

    if audio_obj is None:
        for val in rec.values():
            if isinstance(val, (list, tuple)) and len(val) > 0:
                val = val[0]
            if hasattr(val, "get_all_samples") or "AudioDecoder" in type(val).__name__:
                audio_obj = val
                break
            if isinstance(val, dict) and ("array" in val or "bytes" in val or "path" in val):
                audio_obj = val
                break

    if audio_obj is None:
        return None, TARGET_SAMPLE_RATE

    if isinstance(audio_obj, (list, tuple)) and len(audio_obj) > 0:
        audio_obj = audio_obj[0]

    try:
        # torchcodec AudioDecoder (HF datasets >= 4.0)
        if hasattr(audio_obj, "get_all_samples"):
            samples = audio_obj.get_all_samples()
            data = getattr(samples, "data", samples)
            arr = data.detach().cpu().numpy() if hasattr(data, "detach") else np.array(data)
            arr = arr.astype(np.float32) / 32768.0 if np.issubdtype(arr.dtype, np.integer) else arr.astype(np.float32)
            sr = int(getattr(samples, "sample_rate", TARGET_SAMPLE_RATE) or TARGET_SAMPLE_RATE)
            return arr, sr

        if isinstance(audio_obj, dict):
            if "array" in audio_obj and audio_obj["array"] is not None:
                arr = np.array(audio_obj["array"], dtype=np.float32)
                sr = int(audio_obj.get("sampling_rate", TARGET_SAMPLE_RATE) or TARGET_SAMPLE_RATE)
                return arr, sr
            if "bytes" in audio_obj and audio_obj["bytes"] is not None:
                import io, soundfile as sf
                arr, sr = sf.read(io.BytesIO(audio_obj["bytes"]))
                return arr.astype(np.float32), int(sr)
            if "path" in audio_obj and audio_obj["path"] and os.path.exists(str(audio_obj["path"])):
                return load_audio_file(Path(audio_obj["path"]))

        if isinstance(audio_obj, (str, Path)) and os.path.exists(str(audio_obj)):
            return load_audio_file(Path(audio_obj))

    except Exception as e:
        logger.debug("Failed decoding HF audio object: %s", e)

    return None, TARGET_SAMPLE_RATE


# ---------------------------------------------------------------------------
# Corpus Loaders
# ---------------------------------------------------------------------------

def load_visec(dataset_name: str, sample_limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load ViSEC dataset from Hugging Face."""
    logger.info("Loading ViSEC dataset '%s' from Hugging Face...", dataset_name)
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("The 'datasets' library is required. Install via: pip install datasets")
        sys.exit(1)

    try:
        ds = load_dataset(dataset_name)
    except Exception as e:
        logger.error("Failed to load '%s': %s", dataset_name, e)
        sys.exit(1)

    # Gather rows across all splits
    all_rows = []
    splits = list(ds.keys())
    for sp in splits:
        split_data = ds[sp]
        for row in split_data:
            all_rows.append(row)
            if sample_limit and len(all_rows) >= sample_limit:
                break
        if sample_limit and len(all_rows) >= sample_limit:
            break

    logger.info("Retrieved %d raw records from ViSEC across splits %s.", len(all_rows), splits)

    records = []
    for idx, row in enumerate(all_rows):
        # Extract label
        raw_label = None
        for k in ["label", "emotion", "sentiment", "labels"]:
            if k in row:
                raw_label = row[k]
                break
        if raw_label is None:
            continue

        raw_str = str(raw_label).lower().strip()
        mapped = VISEC_LABEL_MAPPING.get(raw_str)
        if mapped is None:
            for k, v in VISEC_LABEL_MAPPING.items():
                if str(k) in raw_str:
                    mapped = v
                    break
        if mapped is None:
            continue

        primary_label, sub_label = mapped

        # Extract audio
        arr, sr = extract_audio_from_hf_rec(row)
        if arr is None or len(arr) == 0:
            continue

        # Extract transcript
        transcript = ""
        for k in ["transcript", "text", "sentence", "content"]:
            if k in row and row[k]:
                transcript = str(row[k]).strip()
                break

        records.append({
            "sample_id": f"visec_{idx:05d}",
            "audio_arr": arr,
            "orig_sr": sr,
            "primary_label": primary_label,
            "sub_label": sub_label,
            "transcript": transcript,
            "speaker_id": str(row.get("speaker_id", f"visec_spk_{idx % 50}")),
            "source": "ViSEC",
        })

    logger.info("Successfully processed %d valid ViSEC samples.", len(records))
    return records


def load_ravdess_fearful(ravdess_dir: Path) -> List[Dict[str, Any]]:
    """Scan RAVDESS directory and extract only Fearful/Anxiety audio clips (emotion 06)."""
    logger.info("Scanning RAVDESS audio in %s...", ravdess_dir)
    wav_files = list(ravdess_dir.glob("**/*.wav"))
    fearful_files = []

    for f in wav_files:
        # RAVDESS filename format: 03-01-06-01-01-01-01.wav
        parts = f.stem.split("-")
        if len(parts) >= 3 and parts[2] == "06":
            fearful_files.append(f)

    logger.info("Found %d Fearful/Anxiety audio clips in RAVDESS (from %d total wavs).", len(fearful_files), len(wav_files))

    records = []
    for idx, f in enumerate(fearful_files):
        arr, sr = load_audio_file(f)
        if arr is None or len(arr) == 0:
            continue

        parts = f.stem.split("-")
        actor_id = f"Actor_{parts[6]}" if len(parts) >= 7 else "Actor_Unknown"
        intensity = "strong" if len(parts) >= 4 and parts[3] == "02" else "normal"

        records.append({
            "sample_id": f"ravdess_fear_{idx:04d}",
            "audio_arr": arr,
            "orig_sr": sr,
            "primary_label": "ANXIETY",
            "sub_label": "FEAR",
            "transcript": f"RAVDESS Fearful statement ({intensity} intensity)",
            "speaker_id": actor_id,
            "source": "RAVDESS",
        })

    return records


def find_ravdess_dir(user_specified_dir: Optional[str] = None) -> Optional[Path]:
    """Locate RAVDESS dataset directory across common Kaggle and local paths."""
    candidates = []
    if user_specified_dir:
        candidates.append(Path(user_specified_dir))

    candidates.extend([
        Path("/kaggle/input/ravdess-emotional-speech-audio"),
        Path("/kaggle/input/ravdess-emotional-speech-audio/audio_speech_actors_01-24"),
        Path("/kaggle/input/speech-emotion-recognition/ravdess-emotional-speech-audio"),
        Path("/kaggle/input/speech-emotion-recognition"),
        Path("data/ravdess"),
        Path("data/raw/ravdess"),
    ])

    for c in candidates:
        if c.exists() and any(c.glob("**/*-*-06-*.wav")):
            logger.info("Found RAVDESS directory: %s", c)
            return c

    # Deep search inside /kaggle/input if available
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for sub in kaggle_input.iterdir():
            if sub.is_dir():
                # Check if this subfolder contains RAVDESS fearful clips
                matches = list(sub.glob("**/*-*-06-*.wav"))
                if matches:
                    matched_dir = matches[0].parent.parent if matches[0].parent.name.startswith("Actor_") else matches[0].parent
                    logger.info("Auto-discovered RAVDESS via deep search in %s", matched_dir)
                    return matched_dir

    return None


def find_cremad_dir(user_specified_dir: Optional[str] = None) -> Optional[Path]:
    """Locate CREMA-D dataset directory across common Kaggle and local paths."""
    candidates = []
    if user_specified_dir:
        candidates.append(Path(user_specified_dir))

    candidates.extend([
        Path("/kaggle/input/cremad"),
        Path("/kaggle/input/crema-d"),
        Path("/kaggle/input/cremad-dataset"),
        Path("/kaggle/input/crema-d-dataset"),
        Path("/kaggle/input/speech-emotion-recognition/AudioWAV"),
        Path("/kaggle/input/speech-emotion-recognition/crema-d/AudioWAV"),
        Path("/kaggle/input/speech-emotion-recognition"),
        Path("data/cremad"),
        Path("data/raw/cremad"),
    ])

    for c in candidates:
        if c.exists() and any(c.glob("**/*_FEA_*.wav")):
            logger.info("Found CREMA-D directory: %s", c)
            return c

    # Deep search inside /kaggle/input if available
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for sub in kaggle_input.iterdir():
            if sub.is_dir():
                matches = list(sub.glob("**/*_FEA_*.wav"))
                if matches:
                    matched_dir = matches[0].parent
                    logger.info("Auto-discovered CREMA-D via deep search in %s", matched_dir)
                    return matched_dir

    return None


def load_cremad_fear(cremad_dir: Path, sample_limit: int = 600) -> List[Dict[str, Any]]:
    """Scan CREMA-D directory and extract Fearful/Anxiety audio clips (*_FEA_*.wav)."""
    logger.info("Scanning CREMA-D audio in %s...", cremad_dir)
    wav_files = list(cremad_dir.glob("**/*.wav"))
    fear_files = []

    for f in wav_files:
        # CREMA-D format: 1001_DFA_FEA_XX.wav
        parts = f.stem.split("_")
        if len(parts) >= 3 and parts[2].upper() == "FEA":
            fear_files.append(f)

    logger.info("Found %d Fear audio clips in CREMA-D.", len(fear_files))
    if sample_limit and len(fear_files) > sample_limit:
        random.seed(42)
        random.shuffle(fear_files)
        fear_files = fear_files[:sample_limit]
        logger.info("Subsampled to %d clips for balanced distribution.", len(fear_files))

    records = []
    for idx, f in enumerate(fear_files):
        arr, sr = load_audio_file(f)
        if arr is None or len(arr) == 0:
            continue

        parts = f.stem.split("_")
        speaker_id = f"crema_{parts[0]}" if len(parts) >= 1 else "crema_unknown"

        records.append({
            "sample_id": f"cremad_fear_{idx:04d}",
            "audio_arr": arr,
            "orig_sr": sr,
            "primary_label": "ANXIETY",
            "sub_label": "FEAR",
            "transcript": "CREMA-D Fearful speech utterance",
            "speaker_id": speaker_id,
            "source": "CREMA-D",
        })

    return records


# ---------------------------------------------------------------------------
# Dummy Generator (for testing)
# ---------------------------------------------------------------------------

def generate_dummy_crosscorpus(output_dir: Path, total_samples: int = 60) -> None:
    """Generate synthetic 5-class dummy dataset for offline testing."""
    logger.info("Generating %d synthetic dummy samples for all 5 classes...", total_samples)
    classes = [
        ("JOY", "JOY"),
        ("SADNESS", "SADNESS"),
        ("ANXIETY", "FEAR"),
        ("ANGER", "ANGER"),
        ("NEUTRAL", "CALM"),
    ]

    records = []
    for i in range(total_samples):
        primary, sub = classes[i % len(classes)]
        duration_s = 3.0 + (i % 3) * 1.0
        num_samples = int(duration_s * TARGET_SAMPLE_RATE)
        # Generate synthetic speech-like bandpassed noise
        t = np.linspace(0, duration_s, num_samples, endpoint=False)
        audio = 0.1 * np.sin(2 * np.pi * 220.0 * t) + 0.05 * np.random.randn(num_samples)

        records.append({
            "sample_id": f"dummy_{i:04d}",
            "audio_arr": audio.astype(np.float32),
            "orig_sr": TARGET_SAMPLE_RATE,
            "primary_label": primary,
            "sub_label": sub,
            "transcript": f"Synthetic dummy monologue for emotion {primary}",
            "speaker_id": f"speaker_{i % 6:02d}",
            "source": "SyntheticDummy",
        })

    split_and_save(records, output_dir=output_dir, train_ratio=0.8, val_ratio=0.1, seed=42)


# ---------------------------------------------------------------------------
# Stratified Split & Save
# ---------------------------------------------------------------------------

def split_and_save(
    records: List[Dict[str, Any]],
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """Stratify by primary_label, save audio files, and generate metadata.csv."""
    rng = random.Random(seed)

    # Group by primary label
    by_class: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        lbl = r["primary_label"]
        by_class.setdefault(lbl, []).append(r)

    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []
    test_records: List[Dict[str, Any]] = []

    for lbl, items in by_class.items():
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, int(math.floor(n * train_ratio)))
        n_val = max(1, int(math.floor(n * val_ratio)))
        n_test = max(1, n - n_train - n_val)

        train_records.extend(items[:n_train])
        val_records.extend(items[n_train:n_train + n_val])
        test_records.extend(items[n_train + n_val:])

    splits = {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }

    logger.info("=" * 65)
    logger.info("  FINAL 5-CLASS DATASET DISTRIBUTION")
    logger.info("=" * 65)
    all_classes = ["JOY", "SADNESS", "ANXIETY", "ANGER", "NEUTRAL"]
    logger.info("%-12s | %-8s | %-8s | %-8s | %-8s", "Emotion", "Train", "Val", "Test", "Total")
    logger.info("-" * 65)

    for c in all_classes:
        n_tr = sum(1 for r in train_records if r["primary_label"] == c)
        n_va = sum(1 for r in val_records if r["primary_label"] == c)
        n_te = sum(1 for r in test_records if r["primary_label"] == c)
        n_tot = n_tr + n_va + n_te
        logger.info("%-12s | %8d | %8d | %8d | %8d", c, n_tr, n_va, n_te, n_tot)

    logger.info("-" * 65)
    logger.info("%-12s | %8d | %8d | %8d | %8d", "TOTAL", len(train_records), len(val_records), len(test_records), len(records))
    logger.info("=" * 65)

    # Save audio files and write metadata.csv
    for split_name, split_items in splits.items():
        split_dir = output_dir / split_name
        audio_dir = split_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        csv_path = split_dir / "metadata.csv"

        logger.info("Writing %d samples to %s...", len(split_items), split_dir)
        rows_to_write = []

        for item in split_items:
            s_id = item["sample_id"]
            wav_filename = f"{s_id}.wav"
            target_wav_path = audio_dir / wav_filename

            # Resample & save WAV
            arr_16k = resample_audio(item["audio_arr"], orig_sr=item["orig_sr"], target_sr=TARGET_SAMPLE_RATE)
            save_wav(target_wav_path, arr_16k, sample_rate=TARGET_SAMPLE_RATE)

            duration_s = round(float(len(arr_16k)) / TARGET_SAMPLE_RATE, 3)
            rel_audio_path = f"audio/{wav_filename}"

            rows_to_write.append({
                "sample_id": s_id,
                "audio_path": rel_audio_path,
                "primary_label": item["primary_label"],
                "sub_label": item["sub_label"],
                "transcript": item["transcript"],
                "duration_s": duration_s,
                "speaker_id": item["speaker_id"],
            })

        fieldnames = ["sample_id", "audio_path", "primary_label", "sub_label", "transcript", "duration_s", "speaker_id"]
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_to_write)

        logger.info("Successfully generated %s (%d records).", csv_path, len(rows_to_write))


# ---------------------------------------------------------------------------
# Main Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    if args.dummy:
        generate_dummy_crosscorpus(output_dir=output_dir, total_samples=60)
        return

    # 1. Load ViSEC (4 classes: JOY, SADNESS, ANGER, NEUTRAL)
    visec_records = load_visec(dataset_name=args.visec_dataset, sample_limit=args.sample_limit)

    # 2. Ingest RAVDESS Fearful clips
    ravdess_dir = find_ravdess_dir(args.ravdess_dir)
    ravdess_records = []
    if ravdess_dir:
        ravdess_records = load_ravdess_fearful(ravdess_dir)
    else:
        logger.warning(
            "RAVDESS directory not found! Checked '%s'. "
            "Mount 'uwrfkaggler/ravdess-emotional-speech-audio' on Kaggle or provide --ravdess-dir.",
            args.ravdess_dir,
        )

    # 3. Ingest CREMA-D Fear clips (Optional)
    cremad_dir = find_cremad_dir(args.cremad_dir)
    cremad_records = []
    if cremad_dir:
        cremad_records = load_cremad_fear(cremad_dir, sample_limit=args.cremad_limit)
    else:
        logger.info("CREMA-D directory not detected. Proceeding with ViSEC + RAVDESS.")

    # Combine all records
    all_records = visec_records + ravdess_records + cremad_records
    if not all_records:
        logger.error("No valid audio records collected. Aborting.")
        sys.exit(1)

    anxiety_count = sum(1 for r in all_records if r["primary_label"] == "ANXIETY")
    if anxiety_count == 0:
        logger.warning("Caution: ANXIETY sample count is 0! Make sure RAVDESS or CREMA-D is properly mounted.")

    # Stratified split and save
    split_and_save(
        all_records,
        output_dir=output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    logger.info("Cross-Corpus dataset preparation complete! Ready for Stage 1.5 training.")


if __name__ == "__main__":
    main()
