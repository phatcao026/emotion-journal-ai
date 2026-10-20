"""
audio_dataset.py

Dataset class for Vietnamese voice journal emotion recognition.

Supports:
    - Loading real .wav recordings with accompanying transcripts and labels.
    - Generating synthetic dummy samples for offline testing, debugging, and CI.
    - Batch padding with custom collate_fn.
    - Class weight calculation for handling class imbalance.
"""

from __future__ import annotations

import csv
import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Emotion Labels (per VOICE_PIPELINE_SPEC.md)
# ---------------------------------------------------------------------------

class PrimaryEmotion(Enum):
    """Primary emotion labels (Level 1) - 5 classes."""
    JOY = 0
    SADNESS = 1
    ANXIETY = 2
    ANGER = 3
    NEUTRAL = 4


class SubEmotion(Enum):
    """Sub-category emotion labels (Level 2) - 11 fine-grained classes (Annotation Guideline v0.1)."""
    JOY = 0
    CALM = 1
    HOPE = 2
    CONNECTION = 3
    SADNESS = 4
    ANXIETY = 5
    FEAR = 6
    ANGER = 7
    GUILT_SHAME = 8
    LONELINESS = 9
    DISGUST = 10


# Mapping from sub-category emotion -> primary emotion
SUB_TO_PRIMARY: Dict[SubEmotion, PrimaryEmotion] = {
    SubEmotion.JOY: PrimaryEmotion.JOY,
    SubEmotion.CALM: PrimaryEmotion.JOY,
    SubEmotion.HOPE: PrimaryEmotion.JOY,
    SubEmotion.CONNECTION: PrimaryEmotion.JOY,
    SubEmotion.SADNESS: PrimaryEmotion.SADNESS,
    SubEmotion.LONELINESS: PrimaryEmotion.SADNESS,
    SubEmotion.GUILT_SHAME: PrimaryEmotion.SADNESS,
    SubEmotion.ANXIETY: PrimaryEmotion.ANXIETY,
    SubEmotion.FEAR: PrimaryEmotion.ANXIETY,
    SubEmotion.ANGER: PrimaryEmotion.ANGER,
    SubEmotion.DISGUST: PrimaryEmotion.ANGER,
}


# ---------------------------------------------------------------------------
# Dual-Engine Voice Activity Detection (VAD)
# ---------------------------------------------------------------------------

_SILERO_VAD_MODEL = None
_SILERO_VAD_UTILS = None
_SILERO_LOAD_TRIED = False


def _get_silero_vad() -> Tuple[Optional[Any], Optional[Any]]:
    """Lazily load Silero-VAD from torch.hub or local cache (fail-safe)."""
    global _SILERO_VAD_MODEL, _SILERO_VAD_UTILS, _SILERO_LOAD_TRIED
    if _SILERO_VAD_MODEL is not None:
        return _SILERO_VAD_MODEL, _SILERO_VAD_UTILS
    if _SILERO_LOAD_TRIED:
        return None, None
    _SILERO_LOAD_TRIED = True
    try:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        _SILERO_VAD_MODEL = model.eval()
        _SILERO_VAD_UTILS = utils
        return _SILERO_VAD_MODEL, _SILERO_VAD_UTILS
    except Exception as e:
        logger.debug("Silero-VAD lazy load failed (%s); will use Energy-Entropy VAD.", e)
        return None, None


def _energy_vad(waveform: torch.Tensor, sample_rate: int = 16000) -> List[dict]:
    """Lightweight, ultra-fast (<0.2ms) Energy-Entropy VAD fallback.

    Segments audio using local RMS envelope with 30ms window and 15ms hop.
    Speech frames are classified where RMS > max(0.01, 0.10 * rms_max).
    Consecutive speech frames are merged, segments shorter than 100ms are discarded,
    and a 50ms safety padding is applied to each boundary.

    Args:
        waveform: 1D or 2D audio tensor.
        sample_rate: Sampling rate in Hz (default 16000).

    Returns:
        List of dicts [{'start': int, 'end': int}] in sample indices.
    """
    if waveform is None or waveform.numel() == 0:
        return []

    wav = waveform.detach().float()
    if wav.ndim > 1:
        wav = wav.squeeze()
        if wav.ndim > 1:
            wav = wav[0]
    wav = wav.cpu()

    total_samples = wav.numel()
    frame_len = int(0.030 * sample_rate)  # 30ms frame
    hop_len = int(0.015 * sample_rate)    # 15ms hop
    min_speech_samples = int(0.100 * sample_rate)  # 100ms
    pad_samples = int(0.050 * sample_rate)         # 50ms

    if total_samples < frame_len:
        return []

    # Unfold into overlapping frames: (num_frames, frame_len)
    frames = wav.unfold(0, frame_len, hop_len)
    num_frames = frames.size(0)
    if num_frames == 0:
        return []

    # Frame-wise RMS computation
    rms = torch.sqrt(torch.mean(frames ** 2, dim=-1) + 1e-9)
    rms_max = float(torch.max(rms).item())
    threshold = max(0.01, 0.10 * rms_max)

    is_speech = (rms > threshold).tolist()

    # Find contiguous speech segments
    raw_segments: List[Tuple[int, int]] = []
    in_speech = False
    start_sample = 0

    for i, active in enumerate(is_speech):
        if active and not in_speech:
            in_speech = True
            start_sample = i * hop_len
        elif not active and in_speech:
            in_speech = False
            end_sample = min(i * hop_len + frame_len, total_samples)
            if (end_sample - start_sample) >= min_speech_samples:
                raw_segments.append((start_sample, end_sample))

    if in_speech:
        end_sample = min((num_frames - 1) * hop_len + frame_len, total_samples)
        if (end_sample - start_sample) >= min_speech_samples:
            raw_segments.append((start_sample, end_sample))

    if not raw_segments:
        return []

    # Apply 50ms padding and merge overlapping intervals
    merged: List[dict] = []
    for s, e in raw_segments:
        p_start = max(0, s - pad_samples)
        p_end = min(total_samples, e + pad_samples)
        if not merged:
            merged.append({"start": int(p_start), "end": int(p_end)})
        else:
            if p_start <= merged[-1]["end"]:
                merged[-1]["end"] = max(merged[-1]["end"], int(p_end))
            else:
                merged.append({"start": int(p_start), "end": int(p_end)})

    return merged


def extract_speech_timestamps(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    vad_preprocessor: Optional[Any] = None,
) -> List[dict]:
    """Extract speech timestamps using Dual-Engine VAD architecture.

    Engine 1 (Primary): Silero-VAD if provided via vad_preprocessor or cached from torch.hub.
    Engine 2 (Instant Fallback): Deterministic Energy-Entropy VAD (<1ms execution time).

    Guarantees:
        - Device-safe and zero-crash on any input (silent, all-zeros, short tensors).
        - Consistent schema: List of dicts [{'start': int, 'end': int}].

    Args:
        waveform: Tensor of shape (1, T), (T,), or (C, T).
        sample_rate: Sample rate in Hz (default: 16000).
        vad_preprocessor: Optional preprocessor instance or Silero wrapper.

    Returns:
        List of dicts [{'start': int, 'end': int}] specifying speech intervals in samples.
    """
    if waveform is None or waveform.numel() == 0:
        return []

    wav = waveform.detach()
    if wav.ndim > 1:
        wav = wav.squeeze()
        if wav.ndim > 1:
            wav = wav[0]
    wav = wav.float()

    total_samples = wav.numel()
    if total_samples < int(0.100 * sample_rate):
        return []

    # 1. Primary Engine: Silero-VAD via vad_preprocessor
    if vad_preprocessor is not None:
        try:
            if hasattr(vad_preprocessor, "_run_vad"):
                if getattr(vad_preprocessor, "_vad_model", None) is None and hasattr(vad_preprocessor, "load_model"):
                    vad_preprocessor.load_model()
                ts = vad_preprocessor._run_vad(wav.unsqueeze(0))
                if isinstance(ts, list) and len(ts) > 0:
                    return [{"start": int(t["start"]), "end": int(t["end"])} for t in ts]
            elif hasattr(vad_preprocessor, "process_waveform"):
                res = vad_preprocessor.process_waveform(wav.unsqueeze(0), sample_rate)
                if res and hasattr(res, "speech_timestamps") and len(res.speech_timestamps) > 0:
                    return [{"start": int(t["start"]), "end": int(t["end"])} for t in res.speech_timestamps]
        except Exception as e:
            logger.debug("Primary vad_preprocessor failed (%s); trying fallback.", e)

    # 2. Primary Engine: Cached Silero-VAD from torch.hub
    try:
        model, utils = _get_silero_vad()
        if model is not None and utils is not None and wav.numel() >= 512:
            get_speech_timestamps = utils[0]
            device = next(model.parameters()).device
            ts = get_speech_timestamps(
                wav.to(device),
                model,
                sampling_rate=sample_rate,
                threshold=0.5,
                min_speech_duration_ms=100,
                speech_pad_ms=50,
            )
            if isinstance(ts, list) and len(ts) > 0:
                return [{"start": int(t["start"]), "end": int(t["end"])} for t in ts]
    except Exception as e:
        logger.debug("Cached Silero-VAD failed (%s); falling back to Energy VAD.", e)

    # 3. Fallback Engine: Fast Energy-Entropy VAD
    try:
        return _energy_vad(wav, sample_rate=sample_rate)
    except Exception as e:
        logger.debug("Energy VAD fallback failed (%s); returning empty list.", e)
        return []


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class AudioSample:
    """Represents a single audio journal entry.

    Attributes:
        sample_id (str): Unique identifier.
        primary_label (PrimaryEmotion): Ground truth primary class.
        audio_path (Optional[str]): Path to audio file on disk.
        waveform (Optional[torch.Tensor]): Loaded waveform tensor (1, T).
        sample_rate (int): Sampling rate (default 16000).
        duration_s (float): Duration in seconds.
        sub_label (Optional[SubEmotion]): Ground truth sub-category class.
        transcript (Optional[str]): Text transcript of the speech.
        speaker_id (Optional[str]): Speaker identity if available.
        metadata (Dict): Any additional metadata.
    """

    sample_id: str
    primary_label: PrimaryEmotion
    audio_path: Optional[str] = None
    waveform: Optional[torch.Tensor] = None
    sample_rate: int = 16000
    duration_s: float = 0.0
    sub_label: Optional[SubEmotion] = None
    transcript: Optional[str] = None
    speaker_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# VoiceJournalDataset
# ---------------------------------------------------------------------------

class VoiceJournalDataset(Dataset):
    """Dataset for Vietnamese voice journals with hierarchical emotion labels.

    Args:
        data_dir (Optional[Union[str, Path]]): Path to dataset folder.
        split (str): 'train', 'val', or 'test'. Default: 'train'.
        sample_rate (int): Target sampling rate. Default: 16000.
        max_duration_s (float): Cap on sample duration. Default: 300.0.
        dummy_mode (bool): If True, generate synthetic samples for offline testing.
        dummy_size (int): Number of synthetic samples to create. Default: 100.
        dummy_duration_s (float): Duration of each synthetic sample in seconds. Default: 6.0.
        transform (Optional[Callable]): Optional waveform transformation.
        use_sub_labels (bool): Whether to include sub-category labels. Default: True.
    """

    SAMPLE_RATE: int = 16000

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        split: str = "train",
        sample_rate: int = 16000,
        max_duration_s: float = 300.0,
        dummy_mode: bool = False,
        dummy_size: int = 100,
        dummy_duration_s: float = 6.0,
        transform: Optional[Callable] = None,
        use_sub_labels: bool = True,
        vad_preprocessor: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir) if data_dir else None
        self.split = split
        self.sample_rate = sample_rate
        self.max_duration_s = max_duration_s
        self.dummy_mode = dummy_mode
        self.dummy_size = dummy_size
        self.dummy_duration_s = dummy_duration_s
        self.transform = transform
        self.use_sub_labels = use_sub_labels
        self.vad_preprocessor = vad_preprocessor

        self.samples: List[AudioSample] = []

        if dummy_mode:
            self._generate_dummy_samples()
            logger.info(
                "VoiceJournalDataset [DUMMY]: %d samples generated, duration=%.1fs",
                dummy_size,
                dummy_duration_s,
            )
        else:
            if self.data_dir is None:
                raise ValueError("data_dir is required when dummy_mode=False")
            self._load_from_directory()
            logger.info(
                "VoiceJournalDataset [%s]: loaded %d samples from %s",
                split.upper(),
                len(self.samples),
                data_dir,
            )

    def _generate_dummy_samples(self) -> None:
        """Create synthetic audio samples with random waveforms and labels."""
        torch.manual_seed(42)
        num_samples_per_clip = int(self.dummy_duration_s * self.sample_rate)

        dummy_transcripts = [
            "Hôm nay tôi cảm thấy rất vui và tràn đầy năng lượng.",
            "Tôi thấy khá mệt mỏi và chán nản về công việc hiện tại.",
            "Tự dưng cảm thấy lo lắng và bất an không rõ lý do.",
            "Tôi rất bực bội khi mọi chuyện không như ý muốn.",
            "Một ngày bình thường, không có gì đặc biệt diễn ra.",
        ]

        sub_list = list(SubEmotion)
        for i in range(self.dummy_size):
            sub_label = random.choice(sub_list)
            primary_label = SUB_TO_PRIMARY[sub_label]

            # Generate synthetic speech-like bandpassed noise with natural speech pauses
            noise = torch.randn(1, num_samples_per_clip) * 0.1
            if self.dummy_duration_s >= 4.0:
                p1_start = int(1.5 * self.sample_rate)
                p1_end = int(2.2 * self.sample_rate)
                noise[..., p1_start:p1_end] = 0.0
                if self.dummy_duration_s >= 5.5:
                    p2_start = int(3.8 * self.sample_rate)
                    p2_end = int(4.5 * self.sample_rate)
                    noise[..., p2_start:p2_end] = 0.0
            transcript = dummy_transcripts[primary_label.value % len(dummy_transcripts)]

            sample = AudioSample(
                sample_id=f"dummy_{i:04d}",
                primary_label=primary_label,
                sub_label=sub_label,
                waveform=noise,
                sample_rate=self.sample_rate,
                duration_s=self.dummy_duration_s,
                transcript=transcript,
                metadata={"synthetic": True},
            )
            self.samples.append(sample)

    def _load_waveform(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Load audio file and ensure mono output at target sample rate."""
        path = str(audio_path)
        try:
            import torchaudio

            waveform, sr = torchaudio.load(path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != self.sample_rate:
                import torchaudio.transforms as T

                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
                sr = self.sample_rate
            return waveform, sr
        except Exception:
            import soundfile as sf

            data, sr = sf.read(path)
            waveform = torch.from_numpy(data).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.mean(dim=-1, keepdim=True).t()
            return waveform, sr

    def _load_from_directory(self) -> None:
        """Load dataset manifest from data_dir."""
        split_dir = self.data_dir / self.split if (self.data_dir / self.split).exists() else self.data_dir
        manifest_path = split_dir / "metadata.csv"

        if not manifest_path.exists():
            # Scan for all wav files if metadata.csv not found
            wav_files = list(split_dir.glob("**/*.wav"))
            if not wav_files:
                raise FileNotFoundError(f"No .wav files or metadata.csv found in {split_dir}")

            logger.warning("metadata.csv not found in %s; using unlabeled wav files with default labels.", split_dir)
            for i, p in enumerate(wav_files):
                self.samples.append(
                    AudioSample(
                        sample_id=p.stem,
                        audio_path=str(p),
                        primary_label=PrimaryEmotion.NEUTRAL,
                        sub_label=SubEmotion.CALM if self.use_sub_labels else None,
                        sample_rate=self.sample_rate,
                    )
                )
            return

        with open(manifest_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_split = row.get("split", "").strip().lower()
                if row_split and row_split != self.split.lower():
                    continue
                s_id = row.get("sample_id", "")
                a_path = row.get("audio_path", "")
                if not Path(a_path).is_absolute():
                    a_path = str(split_dir / a_path)

                p_str = row.get("primary_label", "NEUTRAL").upper()
                primary = getattr(PrimaryEmotion, p_str, PrimaryEmotion.NEUTRAL)

                sub = None
                if self.use_sub_labels and "sub_label" in row:
                    sub_str = row.get("sub_label", "").upper()
                    sub = getattr(SubEmotion, sub_str, None)

                transcript = row.get("transcript", None)

                self.samples.append(
                    AudioSample(
                        sample_id=s_id,
                        audio_path=a_path,
                        primary_label=primary,
                        sub_label=sub,
                        transcript=transcript,
                        sample_rate=self.sample_rate,
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, str]]:
        sample = self.samples[idx]

        if sample.waveform is not None:
            waveform = sample.waveform.clone()
        elif sample.audio_path is not None:
            waveform, sr = self._load_waveform(sample.audio_path)
            sample.duration_s = float(waveform.shape[-1]) / float(sr)
        else:
            raise RuntimeError(f"Sample {sample.sample_id} has neither waveform nor audio_path")

        if self.transform is not None:
            waveform = self.transform(waveform)

        # Truncate if exceeding max duration
        max_samples = int(self.max_duration_s * self.sample_rate)
        if waveform.shape[-1] > max_samples:
            waveform = waveform[..., :max_samples]

        speech_timestamps = extract_speech_timestamps(
            waveform,
            sample_rate=self.sample_rate,
            vad_preprocessor=getattr(self, "vad_preprocessor", None),
        )

        item: Dict[str, Union[torch.Tensor, int, str, List[dict]]] = {
            "waveform": waveform,  # shape (1, T)
            "primary_label": torch.tensor(sample.primary_label.value, dtype=torch.long),
            "sample_id": sample.sample_id,
            "duration_s": sample.duration_s,
            "transcript": sample.transcript or "",
            "speech_timestamps": speech_timestamps,
        }

        if self.use_sub_labels:
            sub_val = sample.sub_label.value if sample.sub_label is not None else 0
            item["sub_label"] = torch.tensor(sub_val, dtype=torch.long)

        return item

    @staticmethod
    def collate_fn(
        batch: List[Dict[str, Union[torch.Tensor, int, str, List[dict]]]],
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """Pad waveforms to the maximum length in the batch."""
        max_len = max(item["waveform"].shape[-1] for item in batch)
        batch_size = len(batch)

        padded_waveforms = torch.zeros((batch_size, 1, max_len), dtype=torch.float32)
        lengths = torch.zeros(batch_size, dtype=torch.long)

        primary_labels = []
        sub_labels = []
        sample_ids = []
        transcripts = []
        speech_timestamps = []

        has_sub = "sub_label" in batch[0]

        for i, item in enumerate(batch):
            w = item["waveform"]
            cur_len = w.shape[-1]
            padded_waveforms[i, :, :cur_len] = w
            lengths[i] = cur_len
            primary_labels.append(item["primary_label"])
            if has_sub:
                sub_labels.append(item["sub_label"])
            sample_ids.append(item["sample_id"])
            transcripts.append(item["transcript"])
            speech_timestamps.append(item.get("speech_timestamps", []))

        res: Dict[str, Union[torch.Tensor, List]] = {
            "waveforms": padded_waveforms,
            "waveform_lengths": lengths,
            "primary_labels": torch.stack(primary_labels),
            "sample_ids": sample_ids,
            "transcripts": transcripts,
            "speech_timestamps": speech_timestamps,
        }

        if has_sub:
            res["sub_labels"] = torch.stack(sub_labels)

        return res

    def get_class_weights(self) -> torch.Tensor:
        """Calculate balanced class weights based on label frequencies."""
        num_classes = len(PrimaryEmotion)
        counts = [0] * num_classes
        for s in self.samples:
            counts[s.primary_label.value] += 1

        total = len(self.samples)
        weights = torch.zeros(num_classes, dtype=torch.float32)
        for c in range(num_classes):
            if counts[c] > 0:
                weights[c] = total / (num_classes * counts[c])
            else:
                weights[c] = 1.0

        return weights

    def get_split_indices(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Generate stratified or deterministic train / val / test indices."""
        rng = random.Random(seed)
        indices = list(range(len(self.samples)))
        rng.shuffle(indices)

        n = len(indices)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        return train_idx, val_idx, test_idx
