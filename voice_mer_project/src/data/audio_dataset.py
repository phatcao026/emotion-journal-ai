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
from typing import Callable, Dict, List, Optional, Tuple, Union

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
    """Sub-category emotion labels (Level 2) - 10 fine-grained classes."""
    GRATITUDE = 0
    PRIDE = 1
    RELIEF = 2
    DISAPPOINTMENT = 3
    REMORSE = 4
    LONELINESS = 5
    NERVOUSNESS = 6
    FEAR = 7
    ANNOYANCE = 8
    REALIZATION = 9


# Mapping from sub-category emotion -> primary emotion
SUB_TO_PRIMARY: Dict[SubEmotion, PrimaryEmotion] = {
    SubEmotion.GRATITUDE: PrimaryEmotion.JOY,
    SubEmotion.PRIDE: PrimaryEmotion.JOY,
    SubEmotion.RELIEF: PrimaryEmotion.JOY,
    SubEmotion.DISAPPOINTMENT: PrimaryEmotion.SADNESS,
    SubEmotion.REMORSE: PrimaryEmotion.SADNESS,
    SubEmotion.LONELINESS: PrimaryEmotion.SADNESS,
    SubEmotion.NERVOUSNESS: PrimaryEmotion.ANXIETY,
    SubEmotion.FEAR: PrimaryEmotion.ANXIETY,
    SubEmotion.ANNOYANCE: PrimaryEmotion.ANGER,
    SubEmotion.REALIZATION: PrimaryEmotion.NEUTRAL,
}


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

            # Generate synthetic speech-like bandpassed noise
            noise = torch.randn(1, num_samples_per_clip) * 0.1
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
                        sub_label=SubEmotion.REALIZATION if self.use_sub_labels else None,
                        sample_rate=self.sample_rate,
                    )
                )
            return

        with open(manifest_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
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

        item: Dict[str, Union[torch.Tensor, int, str]] = {
            "waveform": waveform,  # shape (1, T)
            "primary_label": torch.tensor(sample.primary_label.value, dtype=torch.long),
            "sample_id": sample.sample_id,
            "duration_s": sample.duration_s,
            "transcript": sample.transcript or "",
        }

        if self.use_sub_labels:
            sub_val = sample.sub_label.value if sample.sub_label is not None else 0
            item["sub_label"] = torch.tensor(sub_val, dtype=torch.long)

        return item

    @staticmethod
    def collate_fn(
        batch: List[Dict[str, Union[torch.Tensor, int, str]]],
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

        res: Dict[str, Union[torch.Tensor, List]] = {
            "waveforms": padded_waveforms,
            "waveform_lengths": lengths,
            "primary_labels": torch.stack(primary_labels),
            "sample_ids": sample_ids,
            "transcripts": transcripts,
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
