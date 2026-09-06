"""
pause_analyzer.py

Extracts the 4-dimensional paralinguistic pause vector from speech:

    P_pause = [SPR, MeanPauseDuration, PauseFrequency, EnergyDrift]  in R^4

Components:
    [0] SPR  — Speech-to-Pause Ratio = t_speech / t_total
    [1] MeanPauseDuration — mean duration of detected pauses (seconds)
    [2] PauseFrequency    — number of pauses per minute
    [3] EnergyDrift       — ratio of RMS energy in the second half to the first half
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PauseFeatures:
    """Container for extracted pause features.

    Attributes:
        speech_pause_ratio:  SPR in [0, 1].
        mean_pause_duration: Mean pause length in seconds.
        pause_frequency:     Pauses per minute.
        energy_drift:        RMS energy ratio (second half / first half).
        pause_intervals_s:   Individual pause durations in seconds.
    """

    speech_pause_ratio: float = 0.0
    mean_pause_duration: float = 0.0
    pause_frequency: float = 0.0
    energy_drift: float = 1.0
    pause_intervals_s: List[float] = field(default_factory=list)

    def __init__(
        self,
        speech_pause_ratio: float = 0.0,
        mean_pause_duration: Optional[float] = None,
        pause_frequency: float = 0.0,
        energy_drift: Optional[float] = None,
        pause_intervals_s: Optional[List[float]] = None,
        mean_pause_duration_s: Optional[float] = None,
        energy_drift_db: Optional[float] = None,
    ) -> None:
        self.speech_pause_ratio = speech_pause_ratio
        if mean_pause_duration is not None:
            self.mean_pause_duration = mean_pause_duration
        elif mean_pause_duration_s is not None:
            self.mean_pause_duration = mean_pause_duration_s
        else:
            self.mean_pause_duration = 0.0

        self.pause_frequency = pause_frequency

        if energy_drift is not None:
            self.energy_drift = energy_drift
        elif energy_drift_db is not None:
            self.energy_drift = energy_drift_db
        else:
            self.energy_drift = 1.0

        self.pause_intervals_s = pause_intervals_s if pause_intervals_s is not None else []

    def to_tensor(self) -> torch.Tensor:
        """Return P_pause as a ``(4,)`` float32 tensor."""
        return torch.tensor(
            [
                self.speech_pause_ratio,
                self.mean_pause_duration,
                self.pause_frequency,
                self.energy_drift,
            ],
            dtype=torch.float32,
        )


class EmotionalPauseAnalyzer:
    """Extracts paralinguistic pause features from a waveform + VAD timestamps.

    Args:
        sample_rate:           Audio sample rate. Default 16000.
        silence_threshold_db:  dB threshold below which a region is silence. Default -40.
        min_pause_duration_ms: Minimum gap duration (ms) to count as a pause. Default 200.
    """

    FEATURE_DIM: int = 4

    def __init__(
        self,
        sample_rate: int = 16000,
        silence_threshold_db: float = -40.0,
        min_pause_duration_ms: int = 200,
    ) -> None:
        self.sample_rate = sample_rate
        self.silence_threshold_db = silence_threshold_db
        self.min_pause_samples = int(min_pause_duration_ms * sample_rate / 1000)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        waveform: torch.Tensor,
        speech_timestamps: List[dict],
    ) -> PauseFeatures:
        """Analyze a single waveform.

        Args:
            waveform: ``(1, T)`` or ``(T,)`` at ``self.sample_rate``.
            speech_timestamps: ``[{'start': int, 'end': int}, ...]`` in samples.

        Returns:
            PauseFeatures with all four components populated.
        """
        wav = waveform.squeeze()
        total_samples = wav.numel()
        total_duration_s = total_samples / self.sample_rate

        if total_duration_s <= 0 or len(speech_timestamps) == 0:
            return PauseFeatures()

        # --- SPR ---
        speech_samples = sum(seg["end"] - seg["start"] for seg in speech_timestamps)
        spr = min(float(speech_samples) / total_samples, 1.0)

        # --- Pause intervals ---
        sorted_ts = sorted(speech_timestamps, key=lambda s: s["start"])
        pause_intervals: List[float] = []

        # Gap before first speech
        if sorted_ts[0]["start"] > self.min_pause_samples:
            pause_intervals.append(sorted_ts[0]["start"] / self.sample_rate)

        # Gaps between consecutive speech segments
        for i in range(len(sorted_ts) - 1):
            gap = sorted_ts[i + 1]["start"] - sorted_ts[i]["end"]
            if gap >= self.min_pause_samples:
                pause_intervals.append(gap / self.sample_rate)

        # Gap after last speech
        tail_gap = total_samples - sorted_ts[-1]["end"]
        if tail_gap >= self.min_pause_samples:
            pause_intervals.append(tail_gap / self.sample_rate)

        mean_dur = float(np.mean(pause_intervals)) if pause_intervals else 0.0
        freq = len(pause_intervals) / (total_duration_s / 60.0) if total_duration_s > 0 else 0.0

        # --- Energy drift ---
        mid = total_samples // 2
        e_first = (wav[:mid] ** 2).mean().clamp(min=1e-9)
        e_second = (wav[mid:] ** 2).mean().clamp(min=1e-9)
        drift = float((e_second / e_first).sqrt().item())

        return PauseFeatures(
            speech_pause_ratio=spr,
            mean_pause_duration=mean_dur,
            pause_frequency=freq,
            energy_drift=drift,
            pause_intervals_s=pause_intervals,
        )

    def analyze_from_timestamps(
        self,
        speech_timestamps: List[dict],
        total_duration_s: float = 6.0,
    ) -> PauseFeatures:
        """Compute pause features directly from speech timestamps.

        Args:
            speech_timestamps: List of [{'start': int, 'end': int}, ...].
            total_duration_s: Expected total duration in seconds. Default: 6.0.

        Returns:
            PauseFeatures: Computed pause features.
        """
        if not speech_timestamps:
            return PauseFeatures(
                speech_pause_ratio=0.8,
                mean_pause_duration=0.5,
                pause_frequency=6.0,
                energy_drift=1.0,
            )

        total_samples = int(total_duration_s * self.sample_rate)
        speech_samples = sum(seg.get("end", 0) - seg.get("start", 0) for seg in speech_timestamps)
        spr = min(float(speech_samples) / max(1, total_samples), 1.0)

        sorted_ts = sorted(speech_timestamps, key=lambda s: s.get("start", 0))
        pause_intervals: List[float] = []

        if sorted_ts and sorted_ts[0].get("start", 0) > self.min_pause_samples:
            pause_intervals.append(sorted_ts[0]["start"] / self.sample_rate)

        for i in range(len(sorted_ts) - 1):
            gap = sorted_ts[i + 1].get("start", 0) - sorted_ts[i].get("end", 0)
            if gap >= self.min_pause_samples:
                pause_intervals.append(gap / self.sample_rate)

        mean_dur = float(np.mean(pause_intervals)) if pause_intervals else 0.0
        freq = len(pause_intervals) / (total_duration_s / 60.0) if total_duration_s > 0 else 0.0

        return PauseFeatures(
            speech_pause_ratio=spr,
            mean_pause_duration=mean_dur,
            pause_frequency=freq,
            energy_drift=1.0,
            pause_intervals_s=pause_intervals,
        )

    def batch_analyze(
        self,
        waveforms: List[torch.Tensor],
        speech_timestamps_list: List[List[dict]],
    ) -> torch.Tensor:
        """Batch version returning ``(B, 4)`` tensor."""
        feats = [
            self.analyze(w, ts).to_tensor()
            for w, ts in zip(waveforms, speech_timestamps_list)
        ]
        return torch.stack(feats, dim=0)

    # ------------------------------------------------------------------
    @staticmethod
    def _rms(wav: torch.Tensor) -> float:
        """Root-mean-square energy of a 1-D tensor."""
        if wav.numel() == 0:
            return 0.0
        return float(torch.sqrt(torch.mean(wav.float() ** 2) + 1e-10))
