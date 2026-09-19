"""
vad_preprocessor.py

Silero-VAD based preprocessor with sliding-window chunking.

Pipeline:
    Raw waveform → resample to 16 kHz → Silero-VAD → speech segments
    → sliding window (6.0 s window, 3.0 s stride, 50 % overlap) → AudioChunks
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torchaudio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AudioChunk:
    """One chunk produced by the sliding-window segmenter.

    Attributes:
        waveform:      ``(1, chunk_samples)`` mono tensor.
        sample_rate:   Always 16 000 after resampling.
        start_sample:  Absolute sample offset in the source recording.
        end_sample:    Absolute end sample in the source recording.
        start_time_s:  Start time in seconds.
        end_time_s:    End time in seconds.
        chunk_id:      Sequential index.
        source_path:   Optional path of the originating file.
    """
    waveform: torch.Tensor
    sample_rate: int
    start_sample: int
    end_sample: int
    start_time_s: float
    end_time_s: float
    chunk_id: int
    source_path: Optional[str] = None

    def duration_s(self) -> float:
        return (self.end_sample - self.start_sample) / self.sample_rate


@dataclass
class VADResult:
    """Output of the full VAD + chunking pipeline.

    Attributes:
        speech_timestamps: ``[{'start': int, 'end': int}, ...]`` in sample indices.
        chunks:            List of ``AudioChunk`` objects.
        total_speech_ratio: Fraction of the recording that is speech.
        source_path:        Optional originating file path.
    """
    speech_timestamps: List[dict] = field(default_factory=list)
    chunks: List[AudioChunk] = field(default_factory=list)
    total_speech_ratio: float = 0.0
    source_path: Optional[str] = None


# ---------------------------------------------------------------------------
# VADPreprocessor
# ---------------------------------------------------------------------------

class VADPreprocessor:
    """Silero-VAD wrapper with sliding-window chunking.

    Args:
        sample_rate:            Target sample rate.  Default 16 000.
        chunk_duration_s:       Window length in seconds.  Default 6.0.
        stride_duration_s:      Stride in seconds (50 % overlap → 3.0).  Default 3.0.
        vad_threshold:          Speech probability threshold.  Default 0.5.
        min_speech_duration_ms: Minimum speech region length (ms).  Default 250.
        speech_pad_ms:          Padding around each region (ms).  Default 100.
        device:                 ``"cpu"`` or ``"cuda"``.
    """

    TARGET_SAMPLE_RATE: int = 16000

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration_s: float = 6.0,
        stride_duration_s: float = 3.0,
        vad_threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        speech_pad_ms: int = 100,
        device: str = "cpu",
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_duration_s = chunk_duration_s
        self.stride_duration_s = stride_duration_s
        self.vad_threshold = vad_threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.device = torch.device(device)
        self._vad_model: Optional[torch.nn.Module] = None
        self._vad_utils = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load Silero-VAD from ``torch.hub`` (lazy, cached)."""
        if self._vad_model is not None:
            return
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        self._vad_model = model.to(self.device)
        self._vad_utils = utils
        logger.info("Silero-VAD loaded on %s", self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, audio_path: Union[str, Path]) -> VADResult:
        """Load a ``.wav`` file and run VAD + chunking."""
        waveform, sr = self._load_audio(audio_path)
        result = self.process_waveform(waveform, sr)
        result.source_path = str(audio_path)
        for c in result.chunks:
            c.source_path = str(audio_path)
        return result

    def process_waveform(self, waveform: torch.Tensor, sample_rate: int) -> VADResult:
        """Process an in-memory waveform tensor."""
        if self._vad_model is None:
            self.load_model()

        # Resample if needed
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)

        # Ensure (1, T) shape
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.size(0) > 1:
            waveform = waveform[:1]  # mono

        speech_ts = self._run_vad(waveform)
        total_samples = waveform.size(1)
        speech_samples = sum(s["end"] - s["start"] for s in speech_ts)
        ratio = speech_samples / max(total_samples, 1)

        chunks = self._create_chunks(waveform, speech_ts)
        return VADResult(
            speech_timestamps=speech_ts,
            chunks=chunks,
            total_speech_ratio=ratio,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_audio(self, audio_path: Union[str, Path]) -> Tuple[torch.Tensor, int]:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        waveform, sr = torchaudio.load(str(path))
        return waveform, sr

    def _run_vad(self, waveform: torch.Tensor) -> List[dict]:
        """Run Silero-VAD and return speech timestamps in sample indices."""
        get_speech_timestamps = self._vad_utils[0]
        wav = waveform.squeeze(0).to(self.device)
        timestamps = get_speech_timestamps(
            wav,
            self._vad_model,
            threshold=self.vad_threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
            sampling_rate=self.sample_rate,
        )
        return [{"start": int(t["start"]), "end": int(t["end"])} for t in timestamps]

    def _create_chunks(
        self,
        waveform: torch.Tensor,
        speech_timestamps: List[dict],
        source_path: Optional[str] = None,
    ) -> List[AudioChunk]:
        """Sliding-window chunking over detected speech regions."""
        if not speech_timestamps:
            return []

        sr = self.sample_rate
        win = int(self.chunk_duration_s * sr)
        stride = int(self.stride_duration_s * sr)
        total = waveform.size(1)

        # Merge speech regions into a single contiguous span for chunking
        global_start = speech_timestamps[0]["start"]
        global_end = speech_timestamps[-1]["end"]

        chunks: List[AudioChunk] = []
        idx = 0
        pos = global_start
        while pos < global_end:
            end = min(pos + win, total)
            seg = waveform[:, pos:end]
            # Zero-pad if shorter than window
            if seg.size(1) < win:
                seg = torch.nn.functional.pad(seg, (0, win - seg.size(1)))
            chunks.append(
                AudioChunk(
                    waveform=seg,
                    sample_rate=sr,
                    start_sample=pos,
                    end_sample=min(pos + win, total),
                    start_time_s=pos / sr,
                    end_time_s=min(pos + win, total) / sr,
                    chunk_id=idx,
                    source_path=source_path,
                )
            )
            idx += 1
            pos += stride
            if end >= total:
                break

        logger.info("Created %d chunks (%.1fs window, %.1fs stride)", len(chunks), self.chunk_duration_s, self.stride_duration_s)
        return chunks
