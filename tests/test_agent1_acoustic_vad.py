import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import pytest
import torch
from src.audio.audio_dataset import extract_speech_timestamps, VoiceJournalDataset
from src.stage2.melody_extractor import ExplicitMelodyExtractor


def test_extract_speech_timestamps_silence():
    # Test all-zeros (silent)
    silence = torch.zeros(1, 32000)
    t0 = time.perf_counter()
    ts_silence = extract_speech_timestamps(silence, sample_rate=16000)
    dt = (time.perf_counter() - t0) * 1000
    print(f"Silence (2s) -> ts: {ts_silence}, time: {dt:.2f}ms")
    assert ts_silence == [], "Expected empty timestamps for silence"


def test_extract_speech_timestamps_short():
    # Test extremely short
    short_wav = torch.zeros(1, 50)
    ts_short = extract_speech_timestamps(short_wav, sample_rate=16000)
    assert ts_short == [], "Expected empty timestamps for short audio"


def test_extract_speech_timestamps_synth():
    # Test synthetic speech (burst of sine wave)
    t = torch.linspace(0, 3, 48000)
    synth = torch.zeros(1, 48000)
    # Add speech in middle: 1s to 2.5s (16000 to 40000)
    synth[0, 16000:40000] = 0.3 * torch.sin(2 * 3.14159 * 200 * t[16000:40000])
    t0 = time.perf_counter()
    ts_synth = extract_speech_timestamps(synth, sample_rate=16000)
    dt = (time.perf_counter() - t0) * 1000
    print(f"Synthetic speech -> intervals: {len(ts_synth)}, ts: {ts_synth}, time: {dt:.2f}ms")
    assert len(ts_synth) > 0, "Expected detected speech"
    for seg in ts_synth:
        assert "start" in seg and "end" in seg, "Schema must have start and end"
        assert isinstance(seg["start"], int) and isinstance(seg["end"], int), "Bounds must be int"
        assert 0 <= seg["start"] < seg["end"] <= 48000, f"Invalid bounds: {seg}"


def test_voice_journal_dataset():
    ds = VoiceJournalDataset(dummy_mode=True, dummy_size=4, dummy_duration_s=3.0)
    item0 = ds[0]
    assert "speech_timestamps" in item0, "Item must contain speech_timestamps"
    assert isinstance(item0["speech_timestamps"], list), "speech_timestamps must be list"

    batch = ds.collate_fn([ds[0], ds[1]])
    assert "speech_timestamps" in batch, "Batch must contain speech_timestamps"
    assert isinstance(batch["speech_timestamps"], list) and len(batch["speech_timestamps"]) == 2
    for ts_list in batch["speech_timestamps"]:
        assert isinstance(ts_list, list)


def test_explicit_melody_extractor():
    extractor = ExplicitMelodyExtractor(sample_rate=16000)

    # Batch test
    x = torch.randn(2, 32000)
    feats = extractor(x)
    assert feats.shape == (2, 32), f"Expected (2, 32), got {feats.shape}"
    assert not torch.isnan(feats).any(), "NaNs detected in melody features"

    # All zeros test
    feats_zero = extractor(torch.zeros(2, 16000))
    assert feats_zero.shape == (2, 32), f"Expected (2, 32), got {feats_zero.shape}"
    assert not torch.isnan(feats_zero).any(), "NaNs detected in zero features"

    # Very short test
    feats_short = extractor(torch.randn(1, 300))
    assert feats_short.shape == (1, 32), f"Expected (1, 32), got {feats_short.shape}"
    assert not torch.isnan(feats_short).any(), "NaNs detected in short features"


def test_f0_median_filter_smoothness():
    extractor = ExplicitMelodyExtractor(sample_rate=16000)
    # Generate 1.0 second of harmonic signal at 150 Hz with an abrupt pitch jump to 300 Hz for 20ms
    sr = 16000
    t = torch.linspace(0, 1.0, sr)
    # 150 Hz base tone
    wav = 0.5 * torch.sin(2 * 3.14159 * 150 * t)
    # Insert high pitch spike around 0.5s (8000 to 8320)
    wav[8000:8320] = 0.5 * torch.sin(2 * 3.14159 * 400 * t[8000:8320])

    feats, raw_dict = extractor._extract_single_with_metrics(wav, torch.device("cpu"))
    assert feats.shape == (32,)
    assert not torch.isnan(feats).any()
    # The smoothed f0_mean should stay close to 150 Hz (not dragged up wildly by 400 Hz spike)
    assert 130.0 <= raw_dict["f0_mean"] <= 180.0, f"f0_mean {raw_dict['f0_mean']} outside expected range"


if __name__ == "__main__":
    test_extract_speech_timestamps_silence()
    test_extract_speech_timestamps_short()
    test_extract_speech_timestamps_synth()
    test_voice_journal_dataset()
    test_explicit_melody_extractor()
    test_f0_median_filter_smoothness()
    print("ALL AGENT 1 TESTS PASSED SUCCESSFULLY!")
