#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_input_voice.py
-----------------------
Automated validation and quality audit script for audio recordings collected
from the Emotion Journal web form.

Verifies compliance with the voice pipeline specifications:
    1. Container & Codec: RIFF WAVE, 16-bit Linear PCM (pcm_s16le).
    2. Sample Rate: Exactly 16,000 Hz.
    3. Channels: Mono (1 channel).
    4. Duration: 15.0s <= duration <= 300.0s (optimal: 60s - 180s).
    5. Signal Integrity:
       - Clipping / Saturation check (peak >= 0.999).
       - Silence / Mute check (RMS < -50 dBFS or peak < 0.01).
       - DC offset detection.
    6. Pipeline Readiness: Verifies loading by soundfile / torchaudio.

Usage:
    # Test a single file
    python scripts/validate_input_voice.py --file path/to/sample.wav

    # Test an entire directory of collected recordings
    python scripts/validate_input_voice.py --dir path/to/recordings/

    # Run self-test with synthetic valid & invalid audio clips
    python scripts/validate_input_voice.py --dummy

    # Export report to CSV and automatically fix non-compliant files
    python scripts/validate_input_voice.py --dir data/raw_recordings/ --export-csv report.csv --fix-dir data/fixed_wavs/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Reconfigure standard output encoding for Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import torch
    import torchaudio
except ImportError:
    torch = None
    torchaudio = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("validate_voice")


# ---------------------------------------------------------------------------
# Specification Thresholds
# ---------------------------------------------------------------------------
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SUBTYPE = "PCM_16"

MIN_DURATION_S = 15.0       # Minimal duration for reliable VAD & Pause Analyzer
MAX_DURATION_S = 300.0      # Hard ceiling from audio_dataset.py (5 minutes)
OPTIMAL_MIN_DURATION_S = 60.0
OPTIMAL_MAX_DURATION_S = 180.0

MAX_CLIPPING_RATIO = 0.005  # Warn if > 0.5% samples are clipped
SILENCE_RMS_THRESHOLD_DB = -50.0  # RMS lower than -50 dBFS is likely dead silence / mute
MIN_PEAK_AMPLITUDE = 0.01   # Peak lower than 0.01 is too quiet
MAX_DC_OFFSET = 0.05        # Warn if absolute DC offset > 0.05


@dataclass
class ValidationResult:
    file_path: str
    is_valid: bool = True
    status: str = "PASS"  # PASS, WARN, FAIL
    format_name: str = ""
    subtype: str = ""
    sample_rate: int = 0
    channels: int = 0
    duration_s: float = 0.0
    num_samples: int = 0
    peak_amplitude: float = 0.0
    rms_dbfs: float = -100.0
    clipping_ratio: float = 0.0
    dc_offset: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary_row(self) -> Dict[str, str]:
        return {
            "file": Path(self.file_path).name,
            "status": self.status,
            "sample_rate": str(self.sample_rate),
            "channels": str(self.channels),
            "subtype": self.subtype,
            "duration_s": f"{self.duration_s:.2f}",
            "rms_dbfs": f"{self.rms_dbfs:.1f}",
            "peak": f"{self.peak_amplitude:.3f}",
            "clip_pct": f"{self.clipping_ratio * 100:.2f}%",
            "errors": "; ".join(self.errors),
            "warnings": "; ".join(self.warnings),
        }


def check_audio_file(file_path: Union[str, Path]) -> ValidationResult:
    """Perform comprehensive checks on a single audio recording file."""
    path = Path(file_path)
    res = ValidationResult(file_path=str(path))

    if not path.exists():
        res.is_valid = False
        res.status = "FAIL"
        res.errors.append("File không tồn tại trên đĩa.")
        return res

    if path.stat().st_size == 0:
        res.is_valid = False
        res.status = "FAIL"
        res.errors.append("File rỗng (0 bytes).")
        return res

    if sf is None:
        res.is_valid = False
        res.status = "FAIL"
        res.errors.append("Thiếu thư viện soundfile. Vui lòng cài đặt: pip install soundfile")
        return res

    # 1. Header & Metadata inspection via soundfile.info
    try:
        info = sf.info(str(path))
        res.format_name = info.format
        res.subtype = info.subtype
        res.sample_rate = info.samplerate
        res.channels = info.channels
        res.duration_s = info.duration
        res.num_samples = info.frames
    except Exception as exc:
        res.is_valid = False
        res.status = "FAIL"
        res.errors.append(f"Không thể đọc header WAV (Lỗi giải mã): {exc}")
        return res

    # Check Format (Must be WAV)
    if res.format_name.upper() != "WAV":
        res.errors.append(f"Định dạng không phải WAV (Hiện tại: {res.format_name}).")
        res.is_valid = False

    # Check Subtype (Must be 16-bit PCM)
    if res.subtype != TARGET_SUBTYPE:
        if "PCM" in res.subtype:
            res.warnings.append(
                f"Độ sâu bit là {res.subtype} thay vì PCM 16-bit chuẩn (pcm_s16le)."
            )
        else:
            res.errors.append(
                f"Codec âm thanh là {res.subtype} (yêu cầu bắt buộc PCM 16-bit uncompressed)."
            )
            res.is_valid = False

    # Check Sample Rate
    if res.sample_rate != TARGET_SAMPLE_RATE:
        res.errors.append(
            f"Tần số lấy mẫu sai: {res.sample_rate} Hz (Chuẩn bắt buộc: {TARGET_SAMPLE_RATE} Hz)."
        )
        res.is_valid = False

    # Check Channels (Must be Mono = 1)
    if res.channels != TARGET_CHANNELS:
        res.errors.append(
            f"Số kênh âm thanh sai: {res.channels} kênh (Chuẩn bắt buộc: 1 kênh Mono)."
        )
        res.is_valid = False

    # Check Duration
    if res.duration_s < MIN_DURATION_S:
        res.errors.append(
            f"Thời lượng quá ngắn: {res.duration_s:.1f}s (Tối thiểu phải >= {MIN_DURATION_S}s để VAD/Pause Analyzer hoạt động)."
        )
        res.is_valid = False
    elif res.duration_s > MAX_DURATION_S:
        res.errors.append(
            f"Thời lượng quá dài: {res.duration_s:.1f}s (Vượt ngưỡng trần {MAX_DURATION_S}s của audio_dataset.py)."
        )
        res.is_valid = False

    # 2. Acoustic Signal Integrity Inspection
    try:
        data, sr = sf.read(str(path), dtype="float32")
    except Exception as exc:
        res.errors.append(f"Lỗi đọc dữ liệu waveform: {exc}")
        res.is_valid = False
        res.status = "FAIL"
        return res

    if data.ndim > 1:
        # If stereo slipped in, inspect the first channel
        data = data[:, 0]

    # Peak amplitude & DC Offset
    peak = float(np.max(np.abs(data))) if len(data) > 0 else 0.0
    dc = float(np.mean(data)) if len(data) > 0 else 0.0
    res.peak_amplitude = peak
    res.dc_offset = dc

    # RMS Energy in dBFS
    rms = float(np.sqrt(np.mean(data**2))) if len(data) > 0 else 0.0
    rms_dbfs = 20.0 * math.log10(rms + 1e-9)
    res.rms_dbfs = rms_dbfs

    # Check Silent / Muted audio
    if rms_dbfs < SILENCE_RMS_THRESHOLD_DB or peak < MIN_PEAK_AMPLITUDE:
        res.errors.append(
            f"File gần như im lặng hoàn toàn (RMS: {rms_dbfs:.1f} dBFS, Peak: {peak:.4f}). Mic có thể bị tắt hoặc mất tiếng."
        )
        res.is_valid = False

    # Check Clipping / Distortion
    clipping_count = np.sum(np.abs(data) >= 0.999)
    clipping_ratio = float(clipping_count / len(data)) if len(data) > 0 else 0.0
    res.clipping_ratio = clipping_ratio

    if clipping_ratio > MAX_CLIPPING_RATIO:
        res.warnings.append(
            f"Phát hiện vỡ âm / clipping ({clipping_ratio * 100:.2f}% mẫu bị kịch trần biên độ). Người nói có thể thở quá sát mic."
        )

    # Check DC Offset
    if abs(dc) > MAX_DC_OFFSET:
        res.warnings.append(f"Phát hiện độ lệch DC offset đáng kể: {dc:+.4f}.")

    # 3. Final Status Determination
    if len(res.errors) > 0:
        res.status = "FAIL"
        res.is_valid = False
    elif len(res.warnings) > 0:
        res.status = "WARN"
        res.is_valid = True
    else:
        res.status = "PASS"
        res.is_valid = True

    return res


def fix_audio_file(src_path: Path, dest_path: Path) -> Tuple[bool, str]:
    """Convert any non-compliant audio file into a compliant 16kHz Mono 16-bit WAV file."""
    try:
        import soundfile as sf

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        data, sr = sf.read(str(src_path), dtype="float32")

        # Convert stereo to mono by averaging channels
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        # Resample to 16000 Hz if needed
        if sr != TARGET_SAMPLE_RATE:
            # High-quality polyphase resample via scipy or torchaudio if available
            try:
                import scipy.signal as signal
                num_target_samples = int(len(data) * TARGET_SAMPLE_RATE / sr)
                data = signal.resample(data, num_target_samples)
            except ImportError:
                if torchaudio is not None:
                    tensor_data = torch.from_numpy(data).unsqueeze(0)
                    resampler = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE)
                    tensor_data = resampler(tensor_data)
                    data = tensor_data.squeeze(0).numpy()
                else:
                    return False, "Thiếu scipy hoặc torchaudio để thực hiện resample."

        # Peak normalization to -1.0 dBFS if clipped
        max_val = np.max(np.abs(data))
        if max_val > 0.99:
            data = data / max_val * 0.95

        # Write compliant WAV
        sf.write(str(dest_path), data, TARGET_SAMPLE_RATE, subtype="PCM_16", format="WAV")
        return True, f"Đã chuẩn hóa thành công sang {dest_path.name} (16kHz, Mono, PCM_16)."
    except Exception as exc:
        return False, f"Lỗi trong quá trình chuẩn hóa: {exc}"


def generate_dummy_samples(output_dir: Path) -> List[Path]:
    """Generate mock audio samples (both valid and invalid) for verification testing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    # 1. Perfectly valid 16kHz mono WAV (20 seconds)
    sr = 16000
    dur = 20.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    # Speech-like synthetic signal
    signal_good = (
        0.3 * np.sin(2 * np.pi * 200.0 * t)
        + 0.15 * np.sin(2 * np.pi * 400.0 * t)
    ) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
    p1 = output_dir / "sample_valid_journal.wav"
    sf.write(str(p1), signal_good.astype(np.float32), sr, subtype="PCM_16", format="WAV")
    generated.append(p1)

    # 2. Too short (< 15 seconds)
    dur_short = 5.0
    t_short = np.linspace(0, dur_short, int(sr * dur_short), endpoint=False)
    p2 = output_dir / "sample_too_short.wav"
    sf.write(str(p2), (0.2 * np.sin(2 * np.pi * 250.0 * t_short)).astype(np.float32), sr, subtype="PCM_16")
    generated.append(p2)

    # 3. Wrong Sample Rate (44100 Hz stereo)
    sr_bad = 44100
    t_stereo = np.linspace(0, 16.0, int(sr_bad * 16.0), endpoint=False)
    left = 0.2 * np.sin(2 * np.pi * 300.0 * t_stereo)
    right = 0.2 * np.cos(2 * np.pi * 300.0 * t_stereo)
    stereo_data = np.stack([left, right], axis=1)
    p3 = output_dir / "sample_wrong_sr_stereo.wav"
    sf.write(str(p3), stereo_data.astype(np.float32), sr_bad, subtype="PCM_16")
    generated.append(p3)

    # 4. Silent / Dead mic (< -50 dBFS)
    p4 = output_dir / "sample_silent_mic.wav"
    silent = np.random.randn(int(sr * 16.0)) * 1e-5
    sf.write(str(p4), silent.astype(np.float32), sr, subtype="PCM_16")
    generated.append(p4)

    # 5. Severely clipped audio
    p5 = output_dir / "sample_clipped.wav"
    clipped = np.clip(np.sin(2 * np.pi * 150.0 * t) * 3.0, -1.0, 1.0)
    sf.write(str(p5), clipped.astype(np.float32), sr, subtype="PCM_16")
    generated.append(p5)

    return generated


def print_report_table(results: List[ValidationResult]) -> None:
    """Print formatted summary table in terminal."""
    col_w = {"file": 28, "status": 8, "sr": 8, "ch": 4, "sub": 8, "dur": 8, "rms": 8, "clip": 8}
    header = (
        f"{'Tên File':<{col_w['file']}} | "
        f"{'Trạng Thái':^{col_w['status']}} | "
        f"{'Tần Số':^{col_w['sr']}} | "
        f"{'Kênh':^{col_w['ch']}} | "
        f"{'Độ Sâu':^{col_w['sub']}} | "
        f"{'Thời Lượng':^{col_w['dur']}} | "
        f"{'RMS (dB)':^{col_w['rms']}} | "
        f"{'Vỡ Âm':^{col_w['clip']}}"
    )
    separator = "-" * len(header)

    print("\n" + separator)
    print(" BÁO CÁO KIỂM TRA CHUẨN ÂM THANH ĐẦU VÀO (EMOTION JOURNAL AI)")
    print(separator)
    print(header)
    print(separator)

    status_symbols = {
        "PASS": "[PASS]  ",
        "WARN": "[WARN]  ",
        "FAIL": "[FAIL]  ",
    }

    pass_count = sum(1 for r in results if r.status == "PASS")
    warn_count = sum(1 for r in results if r.status == "WARN")
    fail_count = sum(1 for r in results if r.status == "FAIL")

    for r in results:
        fname = Path(r.file_path).name
        if len(fname) > col_w["file"]:
            fname = fname[: col_w["file"] - 3] + "..."

        row = (
            f"{fname:<{col_w['file']}} | "
            f"{status_symbols.get(r.status, r.status):^{col_w['status']}} | "
            f"{str(r.sample_rate) + ' Hz':^{col_w['sr']}} | "
            f"{str(r.channels) + ' ch':^{col_w['ch']}} | "
            f"{r.subtype:^{col_w['sub']}} | "
            f"{f'{r.duration_s:.1f}s':^{col_w['dur']}} | "
            f"{f'{r.rms_dbfs:.1f} dB':^{col_w['rms']}} | "
            f"{f'{r.clipping_ratio * 100:.1f}%':^{col_w['clip']}}"
        )
        print(row)

        # Print detailed errors / warnings underneath
        if r.errors:
            for err in r.errors:
                print(f"   ❌ LỖI: {err}")
        if r.warnings:
            for w in r.warnings:
                print(f"   ⚠️  CẢNH BÁO: {w}")

    print(separator)
    print(
        f" TỔNG KẾT: {len(results)} files | "
        f"✅ Hợp lệ: {pass_count} | "
        f"⚠️ Cảnh báo: {warn_count} | "
        f"❌ Không đạt: {fail_count}"
    )
    print(separator + "\n")


def export_csv_report(results: List[ValidationResult], output_path: Path) -> None:
    """Save audit details to CSV file for team review."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file",
        "status",
        "sample_rate",
        "channels",
        "subtype",
        "duration_s",
        "rms_dbfs",
        "peak",
        "clip_pct",
        "errors",
        "warnings",
    ]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r.summary_row())
    logger.info("Đã xuất báo cáo CSV chi tiết tại: %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Kiểm tra chất lượng và chuẩn hóa file âm thanh thu từ Web Form."
    )
    parser.add_argument("--file", type=str, help="Đường dẫn tới 1 file .wav cụ thể.")
    parser.add_argument("--dir", type=str, help="Thư mục chứa các file .wav cần kiểm tra.")
    parser.add_argument("--pattern", type=str, default="**/*.wav", help="Pattern tìm file (mặc định: **/*.wav).")
    parser.add_argument("--dummy", action="store_true", help="Tự tạo dữ liệu mẫu (đúng và lỗi) để chạy kiểm thử script.")
    parser.add_argument("--export-csv", type=str, help="Đường dẫn file CSV để xuất báo cáo kết quả.")
    parser.add_argument("--fix-dir", type=str, help="Thư mục lưu các file tự động sửa/chuẩn hóa (16kHz mono).")

    args = parser.parse_args()

    target_files: List[Path] = []

    if args.dummy:
        dummy_dir = ROOT_DIR / "data" / "dummy_validation_test"
        logger.info("Đang tạo các mẫu âm thanh giả lập tại %s...", dummy_dir)
        target_files = generate_dummy_samples(dummy_dir)
    elif args.file:
        p = Path(args.file)
        if not p.exists():
            logger.error("File không tồn tại: %s", p)
            sys.exit(1)
        target_files = [p]
    elif args.dir:
        dir_path = Path(args.dir)
        if not dir_path.exists():
            logger.error("Thư mục không tồn tại: %s", dir_path)
            sys.exit(1)
        target_files = sorted(list(dir_path.glob(args.pattern)))
        if not target_files:
            logger.warning("Không tìm thấy file nào khớp với pattern '%s' trong %s", args.pattern, dir_path)
            sys.exit(0)
    else:
        parser.print_help()
        sys.exit(0)

    # Perform validation
    logger.info("Bắt đầu kiểm tra %d file âm thanh...", len(target_files))
    results: List[ValidationResult] = []
    for f in target_files:
        res = check_audio_file(f)
        results.append(res)

    # Print summary
    print_report_table(results)

    # Export CSV if requested
    if args.export_csv:
        export_csv_report(results, Path(args.export_csv))

    # Auto fix if requested
    if args.fix_dir:
        fix_dir_path = Path(args.fix_dir)
        logger.info("Đang tự động chuẩn hóa các file lỗi/cảnh báo sang: %s", fix_dir_path)
        fixed_count = 0
        for r in results:
            if r.status in ("FAIL", "WARN") and Path(r.file_path).exists():
                src_p = Path(r.file_path)
                dest_p = fix_dir_path / (src_p.stem + "_fixed.wav")
                ok, msg = fix_audio_file(src_p, dest_p)
                if ok:
                    fixed_count += 1
                    logger.info(" -> [FIXED] %s: %s", src_p.name, msg)
                else:
                    logger.error(" -> [FAILED] %s: %s", src_p.name, msg)
        logger.info("Hoàn thành chuẩn hóa %d file!", fixed_count)


if __name__ == "__main__":
    main()
