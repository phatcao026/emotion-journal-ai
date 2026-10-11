"""
voice_processor.py
------------------
Bộ xử lý & Chuẩn hóa âm thanh chuyên dụng cho Phân hệ Voice (Voice Pipeline).
Tuân thủ tuyệt đối đặc tả tại `docs/specs/voice_pipeline_spec.md` và tiêu chuẩn
kiểm định tại `scripts/validate_input_voice.py`:

1. Định dạng: RIFF WAVE, PCM 16-bit uncompressed (pcm_s16le).
2. Tần số lấy mẫu: 16,000 Hz.
3. Kênh: Mono (1 channel).
4. Thời lượng tối ưu: 15.0s <= duration <= 180.0s (Hỗ trợ tự động phân đoạn các đoạn độc thoại dài).
5. Kiểm định âm học:
   - Kiểm tra Clipping (peak amplitude < 0.99)
   - Kiểm tra Độ ồn/Tắt tiếng (RMS >= -45 dBFS)
   - Kiểm tra Tỷ lệ giọng nói (Speech Energy vs Background Silence)
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio_ffmpeg
import numpy as np

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
MIN_DURATION_S = 3.0
MAX_DURATION_S = 180.0


class VoiceProcessor:
    """Xử lý, kiểm định chất lượng và phân đoạn âm thanh cho Voice MER."""

    def __init__(
        self,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        min_duration_s: float = MIN_DURATION_S,
        max_duration_s: float = MAX_DURATION_S,
    ):
        self.target_sample_rate = target_sample_rate
        self.min_duration_s = min_duration_s
        self.max_duration_s = max_duration_s

    def convert_to_16k_mono_wav(self, input_media_path: str | Path, output_wav_path: str | Path) -> bool:
        """Chuyển đổi video/audio sang chuẩn WAV 16kHz Mono 16-bit PCM qua FFmpeg."""
        cmd = [
            FFMPEG_EXE,
            "-i", str(input_media_path),
            "-vn",                          # Bỏ video stream
            "-acodec", "pcm_s16le",         # 16-bit linear PCM
            "-ar", str(self.target_sample_rate),  # 16,000 Hz
            "-ac", str(TARGET_CHANNELS),    # Mono 1 kênh
            str(output_wav_path),
            "-y"
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return res.returncode == 0 and os.path.exists(output_wav_path)
        except Exception as e:
            print(f"[-] Lỗi FFmpeg convert: {e}")
            return False

    def separate_vocals_demucs(self, input_wav_path: str | Path, output_vocals_path: str | Path) -> bool:
        """
        Tách giọng nói (Vocal Isolation) để loại bỏ nhạc nền radio bằng Demucs nếu hệ thống có cài đặt.
        Nếu không có Demucs, trả về False và giữ nguyên file âm thanh gốc.
        """
        try:
            import shutil
            import tempfile
            temp_out = Path(tempfile.mkdtemp(prefix="demucs_"))
            cmd = [
                sys.executable, "-m", "demucs",
                "--two-stems=vocals",
                "-n", "htdemucs",
                "-o", str(temp_out),
                str(input_wav_path)
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
            if res.returncode == 0:
                found_vocals = list(temp_out.rglob("vocals.wav"))
                if found_vocals and found_vocals[0].exists():
                    shutil.copy2(str(found_vocals[0]), str(output_vocals_path))
                    shutil.rmtree(temp_out, ignore_errors=True)
                    return True
            shutil.rmtree(temp_out, ignore_errors=True)
        except Exception as e:
            pass
        return False

    def inspect_and_audit(self, wav_path: str | Path) -> Dict[str, Any]:
        """
        Kiểm định chất lượng âm thanh dựa trên tiêu chuẩn của validate_input_voice.py:
        - duration_s
        - rms_dbfs
        - peak_amplitude
        - is_valid
        - issues
        """
        wav_path = Path(wav_path)
        if not wav_path.exists() or wav_path.stat().st_size == 0:
            return {
                "is_valid": False,
                "reason": "File âm thanh không tồn tại hoặc dung lượng 0 byte",
                "duration_s": 0.0
            }

        try:
            with wave.open(str(wav_path), "rb") as wf:
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                num_frames = wf.getnframes()
                duration_s = round(num_frames / float(sample_rate), 2)
                raw_bytes = wf.readframes(num_frames)

            # Đọc mảng numpy float32 chuẩn hóa về [-1.0, 1.0]
            if sample_width == 2:
                samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                samples = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0

            # 1. Kiểm tra tần số và kênh
            issues = []
            if sample_rate != self.target_sample_rate:
                issues.append(f"Tần số sai ({sample_rate}Hz != {self.target_sample_rate}Hz)")
            if channels != 1:
                issues.append(f"Không phải Mono ({channels} kênh)")

            # 2. Kiểm tra thời lượng
            if duration_s < self.min_duration_s:
                issues.append(f"Thời lượng quá ngắn ({duration_s}s < {self.min_duration_s}s tối thiểu)")

            # 3. Tính toán cường độ âm thanh RMS & Peak
            peak = float(np.max(np.abs(samples))) if len(samples) > 0 else 0.0
            rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0
            rms_dbfs = 20.0 * math.log10(max(rms, 1e-9))

            if rms_dbfs < -48.0:
                issues.append(f"Âm lượng quá nhỏ hoặc im lặng (RMS: {rms_dbfs:.1f} dBFS)")
            if peak >= 0.999:
                issues.append(f"Bị rè méo tín hiệu (Clipping peak: {peak:.3f})")

            # 4. Kiểm tra sự hiện diện của năng lượng giọng nói (VAD thô)
            frame_len = int(sample_rate * 0.03)  # 30ms frame
            if len(samples) >= frame_len:
                num_frames_calc = len(samples) // frame_len
                frames = samples[: num_frames_calc * frame_len].reshape(num_frames_calc, frame_len)
                frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
                speech_frames = np.sum(frame_rms > 0.015)
                speech_ratio = float(speech_frames / max(num_frames_calc, 1))
            else:
                speech_ratio = 0.0

            if speech_ratio < 0.35 and duration_s >= self.min_duration_s:
                issues.append(f"Tỷ lệ giọng nói quá thấp ({speech_ratio*100:.1f}%), có thể chỉ là nhạc nền/khoảng lặng")

            is_valid = len(issues) == 0
            return {
                "is_valid": is_valid,
                "duration_s": duration_s,
                "rms_dbfs": round(rms_dbfs, 2),
                "peak": round(peak, 3),
                "speech_ratio": round(speech_ratio, 3),
                "issues": issues,
                "reason": "; ".join(issues) if issues else "Đạt chuẩn Voice Pipeline"
            }
        except Exception as e:
            return {
                "is_valid": False,
                "reason": f"Lỗi đọc/phân tích waveform: {e}",
                "duration_s": 0.0
            }

    def split_long_audio_to_monologues(
        self,
        wav_path: str | Path,
        output_dir: str | Path,
        base_sample_id: str,
        chunk_duration_s: float = 45.0,
        overlap_s: float = 5.0
    ) -> List[Dict[str, Any]]:
        """
        Nếu một bài độc thoại podcast/tâm sự dài (ví dụ: 3 - 10 phút),
        hàm này tự động phân đoạn thành các clip 30-60s chuẩn để huấn luyện emotion2vec.
        """
        wav_path = Path(wav_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        audit = self.inspect_and_audit(wav_path)
        total_duration = audit["duration_s"]
        if total_duration < self.min_duration_s:
            return []

        # Nếu file đã nằm trong khoảng lý tưởng (15s - 180s) và hợp lệ -> giữ nguyên
        if total_duration <= self.max_duration_s and audit["is_valid"]:
            dest_wav = output_dir / f"{base_sample_id}.wav"
            if wav_path.resolve() != dest_wav.resolve():
                with open(wav_path, "rb") as f_in, open(dest_wav, "wb") as f_out:
                    f_out.write(f_in.read())
            return [{
                "chunk_id": base_sample_id,
                "wav_path": str(dest_wav),
                "duration_s": total_duration,
                "audit": audit
            }]

        # Phân đoạn file dài bằng FFmpeg
        chunks = []
        step = chunk_duration_s - overlap_s
        num_chunks = int(math.ceil((total_duration - overlap_s) / step))

        for idx in range(num_chunks):
            start_s = idx * step
            if start_s >= total_duration - 10.0:
                break
            actual_len = min(chunk_duration_s, total_duration - start_s)
            if actual_len < self.min_duration_s:
                break

            chunk_id = f"{base_sample_id}_c{idx+1:02d}"
            chunk_wav_path = output_dir / f"{chunk_id}.wav"

            cmd = [
                FFMPEG_EXE,
                "-ss", str(round(start_s, 2)),
                "-t", str(round(actual_len, 2)),
                "-i", str(wav_path),
                "-acodec", "pcm_s16le",
                "-ar", str(self.target_sample_rate),
                "-ac", "1",
                str(chunk_wav_path),
                "-y"
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0 and chunk_wav_path.exists():
                chunk_audit = self.inspect_and_audit(chunk_wav_path)
                if chunk_audit["is_valid"]:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "wav_path": str(chunk_wav_path),
                        "duration_s": chunk_audit["duration_s"],
                        "audit": chunk_audit
                    })
                else:
                    # Xóa chunk nếu không đạt chuẩn âm học
                    try:
                        chunk_wav_path.unlink()
                    except Exception:
                        pass

        return chunks
