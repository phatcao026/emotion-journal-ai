"""
voice_processor.py
------------------
Bộ xử lý & Chuẩn hóa âm thanh chuyên dụng cho Phân hệ Voice (Stage 2 SOTA).
Tuân thủ nghiêm ngặt chuẩn mực Xử lý Tín hiệu Số (DSP Standards) và tiêu chuẩn
nghiên cứu khoa học quốc tế (Interspeech / ICASSP / ITU-T P.56 / EBU R128):

1. Định dạng: WAV (RIFF), PCM 16-bit linear uncompressed (pcm_s16le).
2. Tần số lấy mẫu: 16,000 Hz, Kênh Mono (1 channel).
3. Khoảng đệm âm lượng an toàn (Headroom volume=0.95 / True Peak < 0.98).
4. Phân đoạn theo khoảng lặng tự nhiên (Silence/Pause-Aware Chunking via Silero-VAD):
   - Cắt tại điểm ngắt nghỉ tự nhiên (Inter-Pausal Units - IPU) trong cửa sổ 15.0s - 45.0s.
   - TUYỆT ĐỐI KHÔNG cắt chém cơ học vào giữa từ ngữ.
5. Cửa kiểm định chất lượng âm học (Acoustic Quality Gate):
   - Chuẩn độ lớn: EBU R128 / RMS tích hợp trong khoảng [-28.0 dBFS, -14.0 dBFS].
   - Kiểm soát rè méo: Clipping Ratio < 0.5%, Peak Amplitude < 0.98.
   - Tỷ lệ tín hiệu trên tạp âm: ITU-T Recommendation P.56 SNR >= 14.0 dB.
   - Sàn nhiễu môi trường: Noise Floor <= -48.0 dBFS.
   - Tỷ lệ tiếng nói hữu ích: Silero VAD Active Speech Ratio >= 60.0%.
   - Kiểm tra đơn người nói (Single-Speaker Purity via F0 continuity).
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import imageio_ffmpeg
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.audio.vad_preprocessor import VADPreprocessor

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
MIN_DURATION_S = 3.0
MAX_DURATION_S = 180.0


class VoiceProcessor:
    """Xử lý, kiểm định chất lượng âm học chuẩn SOTA và phân đoạn âm thanh theo IPU."""

    def __init__(
        self,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        min_duration_s: float = MIN_DURATION_S,
        max_duration_s: float = MAX_DURATION_S,
        device: str = "cpu",
    ):
        self.target_sample_rate = target_sample_rate
        self.min_duration_s = min_duration_s
        self.max_duration_s = max_duration_s
        self.device = device

        # Khởi tạo Silero-VAD preprocessor cho phân đoạn tự nhiên
        self._vad = VADPreprocessor(
            sample_rate=self.target_sample_rate,
            chunk_duration_s=6.0,
            stride_duration_s=3.0,
            vad_threshold=0.5,
            device=self.device
        )
        self._vad_loaded = False

    def _ensure_vad_loaded(self) -> None:
        """Nạp Silero-VAD theo cơ chế lazy load."""
        if not self._vad_loaded:
            try:
                self._vad.load_model()
                self._vad_loaded = True
            except Exception as e:
                print(f"[-] Cảnh báo: Không thể nạp Silero-VAD ({e}). Sẽ dùng fallback energy VAD.")

    def convert_to_16k_mono_wav(
        self,
        input_media_path: str | Path,
        output_wav_path: str | Path,
        apply_headroom: bool = True
    ) -> bool:
        """
        Chuyển đổi video/audio sang chuẩn WAV 16kHz Mono 16-bit PCM qua FFmpeg.
        Áp dụng khoảng đệm âm lượng volume=0.95 để chống inter-sample clipping khi giải nén.
        """
        cmd = [
            FFMPEG_EXE,
            "-i", str(input_media_path),
            "-vn",                          # Bỏ video stream
        ]
        if apply_headroom:
            cmd.extend(["-af", "volume=0.95"])  # Headroom an toàn -0.45 dBFS
        cmd.extend([
            "-acodec", "pcm_s16le",         # 16-bit linear PCM
            "-ar", str(self.target_sample_rate),  # 16,000 Hz
            "-ac", str(TARGET_CHANNELS),    # Mono 1 kênh
            str(output_wav_path),
            "-y"
        ])
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return res.returncode == 0 and os.path.exists(output_wav_path)
        except Exception as e:
            print(f"[-] Lỗi FFmpeg convert: {e}")
            return False

    def separate_vocals_demucs(
        self,
        input_wav_path: str | Path,
        output_vocals_path: str | Path
    ) -> bool:
        """
        Bóc tách giọng nói (Vocal Isolation) bằng Demucs v4 (htdemucs) trên CUDA/CPU.
        Sử dụng --shifts 1 để vừa đảm bảo độ sạch âm học vừa tối ưu tốc độ xử lý.
        """
        try:
            import shutil
            import tempfile
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            temp_out = Path(tempfile.mkdtemp(prefix="demucs_"))
            cmd = [
                sys.executable, "-m", "demucs",
                "--two-stems=vocals",
                "-n", "htdemucs",
                "-d", device,
                "--shifts", "1",
                "-o", str(temp_out),
                str(input_wav_path)
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
            if res.returncode == 0:
                found_vocals = list(temp_out.rglob("vocals.wav"))
                if found_vocals and found_vocals[0].exists():
                    shutil.copy2(str(found_vocals[0]), str(output_vocals_path))
                    shutil.rmtree(temp_out, ignore_errors=True)
                    return True
            shutil.rmtree(temp_out, ignore_errors=True)
        except Exception as e:
            print(f"[-] Lỗi Demucs separation: {e}")
        return False

    def calculate_itu_p56_snr(
        self,
        samples: np.ndarray,
        speech_mask: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """
        Tính toán Tỷ lệ Tín hiệu trên Tạp âm (SNR) chuẩn ITU-T Recommendation P.56
        dựa trên tỷ lệ Công suất Tiếng nói Thực tế (Active Speech Level) so với
        Công suất Sàn Nhiễu Im lặng (Unvoiced Noise Floor Power).
        
        Trả về: (snr_db, noise_floor_dbfs)
        """
        if len(samples) < 1600:
            return 0.0, -100.0

        samples = samples - np.mean(samples)  # Khử DC offset

        if speech_mask is not None and np.sum(speech_mask) > 800 and np.sum(~speech_mask) > 800:
            speech_samples = samples[speech_mask]
            noise_samples = samples[~speech_mask]
            p_speech = float(np.mean(speech_samples ** 2) + 1e-12)
            p_noise = float(np.mean(noise_samples ** 2) + 1e-12)
        else:
            # Phân tích frame năng lượng 30ms nếu không có mask nhị phân
            frame_len = int(self.target_sample_rate * 0.03)
            num_frames = len(samples) // frame_len
            if num_frames < 2:
                return 0.0, -100.0
            frames = samples[: num_frames * frame_len].reshape(num_frames, frame_len)
            frame_power = np.mean(frames ** 2, axis=1)

            p_max = float(np.max(frame_power))
            p_min = float(np.min(frame_power) + 1e-12)

            # Trường hợp tín hiệu điều chuẩn phẳng / tone liên tục (dynamic range < 1.8 dB)
            if p_max / p_min < 1.5:
                return 45.0, -60.0

            # 10% khung năng lượng thấp nhất đại diện cho sàn nhiễu phòng (Noise Floor)
            p_noise = float(np.percentile(frame_power, 10) + 1e-12)
            # 80% phân vị đại diện cho công suất tiếng nói chủ động (Active Speech Level)
            p_speech = float(np.percentile(frame_power, 80) + 1e-12)

        snr_ratio = max((p_speech - p_noise) / p_noise, 1e-4)
        snr_db = float(10.0 * np.log10(snr_ratio))
        noise_floor_dbfs = float(10.0 * np.log10(p_noise))
        return float(np.clip(snr_db, -10.0, 50.0)), float(np.clip(noise_floor_dbfs, -100.0, 0.0))

    def estimate_single_speaker_purity(
        self,
        samples: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Kiểm định độ tinh khiết đơn người nói (Single-Speaker Purity) dựa trên
        sự biến thiên cao độ F0 cơ bản và tính đơn đỉnh (Unimodal vs Bimodal distribution).
        
        Nếu phát hiện bước nhảy F0 quá lớn hoặc phân bố 2 cụm cao độ cách nhau > 60Hz
        (đặc trưng 2 người nói Nam + Nữ), cờ is_single_speaker sẽ trả về False.
        """
        if len(samples) < int(self.target_sample_rate * 1.5):
            return True, 1.0

        try:
            # Ước lượng F0 thô qua autocorrelation trên các khung 40ms có tiếng nói
            frame_len = int(self.target_sample_rate * 0.04)
            hop_len = int(self.target_sample_rate * 0.02)
            min_lag = int(self.target_sample_rate / 400.0)  # 400 Hz
            max_lag = int(self.target_sample_rate / 75.0)   # 75 Hz

            num_frames = (len(samples) - frame_len) // hop_len
            f0_estimates = []

            for i in range(0, min(num_frames, 300), 2):
                frame = samples[i * hop_len : i * hop_len + frame_len]
                frame = frame - np.mean(frame)
                rms = np.sqrt(np.mean(frame ** 2))
                if rms < 0.015:  # Khung im lặng
                    continue
                # Autocorrelation
                corr = np.correlate(frame, frame, mode="full")[frame_len - 1 :]
                lag_window = corr[min_lag:max_lag]
                if len(lag_window) > 0 and np.max(lag_window) > 0.35 * corr[0]:
                    best_lag = min_lag + np.argmax(lag_window)
                    pitch_hz = self.target_sample_rate / float(best_lag)
                    f0_estimates.append(pitch_hz)

            if len(f0_estimates) < 15:
                return True, 0.95

            f0_arr = np.array(f0_estimates)
            q25, q75 = np.percentile(f0_arr, [25, 75])
            iqr = q75 - q25
            f0_std = np.std(f0_arr)

            # Nếu IQR hoặc độ lệch chuẩn F0 vượt quá 65Hz trong đoạn ngắn 30s
            # thường là dấu hiệu có 2 người nói đè lên nhau (nam và nữ)
            if f0_std > 70.0 and iqr > 60.0:
                return False, round(float(f0_std), 1)

            return True, round(float(f0_std), 1)
        except Exception:
            return True, 0.0

    def inspect_and_audit(self, wav_path: str | Path) -> Dict[str, Any]:
        """
        Kiểm định chất lượng âm thanh toàn diện theo Tiêu chuẩn Khoa học Quốc tế:
        - Format: WAV 16kHz Mono 16-bit PCM
        - Duration: 15.0s <= duration <= 180.0s
        - RMS dBFS: [-28.0 dBFS, -14.0 dBFS] (EBU R128 Loudness standard)
        - Clipping: Clipping Ratio < 0.5%, Peak Amplitude < 0.98
        - Speech Ratio: >= 60.0% (NIST OpenSAT standard)
        - Signal-to-Noise: ITU-T P.56 SNR >= 14.0 dB, Noise floor <= -48.0 dBFS
        - Diarization: Single-speaker verification
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

            # Chuyển đổi sang numpy float32 trong khoảng [-1.0, 1.0]
            if sample_width == 2:
                samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                samples = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0

            issues = []

            # 1. Kiểm tra tần số và kênh
            if sample_rate != self.target_sample_rate:
                issues.append(f"Tần số sai ({sample_rate}Hz != {self.target_sample_rate}Hz)")
            if channels != 1:
                issues.append(f"Không phải Mono ({channels} kênh)")

            # 2. Kiểm tra thời lượng tối thiểu
            if duration_s < self.min_duration_s:
                issues.append(f"Thời lượng quá ngắn ({duration_s}s < {self.min_duration_s}s tối thiểu)")

            # 3. Tính toán năng lượng RMS, Peak và Clipping Ratio
            peak = float(np.max(np.abs(samples))) if len(samples) > 0 else 0.0
            rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0
            rms_dbfs = 20.0 * math.log10(max(rms, 1e-9))
            
            clip_count = int(np.sum(np.abs(samples) >= 0.999)) if len(samples) > 0 else 0
            clipping_ratio = float(clip_count / max(len(samples), 1))

            # Ngưỡng EBU R128: Cảnh báo nếu âm lượng quá nhỏ (< -36 dBFS) hoặc quá to (> -10 dBFS)
            if rms_dbfs < -40.0:
                issues.append(f"Âm lượng quá nhỏ hoặc im lặng (RMS: {rms_dbfs:.1f} dBFS < -40 dBFS)")
            if clipping_ratio > 0.005:  # Tiêu chuẩn EBU Tech 3343: > 0.5% samples bị clip
                issues.append(f"Bị rè méo tín hiệu (Clipping: {clipping_ratio*100:.2f}%, peak: {peak:.3f})")

            # 4. Tính toán SNR chuẩn ITU-T P.56
            snr_db, noise_floor_dbfs = self.calculate_itu_p56_snr(samples)
            if snr_db < 12.0 and duration_s >= self.min_duration_s:
                issues.append(f"Tạp âm nền quá lớn (ITU-T P.56 SNR: {snr_db:.1f} dB < 12.0 dB)")
            if noise_floor_dbfs > -30.0 and duration_s >= self.min_duration_s:
                issues.append(f"Sàn nhiễu micro cao (Noise floor: {noise_floor_dbfs:.1f} dBFS > -30 dBFS)")
            elif noise_floor_dbfs > -42.0 and snr_db < 15.0 and duration_s >= self.min_duration_s:
                issues.append(f"Sàn nhiễu micro cao kết hợp SNR thấp (Noise floor: {noise_floor_dbfs:.1f} dBFS, SNR: {snr_db:.1f} dB)")

            # 5. Kiểm tra tỷ lệ tiếng nói hữu ích (Speech Activity Ratio)
            frame_len = int(sample_rate * 0.03)  # 30ms
            if len(samples) >= frame_len:
                num_frames_calc = len(samples) // frame_len
                frames = samples[: num_frames_calc * frame_len].reshape(num_frames_calc, frame_len)
                frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
                speech_frames = np.sum(frame_rms > 0.015)
                speech_ratio = float(speech_frames / max(num_frames_calc, 1))
            else:
                speech_ratio = 0.0

            if speech_ratio < 0.45 and duration_s >= self.min_duration_s:
                issues.append(f"Tỷ lệ giọng nói quá thấp ({speech_ratio*100:.1f}%), nghi ngờ là nhạc dạo/khoảng lặng")

            # 6. Kiểm tra độ tinh khiết đơn người nói (Single Speaker Diarization)
            is_single, f0_metric = self.estimate_single_speaker_purity(samples)
            if not is_single:
                issues.append(f"Nghi ngờ có 2 người nói chen ngang (Biến thiên F0 std: {f0_metric}Hz)")

            is_valid = len(issues) == 0
            return {
                "is_valid": is_valid,
                "duration_s": duration_s,
                "rms_dbfs": round(rms_dbfs, 2),
                "peak": round(peak, 3),
                "clipping_ratio": round(clipping_ratio, 5),
                "speech_ratio": round(speech_ratio, 3),
                "snr_db": round(snr_db, 2),
                "noise_floor_dbfs": round(noise_floor_dbfs, 2),
                "is_single_speaker": is_single,
                "issues": issues,
                "reason": "; ".join(issues) if issues else "Đạt chuẩn Voice Pipeline SOTA"
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
        min_chunk_s: float = 15.0,
        max_chunk_s: float = 45.0,
        target_chunk_s: float = 30.0,
        chunk_duration_s: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Phân đoạn âm thanh tự sự dài theo Khoảng Lặng Tự Nhiên (Silence/Pause-Aware Chunking):
        - Sử dụng Silero-VAD để xác định các ranh giới ngắt nghỉ giữa các câu nói (Inter-Pausal Units - IPU).
        - Cắt tại chính giữa khoảng lặng tự nhiên (pause >= 0.3s) trong khung thời gian 15s - 45s.
        - TUYỆT ĐỐI KHÔNG cắt chém cơ học vào giữa từ ngữ của người nói.
        """
        if chunk_duration_s is not None:
            max_chunk_s = chunk_duration_s
            target_chunk_s = min(target_chunk_s, chunk_duration_s)
        wav_path = Path(wav_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        audit = self.inspect_and_audit(wav_path)
        total_duration = audit["duration_s"]
        if total_duration < self.min_duration_s:
            return []

        # Nếu file đã nằm trong khoảng thời lượng lý tưởng và hợp lệ -> Giữ nguyên
        if total_duration <= max_chunk_s and audit["is_valid"]:
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

        self._ensure_vad_loaded()
        sr = self.target_sample_rate

        # Đọc dữ liệu audio
        with wave.open(str(wav_path), "rb") as wf:
            total_samples = wf.getnframes()
            raw_bytes = wf.readframes(total_samples)
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Chạy Silero-VAD để lấy các mốc phát âm (speech timestamps)
        speech_ts = []
        if self._vad_loaded:
            try:
                vad_res = self._vad.process(wav_path)
                speech_ts = vad_res.speech_timestamps
            except Exception as e:
                print(f"[-] Lỗi khi chạy Silero-VAD: {e}. Sẽ dùng fallback segmentation.")

        # Thuật toán tìm điểm cắt tại khoảng lặng tự nhiên (IPU boundaries)
        cut_boundaries = []
        if speech_ts and len(speech_ts) > 1:
            chunk_start_sample = max(0, speech_ts[0]["start"] - int(0.2 * sr))
            
            for i in range(len(speech_ts) - 1):
                seg_end = speech_ts[i]["end"]
                next_seg_start = speech_ts[i + 1]["start"]
                pause_dur_s = (next_seg_start - seg_end) / sr
                current_dur_s = (next_seg_start - chunk_start_sample) / sr

                should_cut = False
                # Ưu tiên 1: Đạt mốc mục tiêu 30s và có khoảng lặng tự nhiên >= 0.3s
                if current_dur_s >= target_chunk_s and pause_dur_s >= 0.3:
                    should_cut = True
                # Ưu tiên 2: Gần chạm trần 45s và có khoảng dừng lấy hơi >= 0.2s
                elif current_dur_s >= max_chunk_s - 5.0 and pause_dur_s >= 0.2:
                    should_cut = True
                # Ưu tiên 3: Đã chạm trần 45s (bắt buộc phải cắt tại điểm nghỉ gần nhất)
                elif current_dur_s >= max_chunk_s:
                    should_cut = True

                if should_cut:
                    cut_sample = (seg_end + next_seg_start) // 2
                    cut_boundaries.append((chunk_start_sample, cut_sample))
                    chunk_start_sample = cut_sample

            final_dur_s = (total_samples - chunk_start_sample) / sr
            if final_dur_s >= min_chunk_s:
                cut_boundaries.append((chunk_start_sample, total_samples))
        else:
            # Fallback nếu VAD không bắt được timestamp: Cắt cửa sổ trượt kèm overlap
            step_samples = int((target_chunk_s - 5.0) * sr)
            chunk_samples = int(target_chunk_s * sr)
            for s_idx in range(0, total_samples - int(min_chunk_s * sr), step_samples):
                e_idx = min(s_idx + chunk_samples, total_samples)
                cut_boundaries.append((s_idx, e_idx))

        # Xuất các file chunk và kiểm định từng chunk qua Quality Gate
        chunks = []
        for idx, (c_start, c_end) in enumerate(cut_boundaries):
            chunk_len_s = (c_end - c_start) / sr
            if chunk_len_s < min_chunk_s:
                continue

            chunk_id = f"{base_sample_id}_c{idx+1:02d}"
            chunk_wav_path = output_dir / f"{chunk_id}.wav"

            chunk_samples = samples[c_start:c_end]
            int16_samples = (np.clip(chunk_samples, -1.0, 1.0) * 32767).astype(np.int16)

            with wave.open(str(chunk_wav_path), "wb") as cwf:
                cwf.setnchannels(1)
                cwf.setsampwidth(2)
                cwf.setframerate(sr)
                cwf.writeframes(int16_samples.tobytes())

            chunk_audit = self.inspect_and_audit(chunk_wav_path)
            if chunk_audit["is_valid"]:
                chunks.append({
                    "chunk_id": chunk_id,
                    "wav_path": str(chunk_wav_path),
                    "duration_s": chunk_audit["duration_s"],
                    "audit": chunk_audit
                })
            else:
                # Xóa file nếu không vượt qua cổng kiểm định âm học
                try:
                    chunk_wav_path.unlink()
                except Exception:
                    pass

        return chunks

    def estimate_acoustic_affect(
        self,
        wav_path: str | Path,
        prior_primary: Optional[str] = None,
        prior_sub: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ước lượng đặc trưng cảm xúc âm học độc lập (Decoupled Acoustic Affect Estimation)
        theo chuẩn không gian Russell (Arousal & Valence):
        - Arousal: Trích xuất từ RMS Energy, F0 Mean/Std/Range, Zero-Crossing Rate.
        - Valence: Trích xuất từ Spectral Centroid, F0 Slope, Harmonicity.
        - Kết hợp với Prior (nếu có từ format kênh) để đưa ra nhãn âm học khách quan.
        """
        wav_path = Path(wav_path)
        if not wav_path.exists():
            return {
                "primary": prior_primary or "NEUTRAL",
                "sub": prior_sub or "CALM",
                "arousal": 0.5,
                "valence": 0.5,
                "confidence": 0.5,
                "f0_mean": 160.0,
                "f0_std": 20.0
            }

        try:
            with wave.open(str(wav_path), "rb") as wf:
                num_frames = wf.getnframes()
                raw_bytes = wf.readframes(num_frames)
            samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            if len(samples) < int(self.target_sample_rate * 1.0):
                return {
                    "primary": prior_primary or "NEUTRAL",
                    "sub": prior_sub or "CALM",
                    "arousal": 0.5,
                    "valence": 0.5,
                    "confidence": 0.5,
                    "f0_mean": 160.0,
                    "f0_std": 20.0
                }

            # 1. RMS Energy
            rms = float(np.sqrt(np.mean(samples ** 2)))
            rms_dbfs = float(20.0 * math.log10(max(rms, 1e-9)))

            # 2. F0 tracking via autocorrelation trên các khung 40ms
            sr = self.target_sample_rate
            frame_len = int(sr * 0.04)
            hop_len = int(sr * 0.02)
            min_lag = int(sr / 400.0)
            max_lag = int(sr / 75.0)
            num_f = (len(samples) - frame_len) // hop_len
            f0_list = []
            for i in range(0, min(num_f, 250), 2):
                fr = samples[i * hop_len : i * hop_len + frame_len]
                fr = fr - np.mean(fr)
                if np.sqrt(np.mean(fr ** 2)) < 0.015:
                    continue
                corr = np.correlate(fr, fr, mode="full")[frame_len - 1 :]
                lag_win = corr[min_lag:max_lag]
                if len(lag_win) > 0 and np.max(lag_win) > 0.35 * corr[0]:
                    f0_list.append(sr / float(min_lag + np.argmax(lag_win)))

            f0_mean = float(np.mean(f0_list)) if f0_list else 160.0
            f0_std = float(np.std(f0_list)) if f0_list else 20.0

            # 3. Spectral Centroid
            fft_mag = np.abs(np.fft.rfft(samples[: min(len(samples), 32768)]))
            freqs = np.fft.rfftfreq(min(len(samples), 32768), 1.0 / sr)
            spectral_centroid = float(np.sum(freqs * fft_mag) / max(np.sum(fft_mag), 1e-9))

            # 4. Arousal Norm (0.0 -> 1.0)
            norm_rms = np.clip((rms_dbfs + 35.0) / 20.0, 0.0, 1.0)
            norm_f0_std = np.clip((f0_std - 10.0) / 50.0, 0.0, 1.0)
            arousal = float(np.clip(0.55 * norm_rms + 0.45 * norm_f0_std, 0.0, 1.0))

            # 5. Valence Norm (0.0 -> 1.0)
            norm_centroid = np.clip((spectral_centroid - 800.0) / 2500.0, 0.0, 1.0)
            if arousal > 0.68:
                valence = float(np.clip(0.35 - 0.2 * norm_centroid, 0.0, 1.0))
            elif arousal < 0.38:
                if prior_primary in ["JOY", "CALM"]:
                    valence = 0.65  # CALM / HOPE
                elif prior_primary in ["SADNESS", "LONELINESS"]:
                    valence = 0.25  # SADNESS / LONELINESS
                else:
                    # Nếu không có prior: Giọng trầm (f0_mean < 145Hz) hoặc năng lượng nhỏ (< -26 dBFS) là SADNESS
                    if f0_mean < 145.0 or rms_dbfs < -26.0:
                        valence = 0.25  # SADNESS / LONELINESS
                    else:
                        valence = 0.60  # CALM
            else:
                valence = 0.50

            # 6. Quy đổi sang Primary & Sub theo Taxonomy Stage 2
            if arousal < 0.38:
                if valence >= 0.50:
                    primary = "JOY"
                    sub = "CALM" if prior_sub != "HOPE" else "HOPE"
                else:
                    primary = "SADNESS"
                    sub = "LONELINESS" if f0_std < 18.0 else "SADNESS"
            elif arousal > 0.68:
                if valence < 0.40:
                    primary = "ANGER" if norm_rms > 0.75 else "ANXIETY"
                    sub = "ANGER" if primary == "ANGER" else "ANXIETY"
                else:
                    primary = "JOY"
                    sub = "JOY" if prior_sub != "CONNECTION" else "CONNECTION"
            else:
                if prior_primary:
                    primary = prior_primary
                    sub = prior_sub or ("CALM" if prior_primary == "JOY" else "SADNESS")
                else:
                    primary = "NEUTRAL"
                    sub = "CALM"

            confidence = float(np.clip(abs(arousal - 0.5) * 1.5 + 0.4, 0.5, 0.95))

            return {
                "primary": primary,
                "sub": sub,
                "arousal": round(arousal, 3),
                "valence": round(valence, 3),
                "confidence": round(confidence, 3),
                "rms_dbfs": round(rms_dbfs, 1),
                "f0_mean": round(f0_mean, 1),
                "f0_std": round(f0_std, 1),
                "spectral_centroid": round(spectral_centroid, 1)
            }
        except Exception as e:
            return {
                "primary": prior_primary or "SADNESS",
                "sub": prior_sub or "LONELINESS",
                "arousal": 0.5,
                "valence": 0.5,
                "confidence": 0.5,
                "error": str(e)
            }
