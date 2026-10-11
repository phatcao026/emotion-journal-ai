"""
crawler_voice_radio.py
----------------------
Pipeline thu thập & Tiền xử lý Dữ liệu Giọng nói (VOICE PIPELINE) chuyên sâu cho:
"Nhật ký & Tâm sự chứa đựng cảm xúc tiếng Việt"

Nguồn dữ liệu:
1. Các kênh Radio Tâm sự / Podcast độc thoại (Người Giữ Kỉ Niệm, Tri Kỷ Cảm Xúc, Kể Cho Tôi Nghe, Trạm Dừng Chân...)
2. Video độc thoại tâm sự / Confessions từ YouTube, TikTok hoặc link trực tiếp qua `yt-dlp`.
3. File âm thanh/video cục bộ.

Tính năng:
- Tự động tải âm thanh độ phân giải cao qua `yt-dlp`.
- Tự động convert về chuẩn 16,000 Hz, Mono, 16-bit PCM WAV qua FFmpeg.
- Tự động cắt các bài nói/radio dài thành các phân đoạn độc thoại 30s - 60s (chunking).
- Kiểm định âm học nghiêm ngặt theo `validate_input_voice.py` (RMS, clipping, speech presence).
- Gán nhãn cảm xúc theo `data/taxonomy.json` (5 Primary, 11 Sub).
- Xuất trực tiếp vào `data/crawled/voice/` và `data/crawled/text_voice/`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import yt_dlp

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.crawler.journal_filter import JournalFilter
from scripts.crawler.dataset_manager import CrawledDatasetManager
from scripts.crawler.voice_processor import VoiceProcessor

# Danh sách gợi ý các kênh Radio/Podcast tâm sự giàu cảm xúc tiếng Việt (Đã bổ sung các kênh cân bằng cảm xúc tích cực)
CURATED_RADIO_SOURCES = [
    {
        "name": "Radio Người Giữ Kỉ Niệm",
        "description": "Đọc thư tâm sự thật của thính giả, giọng trầm buồn, hoài niệm, cô đơn",
        "default_emotion": ("SADNESS", "LONELINESS"),
        "search_query": "Radio Người Giữ Kỉ Niệm tâm sự"
    },
    {
        "name": "Tri Kỷ Cảm Xúc",
        "description": "Độc thoại tâm sự về áp lực cuộc sống, chữa lành, bình yên",
        "default_emotion": ("JOY", "CALM"),
        "search_query": "Tri Kỷ Cảm Xúc podcast tâm sự"
    },
    {
        "name": "Kể Cho Tôi Nghe",
        "description": "Đọc thư trải lòng đêm muộn, tâm sự tình cảm, gia đình",
        "default_emotion": ("SADNESS", "SADNESS"),
        "search_query": "Kể Cho Tôi Nghe tâm sự đêm"
    },
    {
        "name": "Trạm Dừng Chân",
        "description": "Radio nhật ký cảm xúc, những ngày chông chênh",
        "default_emotion": ("ANXIETY", "ANXIETY"),
        "search_query": "Trạm dừng chân radio tâm sự"
    },
    {
        "name": "Radio Hy Vọng & Chữa Lành",
        "description": "Động lực sống, vượt qua nghịch cảnh và áp lực, hy vọng ngày mai",
        "default_emotion": ("JOY", "HOPE"),
        "search_query": "radio hy vọng vượt qua nghịch cảnh tâm sự chữa lành"
    },
    {
        "name": "Tâm Sự Gia Đình & Tri Kỷ",
        "description": "Thư cảm ơn cha mẹ, sự gắn kết yêu thương, ấm áp tình người",
        "default_emotion": ("JOY", "CONNECTION"),
        "search_query": "radio cảm ơn gia đình tri kỷ chia sẻ yêu thương"
    }
]


class VoiceRadioCrawler:
    """Crawler chuyên biệt cho dữ liệu VOICE độc thoại tâm sự & nhật ký."""

    def __init__(
        self,
        output_dir: str = "data/crawled",
        chunk_duration_s: float = 45.0,
        min_duration_s: float = 3.0,
        max_duration_s: float = 180.0,
        enable_asr: bool = False,
        asr_model_id: str = "vinai/phowhisper-base",
        remove_music: bool = False,
        voice_type: str = "acted_radio"
    ):
        self.dm = CrawledDatasetManager(output_dir)
        self.vp = VoiceProcessor(
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s
        )
        self.jf = JournalFilter(min_word_count=10)
        self.chunk_duration_s = chunk_duration_s
        self.enable_asr = enable_asr
        self.remove_music = remove_music
        self.voice_type = voice_type
        self.temp_dir = Path("temp_voice_downloads")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.asr = None
        if self.enable_asr:
            try:
                from src.audio.asr_transcriber import PhoWhisperTranscriber
                print(f"[*] Đang nạp mô hình ASR ({asr_model_id}) để bóc tách transcript cho text_voice...")
                self.asr = PhoWhisperTranscriber(model_id=asr_model_id)
                self.asr.load_model()
                print("[*] Nạp PhoWhisper ASR thành công!")
            except Exception as e:
                print(f"[-] Không thể nạp PhoWhisperTranscriber: {e}. Sẽ chỉ lưu voice thuần.")
                self.enable_asr = False

        self.stats = {
            "processed_sources": 0,
            "created_voice_chunks": 0,
            "created_text_voice_pairs": 0,
            "quarantined_crisis": 0,
            "rejected_audit": 0,
            "primary_counts": {},
            "sub_counts": {}
        }

    def download_audio_with_ytdlp(self, url: str, base_name: str) -> Optional[Path]:
        """Tải âm thanh từ URL video/podcast bằng yt-dlp và chuyển về WAV qua VoiceProcessor."""
        out_template = str(self.temp_dir / f"{base_name}.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": out_template,
            "quiet": True,
            "no_warnings": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                print(f"[*] Đang tải audio từ: {url}")
                ydl.download([url])

            # Tìm file vừa tải về
            candidates = list(self.temp_dir.glob(f"{base_name}.*"))
            if candidates:
                raw_audio = candidates[0]
                converted_wav = self.temp_dir / f"{base_name}.wav"
                if raw_audio.suffix.lower() == ".wav" and raw_audio.resolve() == converted_wav.resolve():
                    return raw_audio

                ok = self.vp.convert_to_16k_mono_wav(raw_audio, converted_wav)
                try:
                    if raw_audio.exists() and raw_audio.resolve() != converted_wav.resolve():
                        raw_audio.unlink()
                except Exception:
                    pass

                if ok and converted_wav.exists():
                    return converted_wav
        except Exception as e:
            print(f"[-] Lỗi yt-dlp khi tải {url}: {e}")
        return None

    def process_voice_url(
        self,
        url: str,
        assigned_primary: Optional[str] = None,
        assigned_sub: Optional[str] = None,
        context_text: str = "",
        author: str = "radio_podcast"
    ):
        """Xử lý một URL âm thanh/podcast: tải -> convert 16kHz mono -> chunking -> validate -> lưu dataset."""
        base_id = f"voice_{uuid.uuid4().hex[:8]}"
        downloaded = self.download_audio_with_ytdlp(url, base_id)
        if not downloaded:
            return

        self.process_local_audio(
            audio_path=downloaded,
            source_url=url,
            assigned_primary=assigned_primary,
            assigned_sub=assigned_sub,
            context_text=context_text,
            author=author
        )

        # Xóa file tải thô
        if downloaded.exists():
            try:
                downloaded.unlink()
            except Exception:
                pass

    def process_local_audio(
        self,
        audio_path: str | Path,
        source_url: str = "",
        assigned_primary: Optional[str] = None,
        assigned_sub: Optional[str] = None,
        context_text: str = "",
        author: str = "radio_podcast"
    ):
        """Xử lý file âm thanh cục bộ: chuẩn hóa 16k -> cắt chunks -> kiểm định -> lưu."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            print(f"[-] File âm thanh không tồn tại: {audio_path}")
            return

        self.stats["processed_sources"] += 1
        base_id = audio_path.stem

        # 1. Convert về chuẩn WAV 16kHz Mono 16-bit PCM
        std_wav_path = self.temp_dir / f"std_{base_id}.wav"
        ok = self.vp.convert_to_16k_mono_wav(audio_path, std_wav_path)
        if not ok:
            print(f"[-] Không thể convert file {audio_path} sang 16kHz mono WAV.")
            return

        # 1b. Tách nhạc nền bằng Demucs nếu bật cờ --remove-music
        clean_wav_path = self.temp_dir / f"clean_{base_id}.wav"
        if self.remove_music:
            print(f"[*] Đang chạy tách nhạc nền (Vocal Isolation) bằng Demucs...")
            separated = self.vp.separate_vocals_demucs(std_wav_path, clean_wav_path)
            if separated and clean_wav_path.exists():
                std_wav_path = clean_wav_path
                print(f"  [+] Đã tách nhạc nền thành công! Sử dụng track vocal sạch.")
            else:
                print(f"  [-] Không thể tách nhạc nền. Tiếp tục với audio ban đầu.")

        author_hash = hashlib.sha256(f"radio_{author}".encode("utf-8")).hexdigest()[:12]

        # 2. Phân tích ngữ cảnh text (nếu có mô tả/tiêu đề tập) để xác định cảm xúc
        if context_text and not (assigned_primary and assigned_sub):
            eval_res = self.jf.evaluate_entry(context_text)
            if eval_res["accepted"]:
                primary_label = eval_res["primary_emotion"]
                sub_label = eval_res["sub_emotion"]
            else:
                primary_label = assigned_primary or "SADNESS"
                sub_label = assigned_sub or "LONELINESS"
        else:
            primary_label = assigned_primary or "SADNESS"
            sub_label = assigned_sub or "LONELINESS"

        # 3. Cắt thành các đoạn monologue (30s - 60s)
        chunks_output_dir = self.temp_dir / f"chunks_{base_id}"
        chunks = self.vp.split_long_audio_to_monologues(
            wav_path=std_wav_path,
            output_dir=chunks_output_dir,
            base_sample_id=base_id,
            chunk_duration_s=self.chunk_duration_s,
            overlap_s=5.0
        )

        print(f"[*] Đã phân đoạn thành công {len(chunks)} đoạn độc thoại tâm sự (chuẩn 15s - {self.chunk_duration_s}s).")

        for c in chunks:
            chunk_id = c["chunk_id"]
            c_wav_path = Path(c["wav_path"])
            c_duration = c["duration_s"]
            audit = c["audit"]

            if not audit["is_valid"]:
                self.stats["rejected_audit"] += 1
                continue

            with open(c_wav_path, "rb") as f:
                wav_bytes = f.read()

            meta = {
                "source_platform": "radio_podcast_monologue",
                "author": author,
                "author_hash": author_hash,
                "source_url": source_url,
                "duration_s": c_duration,
                "voice_type": self.voice_type,
                "rms_dbfs": audit["rms_dbfs"],
                "peak": audit["peak"],
                "speech_ratio": audit["speech_ratio"]
            }

            # 4. Nếu bật ASR (--transcribe), sinh transcript thật từ giọng nói
            spoken_text = ""
            chunk_pri = primary_label
            chunk_sub = sub_label
            is_crisis_chunk = False

            if self.enable_asr and self.asr is not None:
                try:
                    asr_res = self.asr.transcribe(str(c_wav_path))
                    if asr_res and asr_res.text.strip():
                        spoken_text = asr_res.text.strip()
                        eval_asr = self.jf.evaluate_entry(spoken_text)
                        if eval_asr.get("is_crisis"):
                            is_crisis_chunk = True
                            self.dm.save_crisis_sample(
                                sample_id=chunk_id,
                                text=spoken_text,
                                crisis_keywords=eval_asr.get("crisis_keywords", []),
                                primary_label=eval_asr["primary_emotion"],
                                sub_label=eval_asr["sub_emotion"],
                                meta=meta
                            )
                            self.stats["quarantined_crisis"] += 1
                            print(f"  [!] CẢNH BÁO ASR: Phát hiện đoạn nói khủng hoảng tâm lý -> Đã cách ly!")
                        elif eval_asr["accepted"]:
                            chunk_pri = eval_asr["primary_emotion"]
                            chunk_sub = eval_asr["sub_emotion"]
                except Exception as e:
                    print(f"  [-] Lỗi ASR transcribe chunk {chunk_id}: {e}")

            if is_crisis_chunk:
                continue

            # Lưu vào Voice thuần
            voice_id = f"{chunk_id}_v"
            self.dm.save_voice_sample(
                sample_id=voice_id,
                wav_bytes=wav_bytes,
                primary_label=chunk_pri,
                sub_label=chunk_sub,
                emotion_score=0.9,
                duration_s=c_duration,
                meta=meta
            )
            self.stats["created_voice_chunks"] += 1

            # Lưu cặp Text-Voice CHUẨN XÁC nếu có transcript từ ASR hoặc văn bản khớp
            final_text = spoken_text if spoken_text else context_text
            if final_text:
                self.dm.save_text_voice_sample(
                    sample_id=chunk_id,
                    wav_bytes=wav_bytes,
                    text=final_text,
                    primary_label=chunk_pri,
                    sub_label=chunk_sub,
                    emotion_score=0.9,
                    duration_s=c_duration,
                    meta=meta
                )
                self.stats["created_text_voice_pairs"] += 1

            self.stats["primary_counts"][chunk_pri] = self.stats["primary_counts"].get(chunk_pri, 0) + 1
            self.stats["sub_counts"][chunk_sub] = self.stats["sub_counts"].get(chunk_sub, 0) + 1
            print(f"  [+] ĐÃ LƯU VOICE: {voice_id} ({c_duration}s | {chunk_pri}:{chunk_sub} | RMS: {audit['rms_dbfs']}dBFS)")

        # Dọn dẹp thư mục tạm
        for tmp in [std_wav_path, clean_wav_path]:
            if tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass
        if chunks_output_dir.exists():
            import shutil
            shutil.rmtree(chunks_output_dir, ignore_errors=True)

    def print_summary(self):
        print("\n" + "="*65)
        print("TỔNG KẾT THU THẬP DỮ LIỆU VOICE CHO MULTIMODAL JOURNAL MER")
        print("="*65)
        print(f"Tổng nguồn audio đã xử lý:             {self.stats['processed_sources']}")
        print(f"Số mẫu VOICE độc thoại chuẩn (16kHz):   {self.stats['created_voice_chunks']} mẫu")
        print(f"Số cặp Text-Voice đa phương thức:      {self.stats['created_text_voice_pairs']} mẫu")
        print(f"Số đoạn bị từ chối do không đạt chuẩn: {self.stats['rejected_audit']} đoạn")
        print("\nPhân bố cảm xúc theo Taxonomy:")
        print("  Primary Emotions:")
        for em, cnt in self.stats["primary_counts"].items():
            print(f"    * {em:<12}: {cnt} mẫu")
        print("  Sub Emotions:")
        for sub, cnt in self.stats["sub_counts"].items():
            print(f"    * {sub:<14}: {cnt} mẫu")
        print(f"\nMetadata tổng hợp:")
        print(f"  - CSV:   {self.dm.metadata_csv}")
        print(f"  - JSONL: {self.dm.metadata_jsonl}")
        print("="*65)


def main():
    parser = argparse.ArgumentParser(description="Vietnamese Emotional Voice Monologue & Radio Crawler")
    parser.add_argument("--url", type=str, help="URL video/podcast YouTube/TikTok/Soundcloud cần tải và phân đoạn")
    parser.add_argument("--file", type=str, help="Đường dẫn file âm thanh/video cục bộ cần xử lý")
    parser.add_argument("--output-dir", type=str, default="data/crawled", help="Thư mục xuất dữ liệu")
    parser.add_argument("--chunk-duration", type=float, default=45.0, help="Thời lượng mỗi đoạn độc thoại (giây)")
    parser.add_argument("--min-duration", type=float, default=3.0, help="Thời lượng tối thiểu cho một đoạn voice (giây), mặc định 3.0s")
    parser.add_argument("--primary", type=str, default="SADNESS", choices=["JOY", "SADNESS", "ANXIETY", "ANGER", "NEUTRAL"],
                        help="Nhãn Primary gán cho audio")
    parser.add_argument("--sub", type=str, default="LONELINESS",
                        help="Nhãn Sub gán cho audio (CALM, HOPE, CONNECTION, SADNESS, ANXIETY, FEAR, ANGER, GUILT_SHAME, LONELINESS, DISGUST)")
    parser.add_argument("--author", type=str, default="radio_tamsu", help="Tên tác giả / kênh nguồn")
    parser.add_argument("--context", type=str, default="", help="Văn bản tóm tắt/nội dung tâm sự đi kèm")
    parser.add_argument("--transcribe", action="store_true", help="Bật PhoWhisper ASR để sinh transcript chính xác cho cặp text_voice")
    parser.add_argument("--asr-model", type=str, default="vinai/phowhisper-base", help="Model HuggingFace của ASR")
    parser.add_argument("--remove-music", action="store_true", help="Chạy Demucs tách nhạc nền (Vocal Isolation)")
    parser.add_argument("--voice-type", type=str, default="acted_radio", choices=["acted_radio", "natural_monologue"],
                        help="Loại giọng nói: acted_radio (đọc diễn cảm) hoặc natural_monologue (tự sự mộc mạc)")
    parser.add_argument("--list-sources", action="store_true", help="Hiển thị danh sách các nguồn Radio/Podcast gợi ý")

    args = parser.parse_args()

    if args.list_sources:
        print("\n=== DANH SÁCH GỢI Ý CÁC KÊNH RADIO / PODCAST TÂM SỰ CẢM XÚC TIẾNG VIỆT ===")
        for idx, src in enumerate(CURATED_RADIO_SOURCES, 1):
            print(f"\n[{idx}] {src['name']}")
            print(f"    Mô tả:    {src['description']}")
            print(f"    Cảm xúc:  {src['default_emotion'][0]} / {src['default_emotion'][1]}")
            print(f"    Tìm kiếm: '{src['search_query']}'")
        return

    crawler = VoiceRadioCrawler(
        output_dir=args.output_dir,
        chunk_duration_s=args.chunk_duration,
        min_duration_s=args.min_duration,
        enable_asr=args.transcribe,
        asr_model_id=args.asr_model,
        remove_music=args.remove_music,
        voice_type=args.voice_type
    )

    if args.url:
        crawler.process_voice_url(
            url=args.url,
            assigned_primary=args.primary,
            assigned_sub=args.sub,
            context_text=args.context,
            author=args.author
        )
    elif args.file:
        crawler.process_local_audio(
            audio_path=args.file,
            assigned_primary=args.primary,
            assigned_sub=args.sub,
            context_text=args.context,
            author=args.author
        )
    else:
        print("Vui lòng cung cấp tham số --url <URL> hoặc --file <Path> (hoặc --list-sources để xem gợi ý).")
        parser.print_help()
        return

    crawler.print_summary()


if __name__ == "__main__":
    main()
