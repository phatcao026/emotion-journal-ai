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
            "deep_journal_count": 0,
            "general_monologue_count": 0,
            "quarantined_crisis": 0,
            "rejected_audit": 0,
            "primary_counts": {},
            "sub_counts": {}
        }

    def download_audio_with_ytdlp(self, url: str, base_name: str) -> Tuple[Optional[Path], Dict[str, Any]]:
        """Tải âm thanh từ URL video/podcast bằng yt-dlp và chuyển về WAV qua VoiceProcessor."""
        out_template = str(self.temp_dir / f"{base_name}.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": out_template,
            "quiet": True,
            "no_warnings": True,
        }
        info_dict: Dict[str, Any] = {}

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                print(f"[*] Đang tải audio từ: {url}")
                info = ydl.extract_info(url, download=True)
                if info:
                    info_dict = {
                        "title": info.get("title", ""),
                        "description": info.get("description", ""),
                        "uploader": info.get("uploader", ""),
                        "uploader_id": info.get("uploader_id", ""),
                        "channel_id": info.get("channel_id", ""),
                        "channel": info.get("channel", ""),
                        "duration": info.get("duration", 0)
                    }

            # Tìm file vừa tải về
            candidates = list(self.temp_dir.glob(f"{base_name}.*"))
            if candidates:
                raw_audio = candidates[0]
                converted_wav = self.temp_dir / f"{base_name}.wav"
                if raw_audio.suffix.lower() == ".wav" and raw_audio.resolve() == converted_wav.resolve():
                    return raw_audio, info_dict

                ok = self.vp.convert_to_16k_mono_wav(raw_audio, converted_wav)
                try:
                    if raw_audio.exists() and raw_audio.resolve() != converted_wav.resolve():
                        raw_audio.unlink()
                except Exception:
                    pass

                if ok and converted_wav.exists():
                    return converted_wav, info_dict
        except Exception as e:
            print(f"[-] Lỗi yt-dlp khi tải {url}: {e}")
        return None, info_dict

    def search_and_harvest(
        self,
        query: str,
        max_items: int = 3,
        assigned_primary: Optional[str] = None,
        assigned_sub: Optional[str] = None
    ):
        """Tìm kiếm video podcast/tâm sự từ YouTube và tự động tải, phân đoạn (kèm lọc tiêu cực)."""
        search_query = f"ytsearch{max_items}:{query} -karaoke -nhạc -truyện -cover -audiobook"
        ydl_opts = {
            "quiet": True,
            "extract_flat": True,
            "skip_download": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                print(f"[*] Đang tìm kiếm YouTube với truy vấn: '{query}' (lọc karaoke/nhạc/truyện)...")
                res = ydl.extract_info(search_query, download=False)
                entries = res.get("entries", []) if res else []
                print(f"[+] Tìm thấy {len(entries)} video tiềm năng.")
                for entry in entries:
                    if not entry:
                        continue
                    video_url = entry.get("url") or f"https://www.youtube.com/watch?v={entry.get('id')}"
                    title = entry.get("title", "")
                    uploader = entry.get("uploader", "")
                    print(f"\n---> Bắt đầu xử lý: {title} ({video_url})")
                    self.process_voice_url(
                        url=video_url,
                        assigned_primary=assigned_primary,
                        assigned_sub=assigned_sub,
                        context_text=title,
                        author=uploader or "youtube_curated"
                    )
        except Exception as e:
            print(f"[-] Lỗi khi tìm kiếm YouTube: {e}")

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
        downloaded, info_dict = self.download_audio_with_ytdlp(url, base_id)
        if not downloaded:
            return

        # Xác định author chuẩn từ uploader_id/channel_id để gán author_hash duy nhất theo người nói
        real_author = info_dict.get("channel_id") or info_dict.get("uploader_id") or info_dict.get("uploader") or author
        video_context = context_text or info_dict.get("title", "")

        self.process_local_audio(
            audio_path=downloaded,
            source_url=url,
            assigned_primary=assigned_primary,
            assigned_sub=assigned_sub,
            context_text=video_context,
            author=real_author
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

        author_hash = hashlib.sha256(f"spk_{author}".encode("utf-8")).hexdigest()[:12]

        # 2. Xác định Bayesian Prior từ cấu trúc kênh hoặc tham số người dùng
        if assigned_primary and assigned_sub:
            prior_pri = assigned_primary
            prior_sub = assigned_sub
        elif context_text:
            eval_res = self.jf.evaluate_entry(context_text)
            if eval_res["accepted"]:
                prior_pri = eval_res["primary_emotion"]
                prior_sub = eval_res["sub_emotion"]
            else:
                prior_pri = "SADNESS"
                prior_sub = "LONELINESS"
        else:
            prior_pri = "SADNESS"
            prior_sub = "LONELINESS"

        # 3. Cắt thành các đoạn monologue (15s - 45s) theo khoảng lặng tự nhiên IPU via Silero-VAD
        chunks_output_dir = self.temp_dir / f"chunks_{base_id}"
        chunks = self.vp.split_long_audio_to_monologues(
            wav_path=std_wav_path,
            output_dir=chunks_output_dir,
            base_sample_id=base_id,
            chunk_duration_s=self.chunk_duration_s
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

            # 4. Gán nhãn cảm xúc âm học độc lập (Decoupled Acoustic Affect - No Circular Reasoning)
            acoustic_affect = self.vp.estimate_acoustic_affect(
                c_wav_path,
                prior_primary=prior_pri,
                prior_sub=prior_sub
            )
            voice_pri = acoustic_affect["primary"]
            voice_sub = acoustic_affect["sub"]
            emotion_score = acoustic_affect["confidence"]

            meta = {
                "source_platform": "radio_podcast_monologue",
                "author": author,
                "author_hash": author_hash,
                "source_url": source_url,
                "duration_s": c_duration,
                "voice_type": self.voice_type,
                "rms_dbfs": audit["rms_dbfs"],
                "peak": audit["peak"],
                "speech_ratio": audit["speech_ratio"],
                "snr_db": audit.get("snr_db", 0.0),
                "noise_floor_dbfs": audit.get("noise_floor_dbfs", -60.0),
                "acoustic_arousal": acoustic_affect["arousal"],
                "acoustic_valence": acoustic_affect["valence"],
                "f0_mean": acoustic_affect["f0_mean"],
                "f0_std": acoustic_affect["f0_std"],
                "incongruent_masking": False,
                "text_primary": "",
                "text_sub": ""
            }

            # 5. Nếu bật ASR (--transcribe), sinh transcript thật từ giọng nói qua PhoWhisper
            spoken_text = ""
            is_crisis_chunk = False
            chunk_tier = "general_monologue"
            disclosure_score = 0.0

            if self.enable_asr and self.asr is not None:
                try:
                    asr_res = self.asr.transcribe(str(c_wav_path))
                    if asr_res and asr_res.text.strip():
                        spoken_text = asr_res.text.strip()
                        eval_asr = self.jf.evaluate_entry(spoken_text)
                        chunk_tier = eval_asr.get("data_tier", "general_monologue")
                        disclosure_score = eval_asr.get("disclosure_score", 0.0)

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
                            print(f"  [!] CẢNH BÁO ASR: Phát hiện đoạn nói khủng hoảng tâm lý -> Đã cách ly an toàn!")
                        elif eval_asr["accepted"]:
                            text_pri = eval_asr["primary_emotion"]
                            text_sub = eval_asr["sub_emotion"]
                            meta["text_primary"] = text_pri
                            meta["text_sub"] = text_sub

                            # Kiểm tra hiện tượng che giấu cảm xúc (Affective Incongruence / Masking)
                            if text_pri != voice_pri and (text_pri == "JOY" or voice_pri in ["SADNESS", "ANXIETY", "ANGER"]):
                                meta["incongruent_masking"] = True
                                print(f"  [*] PHÁT HIỆN MASKING: Lời nói '{text_pri}' nhưng Giọng nói '{voice_pri}'!")
                except Exception as e:
                    print(f"  [-] Lỗi ASR transcribe chunk {chunk_id}: {e}")

            if is_crisis_chunk:
                continue

            meta["data_tier"] = chunk_tier
            meta["disclosure_score"] = disclosure_score

            # 6. Lưu vào Voice thuần (Nhãn âm học chuẩn xác, độc lập)
            voice_id = f"{chunk_id}_v"
            self.dm.save_voice_sample(
                sample_id=voice_id,
                wav_bytes=wav_bytes,
                primary_label=voice_pri,
                sub_label=voice_sub,
                emotion_score=emotion_score,
                duration_s=c_duration,
                data_tier=chunk_tier,
                meta=meta
            )
            self.stats["created_voice_chunks"] += 1

            # 7. Lưu cặp Text-Voice ĐA PHƯƠNG THỨC:
            # - CHỈ lưu khi transcript ASR hợp lệ (>= 5 từ) VÀ không phải lời chào/quảng cáo rác bị reject
            # - Gắn nhãn data_tier ('deep_journal' hoặc 'general_monologue')
            if spoken_text and len(spoken_text.split()) >= 5 and chunk_tier in ["deep_journal", "general_monologue"]:
                self.dm.save_text_voice_sample(
                    sample_id=chunk_id,
                    wav_bytes=wav_bytes,
                    text=spoken_text,
                    primary_label=voice_pri,
                    sub_label=voice_sub,
                    emotion_score=emotion_score,
                    duration_s=c_duration,
                    data_tier=chunk_tier,
                    meta=meta
                )
                self.stats["created_text_voice_pairs"] += 1

            if chunk_tier == "deep_journal":
                self.stats["deep_journal_count"] += 1
            else:
                self.stats["general_monologue_count"] += 1

            self.stats["primary_counts"][voice_pri] = self.stats["primary_counts"].get(voice_pri, 0) + 1
            self.stats["sub_counts"][voice_sub] = self.stats["sub_counts"].get(voice_sub, 0) + 1
            mask_flag = " [MASKED]" if meta.get("incongruent_masking") else ""
            tier_flag = f" [{chunk_tier.upper()}]"
            print(f"  [+] ĐÃ LƯU VOICE: {voice_id} ({c_duration}s | {voice_pri}:{voice_sub}{mask_flag}{tier_flag} | Arousal: {acoustic_affect['arousal']} | RMS: {audit['rms_dbfs']}dBFS)")

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
        print(f"  * Tier 1 (Deep Affective Journal):   {self.stats['deep_journal_count']} mẫu")
        print(f"  * Tier 2 (General Monologue MER):    {self.stats['general_monologue_count']} mẫu")
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
    parser.add_argument("--search", type=str, help="Từ khóa tìm kiếm YouTube để tự động quét nhiều video")
    parser.add_argument("--max-items", type=int, default=3, help="Số video tối đa cần tải khi dùng --search")
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

    if args.search:
        crawler.search_and_harvest(
            query=args.search,
            max_items=args.max_items,
            assigned_primary=args.primary,
            assigned_sub=args.sub
        )
    elif args.url:
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
        print("Vui lòng cung cấp tham số --search <Query>, --url <URL> hoặc --file <Path> (hoặc --list-sources để xem gợi ý).")
        parser.print_help()
        return

    crawler.print_summary()


if __name__ == "__main__":
    main()
