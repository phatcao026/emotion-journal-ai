"""
crawler_tiktok.py
-----------------
Pipeline thu thập & Tiền xử lý Dữ liệu Giọng nói TikTok chuyên sâu cho:
"Nhật ký & Độc thoại Tâm sự Cảm xúc Đời thường (Spontaneous Voice Journaling)"

Đáp ứng 3 chiến lược khai thác triệt để không phụ thuộc vào Hashtag/Caption:
1. Creator Profiles  : Quét trực tiếp kho video từ các kênh chuyên Voice Journal/Tâm sự đêm.
2. Original Sounds   : Quét các video sử dụng Âm thanh gốc (Original Audio) độc thoại trải lòng.
3. Focused Search    : Tìm kiếm cụm từ tâm sự & bắt trọn API stream qua Playwright Automation.

Tiêu chuẩn Chất lượng:
- Vượt bot an toàn qua `tiktok_cookies.json` / `tiktok_cookies.txt` (Playwright session).
- Bóc tách nhạc nền (Vocal Isolation) bằng Demucs v4.
- Cắt phân đoạn theo khoảng lặng tự nhiên IPU (Silero-VAD).
- Kiểm định âm học DSP: ITU-T P.56 SNR >= 12dB, Clipping < 0.5%, EBU R128 RMS.
- Phiên âm chuẩn xác qua PhoWhisper ASR và kiểm tra tự sự bằng JournalFilter.
- Gán nhãn âm học độc lập (Decoupled Affect) & phát hiện che giấu cảm xúc (Masking).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import yt_dlp
from playwright.async_api import BrowserContext, Page, async_playwright

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.crawler.dataset_manager import CrawledDatasetManager
from scripts.crawler.journal_filter import JournalFilter
from scripts.crawler.voice_processor import VoiceProcessor

# Danh sách gợi ý các kênh chuyên Voice Journal / Tâm sự đêm trên TikTok
CURATED_TIKTOK_CREATORS = [
    {"username": "@tam.su.dem", "description": "Tâm sự đêm khuya, giọng trầm buồn", "default_emotion": ("SADNESS", "LONELINESS")},
    {"username": "@goc.tam.su", "description": "Trải lòng một mình, overthinking", "default_emotion": ("ANXIETY", "ANXIETY")},
    {"username": "@nhatky.cuatoi", "description": "Nhật ký cuộc sống, những ngày bình yên và chữa lành", "default_emotion": ("JOY", "CALM")},
    {"username": "@chuyen.cua.dem", "description": "Độc thoại tâm sự muộn phiền", "default_emotion": ("SADNESS", "SADNESS")},
    {"username": "@loi.tu.su", "description": "Trải nghiệm vượt qua áp lực, hy vọng", "default_emotion": ("JOY", "HOPE")}
]


class TikTokVoiceCrawler:
    """Crawler chuyên biệt cho dữ liệu Spontaneous Voice Journaling trên TikTok."""

    def __init__(
        self,
        output_dir: str = "data/crawled",
        chunk_duration_s: float = 45.0,
        enable_asr: bool = True,
        asr_model_id: str = "vinai/phowhisper-base",
        remove_music: bool = False,
        headless: bool = True,
    ):
        self.output_dir = output_dir
        self.dm = CrawledDatasetManager(output_dir)
        self.vp = VoiceProcessor(min_duration_s=3.0, max_duration_s=180.0)
        self.jf = JournalFilter(min_word_count=10)
        self.chunk_duration_s = chunk_duration_s
        self.enable_asr = enable_asr
        self.remove_music = remove_music
        self.headless = headless

        self.crawler_dir = Path(__file__).resolve().parent
        self.cookie_json_path = self.crawler_dir / "tiktok_cookies.json"
        self.cookie_txt_path = self.crawler_dir / "tiktok_cookies.txt"

        self.temp_dir = Path("temp_tiktok_downloads")
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
            "scanned_videos": 0,
            "processed_videos": 0,
            "created_voice_chunks": 0,
            "created_text_voice_pairs": 0,
            "quarantined_crisis": 0,
            "rejected_audit": 0,
            "rejected_filter": 0,
            "primary_counts": {},
            "sub_counts": {}
        }

    async def _setup_browser_context(self, p) -> Tuple[Any, BrowserContext]:
        """Khởi tạo trình duyệt Playwright với session cookies đã lưu để vượt bot."""
        browser = await p.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage"
            ]
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            locale="vi-VN",
            viewport={"width": 1280, "height": 800}
        )

        if self.cookie_json_path.exists():
            try:
                with open(self.cookie_json_path, "r", encoding="utf-8") as f:
                    cookies = json.load(f)
                await context.add_cookies(cookies)
                print(f"[+] Đã nạp {len(cookies)} session cookies từ {self.cookie_json_path.name}")
            except Exception as e:
                print(f"[-] Lỗi nạp session cookies: {e}")
        else:
            print("[!] Lưu ý: Chưa có tiktok_cookies.json. Khuyến nghị chạy 'python scripts/crawler/save_tiktok_session.py' trước để tránh bị chặn.")

        return browser, context

    async def harvest_from_profile(self, username: str, max_videos: int = 10) -> List[str]:
        """Khai thác trực tiếp video từ hồ sơ Creator chuyên tâm sự (100% không phụ thuộc hashtag)."""
        clean_user = username if username.startswith("@") else f"@{username}"
        profile_url = f"https://www.tiktok.com/{clean_user}"
        print(f"\n[*] Đang quét Creator Profile: {clean_user} ({profile_url})...")

        video_urls: Set[str] = set()

        async with async_playwright() as p:
            browser, context = await self._setup_browser_context(p)
            page = await context.new_page()

            try:
                await page.goto(profile_url, timeout=45000, wait_until="domcontentloaded")
                await page.wait_for_timeout(4000)

                # Cuộn trang để tải thêm danh sách video
                scroll_attempts = 0
                while len(video_urls) < max_videos and scroll_attempts < 6:
                    links = await page.query_selector_all('a[href*="/video/"]')
                    for link in links:
                        href = await link.get_attribute("href")
                        if href and "/video/" in href:
                            full_url = href if href.startswith("http") else f"https://www.tiktok.com{href}"
                            clean_link = full_url.split("?")[0]
                            if not self.dm.is_url_crawled(clean_link):
                                video_urls.add(clean_link)
                            if len(video_urls) >= max_videos:
                                break

                    await page.evaluate("window.scrollBy(0, 1200)")
                    await page.wait_for_timeout(2500)
                    scroll_attempts += 1
            except Exception as e:
                print(f"[-] Lỗi khi quét hồ sơ {clean_user}: {e}")
            finally:
                await browser.close()

        print(f"[+] Tìm thấy {len(video_urls)} video mới từ {clean_user}.")
        return list(video_urls)

    async def harvest_from_sound(self, sound_url: str, max_videos: int = 10) -> List[str]:
        """Khai thác video sử dụng chung Âm thanh gốc (Original Audio) độc thoại trải lòng."""
        print(f"\n[*] Đang quét trang Âm thanh gốc (Original Sound): {sound_url}...")
        video_urls: Set[str] = set()

        async with async_playwright() as p:
            browser, context = await self._setup_browser_context(p)
            page = await context.new_page()

            try:
                await page.goto(sound_url, timeout=45000, wait_until="domcontentloaded")
                await page.wait_for_timeout(4000)

                scroll_attempts = 0
                while len(video_urls) < max_videos and scroll_attempts < 5:
                    links = await page.query_selector_all('a[href*="/video/"]')
                    for link in links:
                        href = await link.get_attribute("href")
                        if href and "/video/" in href:
                            full_url = href if href.startswith("http") else f"https://www.tiktok.com{href}"
                            clean_link = full_url.split("?")[0]
                            if not self.dm.is_url_crawled(clean_link):
                                video_urls.add(clean_link)
                            if len(video_urls) >= max_videos:
                                break

                    await page.evaluate("window.scrollBy(0, 1000)")
                    await page.wait_for_timeout(2000)
                    scroll_attempts += 1
            except Exception as e:
                print(f"[-] Lỗi khi quét Original Sound: {e}")
            finally:
                await browser.close()

        print(f"[+] Tìm thấy {len(video_urls)} video mới từ Original Sound.")
        return list(video_urls)

    async def harvest_from_search(self, query: str, max_videos: int = 10) -> List[str]:
        """Tìm kiếm theo từ khóa kết hợp bắt gói tin API tìm kiếm qua Playwright."""
        encoded_query = query.replace(" ", "%20")
        search_url = f"https://www.tiktok.com/search?q={encoded_query}"
        print(f"\n[*] Đang tìm kiếm TikTok với truy vấn: '{query}'...")
        video_urls: Set[str] = set()

        async with async_playwright() as p:
            browser, context = await self._setup_browser_context(p)
            page = await context.new_page()

            # Bắt gói tin API tìm kiếm
            async def handle_response(response):
                if any(x in response.url for x in ["item_list", "search/item", "api/search"]):
                    try:
                        data = await response.json()
                        items = data.get("item_list") or data.get("data") or []
                        for it in items:
                            if isinstance(it, dict):
                                item_info = it.get("item") or it
                                vid_id = item_info.get("id") or item_info.get("video", {}).get("id")
                                author = item_info.get("author", {}).get("uniqueId") or item_info.get("author", {}).get("secUid", "tiktok_user")
                                if vid_id:
                                    v_url = f"https://www.tiktok.com/@{author}/video/{vid_id}"
                                    if not self.dm.is_url_crawled(v_url):
                                        video_urls.add(v_url)
                    except Exception:
                        pass

            page.on("response", handle_response)

            try:
                await page.goto(search_url, timeout=45000, wait_until="domcontentloaded")
                await page.wait_for_timeout(5000)

                scroll_attempts = 0
                while len(video_urls) < max_videos and scroll_attempts < 6:
                    links = await page.query_selector_all('a[href*="/video/"]')
                    for link in links:
                        href = await link.get_attribute("href")
                        if href and "/video/" in href:
                            full_url = href if href.startswith("http") else f"https://www.tiktok.com{href}"
                            clean_link = full_url.split("?")[0]
                            if not self.dm.is_url_crawled(clean_link):
                                video_urls.add(clean_link)
                            if len(video_urls) >= max_videos:
                                break

                    await page.evaluate("window.scrollBy(0, 1200)")
                    await page.wait_for_timeout(3000)
                    scroll_attempts += 1
            except Exception as e:
                print(f"[-] Lỗi khi tìm kiếm '{query}': {e}")
            finally:
                await browser.close()

        print(f"[+] Tìm thấy {len(video_urls)} video mới từ tìm kiếm '{query}'.")
        return list(video_urls)

    def download_audio_from_tiktok(self, video_url: str, base_name: str) -> Tuple[Optional[Path], Dict[str, Any]]:
        """Tải âm thanh TikTok chất lượng cao qua yt-dlp (kèm session cookie)."""
        out_template = str(self.temp_dir / f"{base_name}.%(ext)s")
        ydl_opts: Dict[str, Any] = {
            "format": "bestaudio/best",
            "outtmpl": out_template,
            "quiet": True,
            "no_warnings": True,
        }
        if self.cookie_txt_path.exists():
            ydl_opts["cookiefile"] = str(self.cookie_txt_path)

        info_dict: Dict[str, Any] = {}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                if info:
                    info_dict = {
                        "uploader": info.get("uploader", ""),
                        "uploader_id": info.get("uploader_id", ""),
                        "title": info.get("title", ""),
                        "description": info.get("description", ""),
                        "duration": info.get("duration", 0),
                    }

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
            print(f"[-] Lỗi yt-dlp tải video {video_url}: {e}")
        return None, info_dict

    def process_video_url(
        self,
        video_url: str,
        assigned_primary: Optional[str] = None,
        assigned_sub: Optional[str] = None,
        author: str = "tiktok_creator"
    ):
        """Xử lý toàn diện một video TikTok: Tải -> Convert -> Demucs -> VAD IPU -> ASR -> Filter -> Lưu."""
        if self.dm.is_url_crawled(video_url):
            print(f"[*] URL đã được xử lý trước đó: {video_url}")
            return

        self.stats["scanned_videos"] += 1
        base_id = f"tt_{uuid.uuid4().hex[:8]}"

        downloaded_wav, info_dict = self.download_audio_from_tiktok(video_url, base_id)
        if not downloaded_wav:
            return

        self.stats["processed_videos"] += 1
        real_author = info_dict.get("uploader_id") or info_dict.get("uploader") or author
        author_hash = hashlib.sha256(f"tiktok_{real_author}".encode("utf-8")).hexdigest()[:12]

        # 1. Tách nhạc nền nếu có cờ --remove-music
        std_wav_path = downloaded_wav
        clean_wav_path = self.temp_dir / f"clean_{base_id}.wav"
        if self.remove_music:
            print(f"[*] Đang tách nhạc nền (Vocal Isolation) bằng Demucs...")
            separated = self.vp.separate_vocals_demucs(std_wav_path, clean_wav_path)
            if separated and clean_wav_path.exists():
                std_wav_path = clean_wav_path
                print(f"  [+] Đã tách nhạc nền thành công! Sử dụng vocal sạch.")

        # 2. Phân đoạn theo khoảng lặng tự nhiên qua Silero-VAD
        chunks_dir = self.temp_dir / f"chunks_{base_id}"
        chunks = self.vp.split_long_audio_to_monologues(
            wav_path=std_wav_path,
            output_dir=chunks_dir,
            base_sample_id=base_id,
            chunk_duration_s=self.chunk_duration_s
        )

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

            # 3. Ước lượng cảm xúc âm học độc lập (Acoustic-first)
            acoustic_affect = self.vp.estimate_acoustic_affect(
                c_wav_path,
                prior_primary=assigned_primary or "SADNESS",
                prior_sub=assigned_sub or "LONELINESS"
            )
            voice_pri = acoustic_affect["primary"]
            voice_sub = acoustic_affect["sub"]
            emotion_score = acoustic_affect["confidence"]

            meta = {
                "source_platform": "tiktok_spontaneous_monologue",
                "author": real_author,
                "author_hash": author_hash,
                "source_url": video_url,
                "duration_s": c_duration,
                "voice_type": "spontaneous_monologue",
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

            # 4. Phiên âm qua PhoWhisper ASR và kiểm định tự sự qua JournalFilter
            spoken_text = ""
            is_crisis_chunk = False
            is_journal_entry = False

            if self.enable_asr and self.asr is not None:
                try:
                    asr_res = self.asr.transcribe(str(c_wav_path))
                    if asr_res and asr_res.text.strip():
                        spoken_text = asr_res.text.strip()
                        eval_asr = self.jf.evaluate_entry(spoken_text)

                        # Kiểm tra khủng hoảng tâm lý
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
                            print(f"  [!] CẢNH BÁO ASR: Phát hiện khủng hoảng -> Đã cách ly an toàn!")

                        # Đánh giá tính tự sự chân thật
                        elif eval_asr["accepted"]:
                            is_journal_entry = True
                            text_pri = eval_asr["primary_emotion"]
                            text_sub = eval_asr["sub_emotion"]
                            meta["text_primary"] = text_pri
                            meta["text_sub"] = text_sub

                            # Kiểm tra hiện tượng che giấu cảm xúc (Affective Incongruence / Masking)
                            if text_pri != voice_pri and (text_pri == "JOY" or voice_pri in ["SADNESS", "ANXIETY", "ANGER"]):
                                meta["incongruent_masking"] = True
                                print(f"  [*] PHÁT HIỆN MASKING: Lời nói '{text_pri}' nhưng Giọng nói '{voice_pri}'!")
                        else:
                            # Từ chối nếu lời nói là quảng cáo, bán hàng online trên TikTok
                            self.stats["rejected_filter"] += 1
                except Exception as e:
                    print(f"  [-] Lỗi PhoWhisper transcribe chunk {chunk_id}: {e}")

            if is_crisis_chunk:
                continue

            chunk_tier = meta.get("data_tier") or ("deep_journal" if is_journal_entry else "general_monologue")
            meta["data_tier"] = chunk_tier

            # 5. Lưu vào kho Voice thuần
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

            # 6. Lưu vào kho Text-Voice ĐA PHƯƠNG THỨC: Chỉ lưu khi lời nói được xác thực là tự sự (>= 5 từ)
            if spoken_text and is_journal_entry and len(spoken_text.split()) >= 5:
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

            self.stats["primary_counts"][voice_pri] = self.stats["primary_counts"].get(voice_pri, 0) + 1
            self.stats["sub_counts"][voice_sub] = self.stats["sub_counts"].get(voice_sub, 0) + 1
            mask_flag = " [MASKED]" if meta.get("incongruent_masking") else ""
            print(f"  [+] ĐÃ LƯU TIKTOK VOICE: {voice_id} ({c_duration}s | {voice_pri}:{voice_sub}{mask_flag} | RMS: {audit['rms_dbfs']}dBFS)")

        # Dọn dẹp file tạm
        for tmp in [downloaded_wav, clean_wav_path]:
            if tmp and tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass
        if chunks_dir.exists():
            import shutil
            shutil.rmtree(chunks_dir, ignore_errors=True)

    def print_summary(self):
        print("\n" + "="*65)
        print("TỔNG KẾT THU THẬP DỮ LIỆU TIKTOK SPONTANEOUS VOICE MER")
        print("="*65)
        print(f"Tổng video quét:                       {self.stats['scanned_videos']}")
        print(f"Tổng video đã xử lý:                   {self.stats['processed_videos']}")
        print(f"Số mẫu VOICE độc thoại tự sự (16kHz):  {self.stats['created_voice_chunks']} mẫu")
        print(f"Số cặp Text-Voice đa phương thức:      {self.stats['created_text_voice_pairs']} mẫu")
        print(f"Số đoạn bị từ chối do không đạt chuẩn: {self.stats['rejected_audit']} đoạn")
        print(f"Số đoạn bị từ chối do quảng cáo/spam:  {self.stats['rejected_filter']} đoạn")
        print(f"Số đoạn khủng hoảng tâm lý (cách ly):  {self.stats['quarantined_crisis']} đoạn")
        print("\nPhân bố cảm xúc theo Taxonomy:")
        print("  Primary Emotions:")
        for em, cnt in self.stats["primary_counts"].items():
            print(f"    * {em:<12}: {cnt} mẫu")
        print("  Sub Emotions:")
        for sub, cnt in self.stats["sub_counts"].items():
            print(f"    * {sub:<14}: {cnt} mẫu")
        print(f"\nMetadata tổng hợp: {self.dm.metadata_csv}")
        print("="*65)


async def main():
    parser = argparse.ArgumentParser(description="TikTok Spontaneous Voice Journaling Harvester")
    parser.add_argument("--profiles", nargs="*", help="Danh sách TikTok username cần quét (ví dụ: @tam.su.dem @goc.tam.su)")
    parser.add_argument("--sound-urls", nargs="*", help="Danh sách URL Original Sound cần quét")
    parser.add_argument("--search", nargs="*", help="Danh sách từ khóa tìm kiếm trên TikTok")
    parser.add_argument("--url", type=str, help="URL video TikTok cụ thể cần xử lý trực tiếp")
    parser.add_argument("--max-items", type=int, default=10, help="Số video tối đa cho mỗi profile/query")
    parser.add_argument("--chunk-duration", type=float, default=45.0, help="Thời lượng mỗi đoạn monologue")
    parser.add_argument("--output-dir", type=str, default="data/crawled")
    parser.add_argument("--transcribe", action="store_true", help="Bật PhoWhisper ASR để trích xuất transcript")
    parser.add_argument("--remove-music", action="store_true", help="Chạy Demucs tách nhạc nền (Vocal Isolation)")
    parser.add_argument("--primary", type=str, default="SADNESS", choices=["JOY", "SADNESS", "ANXIETY", "ANGER", "NEUTRAL"])
    parser.add_argument("--sub", type=str, default="LONELINESS")
    parser.add_argument("--no-headless", action="store_true", help="Chạy trình duyệt có giao diện")
    parser.add_argument("--list-creators", action="store_true", help="Xem danh sách Creator gợi ý")

    args = parser.parse_args()

    if args.list_creators:
        print("\n=== DANH SÁCH GỢI Ý CÁC TIKTOK CREATORS CHUYÊN VOICE JOURNAL & TÂM SỰ ===")
        for idx, c in enumerate(CURATED_TIKTOK_CREATORS, 1):
            print(f"[{idx}] {c['username']} - {c['description']}")
            print(f"    Cảm xúc mặc định: {c['default_emotion'][0]} / {c['default_emotion'][1]}")
        return

    crawler = TikTokVoiceCrawler(
        output_dir=args.output_dir,
        chunk_duration_s=args.chunk_duration,
        enable_asr=args.transcribe,
        remove_music=args.remove_music,
        headless=not args.no_headless
    )

    all_targets: List[str] = []

    if args.url:
        crawler.process_video_url(
            video_url=args.url,
            assigned_primary=args.primary,
            assigned_sub=args.sub
        )
    else:
        # 1. Quét theo Creator Profiles
        if args.profiles:
            for prof in args.profiles:
                vids = await crawler.harvest_from_profile(prof, max_videos=args.max_items)
                for v in vids:
                    crawler.process_video_url(v, assigned_primary=args.primary, assigned_sub=args.sub, author=prof)

        # 2. Quét theo Original Sounds
        if args.sound_urls:
            for s_url in args.sound_urls:
                vids = await crawler.harvest_from_sound(s_url, max_videos=args.max_items)
                for v in vids:
                    crawler.process_video_url(v, assigned_primary=args.primary, assigned_sub=args.sub, author="sound_user")

        # 3. Quét theo Search Queries
        if args.search:
            for q in args.search:
                vids = await crawler.harvest_from_search(q, max_videos=args.max_items)
                for v in vids:
                    crawler.process_video_url(v, assigned_primary=args.primary, assigned_sub=args.sub, author="search_user")

        if not (args.profiles or args.sound_urls or args.search):
            print("Vui lòng cung cấp tham số --profiles, --sound-urls, --search hoặc --url.")
            parser.print_help()
            return

    crawler.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
