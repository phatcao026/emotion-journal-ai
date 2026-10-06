"""
crawler_threads.py
------------------
Pipeline thu thập dữ liệu Nhật ký / Tâm sự tiếng Việt từ Threads cho bài toán MER.

Đặc điểm:
1. Sử dụng Playwright tương tác trình duyệt.
2. Tích hợp `JournalFilter` lọc nghiêm ngặt:
   - Loại trừ hoàn toàn quảng cáo, PR, tin tức showbiz, review công nghệ.
   - Bắt buộc văn phong tự sự ngôi thứ nhất (người thật trải lòng).
   - Phân loại cảm xúc chuẩn hóa theo 5 Primary & 11 Sub taxonomy.
3. Tách biệt 3 luồng modality:
   - `text/`: Bài viết văn bản tâm sự thuần.
   - `voice/`: Trích xuất âm thanh độc thoại giọng người 16kHz mono.
   - `text_voice/`: Cặp đôi âm thanh + transcript văn bản khớp nhau.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.parse
import uuid
import wave
from pathlib import Path
from typing import Dict, List, Optional

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import imageio_ffmpeg
from playwright.async_api import async_playwright

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.crawler.journal_filter import JournalFilter
from scripts.crawler.dataset_manager import CrawledDatasetManager
from scripts.crawler.voice_processor import VoiceProcessor

COOKIE_FILE = Path(__file__).resolve().parent / "threads_cookies.json"

DEFAULT_SEARCH_QUERIES = [
    # 1. Hoạt động đời sống & Cột mốc cá nhân (Life Activities & Milestones)
    "tập gym",
    "concert",
    "hành trình",
    "ngày đầu tiên đi làm",
    "xa nhà",
    "thay đổi bản thân",
    # 2. Trạng thái tâm lý & Cảm xúc (Emotional States)
    "áp lực",
    "áp lực đồng trang lứa",
    "một mình",
    "cô đơn",
    "chữa lành",
    # 3. Văn phong nhật ký & Tự sự (Diary & Introspective Narratives)
    "tâm sự",
    "nhật ký",
    "hôm nay của mình",
    "chợt nhận ra"
]


def clean_post_text(raw_text: str) -> str:
    """Loại bỏ các thành phần rác giao diện (UI artifacts) của Threads:
    - Các dòng điều hướng: 'Xem thêm', 'Dịch', 'Trả lời', 'Theo dõi'...
    - Huy hiệu tác giả: '· Tác giả'
    - Tiền tố thời gian / ngày tháng ở đầu bài: '27/04/2026', '3 giờ trước', '5 ngày'...
    """
    if not raw_text:
        return ""
    lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
    cleaned_lines = []
    for line in lines:
        if line.isdigit() or line.lower() in {"xem thêm", "dịch", "trả lời", "theo dõi", "đã theo dõi", "gợi ý cho bạn"}:
            continue
        line = re.sub(r"^[·•]?\s*tác giả\s*$", "", line, flags=re.IGNORECASE).strip()
        if not line:
            continue
        cleaned_lines.append(line)
    text = " ".join(cleaned_lines)
    # Loại bỏ header thời gian/ngày tháng ở đầu bài
    text = re.sub(
        r"^(?:\d{1,2}/\d{1,2}/\d{4}|\d+\s*(?:giờ|ngày|phút|tuần|tháng|năm|s|m|h|d|w)(?:\s*trước)?)\s*[·•\-–—]?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )
    # Loại bỏ nhãn tác giả còn sót lại
    text = re.sub(r"\s*[·•]\s*tác giả\b", "", text, flags=re.IGNORECASE)
    return text.strip()


class ThreadsJournalCrawler:
    """Crawler chuyên dụng cho dữ liệu Nhật ký & Tâm sự đa phương thức trên Threads."""

    def __init__(self, output_dir: str = "data/crawled", min_word_count: int = 15, min_duration_s: float = 3.0):
        self.dm = CrawledDatasetManager(output_dir)
        self.filter = JournalFilter(min_word_count=min_word_count)
        self.vp = VoiceProcessor(min_duration_s=min_duration_s, max_duration_s=180.0)
        self.stats = {
            "scanned_posts": 0,
            "accepted_text": 0,
            "accepted_voice": 0,
            "accepted_text_voice": 0,
            "rejected_spam": 0,
            "rejected_non_journal": 0,
            "rejected_invalid_voice": 0,
            "primary_emotions": {},
            "sub_emotions": {}
        }

    def convert_video_to_wav(self, video_path: str, output_wav_path: str) -> Optional[float]:
        """Trích xuất stream âm thanh từ video sang chuẩn 16kHz Mono WAV."""
        cmd = [
            FFMPEG_EXE,
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(output_wav_path),
            "-y"
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0 and os.path.exists(output_wav_path):
                with wave.open(str(output_wav_path), "rb") as w:
                    duration = round(w.getnframes() / float(w.getframerate()), 2)
                    return duration
        except Exception as e:
            print(f"  [FFmpeg Error] {e}")
        return None

    async def crawl_profile(self, page, username: str, max_items: int = 15):
        clean_user = username.strip().lstrip("@")
        print(f"\n" + "="*60)
        print(f"[*] Đang quét tài khoản tâm sự / nhật ký: @{clean_user}")
        print("="*60)

        url = f"https://www.threads.net/@{clean_user}"
        try:
            await page.goto(url, wait_until="networkidle", timeout=35000)
        except Exception as e:
            print(f"[-] Không thể mở @{clean_user}: {e}")
            return

        # Cuộn trang tự động để load bài viết
        for _ in range(4):
            await page.mouse.wheel(0, 1500)
            await asyncio.sleep(2)

        # 1. Quét các bài viết có Video/Audio (Voice & Text-Voice)
        videos = await page.query_selector_all("video")
        print(f"[*] Tìm thấy {len(videos)} video trên trang @{clean_user}.")

        for v_idx, v in enumerate(videos[:max_items]):
            src = await v.get_attribute("src")
            if not src:
                continue

            # Lấy text mô tả đi kèm video
            caption_text = ""
            try:
                parent = await v.evaluate_handle("el => el.closest('div[data-pressable-container=\"true\"]')")
                if parent and parent.as_element():
                    raw = await parent.as_element().inner_text()
                    caption_text = clean_post_text(raw)
            except Exception:
                pass

            # Đánh giá qua JournalFilter
            eval_res = self.filter.evaluate_entry(caption_text)
            self.stats["scanned_posts"] += 1

            if not eval_res["accepted"]:
                if "thương mại" in eval_res.get("reason", ""):
                    self.stats["rejected_spam"] += 1
                else:
                    self.stats["rejected_non_journal"] += 1
                print(f"  [-] Bỏ qua video {v_idx+1}: {eval_res['reason']}")
                continue

            pri_label = eval_res["primary_emotion"]
            sub_label = eval_res["sub_emotion"]
            score = eval_res["emotion_score"]

            sample_id = f"threads_{clean_user}_tv_{uuid.uuid4().hex[:8]}"
            print(f"  [+] Chấp nhận video {v_idx+1} ({pri_label}/{sub_label}) -> Đang tải...")

            try:
                resp = await page.request.get(src, timeout=30000)
                if resp.status == 200:
                    video_bytes = await resp.body()
                    temp_mp4 = f"temp_{sample_id}.mp4"
                    temp_wav = f"temp_{sample_id}.wav"

                    with open(temp_mp4, "wb") as f:
                        f.write(video_bytes)

                    ok = self.vp.convert_to_16k_mono_wav(temp_mp4, temp_wav)
                    if ok:
                        audit = self.vp.inspect_and_audit(temp_wav)
                        duration = audit["duration_s"]

                        if not audit["is_valid"]:
                            self.stats["rejected_invalid_voice"] += 1
                            print(f"    [-] Bỏ qua audio video {v_idx+1}: {audit['reason']}")
                        else:
                            with open(temp_wav, "rb") as f:
                                wav_bytes = f.read()

                            meta = {
                                "source_platform": "threads",
                                "author": clean_user,
                                "source_url": url,
                                "narrative_score": eval_res["narrative_score"],
                                "matched_pronouns": eval_res["matched_pronouns"],
                                "matched_reflective": eval_res["matched_reflective"],
                                "matched_keywords": eval_res["matched_keywords"],
                                "rms_dbfs": audit["rms_dbfs"],
                                "speech_ratio": audit["speech_ratio"]
                            }

                            # Lưu [3] text_voice (Multimodal Pair)
                            self.dm.save_text_voice_sample(
                                sample_id=sample_id,
                                wav_bytes=wav_bytes,
                                text=caption_text,
                                primary_label=pri_label,
                                sub_label=sub_label,
                                emotion_score=score,
                                duration_s=duration,
                                meta=meta
                            )
                            self.stats["accepted_text_voice"] += 1

                            # Lưu [2] voice thuần
                            voice_id = f"{sample_id}_v"
                            self.dm.save_voice_sample(
                                sample_id=voice_id,
                                wav_bytes=wav_bytes,
                                primary_label=pri_label,
                                sub_label=sub_label,
                                emotion_score=score,
                                duration_s=duration,
                                meta=meta
                            )
                            self.stats["accepted_voice"] += 1

                            self.stats["primary_emotions"][pri_label] = self.stats["primary_emotions"].get(pri_label, 0) + 1
                            self.stats["sub_emotions"][sub_label] = self.stats["sub_emotions"].get(sub_label, 0) + 1
                            print(f"    -> Đã lưu cặp Text-Voice & Voice ({duration}s, {pri_label}:{sub_label}, RMS: {audit['rms_dbfs']}dB)")

                    for tmp in [temp_mp4, temp_wav]:
                        if os.path.exists(tmp):
                            try:
                                os.remove(tmp)
                            except Exception:
                                pass
            except Exception as e:
                print(f"  [-] Lỗi xử lý media video: {e}")

        # 2. Quét các bài viết Text thuần
        containers = await page.query_selector_all("div[data-pressable-container='true']")
        print(f"[*] Đang quét {len(containers)} bài viết văn bản...")

        for c_idx, c in enumerate(containers[:max_items]):
            has_v = await c.query_selector("video")
            if has_v:
                continue

            raw = await c.inner_text()
            text = clean_post_text(raw)
            if not text:
                continue

            eval_res = self.filter.evaluate_entry(text)
            self.stats["scanned_posts"] += 1

            if not eval_res["accepted"]:
                if "thương mại" in eval_res.get("reason", ""):
                    self.stats["rejected_spam"] += 1
                else:
                    self.stats["rejected_non_journal"] += 1
                continue

            pri_label = eval_res["primary_emotion"]
            sub_label = eval_res["sub_emotion"]
            score = eval_res["emotion_score"]
            author_hash = hashlib.sha256(f"threads_{clean_user}".encode("utf-8")).hexdigest()[:12]

            sample_id = f"threads_{clean_user}_txt_{uuid.uuid4().hex[:8]}"
            meta = {
                "source_platform": "threads",
                "author": clean_user,
                "author_hash": author_hash,
                "source_url": url,
                "narrative_score": eval_res["narrative_score"],
                "matched_pronouns": eval_res["matched_pronouns"],
                "matched_reflective": eval_res["matched_reflective"],
                "matched_keywords": eval_res["matched_keywords"]
            }

            if eval_res.get("is_crisis"):
                self.dm.save_crisis_sample(
                    sample_id=sample_id,
                    text=text,
                    crisis_keywords=eval_res.get("crisis_keywords", []),
                    primary_label=pri_label,
                    sub_label=sub_label,
                    meta=meta
                )
                print(f"  [!] CẢNH BÁO: Phát hiện bài khủng hoảng tâm lý ({sample_id}) -> Đã cách ly vào crisis_quarantine/")
            else:
                self.dm.save_text_sample(
                    sample_id=sample_id,
                    text=text,
                    primary_label=pri_label,
                    sub_label=sub_label,
                    emotion_score=score,
                    meta=meta
                )
                self.stats["accepted_text"] += 1
                self.stats["primary_emotions"][pri_label] = self.stats["primary_emotions"].get(pri_label, 0) + 1
                self.stats["sub_emotions"][sub_label] = self.stats["sub_emotions"].get(sub_label, 0) + 1
                print(f"  [+] THÀNH CÔNG: Lưu bài tâm sự Text ({sample_id}) -> [{pri_label} / {sub_label}] (Score: {score})")

    async def crawl_search_query(self, page, query: str, max_items: int = 15):
        print("\n" + "="*60)
        print(f"[*] Đang tìm kiếm từ khóa tâm sự trên Threads: '{query}'")
        print("="*60)

        encoded_q = urllib.parse.quote(query)
        url = f"https://www.threads.net/search?q={encoded_q}&filter=recent&serp_type=default"
        try:
            await page.goto(url, wait_until="networkidle", timeout=35000)
        except Exception as e:
            print(f"[-] Lỗi mở trang tìm kiếm '{query}': {e}")
            return

        for _ in range(5):
            await page.mouse.wheel(0, 1500)
            await asyncio.sleep(2)

        videos = await page.query_selector_all("video")
        print(f"[*] Tìm thấy {len(videos)} video từ kết quả tìm kiếm '{query}'.")

        for v_idx, v in enumerate(videos[:max_items]):
            src = await v.get_attribute("src")
            if not src:
                continue

            caption_text = ""
            try:
                parent = await v.evaluate_handle("el => el.closest('div[data-pressable-container=\"true\"]')")
                if parent and parent.as_element():
                    raw = await parent.as_element().inner_text()
                    caption_text = clean_post_text(raw)
            except Exception:
                pass

            eval_res = self.filter.evaluate_entry(caption_text)
            self.stats["scanned_posts"] += 1

            if not eval_res["accepted"]:
                if "thương mại" in eval_res.get("reason", ""):
                    self.stats["rejected_spam"] += 1
                else:
                    self.stats["rejected_non_journal"] += 1
                continue

            pri_label = eval_res["primary_emotion"]
            sub_label = eval_res["sub_emotion"]
            score = eval_res["emotion_score"]

            sample_id = f"threads_search_{uuid.uuid4().hex[:8]}"
            try:
                resp = await page.request.get(src, timeout=30000)
                if resp.status == 200:
                    video_bytes = await resp.body()
                    temp_mp4 = f"temp_{sample_id}.mp4"
                    temp_wav = f"temp_{sample_id}.wav"

                    with open(temp_mp4, "wb") as f:
                        f.write(video_bytes)

                    ok = self.vp.convert_to_16k_mono_wav(temp_mp4, temp_wav)
                    if ok:
                        audit = self.vp.inspect_and_audit(temp_wav)
                        duration = audit["duration_s"]

                        if not audit["is_valid"]:
                            self.stats["rejected_invalid_voice"] += 1
                        else:
                            with open(temp_wav, "rb") as f:
                                wav_bytes = f.read()

                            meta = {
                                "source_platform": "threads_search",
                                "search_query": query,
                                "source_url": url,
                                "narrative_score": eval_res["narrative_score"],
                                "matched_pronouns": eval_res["matched_pronouns"],
                                "matched_reflective": eval_res["matched_reflective"],
                                "matched_keywords": eval_res["matched_keywords"],
                                "rms_dbfs": audit["rms_dbfs"],
                                "speech_ratio": audit["speech_ratio"]
                            }

                            self.dm.save_text_voice_sample(
                                sample_id=sample_id,
                                wav_bytes=wav_bytes,
                                text=caption_text,
                                primary_label=pri_label,
                                sub_label=sub_label,
                                emotion_score=score,
                                duration_s=duration,
                                meta=meta
                            )
                            self.stats["accepted_text_voice"] += 1

                            voice_id = f"{sample_id}_v"
                            self.dm.save_voice_sample(
                                sample_id=voice_id,
                                wav_bytes=wav_bytes,
                                primary_label=pri_label,
                                sub_label=sub_label,
                                emotion_score=score,
                                duration_s=duration,
                                meta=meta
                            )
                            self.stats["accepted_voice"] += 1

                            self.stats["primary_emotions"][pri_label] = self.stats["primary_emotions"].get(pri_label, 0) + 1
                            self.stats["sub_emotions"][sub_label] = self.stats["sub_emotions"].get(sub_label, 0) + 1
                            print(f"    -> Đã lưu cặp Text-Voice & Voice ({duration}s, {pri_label}:{sub_label}, RMS: {audit['rms_dbfs']}dB)")

                    for tmp in [temp_mp4, temp_wav]:
                        if os.path.exists(tmp):
                            try:
                                os.remove(tmp)
                            except Exception:
                                pass
            except Exception as e:
                print(f"  [-] Lỗi tải video tìm kiếm: {e}")

        # 2. Quét các bài viết Text thuần từ kết quả tìm kiếm
        containers = await page.query_selector_all("div[data-pressable-container='true']")
        print(f"[*] Đang quét {len(containers)} bài viết văn bản từ kết quả tìm kiếm '{query}'...")

        for c_idx, c in enumerate(containers[:max_items]):
            has_v = await c.query_selector("video")
            if has_v:
                continue

            raw = await c.inner_text()
            text = clean_post_text(raw)
            if not text:
                continue

            eval_res = self.filter.evaluate_entry(text)
            self.stats["scanned_posts"] += 1

            if not eval_res["accepted"]:
                if "thương mại" in eval_res.get("reason", ""):
                    self.stats["rejected_spam"] += 1
                else:
                    self.stats["rejected_non_journal"] += 1
                continue

            pri_label = eval_res["primary_emotion"]
            sub_label = eval_res["sub_emotion"]
            score = eval_res["emotion_score"]
            sample_id = f"threads_search_txt_{uuid.uuid4().hex[:8]}"
            meta = {
                "source_platform": "threads_search",
                "search_query": query,
                "source_url": url,
                "narrative_score": eval_res["narrative_score"],
                "matched_pronouns": eval_res["matched_pronouns"],
                "matched_reflective": eval_res["matched_reflective"],
                "matched_keywords": eval_res["matched_keywords"]
            }

            if eval_res.get("is_crisis"):
                self.dm.save_crisis_sample(
                    sample_id=sample_id,
                    text=text,
                    crisis_keywords=eval_res.get("crisis_keywords", []),
                    primary_label=pri_label,
                    sub_label=sub_label,
                    meta=meta
                )
                print(f"  [!] CẢNH BÁO: Phát hiện bài khủng hoảng tâm lý ({sample_id}) -> Đã cách ly vào crisis_quarantine/")
            else:
                self.dm.save_text_sample(
                    sample_id=sample_id,
                    text=text,
                    primary_label=pri_label,
                    sub_label=sub_label,
                    emotion_score=score,
                    meta=meta
                )
                self.stats["accepted_text"] += 1
                self.stats["primary_emotions"][pri_label] = self.stats["primary_emotions"].get(pri_label, 0) + 1
                self.stats["sub_emotions"][sub_label] = self.stats["sub_emotions"].get(sub_label, 0) + 1
                print(f"  [+] THÀNH CÔNG: Lưu bài tâm sự Text ({sample_id}) -> [{pri_label} / {sub_label}] (Score: {score})")

    def print_summary(self):
        print("\n" + "="*65)
        print("TỔNG KẾT PIPELINE THU THẬP NHẬT KÝ & TÂM SỰ CẢM XÚC")
        print("="*65)
        print(f"Tổng bài viết đã duyệt:                {self.stats['scanned_posts']}")
        print(f"Số bài loại bỏ do dính quảng cáo/PR:  {self.stats['rejected_spam']}")
        print(f"Số bài loại bỏ do không phải nhật ký: {self.stats['rejected_non_journal']}")
        print(f"Số bài loại bỏ do audio không hợp lệ: {self.stats['rejected_invalid_voice']}")
        print(f"- [1] Dữ liệu Text tâm sự chuẩn:       {self.stats['accepted_text']} mẫu")
        print(f"- [2] Dữ liệu Voice độc thoại chuẩn:   {self.stats['accepted_voice']} mẫu")
        print(f"- [3] Cặp Multimodal Text-Voice:       {self.stats['accepted_text_voice']} mẫu")
        print("\nPhân bố cảm xúc theo Taxonomy:")
        print("  Primary Emotions:")
        for em, cnt in self.stats["primary_emotions"].items():
            print(f"    * {em:<12}: {cnt} mẫu")
        print("  Sub Emotions:")
        for sub, cnt in self.stats["sub_emotions"].items():
            print(f"    * {sub:<14}: {cnt} mẫu")
        print(f"\nMetadata tổng hợp:")
        print(f"  - CSV:   {self.dm.metadata_csv}")
        print(f"  - JSONL: {self.dm.metadata_jsonl}")
        print("="*65)


async def main():
    parser = argparse.ArgumentParser(description="Threads Emotion Journal & Confession Crawler")
    parser.add_argument("--targets", nargs="*", default=None,
                        help="Danh sách username Threads cần quét")
    parser.add_argument("--search", nargs="*", default=None,
                        help="Danh sách từ khóa tìm kiếm trên Threads (vd: 'tâm sự' 'nhật ký' 'áp lực')")
    parser.add_argument("--cookies", type=str, default=str(COOKIE_FILE),
                        help="Đường dẫn file cookies json")
    parser.add_argument("--output-dir", type=str, default="data/crawled", help="Thư mục xuất dữ liệu")
    parser.add_argument("--max-items", type=int, default=10, help="Số bài tối đa quét mỗi mục")
    parser.add_argument("--min-words", type=int, default=15, help="Số từ tối thiểu cho một bài nhật ký/tâm sự")
    parser.add_argument("--min-duration", type=float, default=3.0, help="Thời lượng âm thanh tối thiểu (giây), mặc định 3.0s")
    parser.add_argument("--headless", action="store_true", default=True, help="Chạy browser ẩn danh")
    args = parser.parse_args()

    crawler = ThreadsJournalCrawler(
        output_dir=args.output_dir,
        min_word_count=args.min_words,
        min_duration_s=args.min_duration
    )

    cookie_path = Path(args.cookies)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=args.headless,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"]
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            locale="vi-VN"
        )

        # Nạp cookies phiên đăng nhập nếu có
        if cookie_path.exists():
            try:
                with open(cookie_path, "r", encoding="utf-8") as f:
                    cookies = json.load(f)
                    await context.add_cookies(cookies)
                print(f"[*] Đã nạp thành công {len(cookies)} cookies từ {cookie_path.name}")
            except Exception as e:
                print(f"[-] Lỗi nạp cookies: {e}")
        else:
            print(f"[*] Chưa có file cookies ({cookie_path.name}). Chạy ở chế độ khách vãng lai.")

        page = await context.new_page()

        # Quét theo danh sách tài khoản
        if args.targets:
            for target in args.targets:
                await crawler.crawl_profile(page, target, max_items=args.max_items)

        # Quét theo từ khóa tìm kiếm (Search / Hashtags)
        if args.search is not None:
            queries = args.search if len(args.search) > 0 else DEFAULT_SEARCH_QUERIES
            for query in queries:
                await crawler.crawl_search_query(page, query, max_items=args.max_items)

        # Nếu không truyền gì, mặc định quét theo danh sách từ khóa tiêu biểu
        if not args.targets and args.search is None:
            for query in DEFAULT_SEARCH_QUERIES[:4]:
                await crawler.crawl_search_query(page, query, max_items=args.max_items)

        await browser.close()

    crawler.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
