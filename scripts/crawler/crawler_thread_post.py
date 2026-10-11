"""
crawler_thread_post.py
----------------------
Bộ cào chuyên dụng bóc tách Bài viết & Toàn bộ Bình luận (Comments / Replies)
từ một Post Threads cụ thể (ví dụ các post tâm sự, trải lòng, Q&A ẩn danh).

Cơ chế:
1. Truy cập URL bài post (hỗ trợ cả Guest mode lẫn nạp Cookies).
2. Tự động xử lý dialog/popup yêu cầu đăng nhập.
3. Cuộn trang và lắng nghe trực tiếp các phản hồi GraphQL / Relay payloads.
4. Trích xuất kép (Dual Extraction):
   - Quét cấu trúc DOM (các khối reply `div[data-pressable-container='true']`)
   - Bóc tách toàn bộ mảng dữ liệu JSON ngầm trong các thẻ `<script type="application/json">`
5. Áp dụng `JournalFilter` lọc tự sự nội tâm & gán nhãn Taxonomy chuẩn (`taxonomy.json`).
6. Lưu trữ vào `data/crawled/text/` và đồng bộ `metadata.csv` & `metadata.jsonl`.
"""

from __future__ import annotations

import argparse
import asyncio
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

from playwright.async_api import async_playwright

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.crawler.journal_filter import JournalFilter
from scripts.crawler.dataset_manager import CrawledDatasetManager
from scripts.crawler.voice_processor import VoiceProcessor

COOKIE_FILE = Path(__file__).resolve().parent / "threads_cookies.json"


class ThreadPostRepliesCrawler:
    """Cào bài viết và toàn bộ bình luận tâm sự từ một Post Threads."""

    def __init__(self, output_dir: str = "data/crawled", min_word_count: int = 8, min_duration_s: float = 3.0):
        self.dm = CrawledDatasetManager(output_dir)
        self.filter = JournalFilter(min_word_count=min_word_count)
        self.vp = VoiceProcessor(min_duration_s=min_duration_s)
        self.stats = {
            "total_extracted_raw": 0,
            "accepted_confessions": 0,
            "accepted_voice": 0,
            "accepted_text_voice": 0,
            "rejected_non_journal": 0,
            "rejected_spam": 0,
            "primary_counts": {},
            "sub_counts": {}
        }

    def _extract_from_json_tree(self, obj: Any, results: List[Dict[str, Any]]):
        """Duyệt đệ quy cây JSON để tìm tất cả các bài viết và comment."""
        if isinstance(obj, dict):
            caption = obj.get("caption")
            text = ""
            if isinstance(caption, dict) and "text" in caption:
                text = caption.get("text", "")
            elif isinstance(obj.get("text"), str):
                text = obj.get("text")

            user = obj.get("user", {})
            username = user.get("username") if isinstance(user, dict) else ""

            if text and username:
                results.append({
                    "username": username,
                    "text": text.strip(),
                    "id": str(obj.get("id") or obj.get("pk") or uuid.uuid4().hex[:8]),
                    "like_count": obj.get("like_count", 0),
                    "taken_at": obj.get("taken_at")
                })

            for v in obj.values():
                self._extract_from_json_tree(v, results)
        elif isinstance(obj, list):
            for item in obj:
                self._extract_from_json_tree(item, results)

    async def crawl_post(self, post_url: str, max_scrolls: int = 15, cookies_path: Optional[str] = None):
        print("\n" + "="*70)
        print(f"[*] BẮT ĐẦU CÀO BÀI VIẾT & BÌNH LUẬN TÂM SỰ TỪ THREADS")
        print(f"[*] URL: {post_url}")
        print("="*70)

        # Chuẩn hóa domain (threads.com -> threads.net)
        normalized_url = post_url.replace("threads.com", "threads.net")

        raw_comments: List[Dict[str, Any]] = []

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled", "--no-sandbox"]
            )
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                locale="vi-VN"
            )

            # Nạp cookie nếu có
            c_file = Path(cookies_path) if cookies_path else COOKIE_FILE
            if c_file.exists():
                try:
                    with open(c_file, "r", encoding="utf-8") as f:
                        cookies = json.load(f)
                        await context.add_cookies(cookies)
                    print(f"[*] Đã nạp phiên đăng nhập từ: {c_file.name}")
                except Exception as e:
                    print(f"[-] Không thể đọc file cookie: {e}")
            else:
                print(f"[*] Chạy ở chế độ Khách vãng lai (Guest Mode).")

            page = await context.new_page()

            # Lắng nghe GraphQL API responses ngầm
            async def on_response(resp):
                if "graphql" in resp.url or "api" in resp.url:
                    try:
                        ctype = resp.headers.get("content-type", "")
                        if "json" in ctype or "text" in ctype:
                            body = await resp.text()
                            if len(body) > 100:
                                parsed = json.loads(body)
                                self._extract_from_json_tree(parsed, raw_comments)
                    except Exception:
                        pass

            page.on("response", on_response)

            try:
                await page.goto(normalized_url, wait_until="networkidle", timeout=35000)
            except Exception as e:
                print(f"[-] Timeout/Lỗi tải trang: {e}")

            # Đóng bất kỳ login dialog nào nếu xuất hiện
            try:
                await page.keyboard.press("Escape")
                await asyncio.sleep(0.5)
                await page.evaluate("""() => {
                    const dialogs = document.querySelectorAll("div[role='dialog']");
                    dialogs.forEach(d => d.remove());
                    document.body.style.overflow = 'auto';
                    document.documentElement.style.overflow = 'auto';
                }""")
            except Exception:
                pass

            # Bóc tách tất cả JSON nhúng sẵn trong HTML ban đầu
            html = await page.content()
            script_matches = re.findall(r'<script type="application/json"[^>]*>(.*?)</script>', html, re.DOTALL)
            print(f"[*] Tìm thấy {len(script_matches)} khối dữ liệu JSON nhúng trong HTML ban đầu.")
            for sm in script_matches:
                try:
                    data = json.loads(sm)
                    self._extract_from_json_tree(data, raw_comments)
                except Exception:
                    pass

            # Cuộn trang để tải thêm các bình luận tiếp theo
            print(f"[*] Bắt đầu cuộn trang ({max_scrolls} lượt) để nạp thêm bình luận...")
            for s_idx in range(max_scrolls):
                await page.mouse.wheel(0, 2000)
                await asyncio.sleep(1.5)

                # Kiểm tra nút "Xem thêm" hoặc mở rộng trả lời
                try:
                    await page.keyboard.press("Escape")
                except Exception:
                    pass

            # Quét thêm trực tiếp từ cấu trúc DOM và tải video nếu có
            dom_containers = await page.query_selector_all("div[data-pressable-container='true']")
            print(f"[*] Số khối phần tử bình luận hiển thị trên DOM: {len(dom_containers)}")

            for c in dom_containers:
                try:
                    text_raw = await c.inner_text()
                    lines = [l.strip() for l in text_raw.split("\n") if l.strip() and not l.strip().isdigit()]
                    user = lines[0] if len(lines) > 1 else "unknown"
                    body = " ".join(lines[1:]) if len(lines) > 1 else (lines[0] if lines else "")

                    # Kiểm tra xem comment hoặc post có video/audio không
                    has_v = await c.query_selector("video")
                    v_src = await has_v.get_attribute("src") if has_v else None

                    if len(body) > 10:
                        raw_comments.append({
                            "username": user,
                            "text": body,
                            "id": uuid.uuid4().hex[:8],
                            "like_count": 0,
                            "taken_at": None,
                            "video_src": v_src
                        })

                    # Nếu có video/voice trong comment, xử lý tải luôn
                    if v_src:
                        eval_v = self.filter.evaluate_entry(body) if body else {"accepted": False}
                        pri = eval_v.get("primary_emotion", "SADNESS")
                        sub = eval_v.get("sub_emotion", "LONELINESS")
                        v_id = f"threads_cmt_v_{uuid.uuid4().hex[:8]}"

                        try:
                            v_resp = await page.request.get(v_src, timeout=30000)
                            if v_resp.status == 200:
                                v_bytes = await v_resp.body()
                                tmp_mp4 = f"temp_{v_id}.mp4"
                                tmp_wav = f"temp_{v_id}.wav"
                                with open(tmp_mp4, "wb") as f:
                                    f.write(v_bytes)
                                if self.vp.convert_to_16k_mono_wav(tmp_mp4, tmp_wav):
                                    audit = self.vp.inspect_and_audit(tmp_wav)
                                    if audit["is_valid"]:
                                        with open(tmp_wav, "rb") as f:
                                            wav_b = f.read()
                                        meta_v = {
                                            "source_platform": "threads_comment_video",
                                            "source_url": normalized_url,
                                            "author": user,
                                            "duration_s": audit["duration_s"],
                                            "rms_dbfs": audit["rms_dbfs"]
                                        }
                                        self.dm.save_voice_sample(
                                            sample_id=v_id,
                                            wav_bytes=wav_b,
                                            primary_label=pri,
                                            sub_label=sub,
                                            emotion_score=0.9,
                                            duration_s=audit["duration_s"],
                                            meta=meta_v
                                        )
                                        self.stats["accepted_voice"] += 1
                                        if body:
                                            self.dm.save_text_voice_sample(
                                                sample_id=v_id,
                                                wav_bytes=wav_b,
                                                text=body,
                                                primary_label=pri,
                                                sub_label=sub,
                                                emotion_score=0.9,
                                                duration_s=audit["duration_s"],
                                                meta=meta_v
                                            )
                                            self.stats["accepted_text_voice"] += 1
                                        print(f"  [+] ĐÃ LƯU VOICE TỪ COMMENT ({audit['duration_s']}s, {pri}:{sub})")
                                for tmp in [tmp_mp4, tmp_wav]:
                                    if os.path.exists(tmp):
                                        os.remove(tmp)
                        except Exception as e:
                            print(f"  [-] Lỗi tải video comment: {e}")

                except Exception:
                    pass

            await browser.close()

        # Deduplicate các bình luận thu được
        seen: Set[str] = set()
        unique_comments: List[Dict[str, Any]] = []

        for item in raw_comments:
            content = item["text"].strip()
            # Khóa chống trùng: 40 ký tự đầu của nội dung
            dedup_key = content[:50].lower()
            if dedup_key not in seen and len(content) >= 10:
                seen.add(dedup_key)
                unique_comments.append(item)

        self.stats["total_extracted_raw"] = len(unique_comments)
        print(f"\n[*] ĐÃ TRÍCH XUẤT ĐƯỢC {len(unique_comments)} BÌNH LUẬN DUY NHẤT.")
        print(f"[*] Đang chạy qua JournalFilter để lọc tâm sự chuẩn & gán nhãn cảm xúc...")

        # Đánh giá & Lưu trữ vào Dataset
        for c_idx, c in enumerate(unique_comments, 1):
            text = c["text"]
            user = c["username"]

            eval_res = self.filter.evaluate_entry(text)

            if not eval_res["accepted"]:
                if "thương mại" in eval_res.get("reason", ""):
                    self.stats["rejected_spam"] += 1
                else:
                    self.stats["rejected_non_journal"] += 1
                continue

            pri_label = eval_res["primary_emotion"]
            sub_label = eval_res["sub_emotion"]
            score = eval_res["emotion_score"]

            sample_id = f"threads_cmt_{uuid.uuid4().hex[:8]}"
            meta = {
                "source_platform": "threads_comment",
                "source_url": normalized_url,
                "author": user,
                "like_count": c.get("like_count", 0),
                "narrative_score": eval_res["narrative_score"],
                "matched_pronouns": eval_res["matched_pronouns"],
                "matched_reflective": eval_res["matched_reflective"],
                "matched_keywords": eval_res["matched_keywords"]
            }

            self.dm.save_text_sample(
                sample_id=sample_id,
                text=text,
                primary_label=pri_label,
                sub_label=sub_label,
                emotion_score=score,
                meta=meta
            )

            self.stats["accepted_confessions"] += 1
            self.stats["primary_counts"][pri_label] = self.stats["primary_counts"].get(pri_label, 0) + 1
            self.stats["sub_counts"][sub_label] = self.stats["sub_counts"].get(sub_label, 0) + 1

            print(f"  [+] ({self.stats['accepted_confessions']}) Lưu tâm sự @{user} -> [{pri_label} / {sub_label}] (Score: {score})")
            print(f"      Trích đoạn: \"{text[:90]}...\"")

        self.print_summary()

    def print_summary(self):
        print("\n" + "="*70)
        print("TỔNG KẾT KẾT QUẢ CÀO BÌNH LUẬN TÂM SỰ TỪ THREAD")
        print("="*70)
        print(f"Tổng bình luận thô trích xuất được:    {self.stats['total_extracted_raw']}")
        print(f"Số mẫu TÂM SỰ ĐẠT CHUẨN đã lưu:       {self.stats['accepted_confessions']}")
        print(f"Số bình luận loại bỏ (không tự sự):    {self.stats['rejected_non_journal']}")
        print(f"Số bình luận loại bỏ (quảng cáo/spam): {self.stats['rejected_spam']}")
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
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Threads Post & Comments Confession Crawler")
    parser.add_argument("--url", type=str, default="https://www.threads.net/@tieulacdaily/post/DXmGkZ8k5ZF",
                        help="URL bài post Threads cần cào bình luận")
    parser.add_argument("--output-dir", type=str, default="data/crawled", help="Thư mục xuất dữ liệu")
    parser.add_argument("--max-scrolls", type=int, default=10, help="Số lần cuộn trang để tải thêm comment")
    parser.add_argument("--min-words", type=int, default=8, help="Số từ tối thiểu cho một comment tâm sự")
    parser.add_argument("--min-duration", type=float, default=3.0, help="Thời lượng audio tối thiểu (giây), mặc định 3.0s")
    parser.add_argument("--cookies", type=str, default=str(COOKIE_FILE), help="Đường dẫn file cookies json")

    args = parser.parse_args()

    crawler = ThreadPostRepliesCrawler(
        output_dir=args.output_dir,
        min_word_count=args.min_words,
        min_duration_s=args.min_duration
    )
    asyncio.run(crawler.crawl_post(
        post_url=args.url,
        max_scrolls=args.max_scrolls,
        cookies_path=args.cookies
    ))


if __name__ == "__main__":
    main()
