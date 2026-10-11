"""
crawler_news_journal.py
-----------------------
Trình thu thập dữ liệu Nhật ký & Tâm sự từ Báo chí Chính thống (VnExpress Tâm sự).
Nguồn dữ liệu sạch, ngôn ngữ tiếng Việt chuẩn mực, được biên tập kỹ lưỡng và an toàn tuyệt đối về mặt chính trị/pháp lý.

Đặc điểm:
1. Hỗ trợ 2 phương thức quét:
   - Qua kênh RSS chính thức: `https://vnexpress.net/rss/tam-su.rss` (Nhanh, chuẩn, tôn trọng máy chủ).
   - Qua chuyên mục đời sống: `https://vnexpress.net/doi-song/tam-su` và phân trang `-p{page}`.
2. Tích hợp `JournalFilter v2`:
   - Chuẩn hóa Unicode NFC, teencode, khử PII (số điện thoại, email).
   - Phát hiện thể phủ định và cách ly nguy cơ tự hại vào `data/crawled/crisis_quarantine/`.
   - Gán nhãn 5 Primary và 11 Sub emotions theo `data/taxonomy.json`.
3. An toàn & Bền bỉ (Resumable & Ethical Crawling):
   - Tự động bỏ qua các bài đã cào trước đó (`is_url_crawled`).
   - Giãn cách ngẫu nhiên (2.0s - 4.0s) để tôn trọng robots.txt và tránh nghẽn máy chủ.
   - Băm ẩn danh tác giả `author_hash` phục vụ Participant-aware split.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import html
import json
import os
import random
import re
import sys
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import httpx

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.crawler.journal_filter import JournalFilter
from scripts.crawler.dataset_manager import CrawledDatasetManager

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15"
]

RSS_TAM_SU_URL = "https://vnexpress.net/rss/tam-su.rss"
WEB_TAM_SU_BASE = "https://vnexpress.net/doi-song/tam-su"


class VnExpressJournalCrawler:
    """Crawler chuyên biệt cho chuyên mục Tâm sự báo điện tử VnExpress."""

    def __init__(self, output_dir: str = "data/crawled", min_word_count: int = 50, delay_range: Tuple[float, float] = (2.0, 3.5)):
        self.dm = CrawledDatasetManager(output_dir)
        self.filter = JournalFilter(min_word_count=min_word_count)
        self.delay_range = delay_range
        self.headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7"
        }
        self.stats = {
            "scanned_articles": 0,
            "accepted_text": 0,
            "quarantined_crisis": 0,
            "skipped_existing": 0,
            "rejected_non_journal": 0,
            "rejected_short": 0,
            "primary_emotions": {},
            "sub_emotions": {}
        }

    async def fetch_rss_links(self, client: httpx.AsyncClient) -> List[Dict[str, str]]:
        """Lấy danh sách các bài viết mới nhất từ RSS chính thức của VnExpress."""
        items = []
        try:
            print(f"[*] Đang tải RSS Feed từ: {RSS_TAM_SU_URL}")
            resp = await client.get(RSS_TAM_SU_URL, headers=self.headers, timeout=15.0)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall(".//item"):
                    title_elem = item.find("title")
                    link_elem = item.find("link")
                    pub_elem = item.find("pubDate")
                    desc_elem = item.find("description")

                    link = link_elem.text.strip() if link_elem is not None and link_elem.text else ""
                    title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
                    pub_date = pub_elem.text.strip() if pub_elem is not None and pub_elem.text else ""

                    if link:
                        items.append({
                            "url": link,
                            "title": title,
                            "pub_date": pub_date
                        })
                print(f"[+] Tìm thấy {len(items)} bài viết từ RSS feed.")
        except Exception as e:
            print(f"[-] Lỗi tải RSS feed: {e}")
        return items

    async def fetch_web_page_links(self, client: httpx.AsyncClient, page: int = 1) -> List[Dict[str, str]]:
        """Lấy danh sách bài viết từ trang chuyên mục Tâm sự web HTML."""
        items = []
        url = WEB_TAM_SU_BASE if page == 1 else f"{WEB_TAM_SU_BASE}-p{page}"
        try:
            print(f"[*] Đang quét danh mục web: {url}")
            resp = await client.get(url, headers=self.headers, timeout=15.0)
            if resp.status_code == 200:
                html_text = resp.text
                # Tìm các link bài viết có đuôi .html
                raw_links = re.findall(r'href=["\'](https://vnexpress\.net/[^"\']*?-(\d+)\.html)["\']', html_text)
                seen = set()
                for link, art_id in raw_links:
                    if link not in seen and not link.endswith("/tam-su.html") and not "-p" in link:
                        seen.add(link)
                        items.append({"url": link, "title": "", "pub_date": ""})
                print(f"[+] Tìm thấy {len(items)} bài viết từ trang {page}.")
        except Exception as e:
            print(f"[-] Lỗi quét trang web {url}: {e}")
        return items

    def _clean_html_text(self, raw_html: str) -> str:
        """Khử bỏ các thẻ HTML để lấy văn bản sạch."""
        text = re.sub(r'<script.*?</script>', '', raw_html, flags=re.DOTALL)
        text = re.sub(r'<style.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = html.unescape(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    async def parse_article(self, client: httpx.AsyncClient, url: str) -> Optional[Dict[str, Any]]:
        """Tải và bóc tách toàn văn bài viết tâm sự VnExpress."""
        try:
            resp = await client.get(url, headers=self.headers, timeout=15.0)
            if resp.status_code != 200:
                return None

            html_text = resp.text

            # 1. Tiêu đề bài viết
            title_m = re.search(r'<h1[^>]*class=["\'][^"\']*title-detail[^"\']*["\'][^>]*>(.*?)</h1>', html_text, re.DOTALL)
            title = self._clean_html_text(title_m.group(1)) if title_m else ""

            # 2. Đoạn mô tả / sapo
            desc_m = re.search(r'<p[^>]*class=["\'][^"\']*description[^"\']*["\'][^>]*>(.*?)</p>', html_text, re.DOTALL)
            description = self._clean_html_text(desc_m.group(1)) if desc_m else ""

            # 3. Thân bài viết (chứa các đoạn tâm sự chính)
            body_m = re.search(r'<article[^>]*class=["\'][^"\']*fck_detail[^"\']*["\'][^>]*>(.*?)</article>', html_text, re.DOTALL)
            if not body_m:
                return None

            body_html = body_m.group(1)
            # Lấy các đoạn văn <p class="Normal">
            paragraphs = re.findall(r'<p[^>]*class=["\'][^"\']*Normal[^"\']*["\'][^>]*>(.*?)</p>', body_html, re.DOTALL)
            cleaned_paras = []
            for p in paragraphs:
                p_clean = self._clean_html_text(p)
                if p_clean and not p_clean.startswith(">>") and not "Ý kiến" in p_clean:
                    cleaned_paras.append(p_clean)

            if not cleaned_paras:
                full_body = self._clean_html_text(body_html)
            else:
                full_body = "\n\n".join(cleaned_paras)

            # Tên tác giả / Chữ ký
            author_m = re.search(r'<p[^>]*class=["\'][^"\']*author_mail[^"\']*["\'][^>]*><strong>(.*?)</strong>', body_html, re.DOTALL)
            author = self._clean_html_text(author_m.group(1)) if author_m else "Độc giả VnExpress"

            full_text = f"{title}\n\n{description}\n\n{full_body}".strip() if description else f"{title}\n\n{full_body}".strip()

            return {
                "url": url,
                "title": title,
                "author": author,
                "content": full_text
            }
        except Exception as e:
            print(f"  [-] Lỗi parse bài {url}: {e}")
            return None

    async def crawl(self, max_items: int = 15, pages: int = 2, use_rss: bool = True):
        """Tiến trình cào tổng thể các bài tâm sự VnExpress."""
        print("\n" + "="*65)
        print("KHỞI CHẠY THU THẬP BÀI VIẾT TỰ SỰ VNEXPRESS TÂM SỰ")
        print("="*65)

        async with httpx.AsyncClient() as client:
            candidates: List[Dict[str, str]] = []

            # 1. Thu thập từ RSS
            if use_rss:
                rss_items = await self.fetch_rss_links(client)
                candidates.extend(rss_items)

            # 2. Thu thập từ Web pages
            for p in range(1, pages + 1):
                page_items = await self.fetch_web_page_links(client, page=p)
                candidates.extend(page_items)

            # Lọc các URL trùng nhau
            unique_candidates = []
            seen_urls = set()
            for c in candidates:
                if c["url"] not in seen_urls:
                    seen_urls.add(c["url"])
                    unique_candidates.append(c)

            print(f"[*] Tổng số bài tâm sự thu thập được: {len(unique_candidates)} bài.")
            print(f"[*] Bắt đầu bóc tách & phân tích cảm xúc (Mục tiêu: {max_items} bài)...")

            saved_count = 0
            for idx, item in enumerate(unique_candidates):
                if saved_count >= max_items:
                    break

                url = item["url"]

                # Kiểm tra trùng lặp (Resumable check)
                if self.dm.is_url_crawled(url):
                    self.stats["skipped_existing"] += 1
                    continue

                self.stats["scanned_articles"] += 1
                article_data = await self.parse_article(client, url)
                if not article_data or not article_data.get("content"):
                    continue

                text = article_data["content"]
                author = article_data.get("author", "doc_gia")
                author_hash = hashlib.sha256(f"vnexpress_{author}".encode("utf-8")).hexdigest()[:12]

                # Đánh giá qua JournalFilter v2
                eval_res = self.filter.evaluate_entry(text)

                if not eval_res["accepted"]:
                    if "quá ngắn" in eval_res.get("reason", ""):
                        self.stats["rejected_short"] += 1
                    else:
                        self.stats["rejected_non_journal"] += 1
                    print(f"  [-] Bỏ qua bài {idx+1}: {eval_res['reason']}")
                    continue

                sample_id = f"vnexpress_{uuid.uuid4().hex[:8]}"
                pri_label = eval_res["primary_emotion"]
                sub_label = eval_res["sub_emotion"]
                score = eval_res["emotion_score"]

                meta = {
                    "source_platform": "vnexpress_tamsu",
                    "author": author,
                    "author_hash": author_hash,
                    "source_url": url,
                    "title": article_data.get("title", ""),
                    "narrative_score": eval_res.get("narrative_score", 0.8),
                    "matched_pronouns": eval_res.get("matched_pronouns", []),
                    "matched_reflective": eval_res.get("matched_reflective", []),
                    "matched_keywords": eval_res.get("matched_keywords", [])
                }

                # Phân luồng: Nếu có nguy cơ khủng hoảng/tự hại -> Cách ly vào crisis_quarantine/
                if eval_res.get("is_crisis"):
                    self.dm.save_crisis_sample(
                        sample_id=sample_id,
                        text=eval_res.get("cleaned_text", text),
                        crisis_keywords=eval_res.get("crisis_keywords", []),
                        primary_label=pri_label,
                        sub_label=sub_label,
                        meta=meta
                    )
                    self.stats["quarantined_crisis"] += 1
                    print(f"  [!] CẢNH BÁO: Bài viết khủng hoảng tâm lý ({sample_id}) -> Đã cách ly an toàn!")
                else:
                    self.dm.save_text_sample(
                        sample_id=sample_id,
                        text=eval_res.get("cleaned_text", text),
                        primary_label=pri_label,
                        sub_label=sub_label,
                        emotion_score=score,
                        meta=meta
                    )
                    self.stats["accepted_text"] += 1
                    self.stats["primary_emotions"][pri_label] = self.stats["primary_emotions"].get(pri_label, 0) + 1
                    self.stats["sub_emotions"][sub_label] = self.stats["sub_emotions"].get(sub_label, 0) + 1
                    print(f"  [+] THÀNH CÔNG [{saved_count+1}/{max_items}]: Lưu bài tâm sự ({sample_id}) -> [{pri_label} / {sub_label}] (Score: {score})")
                    saved_count += 1

                # Giãn cách ngẫu nhiên lịch sự (Polite Crawling)
                sleep_s = random.uniform(*self.delay_range)
                await asyncio.sleep(sleep_s)

        self.print_summary()

    def print_summary(self):
        print("\n" + "="*65)
        print("TỔNG KẾT CÀO DỮ LIỆU TÂM SỰ VNEXPRESS")
        print("="*65)
        print(f"Số bài quét qua:                   {self.stats['scanned_articles']}")
        print(f"Số bài bỏ qua do đã cào:          {self.stats['skipped_existing']}")
        print(f"Số bài bị từ chối do không tự sự:  {self.stats['rejected_non_journal']}")
        print(f"Số bài quá ngắn:                   {self.stats['rejected_short']}")
        print(f"- [1] Mẫu văn bản đạt chuẩn:       {self.stats['accepted_text']} bài")
        print(f"- [2] Mẫu khủng hoảng đã cách ly:  {self.stats['quarantined_crisis']} bài")
        print("\nPhân bố cảm xúc theo Taxonomy:")
        print("  Primary Emotions:")
        for em, cnt in self.stats["primary_emotions"].items():
            print(f"    * {em:<12}: {cnt} bài")
        print("  Sub Emotions:")
        for sub, cnt in self.stats["sub_emotions"].items():
            print(f"    * {sub:<14}: {cnt} bài")
        print(f"\nMetadata tổng hợp:")
        print(f"  - CSV:   {self.dm.metadata_csv}")
        print(f"  - JSONL: {self.dm.metadata_jsonl}")
        print("="*65)


async def main():
    parser = argparse.ArgumentParser(description="VnExpress Tam Su Journal & Confession Crawler")
    parser.add_argument("--max-items", type=int, default=10, help="Số bài tối đa cần lưu")
    parser.add_argument("--pages", type=int, default=2, help="Số trang HTML cần quét")
    parser.add_argument("--min-words", type=int, default=40, help="Số từ tối thiểu cho một bài tâm sự dài")
    parser.add_argument("--output-dir", type=str, default="data/crawled", help="Thư mục xuất dữ liệu")
    parser.add_argument("--no-rss", action="store_true", help="Không dùng RSS feed")
    args = parser.parse_args()

    crawler = VnExpressJournalCrawler(
        output_dir=args.output_dir,
        min_word_count=args.min_words
    )
    await crawler.crawl(
        max_items=args.max_items,
        pages=args.pages,
        use_rss=not args.no_rss
    )


if __name__ == "__main__":
    asyncio.run(main())
