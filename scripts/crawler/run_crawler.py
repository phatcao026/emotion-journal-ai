"""
run_crawler.py
--------------
Bộ điều khiển trung tâm (Unified CLI Runner) cho hệ thống thu thập dữ liệu Nhật ký Cảm xúc Đa phương thức (MER).
Cung cấp giao diện dòng lệnh duy nhất, tiện lợi, không yêu cầu người dùng phải nhớ từng câu lệnh phức tạp.

Các chế độ hoạt động:
1. `news`    : Cào bài viết tự sự dài chuẩn mực từ chuyên mục Tâm sự VnExpress.
2. `voice`   : Thu thập & phân đoạn audio độc thoại từ YouTube/Podcast (hỗ trợ ASR PhoWhisper & Demucs).
3. `threads` : Cào bài viết & bình luận tự sự từ Meta Threads (hỗ trợ search query & profile).
4. `status`  : Báo cáo dashboard thống kê dataset (số lượng, modality, phân bố nhãn, số người nói, ca khủng hoảng).
5. `balance` : Phân tích độ lệch nhãn và gợi ý các từ khóa tìm kiếm bù trừ để cân bằng tập dữ liệu.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.crawler.dataset_manager import CrawledDatasetManager
from scripts.crawler.journal_filter import JournalFilter


def print_banner():
    print("""
=============================================================================
   EMOTION JOURNAL AI - MULTIMODAL DATA HARVESTING & CURATION SUITE
   Hệ thống thu thập dữ liệu Nhật ký & Tâm sự đa phương thức (Text + Voice)
=============================================================================
""")


def cmd_status(args):
    """Hiển thị dashboard thống kê toàn diện tập dữ liệu đã thu thập."""
    dm = CrawledDatasetManager(args.output_dir)
    stats = dm.get_statistics()

    print("\n" + "="*65)
    print("DASHBOARD THỐNG KÊ TẬP DỮ LIỆU CRAWLED (MER DATASET)")
    print("="*65)
    print(f"Tổng số bản ghi metadata:           {stats['total_records']}")
    print(f"Số lượng tác giả độc lập (Speakers): {stats['unique_authors_count']} người (Participant-aware ready)")
    print(f"Tổng thời lượng âm thanh:           {stats['total_audio_duration_s'] / 60.0:.2f} phút ({stats['total_audio_duration_s']:.1f}s)")
    print("\n[1] Phân bố theo Phương thức (Modalities):")
    for mod, count in stats["modalities"].items():
        pct = (count / max(stats['total_records'], 1)) * 100.0
        bar = "█" * int(pct / 5)
        print(f"    * {mod:<18}: {count:>4} mẫu ({pct:>5.1f}%) {bar}")

    print("\n[2] Phân bố Cảm xúc Chính (Primary 5 Classes):")
    for pri in ["JOY", "SADNESS", "ANXIETY", "ANGER", "NEUTRAL"]:
        cnt = stats["primary_emotions"].get(pri, 0)
        pct = (cnt / max(stats['total_records'], 1)) * 100.0
        bar = "█" * int(pct / 5)
        print(f"    * {pri:<12}: {cnt:>4} mẫu ({pct:>5.1f}%) {bar}")

    print("\n[3] Phân bố Cảm xúc Tinh tế (Sub-emotions 11 Classes):")
    for sub, cnt in sorted(stats["sub_emotions"].items(), key=lambda x: -x[1]):
        pct = (cnt / max(stats['total_records'], 1)) * 100.0
        print(f"    * {sub:<14}: {cnt:>4} mẫu ({pct:>5.1f}%)")

    print("\n[4] Khu vực Cách ly Khủng hoảng (Crisis Quarantine):")
    print(f"    * Số bài chứa nguy cơ tự hại: {stats['crisis_count']} mẫu (Lưu riêng tại data/crawled/crisis_quarantine/)")
    print("="*65)


def cmd_balance(args):
    """Phân tích mức độ mất cân bằng nhãn và đề xuất các từ khóa truy vấn bù trừ."""
    dm = CrawledDatasetManager(args.output_dir)
    stats = dm.get_statistics()
    pri_counts = stats["primary_emotions"]
    total = max(stats["total_records"], 1)

    print("\n" + "="*65)
    print("PHÂN TÍCH ĐỘ LỆCH NHÃN & GỢI Ý TRUY VẤN BÙ TRỪ (LABEL BALANCING)")
    print("="*65)

    positive_sum = pri_counts.get("JOY", 0)
    negative_sum = pri_counts.get("SADNESS", 0) + pri_counts.get("ANXIETY", 0) + pri_counts.get("ANGER", 0)

    print(f"Tỷ lệ Nhóm Tiêu cực (Sadness/Anxiety/Anger): {negative_sum / total * 100:.1f}%")
    print(f"Tỷ lệ Nhóm Tích cực (Joy/Calm/Hope):         {positive_sum / total * 100:.1f}%")

    if positive_sum < negative_sum * 0.35:
        print("\n[!] CẢNH BÁO: Tập dữ liệu đang bị lệch nặng về nhóm Tiêu cực (Sadness/Anxiety).")
        print("    Cần chủ động thu thập thêm các bài viết về sự bình yên, chữa lành, lòng biết ơn.")
        print("\nGợi ý các lệnh cào bù trừ cảm xúc:")
        print("  1. Cào radio podcast bình yên & chữa lành:")
        print("     python scripts/crawler/run_crawler.py voice --url \"<URL>\" --primary JOY --sub CALM")
        print("  2. Quét từ khóa tích cực trên Threads:")
        print("     python scripts/crawler/run_crawler.py threads --search \"bình yên\" \"lòng biết ơn\" \"ngày hôm nay thật đẹp\" \"sau giông bão\"")
        print("  3. Cào thêm bài viết tâm sự nhẹ nhõm:")
        print("     python scripts/crawler/run_crawler.py news --max-items 20")
    else:
        print("\n[+] ĐÁNH GIÁ: Tỷ lệ nhãn đang ở mức cân bằng tương đối tốt giữa các nhóm.")
    print("="*65)


def main():
    print_banner()
    parser = argparse.ArgumentParser(
        description="Emotion Journal AI - Unified Crawler CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Chế độ thu thập hoặc kiểm soát dữ liệu")

    # 1. Subcommand: news
    p_news = subparsers.add_parser("news", help="Cào bài viết tâm sự từ Báo điện tử VnExpress")
    p_news.add_argument("--max-items", type=int, default=10, help="Số bài tối đa cần thu thập")
    p_news.add_argument("--pages", type=int, default=2, help="Số trang web cần quét")
    p_news.add_argument("--min-words", type=int, default=40, help="Số từ tối thiểu cho bài tâm sự")
    p_news.add_argument("--output-dir", type=str, default="data/crawled", help="Thư mục xuất dữ liệu")

    # 2. Subcommand: voice
    p_voice = subparsers.add_parser("voice", help="Thu thập audio độc thoại từ YouTube/TikTok Radio")
    p_voice.add_argument("--url", type=str, help="URL video/podcast cần tải")
    p_voice.add_argument("--file", type=str, help="Đường dẫn file âm thanh cục bộ")
    p_voice.add_argument("--chunk-duration", type=float, default=45.0, help="Thời lượng mỗi chunk (giây)")
    p_voice.add_argument("--primary", type=str, default="SADNESS", choices=["JOY", "SADNESS", "ANXIETY", "ANGER", "NEUTRAL"])
    p_voice.add_argument("--sub", type=str, default="LONELINESS")
    p_voice.add_argument("--author", type=str, default="radio_tamsu")
    p_voice.add_argument("--transcribe", action="store_true", help="Bật PhoWhisper ASR sinh transcript cho text_voice")
    p_voice.add_argument("--remove-music", action="store_true", help="Chạy Demucs tách nhạc nền")
    p_voice.add_argument("--list-sources", action="store_true", help="Xem danh sách kênh gợi ý")
    p_voice.add_argument("--output-dir", type=str, default="data/crawled")

    # 3. Subcommand: threads
    p_threads = subparsers.add_parser("threads", help="Cào bài viết và bình luận tâm sự từ Meta Threads")
    p_threads.add_argument("--targets", nargs="*", help="Danh sách username cần quét")
    p_threads.add_argument("--search", nargs="*", help="Danh sách từ khóa tìm kiếm")
    p_threads.add_argument("--max-items", type=int, default=10, help="Số bài tối đa mỗi mục")
    p_threads.add_argument("--output-dir", type=str, default="data/crawled")

    # 4. Subcommand: status
    p_status = subparsers.add_parser("status", help="Xem báo cáo dashboard thống kê tập dữ liệu đã cào")
    p_status.add_argument("--output-dir", type=str, default="data/crawled")

    # 5. Subcommand: balance
    p_balance = subparsers.add_parser("balance", help="Kiểm tra độ lệch nhãn và gợi ý từ khóa bù trừ")
    p_balance.add_argument("--output-dir", type=str, default="data/crawled")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "status":
        cmd_status(args)
    elif args.command == "balance":
        cmd_balance(args)
    elif args.command == "news":
        from scripts.crawler.crawler_news_journal import VnExpressJournalCrawler
        crawler = VnExpressJournalCrawler(
            output_dir=args.output_dir,
            min_word_count=args.min_words
        )
        asyncio.run(crawler.crawl(max_items=args.max_items, pages=args.pages))
    elif args.command == "voice":
        from scripts.crawler.crawler_voice_radio import VoiceRadioCrawler, CURATED_RADIO_SOURCES
        if args.list_sources:
            print("\n=== DANH SÁCH GỢI Ý CÁC KÊNH RADIO / PODCAST TÂM SỰ ===")
            for idx, src in enumerate(CURATED_RADIO_SOURCES, 1):
                print(f"[{idx}] {src['name']} - Cảm xúc: {src['default_emotion'][0]}/{src['default_emotion'][1]}")
                print(f"    Tìm kiếm: '{src['search_query']}'")
            return
        crawler = VoiceRadioCrawler(
            output_dir=args.output_dir,
            chunk_duration_s=args.chunk_duration,
            enable_asr=args.transcribe,
            remove_music=args.remove_music
        )
        if args.url:
            crawler.process_voice_url(
                url=args.url,
                assigned_primary=args.primary,
                assigned_sub=args.sub,
                author=args.author
            )
        elif args.file:
            crawler.process_local_audio(
                audio_path=args.file,
                assigned_primary=args.primary,
                assigned_sub=args.sub,
                author=args.author
            )
        else:
            print("Vui lòng cung cấp tham số --url hoặc --file (hoặc --list-sources).")
            p_voice.print_help()
    elif args.command == "threads":
        from scripts.crawler.crawler_threads import main as threads_main
        sys.argv = [sys.argv[0]]
        if args.targets:
            sys.argv.extend(["--targets"] + args.targets)
        if args.search:
            sys.argv.extend(["--search"] + args.search)
        sys.argv.extend(["--max-items", str(args.max_items), "--output-dir", args.output_dir])
        asyncio.run(threads_main())


if __name__ == "__main__":
    main()
