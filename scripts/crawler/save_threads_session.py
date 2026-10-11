"""
save_threads_session.py
-----------------------
Công cụ hỗ trợ đăng nhập Threads và tự động lưu Cookies (Session) ra file `threads_cookies.json`.

Cách dùng:
    python scripts/crawler/save_threads_session.py
Trình duyệt sẽ hiện lên -> bạn đăng nhập nick Instagram/Threads của bạn -> nhấn Enter tại console -> script tự lưu cookie!
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from playwright.async_api import async_playwright

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

COOKIE_PATH = Path(__file__).resolve().parent / "threads_cookies.json"


async def login_and_save_cookies():
    print("\n" + "="*60)
    print("KHỞI TẠO PHIÊN ĐĂNG NHẬP THREADS ĐỂ LƯU COOKIES")
    print("="*60)
    print("1. Trình duyệt Chrome sẽ được mở ra.")
    print("2. Vui lòng đăng nhập tài khoản Threads / Instagram của bạn.")
    print("3. Sau khi vào được bảng tin (Feed) thành công, quay lại đây và nhấn phím ENTER.")
    print("="*60 + "\n")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            locale="vi-VN"
        )
        page = await context.new_page()

        await page.goto("https://www.threads.net/login")

        # Chờ người dùng đăng nhập
        await asyncio.to_thread(input, ">> Đăng nhập xong trên trình duyệt, hãy nhấn [ENTER] tại đây để lưu cookie... ")

        cookies = await context.cookies()
        with open(COOKIE_PATH, "w", encoding="utf-8") as f:
            json.dump(cookies, f, ensure_ascii=False, indent=2)

        print(f"\n[+] THÀNH CÔNG: Đã lưu {len(cookies)} cookies vào: {COOKIE_PATH}")
        print("[+] Từ bây giờ crawler sẽ tự động dùng phiên đăng nhập này để quét video/voice!")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(login_and_save_cookies())
