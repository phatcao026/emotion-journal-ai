"""
save_tiktok_session.py
----------------------
Công cụ hỗ trợ đăng nhập TikTok và tự động lưu Cookies (Session) ra file:
- `tiktok_cookies.json` (dành cho Playwright automation)
- `tiktok_cookies.txt` (chuẩn Netscape cookie format dành cho yt-dlp)

Cách dùng:
    python scripts/crawler/save_tiktok_session.py

Trình duyệt Chrome sẽ hiện lên:
1. Bạn đăng nhập tài khoản TikTok của bạn (qua QR code, mật khẩu, hoặc Google).
2. Sau khi vào được bảng tin (For You Feed) thành công, quay lại cửa sổ console và nhấn phím [ENTER].
3. Script sẽ tự động kết xuất cả 2 định dạng cookie để hệ thống crawler sử dụng.
"""

from __future__ import annotations

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

ROOT_DIR = Path(__file__).resolve().parent
COOKIE_JSON_PATH = ROOT_DIR / "tiktok_cookies.json"
COOKIE_TXT_PATH = ROOT_DIR / "tiktok_cookies.txt"


def export_netscape_cookies(cookies: list[dict], txt_path: Path):
    """Xuất cookies sang chuẩn Netscape Cookie File để yt-dlp sử dụng trực tiếp."""
    lines = ["# Netscape HTTP Cookie File", "# https://curl.se/docs/http-cookies.html", ""]
    for c in cookies:
        domain = c.get("domain", "")
        flag = "TRUE" if domain.startswith(".") else "FALSE"
        path = c.get("path", "/")
        secure = "TRUE" if c.get("secure", False) else "FALSE"
        expires = int(c.get("expires", -1))
        if expires < 0:
            expires = 2147483647  # Năm 2038
        name = c.get("name", "")
        value = c.get("value", "")
        lines.append(f"{domain}\t{flag}\t{path}\t{secure}\t{expires}\t{name}\t{value}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


async def login_and_save_cookies():
    print("\n" + "="*65)
    print("KHỞI TẠO PHIÊN ĐĂNG NHẬP TIKTOK ĐỂ VƯỢT BOT & LƯU SESSION COOKIES")
    print("="*65)
    print("1. Trình duyệt Chromium sẽ được mở ra.")
    print("2. Vui lòng đăng nhập tài khoản TikTok của bạn (và giải captcha nếu có).")
    print("3. Sau khi vào được giao diện TikTok thành công, quay lại đây nhấn phím [ENTER].")
    print("="*65 + "\n")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox"
            ]
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            locale="vi-VN",
            viewport={"width": 1280, "height": 800}
        )
        page = await context.new_page()

        print("[*] Đang điều hướng tới trang đăng nhập TikTok...")
        try:
            await page.goto("https://www.tiktok.com/login", timeout=60000)
        except Exception as e:
            print(f"[-] Lưu ý: {e}")

        # Chờ người dùng đăng nhập và nhấn ENTER trên terminal
        await asyncio.to_thread(input, "\n>> Đăng nhập xong trên trình duyệt, hãy nhấn [ENTER] tại đây để lưu cookie... ")

        cookies = await context.cookies()

        # 1. Lưu file JSON cho Playwright
        with open(COOKIE_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(cookies, f, ensure_ascii=False, indent=2)

        # 2. Lưu file TXT chuẩn Netscape cho yt-dlp
        export_netscape_cookies(cookies, COOKIE_TXT_PATH)

        print(f"\n[+] THÀNH CÔNG: Đã lưu {len(cookies)} cookies vào:")
        print(f"    - Playwright session: {COOKIE_JSON_PATH}")
        print(f"    - yt-dlp format:      {COOKIE_TXT_PATH}")
        print("[+] Hệ thống Voice Crawler bây giờ có thể tự do quét video và trích xuất audio!")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(login_and_save_cookies())
