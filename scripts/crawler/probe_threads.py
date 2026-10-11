"""
probe_threads.py
----------------
Kiểm tra các tài khoản tâm sự / nhật ký trên Threads để xem tài khoản nào đang hoạt động
và có nhiều video/bài viết nhất.
"""

import asyncio
import sys
from playwright.async_api import async_playwright

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

candidate_profiles = [
    "gocnhotamhon",
    "tamsudemkhuya",
    "kechotoinghe",
    "nguoigiukinhiem",
    "tramdungchan",
    "chuyencuatoi",
    "nhatkycuatoi",
    "nhung_ngay_chong_chenh",
    "tam_su_genz",
    "trangnhatky",
    "tam_su_tuoi_30",
    "vietcetera",
    "havasip"
]

async def probe():
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"]
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            locale="vi-VN"
        )
        page = await context.new_page()

        print("\n=== KIỂM TRA TRẠNG THÁI CÁC TÀI KHOẢN TÂM SỰ TRÊN THREADS ===")
        active_accounts = []

        for user in candidate_profiles:
            url = f"https://www.threads.net/@{user}"
            try:
                resp = await page.goto(url, wait_until="networkidle", timeout=20000)
                # Cuộn trang nhẹ
                await page.mouse.wheel(0, 1000)
                await asyncio.sleep(1)

                title = await page.title()
                videos = await page.query_selector_all("video")
                posts = await page.query_selector_all("div[data-pressable-container='true']")

                status = resp.status if resp else "N/A"
                print(f"@{user:<22} | Status: {status} | Videos: {len(videos)} | Posts: {len(posts)} | Title: {title[:35]}")
                if len(videos) > 0 or len(posts) > 0:
                    active_accounts.append({
                        "username": user,
                        "videos": len(videos),
                        "posts": len(posts)
                    })
            except Exception as e:
                print(f"@{user:<22} | Lỗi: {e}")

        await browser.close()

        print(f"\n=> Đã tìm thấy {len(active_accounts)} tài khoản đang hoạt động có nội dung.")
        return active_accounts

if __name__ == "__main__":
    asyncio.run(probe())
