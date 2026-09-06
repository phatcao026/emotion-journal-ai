#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_lemotif_vi.py
---------------------
Chương trình tự động hóa quy trình:
1. Tải và đọc dữ liệu thô LE-motif từ repo xaliceli/lemotif.
2. Trích xuất text và 18 nhãn cảm xúc gốc.
3. Dịch sang tiếng Việt bằng Gemini API tuân thủ nghiêm ngặt quy tắc:
   - Giữ nguyên phủ định, mức độ, emoji, sắc thái thân mật/đời thường.
   - Không suy diễn thêm cảm xúc, dịch sát nghĩa.
   - Giữ nguyên mảng nhãn tiếng Anh gốc.
4. Hỗ trợ batching, checkpoint, và 2 chế độ:
   - --mode pilot: Tạo 30-50 mẫu tiêu biểu đa dạng để duyệt phong cách.
   - --mode full: Dịch toàn bộ 1,473 câu ra data/raw/lemotif_vi.jsonl.
"""

import os
import sys
import re
import csv
import json
import time
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Any, Optional

# Đảm bảo in tiếng Việt chuẩn trên Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DATASET_RAW_URL = "https://raw.githubusercontent.com/xaliceli/lemotif/master/assets/data/lemotif-data-cleaned-flat.csv"

# 18 nhãn cảm xúc chính xác của LE-motif
FEELING_COLUMNS = [
    ("Answer.f1.afraid.raw", "Afraid"),
    ("Answer.f1.angry.raw", "Angry"),
    ("Answer.f1.anxious.raw", "Anxious"),
    ("Answer.f1.ashamed.raw", "Ashamed"),
    ("Answer.f1.awkward.raw", "Awkward"),
    ("Answer.f1.bored.raw", "Bored"),
    ("Answer.f1.calm.raw", "Calm"),
    ("Answer.f1.confused.raw", "Confused"),
    ("Answer.f1.disgusted.raw", "Disgusted"),
    ("Answer.f1.excited.raw", "Excited"),
    ("Answer.f1.frustrated.raw", "Frustrated"),
    ("Answer.f1.happy.raw", "Happy"),
    ("Answer.f1.jealous.raw", "Jealous"),
    ("Answer.f1.nostalgic.raw", "Nostalgic"),
    ("Answer.f1.proud.raw", "Proud"),
    ("Answer.f1.sad.raw", "Sad"),
    ("Answer.f1.satisfied.raw", "Satisfied"),
    ("Answer.f1.surprised.raw", "Surprised"),
]

SYSTEM_PROMPT = """Bạn là chuyên gia dịch thuật dữ liệu cho bài toán Huấn luyện Phân loại Cảm xúc (Emotion Classification NLP).
Nhiệm vụ của bạn là dịch các câu văn bản tiếng Anh trong nhật ký cá nhân sang tiếng Việt, tuân thủ nghiêm ngặt các quy tắc bất biến sau:

1. GIỮ NGUYÊN SẮC THÁI CẢM XÚC, MỨC ĐỘ, THỜI GIAN VÀ ĐẶC BIỆT LÀ THỂ PHỦ ĐỊNH:
   - "I am not happy at all." -> "Tôi hoàn toàn không vui chút nào." (TUYỆT ĐỐI KHÔNG dịch thành "Tôi rất buồn", vì như vậy là tự ý gán thêm cảm xúc Sadness mà câu gốc không có).
   - "I didn't feel angry" -> "Tôi đã không cảm thấy tức giận" (không suy diễn sang bình thản hay vui vẻ).

2. GIỮ NGUYÊN EMOJI, DẤU CÂU CẢM THÁN VÀ PHONG CÁCH TỰ NHIÊN:
   - Giữ lại toàn bộ emoji, dấu chấm than (!), dấu hỏi (?), từ viết hoa nhấn mạnh, từ lóng hoặc cách nói chuyện thân mật thường ngày.

3. KHÔNG "LÀM ĐẸP" CÂU VĂN:
   - Không biến văn nói đời thường thành văn phong tiểu thuyết, thơ ca hay nhật ký hoa mỹ nếu bản gốc không như vậy.
   - Không suy diễn hoặc thêm bớt các từ ngữ cảm xúc không có trong văn bản nguồn.

4. VỚI CÂU MƠ HỒ HOẶC KHÓ HIỂU:
   - Dịch sát nghĩa đen nhất có thể, giữ nguyên sự mơ hồ của câu gốc. Không cố tình đoán ý theo nhãn cảm xúc.

5. ĐỊNH DẠNG ĐẦU RA:
   - Bạn sẽ nhận vào một JSON array gồm các object có format: {"id": "...", "text_en": "..."}
   - Trả về DUY NHẤT một JSON array chuẩn cú pháp với format:
     [
       {"id": "...", "text_vi": "bản dịch tiếng Việt tương ứng"},
       ...
     ]
   - Đảm bảo đúng số lượng phần tử và đúng ID tương ứng với từng câu đầu vào."""


def load_env_file(env_path: Path):
    """Đọc file .env nếu có mà không cần thư viện python-dotenv."""
    if not env_path.exists():
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception as e:
        print(f"[Cảnh báo] Không thể đọc file .env: {e}")


def download_raw_data(destination: Path) -> Path:
    """Tải file CSV gốc từ GitHub nếu chưa có trong máy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"[*] Dữ liệu gốc đã tồn tại tại: {destination}")
        return destination

    print(f"[*] Đang tải dữ liệu gốc từ: {DATASET_RAW_URL}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(DATASET_RAW_URL, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as response, open(destination, "wb") as out_file:
        out_file.write(response.read())

    print(f"[+] Tải thành công! Đã lưu tại: {destination}")
    return destination


def parse_raw_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Đọc CSV và chuyển đổi thành danh sách records chuẩn hóa."""
    records = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            text_en = row.get("Answer", "").strip()
            labels = []
            for col_name, feeling_name in FEELING_COLUMNS:
                val = str(row.get(col_name, "")).strip().lower()
                if val in ["true", "1", "yes"]:
                    labels.append(feeling_name)

            rec_id = f"le-{idx:04d}"
            records.append({
                "id": rec_id,
                "text_en": text_en,
                "labels": labels,
                "orig_index": idx
            })
    return records


def select_pilot_samples(records: List[Dict[str, Any]], limit: int = 35) -> List[Dict[str, Any]]:
    """
    Lựa chọn 30-50 mẫu tiêu biểu đa dạng nhất để làm pilot:
    - Có chứa từ phủ định (not, never, don't,...)
    - Có chứa emoji hoặc dấu cảm thán
    - Có nhãn hiếm/nhạy cảm: Awkward, Jealous, Nostalgic, Confused, Bored, Ashamed
    - Có đa nhãn (multi-label)
    - Có câu ngắn và câu dài
    """
    selected_ids = set()
    pilot_list = []

    neg_pattern = re.compile(r"\b(not|never|no|hardly|barely|didn't|don't|couldn't|can't|won't)\b", re.IGNORECASE)
    rare_emotions = {"Awkward", "Jealous", "Nostalgic", "Ashamed", "Confused", "Bored"}

    # 1. Ưu tiên câu có từ phủ định
    for r in records:
        if len(pilot_list) >= limit:
            break
        if r["id"] not in selected_ids and neg_pattern.search(r["text_en"]):
            selected_ids.add(r["id"])
            pilot_list.append(r)

    # 2. Ưu tiên câu có nhãn hiếm
    for r in records:
        if len(pilot_list) >= limit:
            break
        if r["id"] not in selected_ids and any(lbl in rare_emotions for lbl in r["labels"]):
            selected_ids.add(r["id"])
            pilot_list.append(r)

    # 3. Ưu tiên câu có >= 3 nhãn cảm xúc
    for r in records:
        if len(pilot_list) >= limit:
            break
        if r["id"] not in selected_ids and len(r["labels"]) >= 3:
            selected_ids.add(r["id"])
            pilot_list.append(r)

    # 4. Ưu tiên câu có dấu chấm than hoặc emoji
    for r in records:
        if len(pilot_list) >= limit:
            break
        if r["id"] not in selected_ids and ("!" in r["text_en"] or any(ord(char) > 127 for char in r["text_en"])):
            selected_ids.add(r["id"])
            pilot_list.append(r)

    # 5. Nếu vẫn chưa đủ, lấy thêm từ đầu danh sách để đảm bảo tính phổ biến
    for r in records:
        if len(pilot_list) >= limit:
            break
        if r["id"] not in selected_ids:
            selected_ids.add(r["id"])
            pilot_list.append(r)

    # Sắp xếp lại theo thứ tự id ban đầu
    pilot_list.sort(key=lambda x: x["orig_index"])
    return pilot_list


FALLBACK_MODELS = [
    "gemini-3.8-flash",
    "gemini-3-flash-preview",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash"
]


class QuotaExhaustedError(Exception):
    """Ngoại lệ khi một model hoặc key bị hết hạn ngạch ngày (RPD/TPD)."""
    pass


def call_gemini_api_single(
    items_to_translate: List[Dict[str, str]],
    api_key: str,
    model_name: str = "gemini-3.8-flash",
    max_retries: int = 3
) -> Dict[str, str]:
    """
    Gọi Gemini REST API cho một batch với một model_name và api_key cụ thể.
    Nếu gặp 429 Quota Exceeded thì raise QuotaExhaustedError để chuyển model/key.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    user_payload_items = [{"id": it["id"], "text_en": it["text_en"]} for it in items_to_translate]
    user_prompt_text = (
        "Dịch danh sách các câu tiếng Anh sau sang tiếng Việt theo đúng các quy tắc đã đặt ra trong system prompt. "
        "Trả về định dạng JSON Array chứa các object {\"id\": \"...\", \"text_vi\": \"...\"}:\n\n"
        + json.dumps(user_payload_items, ensure_ascii=False, indent=2)
    )

    request_body = {
        "systemInstruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": [
            {
                "parts": [{"text": user_prompt_text}]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json"
        }
    }

    req_data = json.dumps(request_body).encode("utf-8")

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                url,
                data=req_data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=90) as response:
                resp_json = json.loads(response.read().decode("utf-8"))
                
                candidates = resp_json.get("candidates", [])
                if not candidates:
                    raise ValueError(f"Gemini API không trả về candidate nào: {resp_json}")
                
                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    raise ValueError(f"Candidate không có part text: {resp_json}")
                
                raw_text = parts[0].get("text", "").strip()
                translated_array = json.loads(raw_text)

                result_map = {}
                for obj in translated_array:
                    if isinstance(obj, dict) and "id" in obj and "text_vi" in obj:
                        result_map[obj["id"]] = str(obj["text_vi"]).strip()

                missing = [it["id"] for it in items_to_translate if it["id"] not in result_map]
                if missing:
                    print(f"    [Cảnh báo] Thiếu {len(missing)} IDs trong kết quả trả về, đang thử lại...")
                    raise ValueError(f"Thiếu kết quả dịch cho các ID: {missing}")

                return result_map

        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8", errors="ignore")
            # 404 hoặc 429 Quota Exceeded (RPD limit)
            if e.code == 429:
                if "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
                    raise QuotaExhaustedError(f"Model {model_name} hết hạn ngạch quota (RPD).")
                wait_sec = attempt * 10
                print(f"    [Rate Limit 429] Đang chờ {wait_sec}s rồi thử lại lần {attempt}/{max_retries}...")
                time.sleep(wait_sec)
            elif e.code == 404:
                raise QuotaExhaustedError(f"Model {model_name} không tìm thấy (404).")
            else:
                print(f"    [HTTP {e.code}] {err_msg[:200]}")
                time.sleep(4)
        except Exception as ex:
            if isinstance(ex, QuotaExhaustedError):
                raise ex
            print(f"    [Lỗi {type(ex).__name__}] {ex}. Thử lại lần {attempt}/{max_retries}...")
            time.sleep(attempt * 2)

    raise QuotaExhaustedError(f"Model {model_name} không thể hoàn thành batch sau {max_retries} lần thử.")


def run_translation_pipeline(
    records: List[Dict[str, Any]],
    output_path: Path,
    checkpoint_path: Path,
    api_keys: List[str],
    batch_size: int = 35,
    delay_sec: float = 2.0,
    preferred_model: str = "gemini-3.8-flash"
):
    """
    Thực hiện dịch danh sách records theo từng batch có:
    - Tự động xoay tua danh sách Model (gemini-3.8-flash -> gemini-3-flash-preview -> ...)
    - Tự động xoay tua danh sách API Keys khi chạm trần 429
    - Lưu checkpoint liên tục
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Đọc checkpoint nếu đã có
    completed_translations = {}
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        completed_translations[obj["id"]] = obj["text_vi"]
            print(f"[*] Tìm thấy checkpoint: đã dịch {len(completed_translations)}/{len(records)} mẫu.")
        except Exception as e:
            print(f"[Cảnh báo] Lỗi đọc checkpoint ({e}), sẽ bắt đầu lại.")

    # 2. Lọc ra các record cần dịch tiếp
    pending_records = [r for r in records if r["id"] not in completed_translations]
    print(f"[*] Cần dịch tiếp: {len(pending_records)} mẫu (Kích thước batch: {batch_size}).")

    # Tạo danh sách model ưu tiên
    model_queue = [preferred_model] + [m for m in FALLBACK_MODELS if m != preferred_model]
    key_idx = 0
    model_idx = 0

    total_batches = (len(pending_records) + batch_size - 1) // batch_size if pending_records else 0

    with open(checkpoint_path, "a", encoding="utf-8") as cp_file:
        for b_idx in range(total_batches):
            batch = pending_records[b_idx * batch_size : (b_idx + 1) * batch_size]
            b_num = b_idx + 1

            # Vòng lặp thử qua các Model và API Keys nếu bị 429
            success = False
            while not success:
                current_key = api_keys[key_idx]
                current_model = model_queue[model_idx]
                masked_key = current_key[:8] + "..." + current_key[-4:] if len(current_key) > 12 else "KEY"
                print(f"\n -> Dịch Batch {b_num}/{total_batches} ({len(batch)} câu, từ {batch[0]['id']} đến {batch[-1]['id']})")
                print(f"    [Sử dụng] Model: {current_model} | Key #{key_idx + 1} ({masked_key})")

                try:
                    batch_trans = call_gemini_api_single(
                        items_to_translate=batch,
                        api_key=current_key,
                        model_name=current_model
                    )
                    success = True

                    # Ghi checkpoint ngay lập tức
                    for rec in batch:
                        rec_id = rec["id"]
                        text_vi = batch_trans.get(rec_id, "")
                        completed_translations[rec_id] = text_vi
                        checkpoint_entry = {"id": rec_id, "text_vi": text_vi}
                        cp_file.write(json.dumps(checkpoint_entry, ensure_ascii=False) + "\n")
                    cp_file.flush()

                except QuotaExhaustedError as qe:
                    print(f"\n[!] CẢNH BÁO QUOTA: {qe}")
                    # Thử đổi sang model tiếp theo
                    if model_idx + 1 < len(model_queue):
                        model_idx += 1
                        print(f" => [Xoay Model] Chuyển sang model dự phòng: {model_queue[model_idx]}")
                    # Nếu hết model trên key này, thử đổi sang key tiếp theo
                    elif key_idx + 1 < len(api_keys):
                        key_idx += 1
                        model_idx = 0
                        print(f" => [Xoay Key] Chuyển sang API Key #{key_idx + 1}!")
                    else:
                        print("\n[LỖI NGHIÊM TRỌNG] Toàn bộ Model và API Key đều đã hết hạn ngạch ngày (RPD)!")
                        print("Tiến trình đã được lưu an toàn tại checkpoint. Hãy chờ reset ngày mai hoặc thêm key mới.")
                        break

            if not success:
                break

            if b_num < total_batches and delay_sec > 0:
                time.sleep(delay_sec)

    # 3. Tổng hợp thành file JSONL hoàn chỉnh theo đúng thứ tự ban đầu
    print(f"\n[*] Đang tổng hợp dữ liệu ra file đích: {output_path}...")
    with open(output_path, "w", encoding="utf-8") as out_f:
        for r in records:
            rec_id = r["id"]
            final_obj = {
                "id": rec_id,
                "text_vi": completed_translations.get(rec_id, ""),
                "labels": r["labels"]
            }
            out_f.write(json.dumps(final_obj, ensure_ascii=False) + "\n")

    print(f"[+] Hoàn tất! Đã xuất {len(records)} dòng vào: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Chuẩn bị bản dịch tiếng Việt cho LE-motif bằng Gemini API.")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot",
                        help="'pilot' (mẫu 30-50 dòng duyệt trước) hoặc 'full' (toàn bộ 1,473 dòng). Mặc định: pilot")
    parser.add_argument("--limit", type=int, default=35, help="Số lượng mẫu cho chế độ pilot (mặc định: 35).")
    parser.add_argument("--api-key", type=str, default="", help="Gemini API Key (hoặc đặt qua biến GEMINI_API_KEY).")
    parser.add_argument("--model", type=str, default="gemini-3.8-flash", help="Model Gemini ưu tiên (mặc định: gemini-3.8-flash).")
    parser.add_argument("--batch-size", type=int, default=35, help="Số lượng câu cho mỗi request (mặc định: 35).")
    parser.add_argument("--delay", type=float, default=2.0, help="Độ trễ giữa các batch (giây, mặc định: 2.0).")
    parser.add_argument("--test-parser", action="store_true", help="Chỉ chạy thử trích xuất 5 dòng CSV gốc mà không gọi API.")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    
    # Tìm file .env ở thư mục gốc, thư mục scripts/, hoặc thư mục làm việc hiện tại
    env_candidates = [
        project_root / ".env",
        project_root / "scripts" / ".env",
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env"
    ]
    for env_p in env_candidates:
        if env_p.exists():
            load_env_file(env_p)
            break

    raw_csv_path = project_root / "data" / "raw" / "lemotif_raw.csv"
    download_raw_data(raw_csv_path)

    print("[*] Đang đọc và phân tích cấu trúc CSV gốc...")
    all_records = parse_raw_csv(raw_csv_path)
    print(f"[+] Đã đọc {len(all_records)} mẫu từ CSV gốc.")

    if args.test_parser:
        print("\n--- TEST PARSER (5 mẫu đầu tiên) ---")
        for r in all_records[:5]:
            print(f"ID: {r['id']} | Labels: {r['labels']}")
            print(f"Text EN: {r['text_en'][:120]}...\n")
        print("[+] Test parser thành công!")
        return

    # Lấy danh sách API Key (hỗ trợ nhiều key ngăn cách bởi dấu phẩy)
    raw_keys = args.api_key or os.environ.get("GEMINI_API_KEY", "").strip()
    api_keys = [k.strip() for k in re.split(r"[,;\s\n]+", raw_keys) if k.strip()]
    if not api_keys:
        print("\n[LỖI] Chưa tìm thấy Gemini API Key!")
        print("Cách cung cấp key:")
        print("  1. Truyền qua tham số: python scripts/prepare_lemotif_vi.py --api-key YOUR_KEY")
        print("  2. Đặt vào file .env: GEMINI_API_KEY=key_1,key_2,...")
        print("  3. Đặt biến môi trường: set GEMINI_API_KEY=key_1,key_2 (Windows)")
        print("\nBạn có thể lấy key miễn phí tại: https://aistudio.google.com/apikey")
        sys.exit(1)

    print(f"[*] Đã nhận diện {len(api_keys)} API Key và {len(FALLBACK_MODELS)} models trong pool xoay vòng.")

    if args.mode == "pilot":
        target_records = select_pilot_samples(all_records, limit=args.limit)
        output_file = project_root / "data" / "raw" / "lemotif_vi_pilot.jsonl"
        checkpoint_file = project_root / "data" / "raw" / ".checkpoint_lemotif_vi_pilot.jsonl"
        print(f"\n[*] Bắt đầu chế độ PILOT: {len(target_records)} mẫu tiêu biểu.")
    else:
        target_records = all_records
        output_file = project_root / "data" / "raw" / "lemotif_vi.jsonl"
        checkpoint_file = project_root / "data" / "raw" / ".checkpoint_lemotif_vi.jsonl"
        print(f"\n[*] Bắt đầu chế độ FULL: {len(target_records)} mẫu.")

    run_translation_pipeline(
        records=target_records,
        output_path=output_file,
        checkpoint_path=checkpoint_file,
        api_keys=api_keys,
        batch_size=args.batch_size,
        delay_sec=args.delay,
        preferred_model=args.model
    )


if __name__ == "__main__":
    main()
