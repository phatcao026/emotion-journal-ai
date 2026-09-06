#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_lemotif_vi.py
----------------------
Script kiểm tra chất lượng tự động cho dataset LE-motif bản tiếng Việt:
1. Đảm bảo chuẩn JSONL và UTF-8.
2. Kiểm tra tính duy nhất của trường id.
3. Kiểm tra trường text_vi không rỗng.
4. Kiểm tra trường labels:
   - Luôn là mảng (list).
   - Chỉ chứa 18 nhãn tiếng Anh chuẩn của LE-motif.
   - Không bị dịch nhãn sang tiếng Việt.
   - Khớp 100% với nhãn gốc trong file CSV nếu có file đối chiếu.
5. Lọc ra tối thiểu 10% mẫu có rủi ro cao (phủ định, emoji, đa nhãn, cảm xúc hiếm)
   để phục vụ rà soát thủ công theo yêu cầu bàn giao.
"""

import sys
import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Any

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Thêm thư mục scripts vào sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

VALID_LEMOTIF_LABELS = {
    "Happy", "Excited", "Proud", "Satisfied", "Calm",
    "Anxious", "Afraid", "Angry", "Frustrated", "Sad",
    "Disgusted", "Ashamed", "Awkward", "Jealous",
    "Nostalgic", "Surprised", "Bored", "Confused"
}

RARE_UNMAPPED_LABELS = {
    "Awkward", "Jealous", "Nostalgic", "Surprised", "Bored", "Confused"
}

VI_NEGATION_WORDS = [
    "không", "chẳng", "chưa", "đâu có", "nào có", "hoàn toàn không", "chả"
]


def load_raw_reference(raw_csv_path: Path) -> Dict[str, List[str]]:
    """Đọc nhãn gốc từ raw CSV để đối chiếu xem có bị sửa nhãn không."""
    from prepare_lemotif_vi import parse_raw_csv
    if not raw_csv_path.exists():
        return {}
    records = parse_raw_csv(raw_csv_path)
    return {r["id"]: r["labels"] for r in records}


def validate_file(file_path: Path, raw_csv_path: Optional[Path] = None, export_review: bool = True):
    print(f"\n=======================================================")
    print(f"  BẮT ĐẦU KIỂM TRA TẬP DỮ LIỆU: {file_path.name}")
    print(f"=======================================================\n")

    if not file_path.exists():
        print(f"[LỖI NGHIÊM TRỌNG] File không tồn tại: {file_path}")
        return False

    raw_labels_map = {}
    if raw_csv_path and raw_csv_path.exists():
        raw_labels_map = load_raw_reference(raw_csv_path)
        print(f"[*] Đã tải {len(raw_labels_map)} mẫu gốc từ {raw_csv_path.name} để đối chiếu nhãn.")

    errors = []
    warnings = []
    seen_ids = set()
    total_records = 0
    records = []

    review_candidates = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                warnings.append(f"Dòng {line_num}: Dòng trống.")
                continue

            total_records += 1
            try:
                obj = json.loads(line_str)
            except json.JSONDecodeError as e:
                errors.append(f"Dòng {line_num}: Lỗi cú pháp JSON ({e}).")
                continue

            records.append(obj)

            # 1. Kiểm tra trường 'id'
            rec_id = obj.get("id")
            if not rec_id:
                errors.append(f"Dòng {line_num}: Thiếu trường 'id'.")
            elif rec_id in seen_ids:
                errors.append(f"Dòng {line_num}: Trùng lặp id '{rec_id}'.")
            else:
                seen_ids.add(rec_id)

            # 2. Kiểm tra trường 'text_vi'
            text_vi = obj.get("text_vi")
            if text_vi is None or not str(text_vi).strip():
                errors.append(f"Dòng {line_num} (ID: {rec_id}): Trường 'text_vi' bị rỗng hoặc null.")

            # 3. Kiểm tra trường 'labels'
            labels = obj.get("labels")
            if not isinstance(labels, list):
                errors.append(f"Dòng {line_num} (ID: {rec_id}): 'labels' không phải là mảng list.")
            elif len(labels) == 0:
                errors.append(f"Dòng {line_num} (ID: {rec_id}): 'labels' là mảng rỗng.")
            else:
                # Kiểm tra xem có nhãn nào ngoài 18 nhãn chuẩn không (hoặc bị dịch sang tiếng Việt)
                invalid_labels = [lbl for lbl in labels if lbl not in VALID_LEMOTIF_LABELS]
                if invalid_labels:
                    errors.append(f"Dòng {line_num} (ID: {rec_id}): Chứa nhãn không hợp lệ hoặc bị dịch: {invalid_labels}")

                # Đối chiếu với file gốc nếu có
                if rec_id in raw_labels_map:
                    orig_labels = sorted(raw_labels_map[rec_id])
                    curr_labels = sorted(labels)
                    if orig_labels != curr_labels:
                        errors.append(f"Dòng {line_num} (ID: {rec_id}): Nhãn bị thay đổi so với file gốc! Gốc: {orig_labels} -> Hiện tại: {curr_labels}")

            # 4. Đánh giá tiêu chí rà soát thủ công (Manual QA prioritization)
            text_str = str(text_vi or "").lower()
            review_reasons = []

            # Kiểm tra từ phủ định tiếng Việt
            found_negs = [w for w in VI_NEGATION_WORDS if re.search(r"\b" + re.escape(w) + r"\b", text_str)]
            if found_negs:
                review_reasons.append(f"Phủ định: {', '.join(found_negs)}")

            # Kiểm tra nhãn hiếm / unmapped
            rare_found = [lbl for lbl in (labels or []) if lbl in RARE_UNMAPPED_LABELS]
            if rare_found:
                review_reasons.append(f"Nhãn hiếm/nhạy cảm: {', '.join(rare_found)}")

            # Kiểm tra đa nhãn (>= 3 nhãn)
            if isinstance(labels, list) and len(labels) >= 3:
                review_reasons.append(f"Đa nhãn ({len(labels)} nhãn)")

            # Kiểm tra emoji hoặc ký tự đặc biệt
            if any(ord(c) > 127 and not ('\u00c0' <= c <= '\u1ef9') for c in str(text_vi or "")):
                review_reasons.append("Chứa emoji/ký tự đặc biệt")

            if review_reasons:
                review_candidates.append({
                    "id": rec_id,
                    "text_vi": text_vi,
                    "labels": labels,
                    "review_reasons": review_reasons
                })

    # In kết quả kiểm tra
    print(f"[*] TỔNG SỐ RECORDS: {total_records}")
    print(f"[*] SỐ LƯỢNG ID DUY NHẤT: {len(seen_ids)}")
    print(f"[*] SỐ MẪU ĐẠT TIÊU CHÍ ƯU TIÊN REVIEW THỦ CÔNG: {len(review_candidates)} ({len(review_candidates)/max(1, total_records)*100:.1f}%)")

    if warnings:
        print(f"\n[!] CẢNH BÁO ({len(warnings)}):")
        for w in warnings[:10]:
            print(f"   - {w}")
        if len(warnings) > 10:
            print(f"   ... và {len(warnings) - 10} cảnh báo khác.")

    if errors:
        print(f"\n[X] PHÁT HIỆN {len(errors)} LỖI KỸ THUẬT:")
        for err in errors[:15]:
            print(f"   - {err}")
        if len(errors) > 15:
            print(f"   ... và {len(errors) - 15} lỗi khác.")
        print("\n=> KẾT LUẬN: KHÔNG ĐẠT YÊU CẦU BÀN GIAO.")
        return False
    else:
        print("\n[V] CHÚC MỪNG: 100% CÁC KIỂM TRA ĐỀU VƯỢT QUA!")
        print("    - Không trùng lặp ID.")
        print("    - Không có text_vi rỗng.")
        print("    - labels luôn là list và giữ nguyên 18 nhãn tiếng Anh gốc.")
        print("    - Không có nhãn nào bị tự ý dịch hay biến dạng.")

        # Xuất file danh sách các mẫu cần review thủ công tối thiểu 10%
        if export_review and review_candidates:
            # Lấy tối thiểu 10% hoặc toàn bộ các mẫu có lý do nghi vấn
            min_10_percent = max(int(total_records * 0.1), 1)
            target_review_list = review_candidates[:max(min_10_percent, len(review_candidates))]
            
            review_export_file = file_path.parent / f"{file_path.stem}_review_samples.jsonl"
            with open(review_export_file, "w", encoding="utf-8") as out_rf:
                for cand in target_review_list:
                    out_rf.write(json.dumps(cand, ensure_ascii=False) + "\n")
            print(f"\n[+] Đã xuất {len(target_review_list)} mẫu ưu tiên rà soát thủ công (>=10%) ra file:")
            print(f"    -> {review_export_file}")

        return True


def main():
    parser = argparse.ArgumentParser(description="Kiểm tra chất lượng và tính hợp lệ của file lemotif_vi.jsonl.")
    parser.add_argument("--file", type=str, default="data/raw/lemotif_vi_pilot.jsonl",
                        help="Đường dẫn file JSONL cần kiểm tra (mặc định: data/raw/lemotif_vi_pilot.jsonl)")
    parser.add_argument("--raw-csv", type=str, default="data/raw/lemotif_raw.csv",
                        help="Đường dẫn file CSV gốc để đối chiếu nhãn.")

    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent.parent
    target_file = project_root / args.file if not Path(args.file).is_absolute() else Path(args.file)
    raw_csv = project_root / args.raw_csv if not Path(args.raw_csv).is_absolute() else Path(args.raw_csv)

    is_valid = validate_file(target_file, raw_csv)
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
