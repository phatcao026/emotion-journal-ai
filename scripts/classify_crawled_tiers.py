"""
scripts/classify_crawled_tiers.py
---------------------------------
Kiểm định & Phân tầng Dữ liệu Huấn luyện Chuyên sâu (Tier Classification & Dataset Migration):
- Tier 1: `deep_journal` (Tâm sự trải lòng / Nhật ký tự sự ngôi thứ nhất có chiều sâu cảm xúc)
- Tier 2: `general_monologue` (Độc thoại sạch / Podcast chia sẻ / Tản văn đọc diễn cảm cho MER nền tảng)
- Quarantine: `crisis_quarantine` (Cách ly an toàn sinh học & đạo đức)

Tự động cập nhật toàn bộ 288 bản ghi metadata và file json chi tiết.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from collections import Counter

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from scripts.crawler.journal_filter import JournalFilter, AffectiveSelfDisclosureFilter

def main():
    base_dir = ROOT_DIR / "data" / "crawled"
    jsonl_path = base_dir / "metadata.jsonl"
    csv_path = base_dir / "metadata.csv"

    if not jsonl_path.exists():
        print(f"[-] Không tìm thấy {jsonl_path}")
        return

    jf = JournalFilter(min_word_count=5)
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    print(f"[*] Đang kiểm định và phân tầng {len(records)} bản ghi metadata...")

    tier_counts = Counter()
    updated_records = []

    for r in records:
        is_crisis = r.get("is_crisis", False)
        modality = r.get("modality", "voice")
        text = r.get("text_content", "").strip()

        if is_crisis:
            tier = "crisis_quarantine"
            disclosure_score = 1.0
            didactic_markers = []
        elif text:
            # Kiểm định bằng AffectiveSelfDisclosureFilter
            eval_res = jf.evaluate_affective_disclosure(text)
            tier = eval_res["data_tier"]
            if tier == "rejected":
                # Nếu text quá ngắn nhưng audio hợp lệ, vẫn giữ ở Tier 2 làm monologue
                tier = "general_monologue"
            disclosure_score = eval_res["disclosure_score"]
            didactic_markers = eval_res["matched_didactic"]
        else:
            # Voice không có text
            tier = "general_monologue"
            disclosure_score = 0.0
            didactic_markers = []

        r["data_tier"] = tier
        r["disclosure_score"] = disclosure_score
        if didactic_markers:
            r["didactic_markers"] = didactic_markers

        tier_counts[tier] += 1
        updated_records.append(r)

        # Cập nhật file JSON chi tiết nếu tồn tại
        sample_id = r.get("sample_id")
        if sample_id:
            for sub_dir in ["text", "voice", "text_voice", "crisis_quarantine"]:
                sub_json = base_dir / sub_dir / f"{sample_id}.json"
                if sub_json.exists():
                    try:
                        with open(sub_json, "r", encoding="utf-8") as f_sub:
                            jdata = json.load(f_sub)
                        jdata["data_tier"] = tier
                        jdata["disclosure_score"] = disclosure_score
                        with open(sub_json, "w", encoding="utf-8") as f_sub:
                            json.dump(jdata, f_sub, ensure_ascii=False, indent=2)
                    except Exception:
                        pass

    # Ghi đè lại metadata.jsonl
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in updated_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Ghi đè lại metadata.csv với cột data_tier
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id",
            "modality",
            "data_tier",
            "primary_label",
            "sub_label",
            "emotion_score",
            "text_content",
            "audio_filename",
            "duration_s",
            "source_platform",
            "source_url",
            "author",
            "author_hash",
            "is_crisis",
            "voice_type",
            "crawled_at"
        ])
        for r in updated_records:
            writer.writerow([
                r.get("sample_id", ""),
                r.get("modality", ""),
                r.get("data_tier", "general_monologue"),
                r.get("primary_label", ""),
                r.get("sub_label", ""),
                r.get("emotion_score", 0.0),
                r.get("text_content", "").replace("\n", " ").strip(),
                r.get("audio_filename", ""),
                r.get("duration_s", 0.0),
                r.get("source_platform", ""),
                r.get("source_url", ""),
                r.get("author", ""),
                r.get("author_hash", ""),
                r.get("is_crisis", False),
                r.get("voice_type", "unspecified"),
                r.get("crawled_at", "")
            ])

    print("\n" + "="*65)
    print("HOÀN TẤT PHÂN TẦNG DỮ LIỆU HUẤN LUYỆN (TRAINING TIERS REPORT)")
    print("="*65)
    total = len(updated_records)
    for t, cnt in tier_counts.items():
        pct = (cnt / total) * 100
        print(f"  * {t:<20}: {cnt:>4} mẫu ({pct:>5.1f}%)")
    print(f"\n[+] Đã cập nhật đồng bộ {jsonl_path.name} và {csv_path.name}!")
    print("="*65)

if __name__ == "__main__":
    main()
