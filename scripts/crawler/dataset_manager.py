"""
dataset_manager.py
------------------
Quản lý lưu trữ tập dữ liệu thu thập (Social Media Crawled Dataset).
Tổ chức chuẩn hóa theo 3 modalities tương thích với pipeline huấn luyện:
1. text/       : Mẫu văn bản thuần (.json)
2. voice/      : Mẫu âm thanh thuần 16kHz mono (.wav + .json)
3. text_voice/ : Cặp đôi đa phương thức hoàn chỉnh (.wav + .json transcript)

Chỉ mục metadata được đồng bộ tự động tại `metadata.csv` và `metadata.jsonl`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


class CrawledDatasetManager:
    """Quản lý dữ liệu cào cho bài toán MER (Multimodal Emotion Recognition)."""

    def __init__(self, base_dir: str = "data/crawled"):
        self.base_dir = Path(base_dir).resolve()
        self.text_dir = self.base_dir / "text"
        self.voice_dir = self.base_dir / "voice"
        self.text_voice_dir = self.base_dir / "text_voice"
        self.crisis_dir = self.base_dir / "crisis_quarantine"

        self.text_dir.mkdir(parents=True, exist_ok=True)
        self.voice_dir.mkdir(parents=True, exist_ok=True)
        self.text_voice_dir.mkdir(parents=True, exist_ok=True)
        self.crisis_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_csv = self.base_dir / "metadata.csv"
        self.metadata_jsonl = self.base_dir / "metadata.jsonl"

        self._crawled_urls: set[str] = set()
        self._crawled_samples: set[str] = set()

        self._init_csv()
        self._load_existing_indices()

    def _init_csv(self):
        """Khởi tạo file metadata.csv với các cột chuẩn theo taxonomy và bảo vệ danh tính."""
        if not self.metadata_csv.exists():
            with open(self.metadata_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "sample_id",
                    "modality",          # 'text', 'voice', 'text_voice'
                    "primary_label",     # JOY, SADNESS, ANXIETY, ANGER, NEUTRAL
                    "sub_label",         # JOY, CALM, HOPE, CONNECTION, SADNESS, ANXIETY, FEAR, ANGER, GUILT_SHAME, LONELINESS, DISGUST
                    "emotion_score",     # 0.0 -> 1.0
                    "text_content",
                    "audio_filename",
                    "duration_s",
                    "source_platform",   # threads, youtube, vnexpress...
                    "source_url",
                    "author",
                    "author_hash",       # Participant-aware anonymized ID
                    "is_crisis",         # True nếu phát hiện dấu hiệu tự hại / khủng hoảng
                    "voice_type",        # 'acted_radio', 'natural_monologue'
                    "crawled_at"
                ])

    def _load_existing_indices(self):
        """Đọc trước metadata đã cào để hỗ trợ Resumable Crawling (chống cào trùng)."""
        if self.metadata_jsonl.exists():
            try:
                with open(self.metadata_jsonl, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                item = json.loads(line)
                                if "sample_id" in item:
                                    self._crawled_samples.add(item["sample_id"])
                                if "source_url" in item and item["source_url"]:
                                    self._crawled_urls.add(item["source_url"])
                            except Exception:
                                pass
            except Exception:
                pass

    def is_url_crawled(self, url: str) -> bool:
        """Kiểm tra URL đã được cào trước đó chưa."""
        return url in self._crawled_urls

    def is_sample_exists(self, sample_id: str) -> bool:
        """Kiểm tra sample_id đã tồn tại chưa."""
        return sample_id in self._crawled_samples

    def append_metadata(self, entry: Dict[str, Any]):
        """Ghi nối tiếp bản ghi metadata vào CSV và JSONL."""
        now_str = datetime.now(timezone.utc).isoformat()
        entry.setdefault("crawled_at", now_str)

        # Tính author_hash nếu chưa có
        if "author_hash" not in entry or not entry["author_hash"]:
            platform = entry.get("source_platform", "generic")
            author = entry.get("author", "anonymous")
            entry["author_hash"] = hashlib.sha256(f"{platform}_{author}".encode("utf-8")).hexdigest()[:12]

        entry.setdefault("is_crisis", False)
        entry.setdefault("voice_type", "unspecified")

        # Cập nhật index in-memory
        if entry.get("sample_id"):
            self._crawled_samples.add(entry["sample_id"])
        if entry.get("source_url"):
            self._crawled_urls.add(entry["source_url"])

        with open(self.metadata_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        with open(self.metadata_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                entry.get("sample_id", ""),
                entry.get("modality", ""),
                entry.get("primary_label", ""),
                entry.get("sub_label", ""),
                entry.get("emotion_score", 0.0),
                entry.get("text_content", "").replace("\n", " ").strip(),
                entry.get("audio_filename", ""),
                entry.get("duration_s", 0.0),
                entry.get("source_platform", ""),
                entry.get("source_url", ""),
                entry.get("author", ""),
                entry.get("author_hash", ""),
                entry.get("is_crisis", False),
                entry.get("voice_type", "unspecified"),
                entry.get("crawled_at", now_str)
            ])

    def save_text_sample(
        self,
        sample_id: str,
        text: str,
        primary_label: str,
        sub_label: str,
        emotion_score: float,
        meta: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Lưu một mẫu văn bản nhật ký/tâm sự thuần."""
        meta = meta or {}
        filepath = self.text_dir / f"{sample_id}.json"
        data = {
            "sample_id": sample_id,
            "modality": "text",
            "text": text,
            "primary_label": primary_label,
            "sub_label": sub_label,
            "emotion_score": emotion_score,
            **meta
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.append_metadata({
            "sample_id": sample_id,
            "modality": "text",
            "primary_label": primary_label,
            "sub_label": sub_label,
            "emotion_score": emotion_score,
            "text_content": text,
            "audio_filename": "",
            "duration_s": 0.0,
            **meta
        })
        return filepath

    def save_voice_sample(
        self,
        sample_id: str,
        wav_bytes: bytes,
        primary_label: str,
        sub_label: str,
        emotion_score: float,
        duration_s: float,
        meta: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Lưu một mẫu giọng nói độc thoại/tâm sự thuần (WAV 16kHz Mono)."""
        meta = meta or {}
        audio_name = f"{sample_id}.wav"
        wav_path = self.voice_dir / audio_name
        with open(wav_path, "wb") as f:
            f.write(wav_bytes)

        json_path = self.voice_dir / f"{sample_id}.json"
        data = {
            "sample_id": sample_id,
            "modality": "voice",
            "audio_file": audio_name,
            "duration_s": duration_s,
            "primary_label": primary_label,
            "sub_label": sub_label,
            "emotion_score": emotion_score,
            **meta
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.append_metadata({
            "sample_id": sample_id,
            "modality": "voice",
            "primary_label": primary_label,
            "sub_label": sub_label,
            "emotion_score": emotion_score,
            "text_content": "",
            "audio_filename": audio_name,
            "duration_s": duration_s,
            **meta
        })
        return wav_path

    def save_text_voice_sample(
        self,
        sample_id: str,
        wav_bytes: bytes,
        text: str,
        primary_label: str,
        sub_label: str,
        emotion_score: float,
        duration_s: float,
        meta: Optional[Dict[str, Any]] = None
    ) -> Tuple[Path, Path]:
        """Lưu cặp đôi hoàn chỉnh: Audio (.wav) + Nội dung Text transcript tương ứng."""
        meta = meta or {}
        audio_name = f"{sample_id}.wav"
        wav_path = self.text_voice_dir / audio_name
        with open(wav_path, "wb") as f:
            f.write(wav_bytes)

        json_path = self.text_voice_dir / f"{sample_id}.json"
        data = {
            "sample_id": sample_id,
            "modality": "text_voice",
            "text": text,
            "audio_file": audio_name,
            "duration_s": duration_s,
            "primary_label": primary_label,
            "sub_label": sub_label,
            "emotion_score": emotion_score,
            **meta
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.append_metadata({
            "sample_id": sample_id,
            "modality": "text_voice",
            "primary_label": primary_label,
            "sub_label": sub_label,
            "emotion_score": emotion_score,
            "text_content": text,
            "audio_filename": audio_name,
            "duration_s": duration_s,
            **meta
        })
        return wav_path, json_path

    def save_crisis_sample(
        self,
        sample_id: str,
        text: str,
        crisis_keywords: List[str],
        primary_label: str = "SADNESS",
        sub_label: str = "SADNESS",
        meta: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Lưu mẫu bài viết chứa dấu hiệu khủng hoảng / tự hại vào khu vực cách ly (crisis_quarantine).
        Không đưa vào tập dữ liệu huấn luyện cảm xúc thông thường để bảo đảm an toàn đạo đức và tránh nhiễu lâm sàng.
        """
        meta = meta or {}
        meta["is_crisis"] = True
        meta["crisis_keywords"] = crisis_keywords

        filepath = self.crisis_dir / f"{sample_id}.json"
        data = {
            "sample_id": sample_id,
            "modality": "crisis_text",
            "text": text,
            "primary_label": primary_label,
            "sub_label": sub_label,
            "is_crisis": True,
            "crisis_keywords": crisis_keywords,
            **meta
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.append_metadata({
            "sample_id": sample_id,
            "modality": "crisis_quarantine",
            "primary_label": primary_label,
            "sub_label": sub_label,
            "emotion_score": 1.0,
            "text_content": text,
            "audio_filename": "",
            "duration_s": 0.0,
            "is_crisis": True,
            **meta
        })
        return filepath

    def get_statistics(self) -> Dict[str, Any]:
        """Thống kê tổng thể về số lượng mẫu, phân bố modality, cảm xúc và người nói."""
        stats = {
            "total_records": 0,
            "modalities": {"text": 0, "voice": 0, "text_voice": 0, "crisis_quarantine": 0},
            "primary_emotions": {},
            "sub_emotions": {},
            "unique_authors": set(),
            "crisis_count": 0,
            "total_audio_duration_s": 0.0
        }
        if not self.metadata_jsonl.exists():
            return stats

        with open(self.metadata_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    stats["total_records"] += 1
                    mod = item.get("modality", "text")
                    stats["modalities"][mod] = stats["modalities"].get(mod, 0) + 1

                    pri = item.get("primary_label", "UNKNOWN")
                    stats["primary_emotions"][pri] = stats["primary_emotions"].get(pri, 0) + 1

                    sub = item.get("sub_label", "UNKNOWN")
                    stats["sub_emotions"][sub] = stats["sub_emotions"].get(sub, 0) + 1

                    auth = item.get("author_hash") or item.get("author")
                    if auth:
                        stats["unique_authors"].add(auth)

                    if item.get("is_crisis"):
                        stats["crisis_count"] += 1

                    stats["total_audio_duration_s"] += float(item.get("duration_s", 0.0))
                except Exception:
                    pass

        stats["unique_authors_count"] = len(stats["unique_authors"])
        stats.pop("unique_authors", None)
        return stats

