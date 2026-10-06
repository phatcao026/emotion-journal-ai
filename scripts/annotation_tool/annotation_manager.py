"""
Emotion Journal AI - Annotation Manager
Quản lý nạp dữ liệu, lưu trữ nhãn, tính toán thống kê và xuất tập dữ liệu huấn luyện.
"""

import os
import json
import csv
import glob
from datetime import datetime
from typing import List, Dict, Any, Optional

class AnnotationManager:
    def __init__(
        self,
        crawled_meta_path: str = "data/crawled/metadata.jsonl",
        annotation_dir: str = "data/annotated",
        taxonomy_path: str = "data/taxonomy.json"
    ):
        self.crawled_meta_path = crawled_meta_path
        self.annotation_dir = annotation_dir
        self.taxonomy_path = taxonomy_path
        self.annotations_jsonl = os.path.join(annotation_dir, "annotations.jsonl")
        self.annotations_csv = os.path.join(annotation_dir, "annotations.csv")
        os.makedirs(annotation_dir, exist_ok=True)

        self.taxonomy = self._load_taxonomy()
        self.samples: List[Dict[str, Any]] = []
        self.annotations: Dict[str, Dict[str, Any]] = {} # key: (sample_id, annotator_id)
        self.load_all()

    def _load_taxonomy(self) -> Dict[str, Any]:
        """Tải taxonomy cảm xúc phân cấp từ file cấu hình."""
        if os.path.exists(self.taxonomy_path):
            with open(self.taxonomy_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "primary_emotions": {"0": "JOY", "1": "SADNESS", "2": "ANXIETY", "3": "ANGER", "4": "NEUTRAL"},
            "sub_emotions": {"0": "JOY", "1": "CALM", "2": "HOPE", "3": "CONNECTION", "4": "SADNESS", "5": "ANXIETY", "6": "FEAR", "7": "ANGER", "8": "GUILT_SHAME", "9": "LONELINESS", "10": "DISGUST"},
            "sub_to_primary_mapping": {
                "JOY": "JOY", "CALM": "JOY", "HOPE": "JOY", "CONNECTION": "JOY",
                "SADNESS": "SADNESS", "LONELINESS": "SADNESS", "GUILT_SHAME": "SADNESS",
                "ANXIETY": "ANXIETY", "FEAR": "ANXIETY",
                "ANGER": "ANGER", "DISGUST": "ANGER"
            }
        }

    def load_all(self):
        """Nạp toàn bộ mẫu crawled và các lượt gán nhãn đã lưu."""
        self.samples = []
        sample_ids_seen = set()

        # 1. Nạp từ metadata.jsonl
        if os.path.exists(self.crawled_meta_path):
            with open(self.crawled_meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        sid = r.get("sample_id")
                        if sid and sid not in sample_ids_seen:
                            sample_ids_seen.add(sid)
                            self.samples.append(r)

        # 2. Nạp thêm các mẫu từ crisis_quarantine nếu chưa có trong metadata
        quarantine_dir = "data/crawled/crisis_quarantine"
        if os.path.exists(quarantine_dir):
            for q_file in glob.glob(os.path.join(quarantine_dir, "*.json")):
                try:
                    with open(q_file, "r", encoding="utf-8") as f:
                        q_data = json.load(f)
                        sid = q_data.get("sample_id")
                        if sid and sid not in sample_ids_seen:
                            sample_ids_seen.add(sid)
                            self.samples.append(q_data)
                except Exception:
                    pass

        # 3. Nạp các annotations đã lưu
        self.annotations = {}
        if os.path.exists(self.annotations_jsonl):
            with open(self.annotations_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        ann = json.loads(line)
                        sid = ann.get("sample_id")
                        aid = ann.get("annotator_id", "default")
                        self.annotations[f"{sid}___{aid}"] = ann

    def get_audio_path(self, sample: Dict[str, Any]) -> Optional[str]:
        """Tìm đường dẫn tệp âm thanh thực tế trên đĩa."""
        audio_fn = sample.get("audio_filename")
        if not audio_fn:
            return None
        mod = sample.get("modality", "voice")
        candidates = [
            os.path.join("data", "crawled", mod, audio_fn),
            os.path.join("data", "crawled", "voice", audio_fn),
            os.path.join("data", "crawled", "text_voice", audio_fn),
            os.path.join("data", "crawled", audio_fn),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p.replace("\\", "/")
        return None

    def get_filtered_samples(
        self,
        modality_filter: str = "all",
        status_filter: str = "all",
        annotator_id: str = "annotator_01",
        search_query: str = ""
    ) -> List[Dict[str, Any]]:
        """Lọc danh sách mẫu theo phương thức, trạng thái gán nhãn và tìm kiếm."""
        filtered = []
        for s in self.samples:
            sid = s.get("sample_id", "")
            mod = s.get("modality", "text")
            is_crisis = s.get("is_crisis", False) or mod == "crisis_quarantine"
            is_annotated = f"{sid}___{annotator_id}" in self.annotations

            # Lọc theo modality
            if modality_filter == "text" and mod != "text":
                continue
            elif modality_filter == "voice" and mod != "voice":
                continue
            elif modality_filter == "text_voice" and mod != "text_voice":
                continue
            elif modality_filter == "crisis" and not is_crisis:
                continue

            # Lọc theo trạng thái
            if status_filter == "unannotated" and is_annotated:
                continue
            elif status_filter == "annotated" and not is_annotated:
                continue

            # Lọc theo từ khóa tìm kiếm
            if search_query:
                q = search_query.lower()
                text_content = (s.get("text_content") or s.get("text") or "").lower()
                title = (s.get("title") or "").lower()
                author = (s.get("author") or s.get("author_hash") or "").lower()
                if q not in sid.lower() and q not in text_content and q not in title and q not in author:
                    continue

            filtered.append(s)
        return filtered

    def get_existing_annotation(self, sample_id: str, annotator_id: str) -> Optional[Dict[str, Any]]:
        """Lấy bản ghi gán nhãn đã lưu của mẫu theo annotator."""
        return self.annotations.get(f"{sample_id}___{annotator_id}")

    def save_annotation(
        self,
        sample_id: str,
        annotator_id: str,
        primary_label: str,
        sub_label: str,
        valence: float,
        arousal: float,
        congruence: str = "congruent",
        crisis_level: int = 0,
        notes: str = "",
        action_type: str = "annotated" # "annotated", "skipped", "quarantined"
    ) -> Dict[str, Any]:
        """Lưu kết quả gán nhãn vào JSONL và CSV."""
        record = {
            "sample_id": sample_id,
            "annotator_id": annotator_id,
            "primary_label": primary_label,
            "sub_label": sub_label,
            "valence": round(float(valence), 3),
            "arousal": round(float(arousal), 3),
            "congruence": congruence,
            "crisis_level": int(crisis_level),
            "notes": notes.strip(),
            "action_type": action_type,
            "annotated_at": datetime.now().isoformat()
        }

        key = f"{sample_id}___{annotator_id}"
        self.annotations[key] = record

        # 1. Ghi đè file annotations.jsonl đầy đủ
        with open(self.annotations_jsonl, "w", encoding="utf-8") as f:
            for k, ann in self.annotations.items():
                f.write(json.dumps(ann, ensure_ascii=False) + "\n")

        # 2. Xuất bảng CSV song song
        self._export_annotations_csv()

        return record

    def _export_annotations_csv(self):
        """Xuất toàn bộ nhãn đã gán ra file CSV phục vụ phân tích pandas/excel."""
        if not self.annotations:
            return
        fieldnames = [
            "sample_id", "annotator_id", "primary_label", "sub_label",
            "valence", "arousal", "congruence", "crisis_level",
            "notes", "action_type", "annotated_at"
        ]
        with open(self.annotations_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for ann in self.annotations.values():
                writer.writerow({k: ann.get(k, "") for k in fieldnames})

    def get_statistics(self, annotator_id: Optional[str] = None) -> Dict[str, Any]:
        """Tính toán các chỉ số thống kê gán nhãn."""
        total_samples = len(self.samples)
        annotated_subset = list(self.annotations.values())
        if annotator_id:
            annotated_subset = [a for a in annotated_subset if a.get("annotator_id") == annotator_id]

        total_annotated = len(annotated_subset)
        pct = (total_annotated / total_samples * 100) if total_samples > 0 else 0.0

        # Phân bố primary label
        primary_dist: Dict[str, int] = {}
        sub_dist: Dict[str, int] = {}
        crisis_count = 0
        avg_valence = 0.0
        avg_arousal = 0.0

        if total_annotated > 0:
            for a in annotated_subset:
                p = a.get("primary_label", "UNKNOWN")
                s = a.get("sub_label", "UNKNOWN")
                primary_dist[p] = primary_dist.get(p, 0) + 1
                sub_dist[s] = sub_dist.get(s, 0) + 1
                if a.get("crisis_level", 0) >= 2 or a.get("action_type") == "quarantined":
                    crisis_count += 1
                avg_valence += a.get("valence", 0.0)
                avg_arousal += a.get("arousal", 0.0)

            avg_valence = round(avg_valence / total_annotated, 2)
            avg_arousal = round(avg_arousal / total_annotated, 2)

        return {
            "total_samples": total_samples,
            "total_annotated": total_annotated,
            "percentage_done": round(pct, 1),
            "primary_distribution": primary_dist,
            "sub_distribution": sub_dist,
            "crisis_count": crisis_count,
            "avg_valence": avg_valence,
            "avg_arousal": avg_arousal,
            "active_annotators": list(set(a.get("annotator_id", "default") for a in self.annotations.values()))
        }
