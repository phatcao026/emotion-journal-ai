"""
text_dataset.py

PyTorch Dataset and DataCollator for Vietnamese text emotion recognition.
Supports multi-label emotion annotations based on the 11 sub-emotion taxonomy
and 5 primary emotions defined in data/taxonomy.json.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from src.text.text_preprocessor import VietnameseTextPreprocessor

logger = logging.getLogger(__name__)

# Standard 11 sub-emotions aligned with data/taxonomy.json
TAXONOMY_11_LABELS: List[str] = [
    "JOY",
    "CALM",
    "HOPE",
    "CONNECTION",
    "SADNESS",
    "ANXIETY",
    "FEAR",
    "ANGER",
    "GUILT_SHAME",
    "LONELINESS",
    "DISGUST",
]

# Standard 5 primary emotions
TAXONOMY_5_PRIMARY: List[str] = [
    "JOY",
    "SADNESS",
    "ANXIETY",
    "ANGER",
    "NEUTRAL",
]

SUB_TO_PRIMARY: Dict[str, str] = {
    "JOY": "JOY",
    "CALM": "JOY",
    "HOPE": "JOY",
    "CONNECTION": "JOY",
    "SADNESS": "SADNESS",
    "LONELINESS": "SADNESS",
    "GUILT_SHAME": "SADNESS",
    "ANXIETY": "ANXIETY",
    "FEAR": "ANXIETY",
    "ANGER": "ANGER",
    "DISGUST": "ANGER",
}


def load_taxonomy(taxonomy_path: Union[str, Path]) -> Tuple[List[str], List[str], Dict[str, str]]:
    """Load primary and sub-emotion taxonomy from JSON file."""
    path = Path(taxonomy_path)
    if not path.exists():
        logger.warning("Taxonomy file %s not found. Using built-in 11 sub-emotions.", taxonomy_path)
        return TAXONOMY_5_PRIMARY, TAXONOMY_11_LABELS, SUB_TO_PRIMARY

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sub_dict = data.get("sub_emotions", {})
    sub_labels = [sub_dict[str(i)] for i in range(len(sub_dict))] if sub_dict else TAXONOMY_11_LABELS
    primary_dict = data.get("primary_emotions", {})
    primary_labels = [primary_dict[str(i)] for i in range(len(primary_dict))] if primary_dict else TAXONOMY_5_PRIMARY
    mapping = data.get("sub_to_primary_mapping", SUB_TO_PRIMARY)
    return primary_labels, sub_labels, mapping


class TextJournalDataset(Dataset):
    """Dataset for Vietnamese emotion recognition in personal journals and social text.

    Args:
        data_path: Path to a .jsonl or .csv file.
        tokenizer: HuggingFace AutoTokenizer or None.
        sub_labels: List of sub-emotion label names (length 11).
        primary_labels: List of primary emotion label names (length 5).
        max_length: Maximum sequence length for tokenization. Default: 256.
        preprocessor: Optional VietnameseTextPreprocessor instance.
        text_column: Name of the text column. Default: "text".
        labels_column: Name of the labels column. Default: "labels".
    """

    def __init__(
        self,
        data_path: Union[str, Path, List[Dict[str, Any]]],
        tokenizer: Optional[Any] = None,
        sub_labels: Optional[List[str]] = None,
        primary_labels: Optional[List[str]] = None,
        max_length: int = 256,
        preprocessor: Optional[VietnameseTextPreprocessor] = None,
        text_column: str = "text",
        labels_column: str = "labels",
    ) -> None:
        self.tokenizer = tokenizer
        self.sub_labels = sub_labels or TAXONOMY_11_LABELS
        self.primary_labels = primary_labels or TAXONOMY_5_PRIMARY
        self.max_length = max_length
        self.preprocessor = preprocessor or VietnameseTextPreprocessor(normalize_teencode=True)
        self.text_column = text_column
        self.labels_column = labels_column

        self.sub_label_to_idx = {name: i for i, name in enumerate(self.sub_labels)}
        self.primary_label_to_idx = {name: i for i, name in enumerate(self.primary_labels)}

        if isinstance(data_path, (str, Path)):
            self.samples = self._load_file(Path(data_path))
        elif isinstance(data_path, list):
            self.samples = data_path
        else:
            raise ValueError(f"Unsupported data_path type: {type(data_path)}")

        logger.info(
            "TextJournalDataset initialized: %d samples, %d sub-labels, max_length=%d",
            len(self.samples),
            len(self.sub_labels),
            self.max_length,
        )

    def _load_file(self, path: Path) -> List[Dict[str, Any]]:
        """Load samples from .jsonl or .csv."""
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        samples = []
        if path.suffix.lower() == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        samples.append(record)
                    except json.JSONDecodeError as e:
                        logger.warning("Skipping invalid JSON line %d in %s: %s", line_idx, path, e)
        elif path.suffix.lower() == ".csv":
            with open(path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    samples.append(row)
        else:
            raise ValueError(f"Unsupported file format '{path.suffix}'. Use .jsonl or .csv.")
        return samples

    def _parse_labels(self, raw_labels: Any) -> List[str]:
        """Normalize labels into a list of uppercase string tokens."""
        if raw_labels is None:
            return []
        if isinstance(raw_labels, str):
            try:
                parsed = json.loads(raw_labels)
                if isinstance(parsed, list):
                    return [str(item).strip().upper() for item in parsed]
            except json.JSONDecodeError:
                pass
            # Split on comma or pipe
            tokens = [t.strip().upper() for t in raw_labels.replace("|", ",").split(",") if t.strip()]
            return tokens
        if isinstance(raw_labels, (list, tuple, set)):
            return [str(item).strip().upper() for item in raw_labels]
        return [str(raw_labels).strip().upper()]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]

        # Extract text content
        raw_text = item.get(self.text_column) or item.get("journal_content") or item.get("content") or ""
        cleaned_text = self.preprocessor.clean_text(self.preprocessor.normalize_unicode(str(raw_text)))

        # Extract labels
        raw_labels = item.get(self.labels_column) or item.get("sub_emotions") or []
        label_names = self._parse_labels(raw_labels)

        # Multi-label binary vector for 11 sub-emotions
        sub_targets = np.zeros(len(self.sub_labels), dtype=np.float32)
        for lbl in label_names:
            if lbl in self.sub_label_to_idx:
                sub_targets[self.sub_label_to_idx[lbl]] = 1.0

        # Optional primary emotion mapping
        primary_targets = np.zeros(len(self.primary_labels), dtype=np.float32)
        for lbl in label_names:
            prim = SUB_TO_PRIMARY.get(lbl)
            if prim and prim in self.primary_label_to_idx:
                primary_targets[self.primary_label_to_idx[prim]] = 1.0

        sample_id = str(item.get("id", f"sample_{idx}"))

        result: Dict[str, Any] = {
            "id": sample_id,
            "text": cleaned_text,
            "sub_targets": torch.tensor(sub_targets, dtype=torch.float32),
            "primary_targets": torch.tensor(primary_targets, dtype=torch.float32),
            "gold_labels": [l for l in label_names if l in self.sub_label_to_idx],
        }

        # Tokenization if tokenizer is provided
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                cleaned_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,  # Padded in collator
                return_tensors=None,
            )
            result["input_ids"] = encoded["input_ids"]
            result["attention_mask"] = encoded["attention_mask"]

        return result


class MultiLabelTextCollator:
    """Collates a list of TextJournalDataset samples into a batch with dynamic padding."""

    def __init__(self, tokenizer: Optional[Any] = None, max_length: int = 256) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [b["text"] for b in batch]
        ids = [b["id"] for b in batch]
        sub_targets = torch.stack([b["sub_targets"] for b in batch])
        primary_targets = torch.stack([b["primary_targets"] for b in batch])

        out: Dict[str, Any] = {
            "ids": ids,
            "texts": texts,
            "sub_targets": sub_targets,
            "primary_targets": primary_targets,
            "labels": sub_targets,  # Aligned with HuggingFace Trainer API
        }

        if self.tokenizer is not None and "input_ids" in batch[0]:
            # Batch tokenization padding
            input_ids_list = [b["input_ids"] for b in batch]
            attention_mask_list = [b["attention_mask"] for b in batch]

            max_len = min(max(len(ids) for ids in input_ids_list), self.max_length)
            pad_token_id = getattr(self.tokenizer, "pad_token_id", 0) or 0

            padded_input_ids = []
            padded_attention_mask = []
            for seq_ids, seq_mask in zip(input_ids_list, attention_mask_list):
                diff = max_len - len(seq_ids)
                if diff > 0:
                    padded_input_ids.append(seq_ids + [pad_token_id] * diff)
                    padded_attention_mask.append(seq_mask + [0] * diff)
                else:
                    padded_input_ids.append(seq_ids[:max_len])
                    padded_attention_mask.append(seq_mask[:max_len])

            out["input_ids"] = torch.tensor(padded_input_ids, dtype=torch.long)
            out["attention_mask"] = torch.tensor(padded_attention_mask, dtype=torch.long)

        return out
