"""
src/audio/vigoemotions_dataset.py
---------------------------------
Adapter for uitnlp/vigoemotions (28 fine-grained Vietnamese emotion classes)
into Stage 2 (5 Primary x 11 Sub-Emotion) taxonomy with Text-Only Modality Gating.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Maps official uitnlp/vigoemotions label IDs (0..27) -> (primary_idx, sub_idx)
# Matches Stage3InferenceEngine.PRIMARY_NAMES and SUB_NAMES & TAXONOMY_MAP
VIGOEMOTIONS_TO_STAGE2: Dict[int, Optional[Tuple[int, int]]] = {
    0:  (0, 0),   # amusement      -> (JOY, JOY)
    1:  (0, 0),   # excitement     -> (JOY, JOY)
    2:  (0, 0),   # joy            -> (JOY, JOY)
    3:  (0, 3),   # love           -> (JOY, CONNECTION)
    4:  (0, 2),   # desire         -> (JOY, HOPE)
    5:  (0, 2),   # optimism       -> (JOY, HOPE)
    6:  (0, 3),   # caring         -> (JOY, CONNECTION)
    7:  (0, 0),   # pride          -> (JOY, JOY)
    8:  (0, 3),   # admiration     -> (JOY, CONNECTION)
    9:  (0, 3),   # gratitude      -> (JOY, CONNECTION)
    10: (4, 1),   # relief         -> (NEUTRAL, CALM)
    11: (0, 3),   # approval       -> (JOY, CONNECTION)
    12: (4, 1),   # realization    -> (NEUTRAL, CALM)
    13: None,     # surprise       -> Skip (ambiguous valence)
    14: (0, 2),   # curiosity      -> (JOY, HOPE)
    15: (2, 5),   # confusion      -> (ANXIETY, ANXIETY)
    16: (2, 6),   # fear           -> (ANXIETY, FEAR)
    17: (2, 5),   # nervousness    -> (ANXIETY, ANXIETY)
    18: (1, 8),   # remorse        -> (SADNESS, GUILT_SHAME)
    19: (1, 8),   # embarrassment  -> (SADNESS, GUILT_SHAME)
    20: (1, 4),   # disappointment -> (SADNESS, SADNESS)
    21: (1, 4),   # sadness        -> routed to LONELINESS (1, 9) if isolation keywords present
    22: (1, 9),   # grief          -> (SADNESS, LONELINESS)
    23: (3, 10),  # disgust        -> (ANGER, DISGUST)
    24: (3, 7),   # anger          -> (ANGER, ANGER)
    25: (3, 7),   # annoyance      -> (ANGER, ANGER)
    26: (3, 10),  # disapproval    -> (ANGER, DISGUST)
    27: (4, 1),   # neutral        -> (NEUTRAL, CALM)
}

LONELINESS_PATTERN = re.compile(
    r"\b(cô đơn|một mình|trống trải|bơ vơ|lẻ loi|cô độc|không ai hiểu)\b",
    flags=re.IGNORECASE,
)


class ViGoEmotionsStage2Dataset(Dataset):
    """Wraps HuggingFace uitnlp/vigoemotions for Stage 2 Multimodal co-training."""

    def __init__(self, split: str = "train", max_samples: Optional[int] = None) -> None:
        super().__init__()
        self.samples: List[Tuple[str, int, int]] = []
        try:
            from datasets import load_dataset
            ds = load_dataset("uitnlp/vigoemotions", split=split)
            for row in ds:
                text = str(row.get("text", "")).strip()
                labels = row.get("labels", [])
                if not text or not labels:
                    continue
                label_id = int(labels[0]) if isinstance(labels, list) else int(labels)
                mapped = VIGOEMOTIONS_TO_STAGE2.get(label_id)
                if mapped is None:
                    continue
                p_idx, s_idx = mapped
                if label_id == 21 and LONELINESS_PATTERN.search(text):
                    s_idx = 9  # LONELINESS
                self.samples.append((text, p_idx, s_idx))
                if max_samples and len(self.samples) >= max_samples:
                    break
            logger.info("Loaded %d mapped samples from ViGoEmotions (%s).", len(self.samples), split)
        except Exception as e:
            logger.warning("Could not load uitnlp/vigoemotions (%s). Proceeding without external text data.", e)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        text, p_idx, s_idx = self.samples[idx]
        return {
            "waveform": torch.zeros(1, 16000, dtype=torch.float32),  # Shape (1, 16000) for collate_fn
            "primary_label": torch.tensor(p_idx, dtype=torch.long),
            "sub_label": torch.tensor(s_idx, dtype=torch.long),
            "sample_id": f"vigo_{idx}",
            "duration_s": 1.0,
            "transcript": text,
            "speech_timestamps": [],
        }
