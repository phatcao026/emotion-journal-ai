"""
metrics.py

Evaluation metrics for Multimodal Emotion Recognition.

Implements the standard SER/MER metric suite:
    - Weighted Accuracy (WA) — standard accuracy
    - Unweighted Accuracy (UA) — macro-averaged recall
    - Macro-F1
    - Confusion matrix
    - Per-class precision / recall / F1
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _to_numpy(x: Union[torch.Tensor, np.ndarray, List[int]]) -> np.ndarray:
    """Convert predictions/targets to a 1-D int numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(int)
    return np.asarray(x, dtype=int)


# ------------------------------------------------------------------
# Individual metrics
# ------------------------------------------------------------------

def compute_weighted_accuracy(
    predictions: Union[torch.Tensor, np.ndarray, List[int]],
    targets: Union[torch.Tensor, np.ndarray, List[int]],
) -> float:
    """Weighted Accuracy = correct / total."""
    preds = _to_numpy(predictions)
    tgts = _to_numpy(targets)
    if len(preds) != len(tgts):
        raise ValueError("predictions and targets must have the same length")
    if len(preds) == 0:
        raise ValueError("inputs must not be empty")
    return float((preds == tgts).sum()) / len(tgts)


def compute_unweighted_accuracy(
    predictions: Union[torch.Tensor, np.ndarray, List[int]],
    targets: Union[torch.Tensor, np.ndarray, List[int]],
    num_classes: Optional[int] = None,
) -> float:
    """Unweighted Accuracy (Macro Recall) = mean per-class recall."""
    preds = _to_numpy(predictions)
    tgts = _to_numpy(targets)
    if len(preds) != len(tgts):
        raise ValueError("predictions and targets must have the same length")
    if num_classes is None:
        num_classes = int(max(tgts.max(), preds.max())) + 1
    recalls = []
    for c in range(num_classes):
        mask = tgts == c
        if mask.sum() == 0:
            continue
        recalls.append(float((preds[mask] == c).sum()) / float(mask.sum()))
    return float(np.mean(recalls)) if recalls else 0.0


def compute_macro_f1(
    predictions: Union[torch.Tensor, np.ndarray, List[int]],
    targets: Union[torch.Tensor, np.ndarray, List[int]],
    num_classes: Optional[int] = None,
    zero_division: float = 0.0,
) -> float:
    """Macro-F1 = mean of per-class F1 scores."""
    preds = _to_numpy(predictions)
    tgts = _to_numpy(targets)
    if num_classes is None:
        num_classes = int(max(tgts.max(), preds.max())) + 1
    f1_scores = []
    for c in range(num_classes):
        tp = float(((preds == c) & (tgts == c)).sum())
        fp = float(((preds == c) & (tgts != c)).sum())
        fn = float(((preds != c) & (tgts == c)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else zero_division
        recall = tp / (tp + fn) if (tp + fn) > 0 else zero_division
        if precision + recall > 0:
            f1_scores.append(2 * precision * recall / (precision + recall))
        else:
            f1_scores.append(zero_division)
    return float(np.mean(f1_scores)) if f1_scores else zero_division


def compute_confusion_matrix(
    predictions: Union[torch.Tensor, np.ndarray, List[int]],
    targets: Union[torch.Tensor, np.ndarray, List[int]],
    num_classes: int,
    normalize: bool = False,
) -> np.ndarray:
    """Confusion matrix of shape ``(C, C)``.  ``cm[i][j]`` = samples from class *i* predicted as *j*."""
    preds = _to_numpy(predictions)
    tgts = _to_numpy(targets)
    cm = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, p in zip(tgts, preds):
        cm[t, p] += 1
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm / row_sums
    return cm


def compute_per_class_metrics(
    predictions: Union[torch.Tensor, np.ndarray, List[int]],
    targets: Union[torch.Tensor, np.ndarray, List[int]],
    num_classes: int,
    class_names: Optional[List[str]] = None,
) -> List[Dict[str, float]]:
    """Per-class precision, recall, F1, and support."""
    preds = _to_numpy(predictions)
    tgts = _to_numpy(targets)
    results: List[Dict[str, float]] = []
    for c in range(num_classes):
        tp = float(((preds == c) & (tgts == c)).sum())
        fp = float(((preds == c) & (tgts != c)).sum())
        fn = float(((preds != c) & (tgts == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        name = class_names[c] if class_names else str(c)
        results.append({"class": name, "precision": prec, "recall": rec, "f1": f1, "support": int(tp + fn)})
    return results


# ------------------------------------------------------------------
# Aggregate
# ------------------------------------------------------------------

def compute_all_metrics(
    predictions: Union[torch.Tensor, np.ndarray, List[int]],
    targets: Union[torch.Tensor, np.ndarray, List[int]],
    num_classes: Optional[int] = None,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Compute WA, UA, Macro-F1, confusion matrix, and per-class metrics."""
    if num_classes is None:
        if class_names is not None:
            num_classes = len(class_names)
        else:
            preds_arr = _to_numpy(predictions)
            tgts_arr = _to_numpy(targets)
            max_val = max(
                int(preds_arr.max()) if len(preds_arr) > 0 else 0,
                int(tgts_arr.max()) if len(tgts_arr) > 0 else 0,
            )
            num_classes = max(1, max_val + 1)

    return {
        "wa": compute_weighted_accuracy(predictions, targets),
        "ua": compute_unweighted_accuracy(predictions, targets, num_classes),
        "macro_f1": compute_macro_f1(predictions, targets, num_classes),
        "confusion_matrix": compute_confusion_matrix(predictions, targets, num_classes),
        "per_class": compute_per_class_metrics(predictions, targets, num_classes, class_names),
        "num_samples": len(_to_numpy(predictions)),
        "num_classes": num_classes,
    }


def format_metrics_report(
    metrics: Dict,
    epoch: Optional[int] = None,
    phase: str = "val",
    title: Optional[str] = None,
) -> str:
    """Format metrics into a human-readable report string."""
    if title:
        header = f"========== [{title}]"
    else:
        header = f"========== [{phase.upper()}]"
    if epoch is not None:
        header += f" Epoch {epoch}"
    header += " =========="

    lines = [
        header,
        f"  WA:       {metrics['wa']:.4f}",
        f"  UA:       {metrics['ua']:.4f}",
        f"  Macro-F1: {metrics['macro_f1']:.4f}",
        f"  Samples:  {metrics['num_samples']}",
        "  " + "-" * 40,
        f"  {'Class':<16s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'Sup':>5s}",
    ]
    for entry in metrics.get("per_class", []):
        lines.append(
            f"  {str(entry['class']):<16s} "
            f"{entry['precision']:6.3f} {entry['recall']:6.3f} "
            f"{entry['f1']:6.3f} {entry['support']:5d}"
        )
    return "\n".join(lines)
