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


# ------------------------------------------------------------------
# Multi-Label Metrics (Text Subsystem & Hierarchical Classification)
# ------------------------------------------------------------------

def tune_multilabel_thresholds(
    y_true: Union[torch.Tensor, np.ndarray],
    probabilities: Union[torch.Tensor, np.ndarray],
    candidates: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Find per-label optimal F1 thresholds on a validation/dev set.

    Args:
        y_true: Ground truth binary matrix of shape (N, C).
        probabilities: Predicted probabilities in [0, 1] of shape (N, C).
        candidates: Candidate thresholds to evaluate. Default: np.arange(0.05, 0.951, 0.01).

    Returns:
        np.ndarray: Array of shape (C,) with tuned thresholds.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.detach().cpu().numpy()

    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    num_labels = y_true.shape[1]

    if candidates is None:
        candidates = np.arange(0.05, 0.951, 0.01)

    thresholds = np.full(num_labels, 0.5, dtype=float)
    for c in range(num_labels):
        if y_true[:, c].sum() == 0:
            continue
        scores = []
        for val in candidates:
            preds = (probabilities[:, c] >= val).astype(int)
            tp = float(((preds == 1) & (y_true[:, c] == 1)).sum())
            fp = float(((preds == 1) & (y_true[:, c] == 0)).sum())
            fn = float(((preds == 0) & (y_true[:, c] == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            scores.append(f1)
        thresholds[c] = candidates[int(np.argmax(scores))]

    return thresholds


def compute_multilabel_metrics(
    y_true: Union[torch.Tensor, np.ndarray],
    probabilities: Union[torch.Tensor, np.ndarray],
    thresholds: Optional[Union[float, List[float], np.ndarray]] = 0.5,
    label_names: Optional[List[str]] = None,
) -> Dict:
    """Compute standard multi-label evaluation metrics.

    Metrics computed:
        - Macro-F1: Macro-averaged F1 score across all classes.
        - Micro-F1: Micro-averaged F1 score globally over all instances and labels.
        - Exact Match (Subset Accuracy): Ratio of samples where all labels match exactly.
        - Hamming Loss: Fraction of labels that are incorrectly predicted.
        - Per-label precision, recall, F1, support, and threshold.

    Args:
        y_true: Ground truth binary matrix of shape (N, C).
        probabilities: Predicted probabilities in [0, 1] of shape (N, C).
        thresholds: Scalar threshold or array of per-label thresholds. Default: 0.5.
        label_names: Optional list of label names of length C.

    Returns:
        Dict: Dictionary containing macro_f1, micro_f1, exact_match, hamming_loss, per_label.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.detach().cpu().numpy()

    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    n_samples, n_classes = y_true.shape

    if np.isscalar(thresholds):
        thresh_arr = np.full(n_classes, float(thresholds))
    else:
        thresh_arr = np.asarray(thresholds, dtype=float)

    y_pred = (probabilities >= thresh_arr).astype(int)

    # Per-label metrics
    per_label = {}
    f1_list = []
    prec_list = []
    rec_list = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for c in range(n_classes):
        tp = int(((y_pred[:, c] == 1) & (y_true[:, c] == 1)).sum())
        fp = int(((y_pred[:, c] == 1) & (y_true[:, c] == 0)).sum())
        fn = int(((y_pred[:, c] == 0) & (y_true[:, c] == 1)).sum())
        support = int(y_true[:, c].sum())

        total_tp += tp
        total_fp += fp
        total_fn += fn

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)

        name = label_names[c] if label_names and c < len(label_names) else f"label_{c}"
        per_label[name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
            "threshold": round(float(thresh_arr[c]), 4),
        }

    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0
    macro_precision = float(np.mean(prec_list)) if prec_list else 0.0
    macro_recall = float(np.mean(rec_list)) if rec_list else 0.0

    micro_prec = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = float(2 * micro_prec * micro_rec / (micro_prec + micro_rec)) if (micro_prec + micro_rec) > 0 else 0.0

    exact_match = float((y_pred == y_true).all(axis=1).mean()) if n_samples > 0 else 0.0
    hamming_loss = float((y_pred != y_true).mean()) if n_samples > 0 else 0.0

    return {
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "exact_match": round(exact_match, 4),
        "hamming_loss": round(hamming_loss, 4),
        "num_samples": n_samples,
        "num_classes": n_classes,
        "per_label": per_label,
    }


def format_multilabel_report(report: Dict, title: Optional[str] = None) -> str:
    """Format multi-label report dictionary into a clean Markdown / console table."""
    header = f"========== [{title or 'Multi-Label Emotion Evaluation'}] =========="
    lines = [
        header,
        f"  Macro-F1:    {report['macro_f1']:.4f}",
        f"  Micro-F1:    {report['micro_f1']:.4f}",
        f"  Exact Match: {report['exact_match']:.4f}",
        f"  Hamming Loss:{report['hamming_loss']:.4f}",
        f"  Samples:     {report['num_samples']}",
        "  " + "-" * 55,
        f"  {'Emotion':<16s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'Thresh':>7s} {'Sup':>6s}",
    ]
    for name, m in report.get("per_label", {}).items():
        lines.append(
            f"  {name:<16s} {m['precision']:7.3f} {m['recall']:7.3f} {m['f1']:7.3f} {m['threshold']:7.2f} {m['support']:6d}"
        )
    return "\n".join(lines)

