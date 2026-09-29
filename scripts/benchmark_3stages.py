"""
scripts/benchmark_3stages.py
-----------------------------
Comprehensive 3-Way Benchmark Runner:
    1. Stage 1.5 Baseline (Voice-Only Acoustic Backbone)
    2. Stage 2 V1 Baseline (Initial Multimodal Gated Fusion 896D)
    3. Stage 2 V2 SOTA (Tri-Modal 928D + Melody 32D + Masking + LDAM)

Runs on both Local (CPU with --dummy) and Kaggle GPU with real test data.
Produces Markdown and LaTeX ready comparison tables for academic papers.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.audio.audio_dataset import VoiceJournalDataset
from src.common.metrics import compute_all_metrics, compute_confusion_matrix
from src.fusion.hierarchical_mer import HierarchicalMERModel, VoiceOnlyMERModel
from src.stage2.hierarchical_mer_stage2 import (
    Stage2HierarchicalMERModel,
    TAXONOMY_MAP,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Benchmark3Stages")


def calculate_taxonomy_violation_rate(
    primary_preds: torch.Tensor,
    sub_preds: torch.Tensor,
) -> float:
    """Calculate the percentage of sub-emotion predictions that violate taxonomy logic."""
    total = len(primary_preds)
    if total == 0:
        return 0.0

    violations = 0
    p_np = primary_preds.cpu().numpy()
    s_np = sub_preds.cpu().numpy()

    for p, s in zip(p_np, s_np):
        allowed_sub = TAXONOMY_MAP.get(int(p), [])
        if int(s) not in allowed_sub:
            violations += 1

    return (violations / total) * 100.0


@torch.no_grad()
def evaluate_stage1_5(
    checkpoint_path: Optional[str],
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate Stage 1.5 Acoustic-Only model."""
    logger.info("--> Evaluating Stage 1.5 (Acoustic Only)...")
    model = VoiceOnlyMERModel(num_primary_classes=5, device=str(device))
    model.to(device)
    model.eval()

    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded Stage 1.5 checkpoint: missing=%d, unexpected=%d", len(missing), len(unexpected))
    else:
        logger.warning("No Stage 1.5 checkpoint provided/found; running with initialized weights.")
        model.audio_encoder._synthetic = True

    all_p_preds, all_p_targets = [], []
    for batch in dataloader:
        waveforms = batch["waveforms"].squeeze(1).to(device)
        targets = batch["primary_labels"].to(device)
        output = model(waveforms=waveforms)
        all_p_preds.append(output.primary_predictions().cpu())
        all_p_targets.append(targets.cpu())

    y_pred = torch.cat(all_p_preds)
    y_true = torch.cat(all_p_targets)
    metrics = compute_all_metrics(predictions=y_pred, targets=y_true, num_classes=5)

    return {
        "stage": "Stage 1.5 (Voice-Only)",
        "modalities": "Acoustic (768) + Pause (128)",
        "dim_fused": 896,
        "primary_f1": metrics.get("macro_f1", 0.0),
        "primary_wa": metrics.get("accuracy", 0.0),
        "primary_ua": metrics.get("unweighted_accuracy", 0.0),
        "sub_f1": None,
        "violation_rate": "N/A (No Sub-head)",
        "anxiety_recall": metrics.get("class_metrics", {}).get(2, {}).get("recall", 0.0),
        "sadness_recall": metrics.get("class_metrics", {}).get(1, {}).get("recall", 0.0),
    }


@torch.no_grad()
def evaluate_stage2_v1(
    checkpoint_path: Optional[str],
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate Stage 2 V1 Multimodal Gated Fusion model."""
    logger.info("--> Evaluating Stage 2 V1 (Initial Multimodal Gated Fusion)...")
    model = HierarchicalMERModel(num_primary_classes=5, num_sub_classes=11, device=str(device))
    model.to(device)
    model.eval()

    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded Stage 2 V1 checkpoint: missing=%d, unexpected=%d", len(missing), len(unexpected))
    else:
        logger.warning("No Stage 2 V1 checkpoint provided/found; running with initialized weights.")
        model.audio_encoder._synthetic = True

    all_p_preds, all_p_targets = [], []
    all_s_preds, all_s_targets = [], []

    for batch in dataloader:
        waveforms = batch["waveforms"].squeeze(1).to(device)
        p_targets = batch["primary_labels"].to(device)
        s_targets = batch["sub_labels"].to(device)
        transcripts = batch["transcripts"]

        output = model(waveforms=waveforms, texts=transcripts, use_asr=False)
        all_p_preds.append(output.primary_predictions().cpu())
        all_p_targets.append(p_targets.cpu())
        all_s_preds.append(output.sub_predictions().cpu())
        all_s_targets.append(s_targets.cpu())

    y_p_pred = torch.cat(all_p_preds)
    y_p_true = torch.cat(all_p_targets)
    y_s_pred = torch.cat(all_s_preds)
    y_s_true = torch.cat(all_s_targets)

    m_primary = compute_all_metrics(predictions=y_p_pred, targets=y_p_true, num_classes=5)
    m_sub = compute_all_metrics(predictions=y_s_pred, targets=y_s_true, num_classes=11)
    violation_rate = calculate_taxonomy_violation_rate(y_p_pred, y_s_pred)

    return {
        "stage": "Stage 2 V1 (Initial Multimodal)",
        "modalities": "Voice (768) + Text (768) + Pause (128)",
        "dim_fused": 896,
        "primary_f1": m_primary.get("macro_f1", 0.0),
        "primary_wa": m_primary.get("accuracy", 0.0),
        "primary_ua": m_primary.get("unweighted_accuracy", 0.0),
        "sub_f1": m_sub.get("macro_f1", 0.0),
        "violation_rate": f"{violation_rate:.2f}%",
        "anxiety_recall": m_primary.get("class_metrics", {}).get(2, {}).get("recall", 0.0),
        "sadness_recall": m_primary.get("class_metrics", {}).get(1, {}).get("recall", 0.0),
    }


@torch.no_grad()
def evaluate_stage2_v2(
    checkpoint_path: Optional[str],
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate Stage 2 V2 SOTA Multimodal Model."""
    logger.info("--> Evaluating Stage 2 V2 (SOTA Multi-Modal with Melody & Masking)...")
    model = Stage2HierarchicalMERModel(
        num_primary_classes=5,
        num_sub_classes=11,
        use_melody=True,
        dim_melody=32,
        device=str(device),
    )
    model.to(device)
    model.eval()

    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded Stage 2 V2 checkpoint: missing=%d, unexpected=%d", len(missing), len(unexpected))
    else:
        logger.warning("No Stage 2 V2 checkpoint provided/found; running with initialized weights.")
        model.audio_encoder._synthetic = True

    all_p_preds, all_p_targets = [], []
    all_s_preds, all_s_targets = [], []

    for batch in dataloader:
        waveforms = batch["waveforms"].squeeze(1).to(device)
        p_targets = batch["primary_labels"].to(device)
        s_targets = batch["sub_labels"].to(device)
        transcripts = batch["transcripts"]

        output = model(waveforms=waveforms, texts=transcripts, use_asr=False)
        all_p_preds.append(output.primary_predictions().cpu())
        all_p_targets.append(p_targets.cpu())
        all_s_preds.append(output.sub_predictions(enforce_taxonomy=True).cpu())
        all_s_targets.append(s_targets.cpu())

    y_p_pred = torch.cat(all_p_preds)
    y_p_true = torch.cat(all_p_targets)
    y_s_pred = torch.cat(all_s_preds)
    y_s_true = torch.cat(all_s_targets)

    m_primary = compute_all_metrics(predictions=y_p_pred, targets=y_p_true, num_classes=5)
    m_sub = compute_all_metrics(predictions=y_s_pred, targets=y_s_true, num_classes=11)
    violation_rate = calculate_taxonomy_violation_rate(y_p_pred, y_s_pred)

    return {
        "stage": "Stage 2 V2 (SOTA Multi-Modal)",
        "modalities": "Voice (768) + Text (768) + Pause (128) + Melody (32)",
        "dim_fused": 928,
        "primary_f1": m_primary.get("macro_f1", 0.0),
        "primary_wa": m_primary.get("accuracy", 0.0),
        "primary_ua": m_primary.get("unweighted_accuracy", 0.0),
        "sub_f1": m_sub.get("macro_f1", 0.0),
        "violation_rate": f"{violation_rate:.2f}%",
        "anxiety_recall": m_primary.get("class_metrics", {}).get(2, {}).get("recall", 0.0),
        "sadness_recall": m_primary.get("class_metrics", {}).get(1, {}).get("recall", 0.0),
    }


def print_comparison_table(results: List[Dict[str, Any]]) -> None:
    """Format and print an academic comparison table."""
    headers = [
        "Pipeline Version",
        "Fused Dim",
        "Primary Macro-F1",
        "Primary WA",
        "Sub Macro-F1",
        "Hierarchy Violation",
        "Anxiety Recall",
    ]
    print("\n" + "=" * 115)
    print("       ACADEMIC BENCHMARK: 3-WAY PROGRESSION & ABLATION STUDY")
    print("=" * 115)
    header_line = f"{headers[0]:<30} | {headers[1]:<9} | {headers[2]:<16} | {headers[3]:<10} | {headers[4]:<12} | {headers[5]:<19} | {headers[6]}"
    print(header_line)
    print("-" * 115)

    for r in results:
        sub_f1_str = f"{r['sub_f1'] * 100:.2f}%" if r['sub_f1'] is not None else "N/A"
        line = (
            f"{r['stage']:<30} | "
            f"{r['dim_fused']:<9} | "
            f"{r['primary_f1'] * 100:>6.2f}%          | "
            f"{r['primary_wa'] * 100:>6.2f}%   | "
            f"{sub_f1_str:<12} | "
            f"{r['violation_rate']:<19} | "
            f"{r['anxiety_recall'] * 100:>6.2f}%"
        )
        print(line)
    print("=" * 115 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 3-Way Benchmark for Stage 1.5 vs Stage 2 V1 vs Stage 2 V2")
    parser.add_argument("--stage1-checkpoint", type=str, default="checkpoints/voice_only/best_model_stage1_5.pt")
    parser.add_argument("--stage2-v1-checkpoint", type=str, default="checkpoints/multimodal/best_model.pt")
    parser.add_argument("--stage2-v2-checkpoint", type=str, default="checkpoints/stage2/best_model_stage2.pt")
    parser.add_argument("--dummy", action="store_true", help="Run with synthetic dataset on CPU")
    parser.add_argument("--data-dir", type=str, default="data/voice_journals")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    logger.info("Initializing Test Dataloader (dummy=%s, device=%s)...", args.dummy, args.device)
    if args.dummy:
        dataset = VoiceJournalDataset(dummy_mode=True, dummy_size=40, dummy_duration_s=6.0, use_sub_labels=True)
        test_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=VoiceJournalDataset.collate_fn)
    else:
        test_dataset = VoiceJournalDataset(data_dir=args.data_dir, split="test", use_sub_labels=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=VoiceJournalDataset.collate_fn)

    results = []
    # 1. Stage 1.5
    res1 = evaluate_stage1_5(args.stage1_checkpoint, test_loader, device)
    results.append(res1)

    # 2. Stage 2 V1
    res2 = evaluate_stage2_v1(args.stage2_v1_checkpoint, test_loader, device)
    results.append(res2)

    # 3. Stage 2 V2
    res3 = evaluate_stage2_v2(args.stage2_v2_checkpoint, test_loader, device)
    results.append(res3)

    # Print comparison
    print_comparison_table(results)


if __name__ == "__main__":
    main()
