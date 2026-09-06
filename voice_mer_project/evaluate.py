"""
evaluate.py

Evaluation and reporting script for the Voice MER Pipeline.

Computes:
    - Weighted Accuracy (WA)
    - Unweighted Accuracy (UA)
    - Macro-F1
    - Per-class precision, recall, F1
    - Confusion Matrix (normalized & raw)

Usage:
    python evaluate.py --model-path checkpoints/multimodal/best_model.pt --dummy
    python evaluate.py --config configs/multimodal_fusion.yaml --dummy
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.audio_dataset import VoiceJournalDataset
from src.multimodal.hierarchical_mer import HierarchicalMERModel
from src.utils.metrics import compute_all_metrics, format_metrics_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Voice MER Model")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to saved model checkpoint (.pt)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/multimodal_fusion.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Run evaluation on dummy test dataset",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    return parser.parse_args()


def evaluate_model(
    model: HierarchicalMERModel,
    loader: DataLoader,
    device: torch.device,
) -> None:
    model.eval()
    p_preds, p_trues = [], []
    s_preds, s_trues = [], []

    with torch.no_grad():
        for batch in loader:
            waveforms = batch["waveforms"].squeeze(1).to(device)
            p_targets = batch["primary_labels"].to(device)
            s_targets = batch.get("sub_labels", None)
            transcripts = batch["transcripts"]

            output = model(waveforms=waveforms, texts=transcripts, use_asr=False)

            p_preds.append(output.primary_predictions().cpu())
            p_trues.append(p_targets.cpu())
            if s_targets is not None:
                s_preds.append(output.sub_predictions().cpu())
                s_trues.append(s_targets)

    y_p_pred = torch.cat(p_preds)
    y_p_true = torch.cat(p_trues)
    p_metrics = compute_all_metrics(
        predictions=y_p_pred,
        targets=y_p_true,
        num_classes=5,
        class_names=model.PRIMARY_LABELS,
    )

    print("\n" + "=" * 60)
    print("  PRIMARY HEAD EVALUATION REPORT (5 Classes)")
    print("=" * 60)
    print(format_metrics_report(p_metrics, title="Primary Emotion Classification"))

    if s_preds:
        y_s_pred = torch.cat(s_preds)
        y_s_true = torch.cat(s_trues)
        s_metrics = compute_all_metrics(
            predictions=y_s_pred,
            targets=y_s_true,
            num_classes=10,
            class_names=model.SUB_LABELS,
        )
        print("\n" + "=" * 60)
        print("  SUB-CATEGORY HEAD EVALUATION REPORT (10 Classes)")
        print("=" * 60)
        print(format_metrics_report(s_metrics, title="Sub-Category Emotion Classification"))


def main() -> None:
    args = parse_args()
    dev = torch.device(args.device)

    if args.dummy:
        dataset = VoiceJournalDataset(dummy_mode=True, dummy_size=30, dummy_duration_s=6.0, use_sub_labels=True)
    else:
        dataset = VoiceJournalDataset(data_dir="data/voice_journals", split="test", use_sub_labels=True)

    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=VoiceJournalDataset.collate_fn)

    if args.model_path and Path(args.model_path).exists():
        model = HierarchicalMERModel.from_checkpoint(args.model_path, device=args.device)
    else:
        model = HierarchicalMERModel(device=args.device)

    evaluate_model(model, loader, dev)


if __name__ == "__main__":
    main()
