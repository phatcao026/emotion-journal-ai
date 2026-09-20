"""
evaluate_text.py

Evaluation script for Vietnamese Text Emotion Recognition.
Computes Macro-F1, Micro-F1, Exact Match, and Per-Class metrics on test or pilot sets.
Can also output predictions with probabilities for Model-Assisted Annotation review.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.metrics import compute_multilabel_metrics, format_multilabel_report
from src.text.text_dataset import TAXONOMY_11_LABELS, MultiLabelTextCollator, TextJournalDataset
from src.text.text_encoder import TextEmotionClassifier, create_text_encoder
from src.text.text_preprocessor import VietnameseTextPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("evaluate_text")


def evaluate_checkpoint(
    checkpoint_path: str,
    data_path: str,
    thresholds_path: Optional[str] = None,
    output_predictions: Optional[str] = None,
    model_family: str = "visobert",
    model_id: Optional[str] = None,
    batch_size: int = 16,
    device_str: Optional[str] = None,
) -> Dict[str, Any]:
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Evaluating using device: %s", device)

    # 1. Load Dataset
    preprocessor = VietnameseTextPreprocessor(normalize_teencode=True)
    mid = model_id or ("uitnlp/visobert" if model_family == "visobert" else "vinai/phobert-base-v2")

    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(mid)
    except Exception as e:
        logger.warning("Tokenizer could not be loaded from HuggingFace (%s). Using fallback.", e)

    dataset = TextJournalDataset(data_path, tokenizer=tokenizer, preprocessor=preprocessor)
    collator = MultiLabelTextCollator(tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    # 2. Build Model & Load Checkpoint
    encoder = create_text_encoder(
        model_family=model_family,
        model_id=mid,
        device=str(device),
        use_synthetic_fallback=True,
    )
    encoder.load_pretrained()

    classifier = TextEmotionClassifier(
        encoder=encoder,
        num_classes=len(TAXONOMY_11_LABELS),
        num_primary_classes=5,
    ).to(device)

    ckpt = Path(checkpoint_path)
    if ckpt.exists():
        state_dict = torch.load(ckpt, map_location=device)
        classifier.load_state_dict(state_dict)
        logger.info("Loaded checkpoint from %s", ckpt)
    else:
        logger.warning("Checkpoint %s not found. Evaluating randomly initialized / synthetic model.", ckpt)

    classifier.eval()

    # 3. Load Thresholds
    thresholds = np.full(len(TAXONOMY_11_LABELS), 0.5)
    if thresholds_path and Path(thresholds_path).exists():
        with open(thresholds_path, "r", encoding="utf-8") as f:
            t_dict = json.load(f)
            thresholds = np.array([t_dict.get(lbl, 0.5) for lbl in TAXONOMY_11_LABELS])
        logger.info("Loaded tuned thresholds from %s", thresholds_path)

    # 4. Inference loop
    all_probs = []
    all_targets = []
    sample_records = []

    with torch.no_grad():
        for batch in loader:
            targets = batch["sub_targets"].to(device)
            if "input_ids" in batch and batch["input_ids"] is not None:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                out = model = classifier(input_ids=input_ids, attention_mask=attention_mask)
            else:
                texts = batch["texts"]
                out = classifier(texts=texts)

            probs = out["probs"].detach().cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.detach().cpu().numpy())

            for i in range(len(batch["texts"])):
                sample_p = probs[i]
                pred_labels = [lbl for lbl, p, t in zip(TAXONOMY_11_LABELS, sample_p, thresholds) if p >= t]
                sample_records.append({
                    "id": batch["ids"][i],
                    "text": batch["texts"][i],
                    "predicted_labels": pred_labels,
                    "probabilities": {lbl: round(float(p), 4) for lbl, p in zip(TAXONOMY_11_LABELS, sample_p)},
                })

    y_true = np.concatenate(all_targets, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)

    # 5. Compute Metrics
    report = compute_multilabel_metrics(
        y_true, y_prob, thresholds=thresholds, label_names=TAXONOMY_11_LABELS
    )

    print("\n" + format_multilabel_report(report, title=f"Evaluation on {Path(data_path).name}"))

    # 6. Save Predictions if requested
    if output_predictions:
        out_p = Path(output_predictions)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            for rec in sample_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("Predictions saved to %s", out_p)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Vietnamese Text Emotion Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--data", type=str, required=True, help="Path to test.jsonl or pilot data")
    parser.add_argument("--thresholds", type=str, default=None, help="Path to thresholds.json")
    parser.add_argument("--output", type=str, default=None, help="Path to output predictions.jsonl")
    parser.add_argument("--model_family", type=str, default="visobert", choices=["visobert", "phobert"])
    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        thresholds_path=args.thresholds,
        output_predictions=args.output,
        model_family=args.model_family,
    )
