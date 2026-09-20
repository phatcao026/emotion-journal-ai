"""
train_text.py

Training pipeline for the Vietnamese Text Emotion Recognition Subsystem.
Fine-tunes PhoBERT (vinai/phobert-base-v2) or ViSoBERT (uitnlp/visobert)
using Multi-Label BCEWithLogitsLoss or MultilabelFocalLoss with class-weighted loss.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.focal_loss import MultilabelBCEWithLogitsLoss, MultilabelFocalLoss
from src.common.metrics import (
    compute_multilabel_metrics,
    format_multilabel_report,
    tune_multilabel_thresholds,
)
from src.text.text_dataset import (
    TAXONOMY_11_LABELS,
    MultiLabelTextCollator,
    TextJournalDataset,
)
from src.text.text_encoder import TextEmotionClassifier, create_text_encoder
from src.text.text_preprocessor import VietnameseTextPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("train_text")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_pos_weights(dataset: TextJournalDataset, num_classes: int) -> torch.Tensor:
    """Calculate positive class weights to counter extreme multi-label imbalance."""
    all_targets = []
    for i in range(len(dataset)):
        all_targets.append(dataset[i]["sub_targets"].numpy())
    y = np.stack(all_targets)
    positives = y.sum(axis=0)
    negatives = len(y) - positives
    weights = negatives / np.maximum(positives, 1.0)
    weights[positives == 0] = 1.0
    # Soft clip to prevent exploding gradients
    weights = np.clip(weights, 1.0, 50.0)
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        targets = batch["sub_targets"].to(device)

        if "input_ids" in batch and batch["input_ids"] is not None:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            texts = batch["texts"]
            out = model(texts=texts)

        logits = out["logits"]
        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_targets = []

    for batch in loader:
        targets = batch["sub_targets"].to(device)
        if "input_ids" in batch and batch["input_ids"] is not None:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            texts = batch["texts"]
            out = model(texts=texts)

        probs = out["probs"].detach().cpu().numpy()
        all_probs.append(probs)
        all_targets.append(targets.detach().cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    return y_true, y_prob


def run_training(
    config_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    model_family: str = "visobert",
    model_id: Optional[str] = None,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    epochs: int = 5,
    seed: int = 42,
    use_focal_loss: bool = False,
) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load YAML config if present
    cfg: Dict[str, Any] = {}
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    # Resolve paths
    data_path = Path(data_dir or cfg.get("data", {}).get("prepared_dir", "data/manifests/"))
    out_dir = Path(output_dir or cfg.get("training", {}).get("checkpoint_dir", "checkpoints/text_only/"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    mid = model_id or cfg.get("text_encoder", {}).get("model_id")
    if not mid:
        mid = "uitnlp/visobert" if model_family == "visobert" else "vinai/phobert-base-v2"

    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(mid)
        logger.info("Loaded tokenizer from %s", mid)
    except Exception as e:
        logger.warning("Could not load HuggingFace tokenizer (%s). Running with internal fallback.", e)

    # Datasets
    preprocessor = VietnameseTextPreprocessor(normalize_teencode=True)

    train_file = data_path / "train.jsonl"
    dev_file = data_path / "dev.jsonl"
    test_file = data_path / "test.jsonl"

    if not train_file.exists():
        logger.warning("Training file %s not found. Using mock synthetic data for pipeline validation.", train_file)
        mock_data = [
            {"id": "s1", "text": "Hôm nay tôi rất vui và tự hào vì hoàn thành đồ án.", "labels": ["JOY"]},
            {"id": "s2", "text": "Cảm giác cô đơn và thất vọng bủa vây cả buổi tối.", "labels": ["SADNESS", "LONELINESS"]},
            {"id": "s3", "text": "Tôi rất lo lắng và bất an về bài thuyết trình sắp tới.", "labels": ["ANXIETY"]},
            {"id": "s4", "text": "Bực mình và tức giận vô cùng khi bị trễ hẹn.", "labels": ["ANGER"]},
        ] * 4
        train_ds = TextJournalDataset(mock_data, tokenizer=tokenizer, preprocessor=preprocessor)
        dev_ds = TextJournalDataset(mock_data[:4], tokenizer=tokenizer, preprocessor=preprocessor)
        test_ds = TextJournalDataset(mock_data[:4], tokenizer=tokenizer, preprocessor=preprocessor)
    else:
        train_ds = TextJournalDataset(train_file, tokenizer=tokenizer, preprocessor=preprocessor)
        dev_ds = TextJournalDataset(dev_file, tokenizer=tokenizer, preprocessor=preprocessor) if dev_file.exists() else train_ds
        test_ds = TextJournalDataset(test_file, tokenizer=tokenizer, preprocessor=preprocessor) if test_file.exists() else dev_ds

    collator = MultiLabelTextCollator(tokenizer=tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    # Model architecture
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
        dropout=float(cfg.get("classification_head", {}).get("dropout", 0.3)),
    ).to(device)

    # Pos weight & Loss
    pos_weights = calculate_pos_weights(train_ds, len(TAXONOMY_11_LABELS))
    if use_focal_loss:
        criterion = MultilabelFocalLoss(gamma=2.0, pos_weight=pos_weights).to(device)
    else:
        criterion = MultilabelBCEWithLogitsLoss(pos_weight=pos_weights).to(device)

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate, weight_decay=0.01)

    best_macro_f1 = 0.0
    best_thresholds = np.full(len(TAXONOMY_11_LABELS), 0.5)

    logger.info("Starting training: %d epochs, %d batches/epoch", epochs, len(train_loader))

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(classifier, train_loader, optimizer, criterion, device)
        dev_y_true, dev_y_prob = evaluate_epoch(classifier, dev_loader, device)

        # Tune thresholds on dev set
        dev_thresholds = tune_multilabel_thresholds(dev_y_true, dev_y_prob)
        dev_metrics = compute_multilabel_metrics(
            dev_y_true, dev_y_prob, thresholds=dev_thresholds, label_names=TAXONOMY_11_LABELS
        )

        logger.info(
            "Epoch %2d/%2d | Train Loss: %.4f | Dev Macro-F1: %.4f | Dev Micro-F1: %.4f",
            epoch, epochs, train_loss, dev_metrics["macro_f1"], dev_metrics["micro_f1"]
        )

        if dev_metrics["macro_f1"] >= best_macro_f1:
            best_macro_f1 = dev_metrics["macro_f1"]
            best_thresholds = dev_thresholds
            torch.save(classifier.state_dict(), out_dir / "best_model.pt")
            logger.info("--> Saved new best model checkpoint (Macro-F1: %.4f)", best_macro_f1)

    # Final evaluation on test set using best thresholds
    classifier.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    test_y_true, test_y_prob = evaluate_epoch(classifier, test_loader, device)
    test_metrics = compute_multilabel_metrics(
        test_y_true, test_y_prob, thresholds=best_thresholds, label_names=TAXONOMY_11_LABELS
    )

    logger.info("\n" + format_multilabel_report(test_metrics, title="FINAL TEST EVALUATION"))

    # Save artifacts
    with open(out_dir / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump({label: float(val) for label, val in zip(TAXONOMY_11_LABELS, best_thresholds)}, f, indent=2)

    with open(out_dir / "test_report.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    return {
        "best_macro_f1": best_macro_f1,
        "test_metrics": test_metrics,
        "checkpoint_dir": str(out_dir),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vietnamese Text Emotion Recognition model")
    parser.add_argument("--config", type=str, default="configs/text/text_only.yaml", help="Path to config yaml")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to prepared dataset directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints/text_only/", help="Output checkpoint directory")
    parser.add_argument("--model_family", type=str, default="visobert", choices=["visobert", "phobert"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--focal", action="store_true", help="Use MultilabelFocalLoss")
    args = parser.parse_args()

    run_training(
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_family=args.model_family,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_focal_loss=args.focal,
    )
