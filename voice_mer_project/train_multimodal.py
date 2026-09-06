"""
train_multimodal.py

Training script for the Hierarchical Multimodal Emotion Recognition (MER) Model.

Pipeline:
    1. Load configuration from configs/multimodal_fusion.yaml (or multimodal_config.yaml)
    2. Build VoiceJournalDataset and DataLoader (with sub_labels=True)
    3. Initialize HierarchicalMERModel with Dual Classification Heads
    4. Compute weighted joint loss: L_total = w_primary * L_primary + w_sub * L_sub
    5. Evaluate periodically on WA, UA, and Macro-F1 across both heads
    6. Save best checkpoint based on validation performance

Usage:
    python train_multimodal.py --config configs/multimodal_fusion.yaml
    python train_multimodal.py --dummy --epochs 2
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Subset

from src.data.audio_dataset import VoiceJournalDataset
from src.losses.focal_loss import MultiClassFocalLoss
from src.multimodal.hierarchical_mer import HierarchicalMERModel
from src.utils.metrics import compute_all_metrics, format_metrics_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the Hierarchical Multimodal MER Model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/multimodal_fusion.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use synthetic dummy data to test the pipeline",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device (cuda / cpu)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def build_dataloaders(
    config: dict,
    dummy: bool = False,
    batch_size_override: Optional[int] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders."""
    train_cfg = config.get("training", {})
    batch_size = batch_size_override or train_cfg.get("batch_size", 8)

    if dummy:
        dataset = VoiceJournalDataset(
            dummy_mode=True, dummy_size=60, dummy_duration_s=6.0, use_sub_labels=True
        )
        train_idx, val_idx, test_idx = dataset.get_split_indices(
            train_ratio=0.7, val_ratio=0.15, seed=seed
        )
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)
        test_set = Subset(dataset, test_idx)
    else:
        data_dir = config.get("data", {}).get("data_dir", "data/voice_journals")
        train_set = VoiceJournalDataset(data_dir=data_dir, split="train", use_sub_labels=True)
        val_set = VoiceJournalDataset(data_dir=data_dir, split="val", use_sub_labels=True)
        test_set = VoiceJournalDataset(data_dir=data_dir, split="test", use_sub_labels=True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=VoiceJournalDataset.collate_fn,
        drop_last=len(train_set) > batch_size,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=VoiceJournalDataset.collate_fn,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=VoiceJournalDataset.collate_fn,
    )

    return train_loader, val_loader, test_loader


def build_model(config: dict, device: str) -> HierarchicalMERModel:
    """Initialize HierarchicalMERModel from configuration."""
    fusion_cfg = config.get("gated_fusion", {})
    heads_cfg = config.get("classification_heads", {})
    asr_cfg = config.get("asr", {})
    text_cfg = config.get("text_encoder", {})

    model = HierarchicalMERModel(
        num_primary_classes=heads_cfg.get("primary_head", {}).get("num_classes", 5),
        num_sub_classes=heads_cfg.get("sub_head", {}).get("num_classes", 10),
        fusion_dropout=fusion_cfg.get("dropout", 0.2),
        head_dropout=heads_cfg.get("primary_head", {}).get("dropout", 0.3),
        asr_model_id=asr_cfg.get("model_id", "vinai/phowhisper-base"),
        text_model_id=text_cfg.get("model_id", "vinai/phobert-base-v2"),
        device=device,
    )
    model.to(torch.device(device))
    return model


def train_one_epoch(
    model: HierarchicalMERModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_primary_fn: MultiClassFocalLoss,
    loss_sub_fn: MultiClassFocalLoss,
    primary_weight: float,
    sub_weight: float,
    device: torch.device,
    epoch: int,
    gradient_clip: float = 1.0,
) -> Dict[str, float]:
    """Train hierarchical model for one epoch."""
    model.train()
    total_loss = 0.0
    all_p_preds, all_p_targets = [], []
    all_s_preds, all_s_targets = [], []

    for batch in loader:
        waveforms = batch["waveforms"].squeeze(1).to(device)  # (B, T)
        p_targets = batch["primary_labels"].to(device)
        s_targets = batch["sub_labels"].to(device)
        transcripts = batch["transcripts"]

        optimizer.zero_grad()
        output = model(waveforms=waveforms, texts=transcripts, use_asr=False)

        l_primary = loss_primary_fn(output.primary_logits, p_targets)
        l_sub = loss_sub_fn(output.sub_logits, s_targets)
        loss = primary_weight * l_primary + sub_weight * l_sub

        loss.backward()
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        total_loss += loss.item()
        all_p_preds.append(output.primary_predictions().cpu())
        all_p_targets.append(p_targets.cpu())
        all_s_preds.append(output.sub_predictions().cpu())
        all_s_targets.append(s_targets.cpu())

    avg_loss = total_loss / max(1, len(loader))
    y_p_pred = torch.cat(all_p_preds) if all_p_preds else torch.empty(0)
    y_p_true = torch.cat(all_p_targets) if all_p_targets else torch.empty(0)
    m_primary = (
        compute_all_metrics(predictions=y_p_pred, targets=y_p_true, num_classes=5)
        if len(y_p_true) > 0
        else {}
    )

    y_s_pred = torch.cat(all_s_preds) if all_s_preds else torch.empty(0)
    y_s_true = torch.cat(all_s_targets) if all_s_targets else torch.empty(0)
    m_sub = (
        compute_all_metrics(predictions=y_s_pred, targets=y_s_true, num_classes=10)
        if len(y_s_true) > 0
        else {}
    )

    logger.info(
        "Epoch %02d [Train] Loss: %.4f | Primary F1: %.4f | Sub F1: %.4f",
        epoch,
        avg_loss,
        m_primary.get("macro_f1", 0.0),
        m_sub.get("macro_f1", 0.0),
    )
    return {
        "loss": avg_loss,
        "primary_f1": m_primary.get("macro_f1", 0.0),
        "sub_f1": m_sub.get("macro_f1", 0.0),
    }


@torch.no_grad()
def evaluate(
    model: HierarchicalMERModel,
    loader: DataLoader,
    loss_primary_fn: MultiClassFocalLoss,
    loss_sub_fn: MultiClassFocalLoss,
    primary_weight: float,
    sub_weight: float,
    device: torch.device,
    phase: str = "val",
) -> Dict[str, float]:
    """Evaluate hierarchical model on validation or test set."""
    model.eval()
    total_loss = 0.0
    all_p_preds, all_p_targets = [], []
    all_s_preds, all_s_targets = [], []

    for batch in loader:
        waveforms = batch["waveforms"].squeeze(1).to(device)
        p_targets = batch["primary_labels"].to(device)
        s_targets = batch["sub_labels"].to(device)
        transcripts = batch["transcripts"]

        output = model(waveforms=waveforms, texts=transcripts, use_asr=False)

        l_primary = loss_primary_fn(output.primary_logits, p_targets)
        l_sub = loss_sub_fn(output.sub_logits, s_targets)
        loss = primary_weight * l_primary + sub_weight * l_sub

        total_loss += loss.item()
        all_p_preds.append(output.primary_predictions().cpu())
        all_p_targets.append(p_targets.cpu())
        all_s_preds.append(output.sub_predictions().cpu())
        all_s_targets.append(s_targets.cpu())

    avg_loss = total_loss / max(1, len(loader))
    y_p_pred = torch.cat(all_p_preds) if all_p_preds else torch.empty(0)
    y_p_true = torch.cat(all_p_targets) if all_p_targets else torch.empty(0)
    m_primary = (
        compute_all_metrics(predictions=y_p_pred, targets=y_p_true, num_classes=5)
        if len(y_p_true) > 0
        else {}
    )

    y_s_pred = torch.cat(all_s_preds) if all_s_preds else torch.empty(0)
    y_s_true = torch.cat(all_s_targets) if all_s_targets else torch.empty(0)
    m_sub = (
        compute_all_metrics(predictions=y_s_pred, targets=y_s_true, num_classes=10)
        if len(y_s_true) > 0
        else {}
    )

    logger.info(
        "[%s] Loss: %.4f | Primary F1: %.4f (WA: %.4f) | Sub F1: %.4f",
        phase.upper(),
        avg_loss,
        m_primary.get("macro_f1", 0.0),
        m_primary.get("wa", 0.0),
        m_sub.get("macro_f1", 0.0),
    )
    return {
        "loss": avg_loss,
        "primary_f1": m_primary.get("macro_f1", 0.0),
        "primary_wa": m_primary.get("wa", 0.0),
        "primary_ua": m_primary.get("ua", 0.0),
        "sub_f1": m_sub.get("macro_f1", 0.0),
        "sub_wa": m_sub.get("wa", 0.0),
        "sub_ua": m_sub.get("ua", 0.0),
    }


def train(
    config: dict,
    dummy: bool = False,
    resume_path: Optional[str] = None,
    device: str = "cpu",
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    seed: int = 42,
) -> None:
    """Run full multimodal training workflow."""
    set_seed(seed)
    dev = torch.device(device)

    train_loader, val_loader, test_loader = build_dataloaders(
        config, dummy=dummy, batch_size_override=batch_size_override, seed=seed
    )

    model = build_model(config, device=device)

    train_cfg = config.get("training", {})
    lr = float(train_cfg.get("learning_rate", 1.0e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1.0e-2))
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    loss_cfg = config.get("loss", {})
    focal_gamma = float(loss_cfg.get("focal_gamma", 2.0))
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
    primary_weight = float(loss_cfg.get("primary_weight", 0.7))
    sub_weight = float(loss_cfg.get("sub_weight", 0.3))

    loss_primary = MultiClassFocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing)
    loss_sub = MultiClassFocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing)

    start_epoch = 1
    num_epochs = epochs_override or train_cfg.get("num_epochs", 60)
    grad_clip = float(train_cfg.get("gradient_clip", 1.0))
    patience = train_cfg.get("early_stopping_patience", 10)
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints/multimodal/"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer_state" in ckpt and ckpt["optimizer_state"]:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info("Resumed from %s at epoch %d", resume_path, start_epoch)

    best_combined_f1 = -1.0
    patience_counter = 0

    logger.info("Starting Hierarchical Multimodal training for %d epochs...", num_epochs)
    for epoch in range(start_epoch, num_epochs + 1):
        train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_primary_fn=loss_primary,
            loss_sub_fn=loss_sub,
            primary_weight=primary_weight,
            sub_weight=sub_weight,
            device=dev,
            epoch=epoch,
            gradient_clip=grad_clip,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            loss_primary_fn=loss_primary,
            loss_sub_fn=loss_sub,
            primary_weight=primary_weight,
            sub_weight=sub_weight,
            device=dev,
            phase="val",
        )

        combined_f1 = 0.7 * val_metrics.get("primary_f1", 0.0) + 0.3 * val_metrics.get("sub_f1", 0.0)
        if combined_f1 > best_combined_f1:
            best_combined_f1 = combined_f1
            patience_counter = 0
            best_ckpt_path = ckpt_dir / "best_model.pt"
            model.save_checkpoint(
                path=best_ckpt_path,
                epoch=epoch,
                optimizer_state=optimizer.state_dict(),
                metrics=val_metrics,
            )
            logger.info("Saved new best model to %s (Combined F1=%.4f)", best_ckpt_path, best_combined_f1)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered after %d epochs without improvement.", patience)
                break

    # Final test evaluation
    best_path = ckpt_dir / "best_model.pt"
    if best_path.exists():
        model = HierarchicalMERModel.from_checkpoint(best_path, device=device)
    logger.info("Final evaluation of best model on test split:")
    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        loss_primary_fn=loss_primary,
        loss_sub_fn=loss_sub,
        primary_weight=primary_weight,
        sub_weight=sub_weight,
        device=dev,
        phase="test",
    )
    logger.info("Test results: %s", test_metrics)


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    config = load_config(args.config)
    train(
        config=config,
        dummy=args.dummy,
        resume_path=args.resume,
        device=args.device,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
