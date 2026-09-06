"""
train_voice.py

Training script for the Voice-Only MER Model.

Pipeline:
    1. Load configuration from configs/voice_config.yaml (or voice_only.yaml)
    2. Build VoiceJournalDataset and DataLoader
    3. Initialize VoiceOnlyMERModel with LoRAEmotion2Vec
    4. Train with MultiClassFocalLoss
    5. Evaluate periodically using WA, UA, and Macro-F1
    6. Save the best checkpoint

Usage:
    python train_voice.py --config configs/voice_config.yaml
    python train_voice.py --dummy --epochs 2
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

from src.audio.emotion2vec_lora import LoRAConfig
from src.data.audio_dataset import VoiceJournalDataset
from src.losses.focal_loss import MultiClassFocalLoss
from src.multimodal.hierarchical_mer import VoiceOnlyMERModel
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
    """Load configuration from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train the Voice-Only MER Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/voice_config.yaml",
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
    batch_size = batch_size_override or train_cfg.get("batch_size", 16)

    if dummy:
        dataset = VoiceJournalDataset(dummy_mode=True, dummy_size=60, dummy_duration_s=6.0)
        train_idx, val_idx, test_idx = dataset.get_split_indices(
            train_ratio=0.7, val_ratio=0.15, seed=seed
        )
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)
        test_set = Subset(dataset, test_idx)
    else:
        data_dir = config.get("data", {}).get("data_dir", "data/voice_journals")
        train_set = VoiceJournalDataset(data_dir=data_dir, split="train")
        val_set = VoiceJournalDataset(data_dir=data_dir, split="val")
        test_set = VoiceJournalDataset(data_dir=data_dir, split="test")

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


def build_model(config: dict, device: str) -> VoiceOnlyMERModel:
    """Initialize VoiceOnlyMERModel from configuration."""
    lora_dict = config.get("lora", {})
    lora_config = LoRAConfig(
        r=lora_dict.get("r", 8),
        lora_alpha=lora_dict.get("lora_alpha", 16),
        lora_dropout=lora_dict.get("lora_dropout", 0.1),
        bias=lora_dict.get("bias", "none"),
        target_modules=lora_dict.get("target_modules", ["q_proj", "v_proj", "k_proj", "out_proj"]),
    )

    attn_dict = config.get("attention_pooling", {})
    train_dict = config.get("training", {})

    model = VoiceOnlyMERModel(
        lora_config=lora_config,
        num_primary_classes=train_dict.get("num_classes_primary", 5),
        attention_hidden_dim=attn_dict.get("hidden_dim", 256),
        dropout=attn_dict.get("dropout", 0.1),
        device=device,
    )
    model.to(torch.device(device))
    return model


def build_optimizer(model: VoiceOnlyMERModel, config: dict) -> optim.Optimizer:
    """Build AdamW optimizer for trainable parameters."""
    train_cfg = config.get("training", {})
    lr = float(train_cfg.get("learning_rate", 3.0e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-2))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)


def train_one_epoch(
    model: VoiceOnlyMERModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: MultiClassFocalLoss,
    device: torch.device,
    epoch: int,
    gradient_clip: float = 1.0,
) -> Dict[str, float]:
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch_idx, batch in enumerate(loader):
        waveforms = batch["waveforms"].squeeze(1).to(device)  # (B, T)
        targets = batch["primary_labels"].to(device)

        optimizer.zero_grad()
        output = model(waveforms=waveforms)
        loss = loss_fn(output.primary_logits, targets)

        loss.backward()
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        total_loss += loss.item()
        preds = output.primary_predictions().cpu()
        all_preds.append(preds)
        all_targets.append(targets.cpu())

    avg_loss = total_loss / max(1, len(loader))
    y_pred = torch.cat(all_preds) if all_preds else torch.empty(0)
    y_true = torch.cat(all_targets) if all_targets else torch.empty(0)
    metrics = (
        compute_all_metrics(predictions=y_pred, targets=y_true, num_classes=5)
        if len(y_true) > 0
        else {}
    )
    metrics["loss"] = avg_loss

    logger.info(
        "Epoch %02d [Train] Loss: %.4f | WA: %.4f | UA: %.4f | Macro-F1: %.4f",
        epoch,
        avg_loss,
        metrics.get("wa", 0.0),
        metrics.get("ua", 0.0),
        metrics.get("macro_f1", 0.0),
    )
    return metrics


@torch.no_grad()
def evaluate(
    model: VoiceOnlyMERModel,
    loader: DataLoader,
    loss_fn: MultiClassFocalLoss,
    device: torch.device,
    phase: str = "val",
) -> Dict[str, float]:
    """Evaluate model on validation or test set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in loader:
        waveforms = batch["waveforms"].squeeze(1).to(device)
        targets = batch["primary_labels"].to(device)

        output = model(waveforms=waveforms)
        loss = loss_fn(output.primary_logits, targets)

        total_loss += loss.item()
        all_preds.append(output.primary_predictions().cpu())
        all_targets.append(targets.cpu())

    avg_loss = total_loss / max(1, len(loader))
    y_pred = torch.cat(all_preds) if all_preds else torch.empty(0)
    y_true = torch.cat(all_targets) if all_targets else torch.empty(0)
    metrics = (
        compute_all_metrics(predictions=y_pred, targets=y_true, num_classes=5)
        if len(y_true) > 0
        else {}
    )
    metrics["loss"] = avg_loss

    logger.info(
        "[%s] Loss: %.4f | WA: %.4f | UA: %.4f | Macro-F1: %.4f",
        phase.upper(),
        avg_loss,
        metrics.get("wa", 0.0),
        metrics.get("ua", 0.0),
        metrics.get("macro_f1", 0.0),
    )
    return metrics


def train(
    config: dict,
    dummy: bool = False,
    resume_path: Optional[str] = None,
    device: str = "cpu",
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    seed: int = 42,
) -> None:
    """Execute training pipeline."""
    set_seed(seed)
    dev = torch.device(device)

    train_loader, val_loader, test_loader = build_dataloaders(
        config, dummy=dummy, batch_size_override=batch_size_override, seed=seed
    )

    model = build_model(config, device=device)
    optimizer = build_optimizer(model, config)

    focal_cfg = config.get("focal_loss", {})
    loss_fn = MultiClassFocalLoss(
        gamma=float(focal_cfg.get("gamma", 2.0)),
        reduction=focal_cfg.get("reduction", "mean"),
    )

    start_epoch = 1
    train_cfg = config.get("training", {})
    num_epochs = epochs_override or train_cfg.get("num_epochs", 50)
    grad_clip = float(train_cfg.get("gradient_clip", 1.0))
    patience = train_cfg.get("early_stopping_patience", 7)
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints/voice_only/"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer_state" in ckpt and ckpt["optimizer_state"]:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info("Resumed from %s at epoch %d", resume_path, start_epoch)

    best_val_f1 = -1.0
    patience_counter = 0

    logger.info("Starting Voice-Only training for %d epochs...", num_epochs)
    for epoch in range(start_epoch, num_epochs + 1):
        train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=dev,
            epoch=epoch,
            gradient_clip=grad_clip,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=dev,
            phase="val",
        )

        val_f1 = val_metrics.get("macro_f1", 0.0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_ckpt_path = ckpt_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "metrics": val_metrics,
                },
                best_ckpt_path,
            )
            logger.info("Saved new best model checkpoint to %s (Macro-F1=%.4f)", best_ckpt_path, best_val_f1)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered after %d epochs without improvement.", patience)
                break

    # Final evaluation on test split
    best_path = ckpt_dir / "best_model.pt"
    if best_path.exists():
        best_ckpt = torch.load(best_path, map_location=dev, weights_only=False)
        model.load_state_dict(best_ckpt["state_dict"])
    logger.info("Evaluating best model on test split:")
    test_metrics = evaluate(model=model, loader=test_loader, loss_fn=loss_fn, device=dev, phase="test")
    logger.info("\n%s", format_metrics_report(test_metrics))


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
