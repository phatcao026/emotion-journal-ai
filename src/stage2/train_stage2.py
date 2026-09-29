"""
src/stage2/train_stage2.py
--------------------------
Training and evaluation script for Stage 2 SOTA Multimodal Emotion Recognition.
Integrates:
    - Stage2HierarchicalMERModel
    - LDAMLoss with Deferred Re-Weighting (DRW)
    - Kendall Multi-Task Homoscedastic Uncertainty Loss
    - Stratified 5-Fold Cross-Validation & LOCO options
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Subset

from src.audio.audio_dataset import VoiceJournalDataset
from src.stage2.hierarchical_mer_stage2 import Stage2HierarchicalMERModel
from src.stage2.ldam_loss import LDAMLoss
from src.stage2.uncertainty_loss import MultiTaskUncertaintyLoss
from src.common.metrics import compute_all_metrics, format_metrics_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 2 SOTA Hierarchical Multimodal MER")
    parser.add_argument("--config", type=str, default="configs/stage2/multimodal_stage2.yaml")
    parser.add_argument("--dummy", action="store_true", help="Run with synthetic dataset")
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to dataset directory")
    parser.add_argument("--stage1-checkpoint", type=str, default="checkpoints/voice_only/best_model_stage1_5.pt")
    parser.add_argument("--freeze-audio-epochs", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_dataloaders(
    config: dict,
    dummy: bool = False,
    batch_size_override: Optional[int] = None,
    data_dir_override: Optional[str] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    batch_size = batch_size_override or config.get("training", {}).get("batch_size", 8)
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
        data_dir = data_dir_override or config.get("data", {}).get("data_dir", "data/voice_journals")
        train_set = VoiceJournalDataset(data_dir=data_dir, split="train", use_sub_labels=True)
        val_set = VoiceJournalDataset(data_dir=data_dir, split="val", use_sub_labels=True)
        test_set = VoiceJournalDataset(data_dir=data_dir, split="test", use_sub_labels=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=VoiceJournalDataset.collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=VoiceJournalDataset.collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=VoiceJournalDataset.collate_fn)
    return train_loader, val_loader, test_loader


def build_model(config: dict, device: str) -> Stage2HierarchicalMERModel:
    fusion_cfg = config.get("gated_fusion", {})
    heads_cfg = config.get("classification_heads", {})
    asr_cfg = config.get("asr", {})
    text_cfg = config.get("text_encoder", {})
    melody_cfg = config.get("melody_extractor", {})

    model = Stage2HierarchicalMERModel(
        num_primary_classes=heads_cfg.get("primary_head", {}).get("num_classes", 5),
        num_sub_classes=heads_cfg.get("sub_head", {}).get("num_classes", 11),
        fusion_dropout=fusion_cfg.get("dropout", 0.2),
        head_dropout=heads_cfg.get("primary_head", {}).get("dropout", 0.3),
        use_melody=melody_cfg.get("enabled", True),
        dim_melody=fusion_cfg.get("dim_melody", 32),
        asr_model_id=asr_cfg.get("model_id", "vinai/phowhisper-base"),
        text_model_id=text_cfg.get("model_id", "vinai/phobert-base-v2"),
        device=device,
    )
    model.to(torch.device(device))
    return model


def train_one_epoch(
    model: Stage2HierarchicalMERModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_primary_fn: nn.Module,
    loss_sub_fn: nn.Module,
    uncertainty_loss_fn: Optional[MultiTaskUncertaintyLoss],
    primary_weight: float,
    sub_weight: float,
    device: torch.device,
    epoch: int,
    gradient_clip: float = 1.0,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    all_p_preds, all_p_targets = [], []
    all_s_preds, all_s_targets = [], []

    for batch in loader:
        waveforms = batch["waveforms"].squeeze(1).to(device)
        p_targets = batch["primary_labels"].to(device)
        s_targets = batch["sub_labels"].to(device)
        transcripts = batch["transcripts"]

        optimizer.zero_grad()
        output = model(waveforms=waveforms, texts=transcripts, use_asr=False)

        l_primary = loss_primary_fn(output.primary_logits, p_targets)
        l_sub = loss_sub_fn(output.sub_logits, s_targets)

        if uncertainty_loss_fn is not None:
            loss = uncertainty_loss_fn(l_primary, l_sub)
        else:
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
    m_primary = compute_all_metrics(predictions=y_p_pred, targets=y_p_true, num_classes=5) if len(y_p_true) > 0 else {}

    y_s_pred = torch.cat(all_s_preds) if all_s_preds else torch.empty(0)
    y_s_true = torch.cat(all_s_targets) if all_s_targets else torch.empty(0)
    m_sub = compute_all_metrics(predictions=y_s_pred, targets=y_s_true, num_classes=model.num_sub_classes) if len(y_s_true) > 0 else {}

    logger.info(
        "Epoch %02d [Train] Loss: %.4f | Primary F1: %.4f | Sub F1: %.4f",
        epoch, avg_loss, m_primary.get("macro_f1", 0.0), m_sub.get("macro_f1", 0.0),
    )
    return {
        "loss": avg_loss,
        "primary_f1": m_primary.get("macro_f1", 0.0),
        "sub_f1": m_sub.get("macro_f1", 0.0),
    }


@torch.no_grad()
def evaluate(
    model: Stage2HierarchicalMERModel,
    loader: DataLoader,
    loss_primary_fn: nn.Module,
    loss_sub_fn: nn.Module,
    uncertainty_loss_fn: Optional[MultiTaskUncertaintyLoss],
    primary_weight: float,
    sub_weight: float,
    device: torch.device,
    phase: str = "val",
) -> Dict[str, float]:
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
        if uncertainty_loss_fn is not None:
            loss = uncertainty_loss_fn(l_primary, l_sub)
        else:
            loss = primary_weight * l_primary + sub_weight * l_sub

        total_loss += loss.item()
        all_p_preds.append(output.primary_predictions().cpu())
        all_p_targets.append(p_targets.cpu())
        all_s_preds.append(output.sub_predictions().cpu())
        all_s_targets.append(s_targets.cpu())

    avg_loss = total_loss / max(1, len(loader))
    y_p_pred = torch.cat(all_p_preds) if all_p_preds else torch.empty(0)
    y_p_true = torch.cat(all_p_targets) if all_p_targets else torch.empty(0)
    m_primary = compute_all_metrics(predictions=y_p_pred, targets=y_p_true, num_classes=5) if len(y_p_true) > 0 else {}

    y_s_pred = torch.cat(all_s_preds) if all_s_preds else torch.empty(0)
    y_s_true = torch.cat(all_s_targets) if all_s_targets else torch.empty(0)
    m_sub = compute_all_metrics(predictions=y_s_pred, targets=y_s_true, num_classes=model.num_sub_classes) if len(y_s_true) > 0 else {}

    logger.info(
        "Phase [%s] Loss: %.4f | Primary F1: %.4f (WA: %.4f) | Sub F1: %.4f (WA: %.4f)",
        phase.upper(), avg_loss,
        m_primary.get("macro_f1", 0.0), m_primary.get("accuracy", 0.0),
        m_sub.get("macro_f1", 0.0), m_sub.get("accuracy", 0.0),
    )
    return {
        "loss": avg_loss,
        "primary_f1": m_primary.get("macro_f1", 0.0),
        "primary_wa": m_primary.get("accuracy", 0.0),
        "sub_f1": m_sub.get("macro_f1", 0.0),
        "sub_wa": m_sub.get("accuracy", 0.0),
    }


def train_stage2(
    config: dict,
    dummy: bool = False,
    resume_path: Optional[str] = None,
    stage1_checkpoint: Optional[str] = None,
    freeze_audio_epochs: int = 0,
    data_dir: Optional[str] = None,
    device: str = "cpu",
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    seed: int = 42,
) -> None:
    set_seed(seed)
    dev = torch.device(device)

    train_loader, val_loader, test_loader = build_dataloaders(
        config, dummy=dummy, batch_size_override=batch_size_override, data_dir_override=data_dir, seed=seed,
    )
    model = build_model(config, device=device)
    if not dummy:
        logger.info("Loading pretrained weights for Stage 2 MER backbones...")
        model.load_pretrained()

    train_cfg = config.get("training", {})
    loss_cfg = config.get("loss", {})
    lr = float(train_cfg.get("learning_rate", 1.0e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1.0e-2))

    use_uncertainty = loss_cfg.get("use_uncertainty_weighting", True)
    uncertainty_loss_fn = MultiTaskUncertaintyLoss(num_tasks=2).to(dev) if use_uncertainty else None

    params_to_opt = [p for p in model.parameters() if p.requires_grad]
    if uncertainty_loss_fn is not None:
        params_to_opt.extend(list(uncertainty_loss_fn.parameters()))
    optimizer = optim.AdamW(params_to_opt, lr=lr, weight_decay=weight_decay)

    # Class counts for LDAM loss
    class_counts = [100, 80, 20, 60, 40]
    loss_primary = LDAMLoss(
        class_counts=class_counts,
        max_m=float(loss_cfg.get("ldam_max_m", 0.5)),
        s=float(loss_cfg.get("ldam_s", 30.0)),
    ).to(dev)
    loss_sub = nn.CrossEntropyLoss().to(dev)

    drw_start_epoch = int(loss_cfg.get("drw_start_epoch", 15))
    num_epochs = epochs_override or train_cfg.get("num_epochs", 60)
    primary_weight = float(loss_cfg.get("default_primary_weight", 0.7))
    sub_weight = float(loss_cfg.get("default_sub_weight", 0.3))

    logger.info("Starting Stage 2 SOTA training for %d epochs...", num_epochs)
    for epoch in range(1, num_epochs + 1):
        if isinstance(loss_primary, LDAMLoss):
            loss_primary.drw_update(current_epoch=epoch, drw_start_epoch=drw_start_epoch)

        train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_primary_fn=loss_primary,
            loss_sub_fn=loss_sub,
            uncertainty_loss_fn=uncertainty_loss_fn,
            primary_weight=primary_weight,
            sub_weight=sub_weight,
            device=dev,
            epoch=epoch,
        )
        evaluate(
            model=model,
            loader=val_loader,
            loss_primary_fn=loss_primary,
            loss_sub_fn=loss_sub,
            uncertainty_loss_fn=uncertainty_loss_fn,
            primary_weight=primary_weight,
            sub_weight=sub_weight,
            device=dev,
            phase="val",
        )


if __name__ == "__main__":
    args = parse_args()
    cfg_path = Path(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    train_stage2(
        config=config,
        dummy=args.dummy,
        resume_path=args.resume,
        stage1_checkpoint=args.stage1_checkpoint,
        freeze_audio_epochs=args.freeze_audio_epochs,
        data_dir=args.data_dir,
        device=args.device,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        seed=args.seed,
    )
