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
from typing import Dict, List, Optional, Tuple

# Mitigate CUDA memory fragmentation on 15-16GB GPUs (Tesla T4)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    parser.add_argument("--reset-optimizer", action="store_true", help="Reset optimizer states when resuming checkpoint")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to dataset directory")
    parser.add_argument(
        "--stage1-checkpoint",
        type=str,
        default="checkpoints/visec_stage1_pretrained.pt" if Path("checkpoints/visec_stage1_pretrained.pt").exists() else "checkpoints/voice_only/best_model_stage1_5.pt",
        help="Path to pretrained acoustic encoder checkpoint",
    )
    parser.add_argument("--freeze-audio-epochs", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="checkpoints/stage2_5_retrain", help="Directory to save checkpoints")
    parser.add_argument("--model-name", type=str, default="stage2_5", help="Name tag for checkpoint files (e.g. best_model_{name}.pt)")
    parser.add_argument("--use-vigoemotions", action="store_true", help="Co-train with uitnlp/vigoemotions via Text-Only Modality Gate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_dataloaders(
    config: dict,
    dummy: bool = False,
    batch_size_override: Optional[int] = None,
    data_dir_override: Optional[str] = None,
    use_vigoemotions: bool = False,
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

        if use_vigoemotions or config.get("data", {}).get("use_vigoemotions", False):
            try:
                from src.audio.vigoemotions_dataset import ViGoEmotionsStage2Dataset
                from torch.utils.data import ConcatDataset
                vigo_train = ViGoEmotionsStage2Dataset(split="train")
                if len(vigo_train) > 0:
                    train_set = ConcatDataset([train_set, vigo_train])
                    logger.info("Co-training enabled: ConcatDataset contains %d samples (%d from ViGoEmotions).", len(train_set), len(vigo_train))
            except Exception as vigo_err:
                logger.warning("Could not co-train with ViGoEmotions (%s). Continuing with primary dataset only.", vigo_err)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=VoiceJournalDataset.collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=VoiceJournalDataset.collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=VoiceJournalDataset.collate_fn)
    return train_loader, val_loader, test_loader


def compute_dataset_class_counts(loader: DataLoader, num_primary: int = 5, num_sub: int = 11) -> Tuple[List[int], List[int]]:
    """Count actual Primary and Sub-Emotion samples from train_loader instead of hardcoded arrays."""
    p_counts = [0] * num_primary
    s_counts = [0] * num_sub
    for batch in loader:
        for p in batch["primary_labels"].tolist():
            if 0 <= p < num_primary:
                p_counts[p] += 1
        for s in batch["sub_labels"].tolist():
            if 0 <= s < num_sub:
                s_counts[s] += 1
    # Ensure minimum count of 1 for LDAM margin stability
    p_counts = [max(1, c) for c in p_counts]
    s_counts = [max(1, c) for c in s_counts]
    return p_counts, s_counts


def build_model(config: dict, device: str) -> Stage2HierarchicalMERModel:
    fusion_cfg = config.get("gated_fusion", {})
    heads_cfg = config.get("classification_heads", {})
    asr_cfg = config.get("asr", {})
    text_cfg = config.get("text_encoder", {})
    melody_cfg = config.get("melody_extractor", {})

    model = Stage2HierarchicalMERModel(
        num_primary_classes=heads_cfg.get("primary_head", {}).get("num_classes", 5),
        num_sub_classes=heads_cfg.get("sub_head", {}).get("num_classes", 11),
        sub_head_input_dim=heads_cfg.get("sub_head", {}).get("input_dim", None),
        fusion_dropout=fusion_cfg.get("dropout", 0.2),
        head_dropout=heads_cfg.get("primary_head", {}).get("dropout", 0.3),
        use_melody=melody_cfg.get("enabled", True),
        dim_melody=fusion_cfg.get("dim_melody", 32),
        modality_dropout=fusion_cfg.get("modality_dropout", 0.15),
        gate_temperature=fusion_cfg.get("gate_temperature", 0.8),
        asr_model_id=asr_cfg.get("model_id", "vinai/phowhisper-base"),
        text_model_id=text_cfg.get("model_id", "vinai/phobert-base-v2"),
        device=device,
    )
    model.to(torch.device(device))
    return model


def freeze_phobert_layers(
    model: Stage2HierarchicalMERModel,
    num_freeze_layers: int = 10,
) -> None:
    """Freeze lower layers of text encoder while allowing adaptation in top layers.

    PhoBERT architecture (vinai/phobert-base-v2):
        - Embeddings: requires_grad = False
        - Layers 0 to num_freeze_layers - 1: requires_grad = False
        - Layers num_freeze_layers to 11: requires_grad = True (adapted for journal domain)
    Synthetic backbone fallback:
        - embedding: requires_grad = False
        - proj: requires_grad = True
    """
    text_encoder = getattr(model, "text_encoder", None)
    if text_encoder is None:
        logger.warning("freeze_phobert_layers: model has no text_encoder attribute.")
        return

    bert = getattr(text_encoder, "bert_model", None)
    is_synth = getattr(text_encoder, "is_synthetic", False)

    if bert is not None and not is_synth:
        # Real PhoBERT backbone
        if hasattr(bert, "embeddings"):
            for p in bert.embeddings.parameters():
                p.requires_grad = False
        if hasattr(bert, "encoder") and hasattr(bert.encoder, "layer"):
            layers = bert.encoder.layer
            for i, layer in enumerate(layers):
                requires_grad = (i >= num_freeze_layers)
                for p in layer.parameters():
                    p.requires_grad = requires_grad
            logger.info(
                "PhoBERT layer freezing: embeddings & layers 0..%d frozen, layers %d..%d trainable.",
                num_freeze_layers - 1,
                num_freeze_layers,
                len(layers) - 1,
            )
        if hasattr(bert, "pooler") and bert.pooler is not None:
            for p in bert.pooler.parameters():
                p.requires_grad = False
    elif hasattr(text_encoder, "_synthetic_backbone") and text_encoder._synthetic_backbone is not None:
        synth = text_encoder._synthetic_backbone
        if hasattr(synth, "embedding"):
            for p in synth.embedding.parameters():
                p.requires_grad = False
        if hasattr(synth, "proj"):
            for p in synth.proj.parameters():
                p.requires_grad = True
        logger.info("Synthetic PhoBERT freezing: embedding frozen, proj trainable.")
    else:
        logger.warning("freeze_phobert_layers: unrecognized text_encoder structure.")


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
    grad_accum_steps: int = 1,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    all_p_preds, all_p_targets = [], []
    all_s_preds, all_s_targets = [], []

    try:
        from tqdm import tqdm
        pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [Train]", leave=False, dynamic_ncols=True)
    except ImportError:
        pbar = loader

    optimizer.zero_grad()
    accum_steps = max(1, grad_accum_steps)

    for batch_idx, batch in enumerate(pbar):
        waveforms = batch["waveforms"].squeeze(1).to(device)
        p_targets = batch["primary_labels"].to(device)
        s_targets = batch["sub_labels"].to(device)
        transcripts = batch["transcripts"]
        speech_ts = batch.get("speech_timestamps", None)

        output = model(
            waveforms=waveforms,
            speech_timestamps=speech_ts,
            texts=transcripts,
            use_asr=False,
        )

        l_primary = loss_primary_fn(output.primary_logits, p_targets)
        l_sub = loss_sub_fn(output.sub_logits, s_targets)

        if uncertainty_loss_fn is not None:
            loss = uncertainty_loss_fn(l_primary, l_sub)
        else:
            loss = primary_weight * l_primary + sub_weight * l_sub

        loss_scaled = loss / accum_steps
        loss_scaled.backward()

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        all_p_preds.append(output.primary_predictions().cpu())
        all_p_targets.append(p_targets.cpu())
        all_s_preds.append(output.sub_predictions().cpu())
        all_s_targets.append(s_targets.cpu())

        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
            print(f"  --> [Epoch {epoch:02d}] Batch {batch_idx + 1:3d}/{len(loader)} | Current Loss: {loss.item():.4f}", flush=True)

        if (batch_idx + 1) % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_loss = total_loss / max(1, len(loader))
    y_p_pred = torch.cat(all_p_preds) if all_p_preds else torch.empty(0)
    y_p_true = torch.cat(all_p_targets) if all_p_targets else torch.empty(0)
    m_primary = compute_all_metrics(predictions=y_p_pred, targets=y_p_true, num_classes=5) if len(y_p_true) > 0 else {}

    y_s_pred = torch.cat(all_s_preds) if all_s_preds else torch.empty(0)
    y_s_true = torch.cat(all_s_targets) if all_s_targets else torch.empty(0)
    valid_s_mask = (y_s_true >= 0) & (y_s_true < model.num_sub_classes)
    m_sub = compute_all_metrics(predictions=y_s_pred[valid_s_mask], targets=y_s_true[valid_s_mask], num_classes=model.num_sub_classes) if valid_s_mask.any() else {}

    summary_msg = f"\n>>> [EPOCH {epoch:02d} TRAIN COMPLETED] Loss: {avg_loss:.4f} | Primary F1: {m_primary.get('macro_f1', 0.0):.4f} | Sub F1: {m_sub.get('macro_f1', 0.0):.4f}"
    logger.info(summary_msg)
    print(summary_msg, flush=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
        speech_ts = batch.get("speech_timestamps", None)

        output = model(
            waveforms=waveforms,
            speech_timestamps=speech_ts,
            texts=transcripts,
            use_asr=False,
        )

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
    valid_s_mask = (y_s_true >= 0) & (y_s_true < model.num_sub_classes)
    m_sub = compute_all_metrics(predictions=y_s_pred[valid_s_mask], targets=y_s_true[valid_s_mask], num_classes=model.num_sub_classes) if valid_s_mask.any() else {}

    eval_msg = f"    >>> Phase [{phase.upper()}] Loss: {avg_loss:.4f} | Primary F1: {m_primary.get('macro_f1', 0.0):.4f} (WA: {m_primary.get('wa', 0.0):.4f}) | Sub F1: {m_sub.get('macro_f1', 0.0):.4f} (WA: {m_sub.get('wa', 0.0):.4f})"
    logger.info(eval_msg)
    print(eval_msg, flush=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "loss": avg_loss,
        "primary_f1": m_primary.get("macro_f1", 0.0),
        "primary_wa": m_primary.get("wa", 0.0),
        "sub_f1": m_sub.get("macro_f1", 0.0),
        "sub_wa": m_sub.get("wa", 0.0),
    }


def train_stage2(
    config: dict,
    dummy: bool = False,
    resume_path: Optional[str] = None,
    reset_optimizer: bool = False,
    stage1_checkpoint: Optional[str] = None,
    freeze_audio_epochs: int = 0,
    output_dir: Optional[str] = "checkpoints/stage2_5_retrain",
    model_name: str = "stage2_5",
    data_dir: Optional[str] = None,
    use_vigoemotions: bool = False,
    device: str = "cpu",
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    grad_accum_steps: int = 1,
    seed: int = 42,
) -> None:
    set_seed(seed)
    dev = torch.device(device)

    train_loader, val_loader, test_loader = build_dataloaders(
        config, dummy=dummy, batch_size_override=batch_size_override, data_dir_override=data_dir, use_vigoemotions=use_vigoemotions, seed=seed,
    )
    model = build_model(config, device=device)
    if not dummy:
        logger.info("Loading pretrained weights for Stage 2 MER backbones...")
        model.load_pretrained()

    start_epoch = 1
    best_val_f1 = -1.0
    ckpt_dir = Path(output_dir or "checkpoints/stage2_5_retrain")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Check if resuming from an existing Stage 2 checkpoint
    resumed = False
    if resume_path and Path(resume_path).exists():
        logger.info("Resuming Stage 2 model from checkpoint: '%s'...", resume_path)
        ckpt = torch.load(resume_path, map_location=dev, weights_only=False)
        model.load_checkpoint_compatible(resume_path, map_location=dev)
        saved_epoch = ckpt.get("epoch", 0)
        start_epoch = saved_epoch + 1
        best_val_f1 = float(ckpt.get("val_primary_f1", -1.0))
        resumed = True
        resume_msg = f"[*] Successfully resumed from '{Path(resume_path).name}': continuing from Epoch {start_epoch:02d} (prior best val Primary Macro-F1 = {best_val_f1:.4f})"
        logger.info(resume_msg)
        print(resume_msg, flush=True)
    else:
        if stage1_checkpoint and not Path(stage1_checkpoint).exists():
            kaggle_input = Path("/kaggle/input")
            if kaggle_input.exists():
                candidates = list(kaggle_input.glob("**/visec_stage1_pretrained.pt")) + list(kaggle_input.glob("**/best_model_stage1_5.pt"))
                if candidates:
                    stage1_checkpoint = str(candidates[0])
                    logger.info("Auto-discovered Stage 1.5 acoustic checkpoint at '%s'.", stage1_checkpoint)
        if stage1_checkpoint and Path(stage1_checkpoint).exists():
            # Load Stage 1.5 acoustic checkpoint if starting fresh
            logger.info("Loading Stage 1.5 acoustic weights from '%s'...", stage1_checkpoint)
            ckpt = torch.load(stage1_checkpoint, map_location=dev, weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
            model_dict = model.state_dict()
            filtered_dict = {
                k: v for k, v in state_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            model.load_state_dict(filtered_dict, strict=False)
            logger.info("Stage 1.5 weights injected: successfully loaded %d compatible tensors.", len(filtered_dict))

    # Initialize semantic sub-emotion prototypes if not resuming from an existing Stage 2 checkpoint
    if not dummy and not resumed and hasattr(model, "init_semantic_sub_prototypes"):
        try:
            model.init_semantic_sub_prototypes()
        except Exception as proto_err:
            logger.warning("Could not initialize semantic sub prototypes (%s). Continuing with Xavier init.", proto_err)

    # 2. Audio freezing logic (only freezes/unfreezes LoRA adapters; foundation backbone is always frozen)
    if freeze_audio_epochs > 0 and start_epoch <= freeze_audio_epochs:
        logger.info("Freezing audio encoder LoRA adapters for Epochs %d-%d...", start_epoch, freeze_audio_epochs)
        model.set_audio_trainable(False)
    else:
        logger.info("Audio LoRA adapters active (trainable). Foundation backbone remains strictly frozen.")
        model.set_audio_trainable(True)

    train_cfg = config.get("training", {})
    loss_cfg = config.get("loss", {})

    # 3. PhoBERT freezing logic (Freeze bottom layers, adapt top layers 10-11)
    phobert_freeze_layers = int(train_cfg.get("phobert_freeze_layers", 10))
    freeze_phobert_layers(model, num_freeze_layers=phobert_freeze_layers)

    # 4. Optimizer with 3 isolated parameter groups (heads/fusion, audio LoRA, PhoBERT top layers)
    lr_heads = float(train_cfg.get("learning_rate", 1.0e-4))
    lr_audio_lora = float(train_cfg.get("lr_audio_lora", 5.0e-5))
    lr_text_encoder = float(train_cfg.get("lr_text_encoder", 1.5e-5))
    weight_decay = float(train_cfg.get("weight_decay", 1.0e-2))

    use_uncertainty = loss_cfg.get("use_uncertainty_weighting", True)
    uncertainty_loss_fn = MultiTaskUncertaintyLoss(num_tasks=2).to(dev) if use_uncertainty else None

    # Group 2: Audio LoRA parameters
    audio_lora_params = [
        p for n, p in model.audio_encoder.named_parameters()
        if "lora_" in n.lower() or getattr(model.audio_encoder, "_synthetic", False)
    ]

    # Group 3: PhoBERT top layers (trainable parameters of text_encoder)
    text_encoder_params = [
        p for p in model.text_encoder.parameters()
        if p.requires_grad
    ]

    # Group 1: Heads, fusion, pause, melody, pooling, and any other trainable parameters
    audio_param_ids = {id(p) for p in model.audio_encoder.parameters()}
    text_param_ids = {id(p) for p in model.text_encoder.parameters()}
    heads_and_fusion_params = [
        p for p in model.parameters()
        if id(p) not in audio_param_ids and id(p) not in text_param_ids and p.requires_grad
    ]
    if uncertainty_loss_fn is not None:
        heads_and_fusion_params.extend([p for p in uncertainty_loss_fn.parameters() if p.requires_grad])

    param_groups = [
        {"name": "heads_and_fusion", "params": heads_and_fusion_params, "lr": lr_heads, "weight_decay": weight_decay},
        {"name": "audio_lora", "params": audio_lora_params, "lr": lr_audio_lora, "weight_decay": weight_decay},
        {"name": "phobert_top_layers", "params": text_encoder_params, "lr": lr_text_encoder, "weight_decay": weight_decay},
    ]
    optimizer = optim.AdamW(param_groups)

    logger.info(
        "Optimizer AdamW initialized with 3 parameter groups:\n"
        "  1. heads_and_fusion: %d params, lr=%.2e, weight_decay=%.2e\n"
        "  2. audio_lora: %d params, lr=%.2e, weight_decay=%.2e\n"
        "  3. phobert_top_layers: %d params, lr=%.2e, weight_decay=%.2e",
        len(heads_and_fusion_params), lr_heads, weight_decay,
        len(audio_lora_params), lr_audio_lora, weight_decay,
        len(text_encoder_params), lr_text_encoder, weight_decay,
    )

    if resumed and resume_path:
        ckpt = torch.load(resume_path, map_location=dev, weights_only=False)
        opt_state = ckpt.get("optimizer_state_dict", None)
        if opt_state is not None:
            ckpt_param_groups = opt_state.get("param_groups", [])
            num_ckpt_groups = len(ckpt_param_groups)
            num_curr_groups = len(optimizer.param_groups)

            if reset_optimizer:
                logger.warning(
                    "[*] Flag --reset-optimizer is active. Skipping checkpoint optimizer_state_dict and retaining fresh 3-group optimizer state (LR: heads=%.2e, audio_lora=%.2e, phobert=%.2e).",
                    lr_heads, lr_audio_lora, lr_text_encoder,
                )
            elif num_ckpt_groups != num_curr_groups:
                logger.warning(
                    "[*] Optimizer group mismatch: checkpoint contains %d param groups, but current tiered optimizer requires %d groups. "
                    "Skipping legacy optimizer_state_dict to preserve tiered learning rates (LR: heads=%.2e, audio_lora=%.2e, phobert=%.2e).",
                    num_ckpt_groups, num_curr_groups, lr_heads, lr_audio_lora, lr_text_encoder,
                )
            else:
                try:
                    optimizer.load_state_dict(opt_state)
                    logger.info("Restored optimizer state successfully from checkpoint (%d param groups).", num_curr_groups)
                except Exception as opt_err:
                    logger.warning(
                        "Could not restore optimizer state (%s). Retaining fresh 3-group optimizer.", opt_err
                    )

    # Class counts for Primary & Sub LDAM losses
    if dummy:
        class_counts = [12, 12, 12, 12, 12]
        sub_counts = [5] * 11
    else:
        logger.info("Computing actual Primary and Sub-Emotion sample counts from train_loader...")
        class_counts, sub_counts = compute_dataset_class_counts(train_loader, num_primary=5, num_sub=11)
        logger.info("Dynamic sample counts: Primary=%s, Sub=%s", class_counts, sub_counts)

    loss_primary = LDAMLoss(
        class_counts=class_counts,
        max_m=float(loss_cfg.get("ldam_max_m", 0.5)),
        s=float(loss_cfg.get("ldam_s", 30.0)),
        ignore_index=-100,
    ).to(dev)

    loss_sub = LDAMLoss(
        class_counts=sub_counts,
        max_m=float(loss_cfg.get("ldam_max_m", 0.5)),
        s=float(loss_cfg.get("ldam_s", 30.0)),
        ignore_index=-100,
    ).to(dev)

    drw_start_epoch = int(loss_cfg.get("drw_start_epoch", 15))
    num_epochs = epochs_override or train_cfg.get("num_epochs", 60)
    primary_weight = float(loss_cfg.get("default_primary_weight", 0.7))
    sub_weight = float(loss_cfg.get("default_sub_weight", 0.3))

    logger.info("Starting Stage 2 SOTA training from Epoch %d to %d (grad_accum_steps=%d)...", start_epoch, num_epochs, grad_accum_steps)
    for epoch in range(start_epoch, num_epochs + 1):
        if epoch == freeze_audio_epochs + 1 and freeze_audio_epochs > 0:
            logger.info("Unfreezing audio LoRA parameters at epoch %d...", epoch)
            model.set_audio_trainable(True)

        if isinstance(loss_primary, LDAMLoss):
            loss_primary.drw_update(current_epoch=epoch, drw_start_epoch=drw_start_epoch)
        if isinstance(loss_sub, LDAMLoss):
            loss_sub.drw_update(current_epoch=epoch, drw_start_epoch=drw_start_epoch)

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
            grad_accum_steps=grad_accum_steps,
        )
        val_res = evaluate(
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

        val_f1 = val_res.get("primary_f1", 0.0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_path = ckpt_dir / f"best_model_{model_name}.pt"
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_primary_f1": val_f1,
                "val_metrics": val_res,
                "config": config,
            }, str(best_model_path))
            save_msg = f"    [*] [SAVED BEST MODEL] Val Primary Macro-F1 = {val_f1:.4f} -> {best_model_path.name}\n"
            logger.info(save_msg)
            print(save_msg, flush=True)

        # Always save latest model checkpoint for crash recovery
        latest_model_path = ckpt_dir / f"last_model_{model_name}.pt"
        torch.save({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_primary_f1": val_f1,
            "val_metrics": val_res,
            "config": config,
        }, str(latest_model_path))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("Training complete! Running final evaluation on Test Set...")
    evaluate(
        model=model,
        loader=test_loader,
        loss_primary_fn=loss_primary,
        loss_sub_fn=loss_sub,
        uncertainty_loss_fn=uncertainty_loss_fn,
        primary_weight=primary_weight,
        sub_weight=sub_weight,
        device=dev,
        phase="test",
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
        reset_optimizer=args.reset_optimizer,
        stage1_checkpoint=args.stage1_checkpoint,
        freeze_audio_epochs=args.freeze_audio_epochs,
        output_dir=args.output_dir,
        model_name=args.model_name,
        data_dir=args.data_dir,
        use_vigoemotions=args.use_vigoemotions,
        device=args.device,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        seed=args.seed,
    )
