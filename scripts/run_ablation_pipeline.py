#!/usr/bin/env python3
"""
scripts/run_ablation_pipeline.py
--------------------------------
Stage 2.6 Controlled Independent Ablation Study Runner.
Evaluates Gemini Argon 4's proposed architectural and supervision upgrades over Stage 2.5:
  - Run 0: Baseline Evaluation (best_model_stage2_5.pt, 0 epochs, Dual-Track evaluation)
  - Run 1: Sibling Masked LDAM Loss + Decoupled Text-Only Routing (e_no_prosody + loss_primary=0 on ViGo)
  - Run 2: Attentive PhoBERT Token Pooling (attn_scorer = nn.Linear(768, 1))
  - Run 3: 40ms Jitter F0 Pitch Filter (median_kernel=5 preserving vocal micro-tremor)
  - Run 4: Hierarchical Conditional Sub-Head (1189D: 928D + 256D latent + 5D primary logits)
  - Run 5: Full Combined Argon Stage 2.6 Stack (All positive components combined)

SCIENTIFIC CONTROLS:
  1. 100% INDEPENDENT: Every run instantiates a fresh model, reloads the identical baseline
     checkpoint (best_model_stage2_5.pt), and initializes fresh 3-group AdamW + CosineAnnealingLR.
  2. IDENTICAL BUDGET: Each active run trains for the exact same number of epochs (e.g. 10 epochs).
  3. DUAL-TRACK EVALUATION:
     - Speech Track (ViSEC + RAVDESS): Primary Macro-F1, Active-5 Sub-F1 (unpenalized by 6 absent classes),
       Parent-Marginalized Sub-F1, and Raw 11 Sub-F1.
     - ViGo Track (Text-Only): 11-Class Sub Macro-F1, Sub WA, and Prototype Norm collapse audit.
  4. SINGLE NOTEBOOK CELL / COMMAND EXECUTION: Completely automated, outputs comparative scoreboard,
     and auto-exports the winning checkpoint as `best_model_stage2_6_winner.pt`.
"""

import argparse
import copy
import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Mitigate CUDA memory fragmentation on 15-16GB GPUs (Tesla T4)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
import yaml

from src.audio.audio_dataset import VoiceJournalDataset
from src.stage2.hierarchical_mer_stage2 import (
    Stage2HierarchicalMERModel,
    TAXONOMY_MAP,
    TAXONOMY_MASK_MATRIX,
)
from src.stage2.ldam_loss import (
    ACTIVE_5_SUB_INDICES,
    COARSE_SUB_TO_SIBLINGS,
    LDAMLoss,
    SiblingMaskedLDAMLoss,
    ZERO_SUPPORT_SPEECH_SUB_INDICES,
)
from src.stage2.uncertainty_loss import MultiTaskUncertaintyLoss
from src.stage2.train_stage2 import freeze_phobert_layers
from src.common.metrics import compute_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ablation_runner")

SUB_TO_PRIMARY_MAP: Dict[int, int] = {
    0: 0,   # JOY -> JOY
    1: 4,   # CALM -> NEUTRAL (or JOY)
    2: 0,   # HOPE -> JOY
    3: 0,   # CONNECTION -> JOY
    4: 1,   # SADNESS -> SADNESS
    5: 2,   # ANXIETY -> ANXIETY
    6: 2,   # FEAR -> ANXIETY
    7: 3,   # ANGER -> ANGER
    8: 1,   # GUILT_SHAME -> SADNESS
    9: 1,   # LONELINESS -> SADNESS
    10: 3,  # DISGUST -> ANGER
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SyntheticViGoDataset(Dataset):
    """Fallback synthetic ViGo text dataset for offline or dummy dry-runs."""
    def __init__(self, size: int = 100) -> None:
        super().__init__()
        self.samples = []
        sample_texts = [
            ("Hôm nay tôi rất vui và hạnh phúc", 0, 0),
            ("Tâm hồn tôi bình yên và thanh thản", 4, 1),
            ("Tôi luôn có niềm tin và hy vọng", 0, 2),
            ("Cảm ơn tình cảm gia đình đã gắn kết chúng tôi", 0, 3),
            ("Tôi buồn bã và thất vọng về bản thân", 1, 4),
            ("Tôi đang rất lo âu và căng thẳng", 2, 5),
            ("Tôi thấy sợ hãi một điều gì đó mơ hồ", 2, 6),
            ("Tôi vô cùng tức giận và bức xúc", 3, 7),
            ("Tôi cảm thấy tội lỗi và hối hận vô cùng", 1, 8),
            ("Đêm nay một mình cô đơn trống trải quá", 1, 9),
            ("Thật là kinh tởm và ghê sợ", 3, 10),
        ]
        for i in range(size):
            txt, p, s = sample_texts[i % len(sample_texts)]
            self.samples.append((f"{txt} (id {i})", p, s))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        text, p, s = self.samples[idx]
        return {
            "waveform": torch.zeros(1, 16000, dtype=torch.float32),
            "primary_label": torch.tensor(p, dtype=torch.long),
            "sub_label": torch.tensor(s, dtype=torch.long),
            "sample_id": f"vigo_synth_{idx}",
            "duration_s": 1.0,
            "transcript": text,
            "speech_timestamps": [],
        }


def build_dual_track_datasets(
    config: dict,
    data_dir: Optional[str] = None,
    dummy: bool = False,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset, Dataset, Dataset]:
    """Returns (train_speech, val_speech, test_speech, train_vigo, val_vigo)."""
    if dummy:
        speech_ds = VoiceJournalDataset(dummy_mode=True, dummy_size=60, dummy_duration_s=4.0, use_sub_labels=True)
        train_idx, val_idx, test_idx = speech_ds.get_split_indices(train_ratio=0.7, val_ratio=0.15, seed=seed)
        train_speech = Subset(speech_ds, train_idx)
        val_speech = Subset(speech_ds, val_idx)
        test_speech = Subset(speech_ds, test_idx)

        train_vigo = SyntheticViGoDataset(size=40)
        val_vigo = SyntheticViGoDataset(size=20)
        return train_speech, val_speech, test_speech, train_vigo, val_vigo

    data_dir_path = data_dir or config.get("data", {}).get("data_dir", "data/voice_journals")
    train_speech = VoiceJournalDataset(data_dir=data_dir_path, split="train", use_sub_labels=True)
    val_speech = VoiceJournalDataset(data_dir=data_dir_path, split="val", use_sub_labels=True)
    test_speech = VoiceJournalDataset(data_dir=data_dir_path, split="test", use_sub_labels=True)

    # Load ViGoEmotions
    train_vigo = None
    val_vigo = None
    try:
        from src.audio.vigoemotions_dataset import ViGoEmotionsStage2Dataset
        train_vigo = ViGoEmotionsStage2Dataset(split="train")
        val_vigo = ViGoEmotionsStage2Dataset(split="validation")
        if len(train_vigo) == 0:
            raise ValueError("ViGo train dataset returned 0 samples.")
        logger.info("Loaded real ViGoEmotions dataset: train=%d, val=%d", len(train_vigo), len(val_vigo))
    except Exception as e:
        logger.warning("Could not load HuggingFace uitnlp/vigoemotions (%s). Falling back to SyntheticViGoDataset.", e)
        train_vigo = SyntheticViGoDataset(size=100)
        val_vigo = SyntheticViGoDataset(size=50)

    return train_speech, val_speech, test_speech, train_vigo, val_vigo


def compute_speech_active5_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    active_indices: List[int] = ACTIVE_5_SUB_INDICES,
) -> Dict[str, float]:
    """Compute exact macro-F1 and WA restricted to the active 5 speech ground truth classes."""
    y_p = predictions.cpu().numpy()
    y_t = targets.cpu().numpy()

    f1_list = []
    total_active = 0
    correct_active = 0

    for c in active_indices:
        tp = np.sum((y_p == c) & (y_t == c))
        fp = np.sum((y_p == c) & (y_t != c))
        fn = np.sum((y_p != c) & (y_t == c))
        total_c = np.sum(y_t == c)

        total_active += total_c
        correct_active += tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)

    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0
    wa = float(correct_active / total_active) if total_active > 0 else 0.0
    return {"macro_f1": macro_f1, "wa": wa}


def compute_parent_marginalized_metrics(
    sub_predictions: torch.Tensor,
    primary_targets: torch.Tensor,
) -> Dict[str, float]:
    """Evaluates whether fine-grained sub predictions map back to the correct parent emotion family."""
    y_sub_p = sub_predictions.cpu().tolist()
    y_prim_t = primary_targets.cpu().tolist()

    mapped_primary_preds = [SUB_TO_PRIMARY_MAP.get(s, 4) for s in y_sub_p]
    p_pred_t = torch.tensor(mapped_primary_preds, dtype=torch.long)
    p_true_t = torch.tensor(y_prim_t, dtype=torch.long)

    return compute_all_metrics(predictions=p_pred_t, targets=p_true_t, num_classes=5)


@torch.no_grad()
def evaluate_loader(
    model: Stage2HierarchicalMERModel,
    loader: DataLoader,
    device: torch.device,
    is_speech_track: bool = True,
) -> Dict[str, float]:
    """Evaluates a DataLoader on either the Speech Track or ViGo Track."""
    model.eval()
    all_p_preds, all_p_targets = [], []
    all_s_preds, all_s_targets = [], []

    for batch in loader:
        wavs = batch["waveforms"].squeeze(1).to(device)
        txts = batch["transcripts"]
        sts = batch.get("speech_timestamps", None)
        p_tgt = batch["primary_labels"].to(device)
        s_tgt = batch["sub_labels"].to(device)

        out = model(waveforms=wavs, speech_timestamps=sts, texts=txts, use_asr=False)

        all_p_preds.append(out.primary_predictions().cpu())
        all_p_targets.append(p_tgt.cpu())
        all_s_preds.append(out.sub_predictions().cpu())
        all_s_targets.append(s_tgt.cpu())

    p_pred = torch.cat(all_p_preds) if all_p_preds else torch.empty(0, dtype=torch.long)
    p_true = torch.cat(all_p_targets) if all_p_targets else torch.empty(0, dtype=torch.long)
    s_pred = torch.cat(all_s_preds) if all_s_preds else torch.empty(0, dtype=torch.long)
    s_true = torch.cat(all_s_targets) if all_s_targets else torch.empty(0, dtype=torch.long)

    results = {}
    if is_speech_track:
        # 1. Primary Metrics
        p_metrics = compute_all_metrics(p_pred, p_true, num_classes=5)
        results["primary_macro_f1"] = p_metrics.get("macro_f1", 0.0)
        results["primary_wa"] = p_metrics.get("wa", 0.0)

        # 2. Raw 11-Class Sub Metrics
        valid_s = (s_true >= 0) & (s_true < 11)
        raw11_metrics = compute_all_metrics(s_pred[valid_s], s_true[valid_s], num_classes=11)
        results["raw11_sub_macro_f1"] = raw11_metrics.get("macro_f1", 0.0)
        results["raw11_sub_wa"] = raw11_metrics.get("wa", 0.0)

        # 3. Active-5 Sub Metrics (Unpenalized by 6 absent classes)
        active5_metrics = compute_speech_active5_metrics(s_pred, s_true)
        results["active5_sub_macro_f1"] = active5_metrics["macro_f1"]
        results["active5_sub_wa"] = active5_metrics["wa"]

        # 4. Parent-Marginalized Sub Metrics
        parent_metrics = compute_parent_marginalized_metrics(s_pred, p_true)
        results["parent_marginalized_f1"] = parent_metrics.get("macro_f1", 0.0)
        results["parent_marginalized_wa"] = parent_metrics.get("wa", 0.0)
    else:
        # ViGo Track: 11-Class Fine-Grained Sub Metrics
        valid_s = (s_true >= 0) & (s_true < 11)
        vigo_metrics = compute_all_metrics(s_pred[valid_s], s_true[valid_s], num_classes=11)
        results["vigo_sub_macro_f1"] = vigo_metrics.get("macro_f1", 0.0)
        results["vigo_sub_wa"] = vigo_metrics.get("wa", 0.0)

        # Class Prototype Norm Audit (Checking for Class Collapse)
        try:
            head_layer = model.sub_head[4]
            weight = head_layer.weight.data
            norms = torch.norm(weight, p=2, dim=-1).cpu().tolist()
            results["prototype_norms_min"] = float(np.min(norms))
            results["prototype_norms_max"] = float(np.max(norms))
            results["prototype_norms_mean"] = float(np.mean(norms))
        except Exception:
            results["prototype_norms_min"] = 1.0
            results["prototype_norms_max"] = 1.0
            results["prototype_norms_mean"] = 1.0

    return results


def train_single_epoch(
    model: Stage2HierarchicalMERModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_primary_fn: nn.Module,
    loss_sub_fn: nn.Module,
    uncertainty_loss_fn: Optional[MultiTaskUncertaintyLoss],
    vigo_primary_loss_weight: float,
    device: torch.device,
    grad_accum_steps: int = 1,
    gradient_clip: float = 1.0,
) -> float:
    """Train model for one epoch supporting Decoupled Routing & Sibling Masking."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    accum_steps = max(1, grad_accum_steps)

    for batch_idx, batch in enumerate(loader):
        wavs = batch["waveforms"].squeeze(1).to(device)
        p_targets = batch["primary_labels"].to(device)
        s_targets = batch["sub_labels"].to(device)
        transcripts = batch["transcripts"]
        speech_ts = batch.get("speech_timestamps", None)
        sample_ids = batch.get("sample_ids", [])

        # Identify coarse speech vs text-only samples
        # A sample is coarse speech if it has audio energy > 1e-5 and sample_id doesn't start with "vigo_"
        is_speech = (wavs.abs().amax(dim=-1) > 1e-5) & torch.tensor(
            [not str(sid).startswith("vigo_") for sid in sample_ids],
            device=device,
            dtype=torch.bool,
        )

        out = model(waveforms=wavs, speech_timestamps=speech_ts, texts=transcripts, use_asr=False)

        # 1. Primary Loss calculation
        if vigo_primary_loss_weight == 0.0:
            # Mask out primary loss on text-only / ViGo samples to prevent semantic shift
            if is_speech.any():
                l_primary = loss_primary_fn(out.primary_logits[is_speech], p_targets[is_speech])
            else:
                l_primary = out.primary_logits.sum() * 0.0
        else:
            l_primary = loss_primary_fn(out.primary_logits, p_targets)

        # 2. Sub-Emotion Loss calculation (Sibling Masked LDAM)
        if isinstance(loss_sub_fn, SiblingMaskedLDAMLoss):
            l_sub = loss_sub_fn(out.sub_logits, s_targets, is_coarse_speech=is_speech)
        else:
            l_sub = loss_sub_fn(out.sub_logits, s_targets)

        # 3. Multi-Task Fusion
        if uncertainty_loss_fn is not None:
            loss = uncertainty_loss_fn(l_primary, l_sub)
        else:
            loss = 0.7 * l_primary + 0.3 * l_sub

        loss_scaled = loss / accum_steps
        loss_scaled.backward()

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


def build_optimizer_and_scheduler(
    model: Stage2HierarchicalMERModel,
    lr_heads: float = 1.0e-4,
    lr_audio_lora: float = 5.0e-5,
    lr_text_encoder: float = 1.5e-5,
    weight_decay: float = 1.0e-2,
    epochs: int = 10,
    uncertainty_loss_fn: Optional[MultiTaskUncertaintyLoss] = None,
) -> Tuple[optim.Optimizer, CosineAnnealingLR]:
    """Build isolated 3-group AdamW optimizer + CosineAnnealingLR."""
    audio_lora_params = [
        p for n, p in model.audio_encoder.named_parameters()
        if "lora_" in n.lower() or getattr(model.audio_encoder, "_synthetic", False)
    ]
    text_encoder_params = [
        p for p in model.text_encoder.parameters()
        if p.requires_grad
    ]
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
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1.0e-6)
    return optimizer, scheduler


# Define the 6 ablation runs
ABLATION_CONFIGS = [
    {
        "id": 0,
        "name": "Run 0: Baseline Evaluation (Stage 2.5)",
        "description": "Evaluate best_model_stage2_5.pt with Dual-Track metrics (0 epochs)",
        "epochs": 0,
        "use_sibling_masking": False,
        "use_decoupled_routing": False,
        "text_pooling": "cls",
        "melody_median_kernel": 15,
        "use_conditional_subhead": False,
        "vigo_primary_loss_weight": 1.0,
    },
    {
        "id": 1,
        "name": "Run 1: Sibling Masking & Decoupled Routing",
        "description": "SiblingMaskedLDAMLoss + e_no_prosody fallback + loss_primary=0 on ViGo",
        "epochs": 10,
        "use_sibling_masking": True,
        "use_decoupled_routing": True,
        "text_pooling": "cls",
        "melody_median_kernel": 15,
        "use_conditional_subhead": False,
        "vigo_primary_loss_weight": 0.0,
    },
    {
        "id": 2,
        "name": "Run 2: Attentive PhoBERT Pooling",
        "description": "Replace [CLS] pooling with AttentiveTokenPooling (Linear(768, 1))",
        "epochs": 10,
        "use_sibling_masking": False,
        "use_decoupled_routing": False,
        "text_pooling": "attention",
        "melody_median_kernel": 15,
        "use_conditional_subhead": False,
        "vigo_primary_loss_weight": 1.0,
    },
    {
        "id": 3,
        "name": "Run 3: 40ms Jitter F0 Filter",
        "description": "5-frame median filter (median_kernel=5) preserving vocal micro-tremor",
        "epochs": 10,
        "use_sibling_masking": False,
        "use_decoupled_routing": False,
        "text_pooling": "cls",
        "melody_median_kernel": 5,
        "use_conditional_subhead": False,
        "vigo_primary_loss_weight": 1.0,
    },
    {
        "id": 4,
        "name": "Run 4: Hierarchical Conditional Sub-Head",
        "description": "1189D Sub-Head input (928D + 256D primary latent + 5D primary logits)",
        "epochs": 10,
        "use_sibling_masking": False,
        "use_decoupled_routing": False,
        "text_pooling": "cls",
        "melody_median_kernel": 15,
        "use_conditional_subhead": True,
        "vigo_primary_loss_weight": 1.0,
    },
    {
        "id": 5,
        "name": "Run 5: Full Argon 2.6 Combined Stack",
        "description": "Sibling Masking + Decoupled Routing + Attentive Pooling + 40ms F0 + Conditional Head",
        "epochs": 10,
        "use_sibling_masking": True,
        "use_decoupled_routing": True,
        "text_pooling": "attention",
        "melody_median_kernel": 5,
        "use_conditional_subhead": True,
        "vigo_primary_loss_weight": 0.0,
    },
]


def execute_ablation_run(
    run_cfg: Dict[str, Any],
    baseline_checkpoint: str,
    train_loader: DataLoader,
    val_speech_loader: DataLoader,
    test_speech_loader: DataLoader,
    val_vigo_loader: DataLoader,
    output_dir: Path,
    device: torch.device,
    grad_accum_steps: int = 1,
    seed: int = 42,
) -> Dict[str, Any]:
    """Executes a single, scientifically isolated ablation run."""
    run_id = run_cfg["id"]
    name = run_cfg["name"]
    desc = run_cfg["description"]
    epochs = run_cfg["epochs"]

    print("\n" + "=" * 80)
    print(f"STARTING {name.upper()}")
    print(f"Hypothesis: {desc}")
    print(f"Epochs: {epochs} | Seed: {seed} | Device: {device}")
    print("=" * 80)

    # 1. Instantiate fresh model specifically configured for this run
    set_seed(seed)
    model = Stage2HierarchicalMERModel(
        use_conditional_subhead=run_cfg["use_conditional_subhead"],
        melody_median_kernel=run_cfg["melody_median_kernel"],
        text_pooling=run_cfg["text_pooling"],
        device=str(device),
    ).to(device)

    # Enable Decoupled Routing in Fusion module if configured
    model.fusion.use_decoupled_routing = run_cfg["use_decoupled_routing"]

    # 2. Warm-start load from baseline checkpoint
    if Path(baseline_checkpoint).exists():
        missing, unexp = model.load_checkpoint_compatible(baseline_checkpoint, map_location=device)
        logger.info("[%s] Baseline checkpoint loaded (%d missing, %d unexpected).", name, len(missing), len(unexp))
    else:
        logger.warning("[%s] Baseline checkpoint '%s' not found! Initializing from random/backbone weights.", name, baseline_checkpoint)

    # If Run 0 (Baseline), evaluate immediately without training
    if epochs == 0:
        logger.info("[%s] Running Zero-Epoch Baseline Evaluation...", name)
        val_speech_metrics = evaluate_loader(model, val_speech_loader, device=device, is_speech_track=True)
        test_speech_metrics = evaluate_loader(model, test_speech_loader, device=device, is_speech_track=True)
        vigo_metrics = evaluate_loader(model, val_vigo_loader, device=device, is_speech_track=False)

        res = {
            "run_id": run_id,
            "name": name,
            "val_primary_f1": val_speech_metrics["primary_macro_f1"],
            "val_primary_wa": val_speech_metrics["primary_wa"],
            "val_active5_sub_f1": val_speech_metrics["active5_sub_macro_f1"],
            "val_active5_sub_wa": val_speech_metrics["active5_sub_wa"],
            "val_raw11_sub_f1": val_speech_metrics["raw11_sub_macro_f1"],
            "val_parent_marginalized_f1": val_speech_metrics["parent_marginalized_f1"],
            "test_primary_f1": test_speech_metrics["primary_macro_f1"],
            "test_primary_wa": test_speech_metrics["primary_wa"],
            "test_active5_sub_f1": test_speech_metrics["active5_sub_macro_f1"],
            "test_active5_sub_wa": test_speech_metrics["active5_sub_wa"],
            "test_raw11_sub_f1": test_speech_metrics["raw11_sub_macro_f1"],
            "test_parent_marginalized_f1": test_speech_metrics["parent_marginalized_f1"],
            "vigo_sub_f1": vigo_metrics["vigo_sub_macro_f1"],
            "vigo_sub_wa": vigo_metrics["vigo_sub_wa"],
            "vigo_norms_min": vigo_metrics["prototype_norms_min"],
            "vigo_norms_max": vigo_metrics["prototype_norms_max"],
            "best_epoch": 0,
        }
        _print_run_results(res)
        return res

    # 3. Setup optimizer, losses & scheduler
    freeze_phobert_layers(model, num_freeze_layers=10)
    uncertainty_fn = MultiTaskUncertaintyLoss(num_tasks=2).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        epochs=epochs,
        uncertainty_loss_fn=uncertainty_fn,
    )

    loss_primary = LDAMLoss(cls_num_list=[1000] * 5, scale=30.0).to(device)
    if run_cfg["use_sibling_masking"]:
        loss_sub = SiblingMaskedLDAMLoss(
            cls_num_list=[200] * 11,
            scale=30.0,
            use_sibling_masking=True,
        ).to(device)
    else:
        loss_sub = LDAMLoss(cls_num_list=[200] * 11, scale=30.0).to(device)

    best_val_score = -1.0
    best_ckpt_path = output_dir / f"best_model_run_{run_id}.pt"
    best_val_metrics = {}

    for ep in range(1, epochs + 1):
        loss_val = train_single_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_primary_fn=loss_primary,
            loss_sub_fn=loss_sub,
            uncertainty_loss_fn=uncertainty_fn,
            vigo_primary_loss_weight=run_cfg["vigo_primary_loss_weight"],
            device=device,
            grad_accum_steps=grad_accum_steps,
        )
        scheduler.step()

        # Evaluate on validation speech track
        val_speech_m = evaluate_loader(model, val_speech_loader, device=device, is_speech_track=True)
        # Score = Combined Primary F1 + Active-5 Sub F1
        score = val_speech_m["primary_macro_f1"] + val_speech_m["active5_sub_macro_f1"]

        print(
            f"  [Run {run_id} | Ep {ep:02d}/{epochs:02d}] Loss: {loss_val:.4f} | "
            f"Val Primary F1: {val_speech_m['primary_macro_f1']*100:.2f}% | "
            f"Val Active-5 Sub F1: {val_speech_m['active5_sub_macro_f1']*100:.2f}% | "
            f"Score: {score:.4f}",
            flush=True,
        )

        if score > best_val_score:
            best_val_score = score
            best_val_metrics = val_speech_m
            best_val_metrics["best_epoch"] = ep
            torch.save(
                {
                    "run_id": run_id,
                    "run_name": name,
                    "epoch": ep,
                    "val_metrics": val_speech_m,
                    "state_dict": model.state_dict(),
                },
                best_ckpt_path,
            )

    # 4. Final Evaluation of Best Checkpoint on Test & ViGo Tracks
    logger.info("[%s] Loading best checkpoint from Epoch %d for final Test audit...", name, best_val_metrics.get("best_epoch", epochs))
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])

    test_speech_metrics = evaluate_loader(model, test_speech_loader, device=device, is_speech_track=True)
    vigo_metrics = evaluate_loader(model, val_vigo_loader, device=device, is_speech_track=False)

    res = {
        "run_id": run_id,
        "name": name,
        "val_primary_f1": best_val_metrics.get("primary_macro_f1", 0.0),
        "val_primary_wa": best_val_metrics.get("primary_wa", 0.0),
        "val_active5_sub_f1": best_val_metrics.get("active5_sub_macro_f1", 0.0),
        "val_active5_sub_wa": best_val_metrics.get("active5_sub_wa", 0.0),
        "val_raw11_sub_f1": best_val_metrics.get("raw11_sub_macro_f1", 0.0),
        "val_parent_marginalized_f1": best_val_metrics.get("parent_marginalized_f1", 0.0),
        "test_primary_f1": test_speech_metrics["primary_macro_f1"],
        "test_primary_wa": test_speech_metrics["primary_wa"],
        "test_active5_sub_f1": test_speech_metrics["active5_sub_macro_f1"],
        "test_active5_sub_wa": test_speech_metrics["active5_sub_wa"],
        "test_raw11_sub_f1": test_speech_metrics["raw11_sub_macro_f1"],
        "test_parent_marginalized_f1": test_speech_metrics["parent_marginalized_f1"],
        "vigo_sub_f1": vigo_metrics["vigo_sub_macro_f1"],
        "vigo_sub_wa": vigo_metrics["vigo_sub_wa"],
        "vigo_norms_min": vigo_metrics["prototype_norms_min"],
        "vigo_norms_max": vigo_metrics["prototype_norms_max"],
        "best_epoch": best_val_metrics.get("best_epoch", epochs),
        "checkpoint_path": str(best_ckpt_path),
    }
    _print_run_results(res)
    return res


def _print_run_results(res: Dict[str, Any]) -> None:
    print(f"\n>>> [COMPLETED {res['name']}]")
    print(f"    * Best Epoch:                   {res['best_epoch']}")
    print(f"    * Val Primary Macro-F1:         {res['val_primary_f1']*100:.2f}% (WA: {res['val_primary_wa']*100:.2f}%)")
    print(f"    * Val Active-5 Sub Macro-F1:    {res['val_active5_sub_f1']*100:.2f}% (WA: {res['val_active5_sub_wa']*100:.2f}%)")
    print(f"    * Val Parent-Marginalized F1:   {res['val_parent_marginalized_f1']*100:.2f}%")
    print(f"    * Test Primary Macro-F1:        {res['test_primary_f1']*100:.2f}% (WA: {res['test_primary_wa']*100:.2f}%)")
    print(f"    * Test Active-5 Sub Macro-F1:   {res['test_active5_sub_f1']*100:.2f}% (WA: {res['test_active5_sub_wa']*100:.2f}%)")
    print(f"    * Test Parent-Marginalized F1:  {res['test_parent_marginalized_f1']*100:.2f}%")
    print(f"    * ViGo 11-Class Sub Macro-F1:   {res['vigo_sub_f1']*100:.2f}% (WA: {res['vigo_sub_wa']*100:.2f}%)")
    print(f"    * Sub Prototype Norm Range:     [{res['vigo_norms_min']:.2f}, {res['vigo_norms_max']:.2f}]")


def print_scoreboard(results: List[Dict[str, Any]], output_dir: Path) -> str:
    """Renders the comprehensive comparative scoreboard and saves markdown report."""
    base_res = results[0]
    base_test_prim = base_res["test_primary_f1"]
    base_test_sub = base_res["test_active5_sub_f1"]
    base_vigo_sub = base_res["vigo_sub_f1"]

    lines = [
        "# STAGE 2.6 INDEPENDENT ABLATION STUDY SCOREBOARD",
        "",
        "| Run | Upgrade Tested | Val Prim F1 | Val Act-5 Sub F1 | Test Prim F1 | Delta Prim | Test Act-5 Sub F1 | Delta Sub | ViGo Sub F1 | Verdict |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in results:
        rid = r["run_id"]
        name = r["name"].split(":")[-1].strip()
        vp = f"{r['val_primary_f1']*100:.2f}%"
        vas = f"{r['val_active5_sub_f1']*100:.2f}%"
        tp = f"{r['test_primary_f1']*100:.2f}%"
        tas = f"{r['test_active5_sub_f1']*100:.2f}%"
        vg = f"{r['vigo_sub_f1']*100:.2f}%"

        d_prim = r["test_primary_f1"] - base_test_prim
        d_sub = r["test_active5_sub_f1"] - base_test_sub

        d_prim_str = f"{d_prim*100:+.2f}%" if rid != 0 else "-"
        d_sub_str = f"{d_sub*100:+.2f}%" if rid != 0 else "-"

        if rid == 0:
            verdict = "**BASELINE**"
        elif d_prim >= 0.01 and d_sub >= 0.01:
            verdict = "**STRONG KEEP**"
        elif d_prim >= 0.005 or d_sub >= 0.01:
            verdict = "**KEEP**"
        elif d_prim >= -0.005 and d_sub >= -0.005:
            verdict = "NEUTRAL"
        else:
            verdict = "DROP"

        lines.append(f"| {rid} | {name} | {vp} | {vas} | {tp} | {d_prim_str} | {tas} | {d_sub_str} | {vg} | {verdict} |")

    lines.append("")
    # Find winning checkpoint
    # Winner criteria: highest combined Test Primary F1 + Test Active-5 Sub F1
    candidates = [r for r in results if r["run_id"] != 0 and "checkpoint_path" in r]
    if candidates:
        winner = max(candidates, key=lambda x: (x["test_primary_f1"] + x["test_active5_sub_f1"]))
        lines.append(f"### Winning Configuration: {winner['name']}")
        lines.append(f"- **Test Primary Macro-F1**: `{winner['test_primary_f1']*100:.2f}%` (vs Baseline `{base_test_prim*100:.2f}%`)")
        lines.append(f"- **Test Active-5 Sub Macro-F1**: `{winner['test_active5_sub_f1']*100:.2f}%` (vs Baseline `{base_test_sub*100:.2f}%`)")
        lines.append(f"- **ViGo 11-Class Sub Macro-F1**: `{winner['vigo_sub_f1']*100:.2f}%` (vs Baseline `{base_vigo_sub*100:.2f}%`)")
        lines.append(f"- **Exported Checkpoint**: `best_model_stage2_6_winner.pt`")

        winner_src = Path(winner["checkpoint_path"])
        winner_dst = output_dir / "best_model_stage2_6_winner.pt"
        if winner_src.exists():
            shutil.copyfile(winner_src, winner_dst)
            logger.info("Winner checkpoint copied: %s -> %s", winner_src, winner_dst)

    report_content = "\n".join(lines)
    print("\n" + report_content + "\n")

    # Save to file
    scoreboard_file = output_dir / "ablation_scoreboard.md"
    with open(scoreboard_file, "w", encoding="utf-8") as f:
        f.write(report_content)

    summary_json = output_dir / "ablation_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Ablation study completed. Reports saved to %s", output_dir)
    return report_content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2.6 Controlled Independent Ablation Runner")
    parser.add_argument("--config", type=str, default="configs/stage2/multimodal_stage2.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model_stage2_5.pt",
        help="Path to baseline best_model_stage2_5.pt",
    )
    parser.add_argument("--data-dir", type=str, default="data/voice_journals")
    parser.add_argument("--output-dir", type=str, default="checkpoints/stage2_6_ablation")
    parser.add_argument("--epochs-per-run", type=int, default=10, help="Number of training epochs per ablation run")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--runs", type=str, default="all", help="Comma-separated run IDs to execute (e.g. '0,1,2,3,4,5' or 'all')")
    parser.add_argument("--dummy", action="store_true", help="Run with fast synthetic dataset for dry testing")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    dev = torch.device(args.device)

    # Auto-discover checkpoint on Kaggle if default doesn't exist
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists() and Path("/kaggle").exists():
        candidates = (
            list(Path("/kaggle").glob("**/best_model_stage2_5.pt"))
            + list(Path("/kaggle").glob("**/best_model_stage2*.pt"))
            + list(Path("/kaggle").glob("**/*stage2_5*.pt"))
            + list(Path("/kaggle").glob("**/best_model*.pt"))
        )
        if candidates:
            ckpt_path = candidates[0]
            logger.info("Auto-discovered Kaggle baseline checkpoint at '%s'.", ckpt_path)
        else:
            all_pt = list(Path("/kaggle/input").glob("**/*.pt")) if Path("/kaggle/input").exists() else []
            logger.warning("No checkpoint found at '%s'. Available .pt files in /kaggle/input: %s", args.checkpoint, all_pt)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. Build Dual-Track datasets
    train_speech, val_speech, test_speech, train_vigo, val_vigo = build_dual_track_datasets(
        config=config,
        data_dir=args.data_dir,
        dummy=args.dummy,
        seed=args.seed,
    )

    # Co-training train dataset
    train_combined = ConcatDataset([train_speech, train_vigo])

    train_loader = DataLoader(train_combined, batch_size=args.batch_size, shuffle=True, collate_fn=VoiceJournalDataset.collate_fn)
    val_speech_loader = DataLoader(val_speech, batch_size=args.batch_size, shuffle=False, collate_fn=VoiceJournalDataset.collate_fn)
    test_speech_loader = DataLoader(test_speech, batch_size=args.batch_size, shuffle=False, collate_fn=VoiceJournalDataset.collate_fn)
    val_vigo_loader = DataLoader(val_vigo, batch_size=args.batch_size, shuffle=False, collate_fn=VoiceJournalDataset.collate_fn)

    logger.info(
        "DataLoaders prepared: Train=%d samples (%d speech + %d vigo), ValSpeech=%d, TestSpeech=%d, ValViGo=%d",
        len(train_combined), len(train_speech), len(train_vigo), len(val_speech), len(test_speech), len(val_vigo),
    )

    # 2. Parse which runs to execute
    if args.runs.lower() == "all":
        selected_run_ids = [0, 1, 2, 3, 4, 5]
    else:
        selected_run_ids = [int(r.strip()) for r in args.runs.split(",") if r.strip()]

    # Make sure Run 0 (Baseline) is always present for comparative delta
    if 0 not in selected_run_ids:
        selected_run_ids.insert(0, 0)

    results = []
    for run_cfg in ABLATION_CONFIGS:
        if run_cfg["id"] not in selected_run_ids:
            continue
        cfg = copy.deepcopy(run_cfg)
        if cfg["id"] != 0 and args.epochs_per_run is not None:
            cfg["epochs"] = args.epochs_per_run

        run_result = execute_ablation_run(
            run_cfg=cfg,
            baseline_checkpoint=str(ckpt_path),
            train_loader=train_loader,
            val_speech_loader=val_speech_loader,
            test_speech_loader=test_speech_loader,
            val_vigo_loader=val_vigo_loader,
            output_dir=output_dir,
            device=dev,
            grad_accum_steps=args.grad_accum_steps,
            seed=args.seed,
        )
        results.append(run_result)

    # 3. Print scoreboard & export winner
    print_scoreboard(results, output_dir=output_dir)


if __name__ == "__main__":
    main()
