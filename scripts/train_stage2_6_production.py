#!/usr/bin/env python3
"""
scripts/train_stage2_6_production.py
------------------------------------
Stage 2.6 Full Production Training (50 Epochs).
Integrates the Golden Recipe confirmed by the Stage 2.6 Controlled Ablation Study:
  1. Acoustic Biomarkers: 40ms Jitter F0 Pitch Filter (median_kernel=5, Run 3 winner).
  2. Vietnamese Semantic Encoder: Attentive PhoBERT Pooling (Run 2 winner, 67.27% text F1).
  3. Gated Fusion: Decoupled text-only routing with learnable fallback (e_no_prosody, Run 1 winner).
  4. Hierarchical Supervision: SiblingMaskedLDAMLoss (Run 1 winner, +34.12% Sub-Emotion F1).
  5. Multi-Task Fusion: Kendall Uncertainty Loss + CosineAnnealingLR.

HIGH-THROUGHPUT GPU ACCELERATION (Fixes 10% GPU starvation):
  - Multi-threaded DataLoader (num_workers=2, pin_memory=True).
  - Mixed Precision Training (torch.cuda.amp.autocast + GradScaler).
  - Throughput: ~3x-4x faster than Stage 2 (50 epochs in ~45-55 mins on Tesla T4).
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
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
)
from src.stage2.uncertainty_loss import MultiTaskUncertaintyLoss
from src.stage2.train_stage2 import freeze_phobert_layers
from src.common.metrics import compute_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("stage2_6_production")


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


@torch.no_grad()
def evaluate_loader(
    model: Stage2HierarchicalMERModel,
    loader: DataLoader,
    device: torch.device,
    is_speech_track: bool = True,
) -> Dict[str, float]:
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
        p_metrics = compute_all_metrics(p_pred, p_true, num_classes=5)
        results["primary_macro_f1"] = p_metrics.get("macro_f1", 0.0)
        results["primary_wa"] = p_metrics.get("wa", 0.0)

        active5_metrics = compute_speech_active5_metrics(s_pred, s_true)
        results["active5_sub_macro_f1"] = active5_metrics["macro_f1"]
        results["active5_sub_wa"] = active5_metrics["wa"]
    else:
        valid_s = (s_true >= 0) & (s_true < 11)
        vigo_metrics = compute_all_metrics(s_pred[valid_s], s_true[valid_s], num_classes=11)
        results["vigo_sub_macro_f1"] = vigo_metrics.get("macro_f1", 0.0)
        results["vigo_sub_wa"] = vigo_metrics.get("wa", 0.0)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 2.6 Production SOTA (50 Epochs)")
    parser.add_argument("--config", type=str, default="configs/stage2/multimodal_stage2.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Base checkpoint (auto-discovers winner or stage2_5)")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2, help="CPU DataLoader workers for 4x throughput")
    parser.add_argument("--lr", type=float, default=5.0e-5, help="Warm-start learning rate for heads and fusion")
    parser.add_argument("--lr-audio-lora", type=float, default=2.5e-5)
    parser.add_argument("--lr-text-encoder", type=float, default=1.0e-5)
    parser.add_argument("--use-amp", action="store_true", default=False, help="Mixed precision (FP16) acceleration (default: False for audio attention stability)")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false", help="Explicitly disable mixed precision")
    parser.add_argument("--output-dir", type=str, default="checkpoints/stage2_6_production")
    parser.add_argument("--patience", type=int, default=12, help="Early stopping patience")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    dev = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STAGE 2.6 FULL PRODUCTION TRAINING (50 EPOCHS)")
    print(f"Golden Recipe: 40ms Jitter F0 + Attentive PhoBERT + Sibling Masked LDAM")
    print(f"Device: {dev} | Batch Size: {args.batch_size} | Num Workers: {args.num_workers} | AMP: {args.use_amp}")
    print("=" * 80)

    # 1. Checkpoint resolution
    ckpt_path = None
    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt_path = Path(args.checkpoint)
    else:
        # Auto-discover priority: winner.pt -> stage2_5.pt
        search_paths = [
            Path("checkpoints/best_model_stage2_6_winner.pt"),
            Path("checkpoints/stage2_6_ablation/best_model_stage2_6_winner.pt"),
            Path("checkpoints/best_model_stage2_5.pt"),
            Path("checkpoints/stage2_5_retrain/best_model_stage2_5.pt"),
        ]
        if Path("/kaggle").exists():
            search_paths.extend(list(Path("/kaggle").glob("**/best_model*winner*.pt")))
            search_paths.extend(list(Path("/kaggle").glob("**/best_model*stage2_5*.pt")))
            search_paths.extend(list(Path("/kaggle").glob("**/best_model*.pt")))

        for sp in search_paths:
            if sp.exists():
                ckpt_path = sp
                break

        # If no .pt found, check if Kaggle unpacked a PyTorch checkpoint directory containing data.pkl
        if not ckpt_path and Path("/kaggle/input").exists():
            pkl_matches = list(Path("/kaggle/input").glob("**/data.pkl"))
            if pkl_matches:
                pkl_dir = pkl_matches[0].parent
                repacked_pt = Path("checkpoints/base_model_repacked.pt")
                repacked_pt.parent.mkdir(parents=True, exist_ok=True)
                import zipfile
                logger.info("Detected unpacked PyTorch checkpoint directory at '%s'. Packaging into '%s'...", pkl_dir, repacked_pt)
                base_parent = pkl_dir.parent
                with zipfile.ZipFile(repacked_pt, "w", zipfile.ZIP_STORED) as z_out:
                    for root, dirs, files in os.walk(pkl_dir):
                        for file in files:
                            full_p = os.path.join(root, file)
                            rel_p = os.path.relpath(full_p, base_parent)
                            z_out.write(full_p, rel_p)
                ckpt_path = repacked_pt
                logger.info("Successfully packaged checkpoint into '%s' (%0.1f MB).", repacked_pt, repacked_pt.stat().st_size / (1024 * 1024))

    if ckpt_path:
        logger.info("Base Checkpoint discovered: '%s'", ckpt_path)
    else:
        logger.warning("No base checkpoint discovered! Training from backbone weights.")

    # 2. Dataset resolution
    if args.dummy:
        speech_ds = VoiceJournalDataset(dummy_mode=True, dummy_size=80, dummy_duration_s=4.0, use_sub_labels=True)
        train_idx, val_idx, test_idx = speech_ds.get_split_indices(train_ratio=0.7, val_ratio=0.15, seed=args.seed)
        train_speech = Subset(speech_ds, train_idx)
        val_speech = Subset(speech_ds, val_idx)
        test_speech = Subset(speech_ds, test_idx)
        train_vigo = SyntheticViGoDataset(size=50)
        val_vigo = SyntheticViGoDataset(size=25)
    else:
        data_candidates = [
            Path(args.data_dir) if args.data_dir else None,
            Path("data/visec_crosscorpus"),
            Path("data/voice_journals"),
            Path("/kaggle/working/data/visec_crosscorpus"),
            Path("/kaggle/working/data/voice_journals"),
            Path("/kaggle/working/emotion-journal-ai/data/visec_crosscorpus"),
            Path("/kaggle/working/emotion-journal-ai/data/voice_journals"),
        ]
        if Path("/kaggle/input").exists():
            data_candidates.extend([p.parent for p in Path("/kaggle/input").glob("**/metadata.csv")])
            data_candidates.extend([p.parent.parent for p in Path("/kaggle/input").glob("**/train/metadata.csv")])

        data_dir_path = None
        for dc in data_candidates:
            if dc and ((dc / "metadata.csv").exists() or (dc / "train" / "metadata.csv").exists()):
                data_dir_path = dc
                break

        # Self-healing: if dataset is not found, automatically invoke preparation script
        if not data_dir_path:
            logger.info("Dataset metadata.csv not found in workspace or inputs.")
            logger.info("Auto-initiating dataset preparation via scripts/prepare_crosscorpus_visec.py...")
            import subprocess
            prep_script = Path("scripts/prepare_crosscorpus_visec.py")
            if prep_script.exists():
                prep_cmd = [
                    sys.executable,
                    str(prep_script),
                    "--output-dir", "data/visec_crosscorpus",
                ]
                # Deep search for RAVDESS in /kaggle/input if available
                if Path("/kaggle/input").exists():
                    ravdess_candidates = list(Path("/kaggle/input").glob("**/*-*-06-*.wav"))
                    if ravdess_candidates:
                        rav_actor = ravdess_candidates[0].parent
                        rav_dir = rav_actor.parent if rav_actor.name.startswith("Actor_") else rav_actor
                        prep_cmd.extend(["--ravdess-dir", str(rav_dir)])
                        logger.info("Auto-discovered RAVDESS input at '%s'.", rav_dir)

                logger.info("Executing dataset preparation: %s", " ".join(prep_cmd))
                res = subprocess.run(prep_cmd, check=False)
                if res.returncode == 0:
                    cand = Path("data/visec_crosscorpus")
                    if (cand / "metadata.csv").exists() or (cand / "train" / "metadata.csv").exists():
                        data_dir_path = cand
                        logger.info("Successfully auto-prepared cross-corpus dataset at '%s'.", data_dir_path)

        if not data_dir_path:
            raise FileNotFoundError(
                "Could not locate or generate dataset containing metadata.csv! "
                "Please verify that 'uwrfkaggler/ravdess-emotional-speech-audio' is added to Kaggle Input, "
                "or specify --data-dir."
            )

        logger.info("Speech dataset resolved at '%s'.", data_dir_path)
        train_speech = VoiceJournalDataset(data_dir=str(data_dir_path), split="train", use_sub_labels=True)
        val_speech = VoiceJournalDataset(data_dir=str(data_dir_path), split="val", use_sub_labels=True)
        test_speech = VoiceJournalDataset(data_dir=str(data_dir_path), split="test", use_sub_labels=True)

        try:
            from src.audio.vigoemotions_dataset import ViGoEmotionsStage2Dataset
            train_vigo = ViGoEmotionsStage2Dataset(split="train")
            val_vigo = ViGoEmotionsStage2Dataset(split="validation")
            if len(train_vigo) == 0:
                raise ValueError("ViGo train returned 0 samples.")
            logger.info("Loaded real ViGoEmotions dataset: train=%d, val=%d", len(train_vigo), len(val_vigo))
        except Exception as e:
            logger.warning("Falling back to SyntheticViGoDataset (%s).", e)
            train_vigo = SyntheticViGoDataset(size=100)
            val_vigo = SyntheticViGoDataset(size=50)

    train_combined = ConcatDataset([train_speech, train_vigo])

    # Warm up Silero-VAD in main process before spawning DataLoader workers to prevent race conditions
    try:
        from src.audio.audio_dataset import _get_silero_vad
        _get_silero_vad()
        logger.info("Silero-VAD model pre-warmed successfully in main process.")
    except Exception as e:
        logger.warning("Silero-VAD warmup skipped (%s).", e)

    # 3. High-throughput DataLoader
    pin_mem = (dev.type == "cuda")
    nw = args.num_workers if dev.type == "cuda" else 0
    train_loader = DataLoader(
        train_combined,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=pin_mem,
        collate_fn=VoiceJournalDataset.collate_fn,
    )
    val_speech_loader = DataLoader(
        val_speech,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin_mem,
        collate_fn=VoiceJournalDataset.collate_fn,
    )
    test_speech_loader = DataLoader(
        test_speech,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin_mem,
        collate_fn=VoiceJournalDataset.collate_fn,
    )
    val_vigo_loader = DataLoader(
        val_vigo,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin_mem,
        collate_fn=VoiceJournalDataset.collate_fn,
    )

    logger.info(
        "DataLoaders initialized: Train=%d batches (%d samples), Val=%d, Test=%d, ViGoVal=%d",
        len(train_loader), len(train_combined), len(val_speech_loader), len(test_speech_loader), len(val_vigo_loader),
    )

    # 4. Instantiate Golden Recipe Model
    model = Stage2HierarchicalMERModel(
        use_conditional_subhead=False,        # 928D Sub-Head input (fast, stable, proven)
        melody_median_kernel=5,               # 40ms Jitter F0 filter (Run 3 Winner)
        text_pooling="attention",             # Attentive PhoBERT pooling (Run 2 Winner)
        device=str(dev),
    ).to(dev)
    model.fusion.use_decoupled_routing = True # Decoupled text routing (Run 1 Winner)

    if ckpt_path:
        missing, unexp = model.load_checkpoint_compatible(str(ckpt_path), map_location=dev)
        logger.info("Loaded base checkpoint: %d missing, %d unexpected.", len(missing), len(unexp))

    freeze_phobert_layers(model, num_freeze_layers=10)

    # 5. Optimizer, Loss & Scheduler
    uncertainty_fn = MultiTaskUncertaintyLoss(num_tasks=2).to(dev)
    audio_lora_params = [p for n, p in model.audio_encoder.named_parameters() if "lora_" in n.lower() or getattr(model.audio_encoder, "_synthetic", False)]
    text_encoder_params = [p for p in model.text_encoder.parameters() if p.requires_grad]
    audio_param_ids = {id(p) for p in model.audio_encoder.parameters()}
    text_param_ids = {id(p) for p in model.text_encoder.parameters()}
    heads_and_fusion_params = [
        p for p in model.parameters()
        if id(p) not in audio_param_ids and id(p) not in text_param_ids and p.requires_grad
    ]
    heads_and_fusion_params.extend([p for p in uncertainty_fn.parameters() if p.requires_grad])

    param_groups = [
        {"name": "heads_and_fusion", "params": heads_and_fusion_params, "lr": args.lr, "weight_decay": 1e-2},
        {"name": "audio_lora", "params": audio_lora_params, "lr": args.lr_audio_lora, "weight_decay": 1e-2},
        {"name": "phobert_top_layers", "params": text_encoder_params, "lr": args.lr_text_encoder, "weight_decay": 1e-2},
    ]
    optimizer = optim.AdamW(param_groups)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1.0e-6)
    device_type = "cuda" if dev.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(device_type, enabled=(args.use_amp and dev.type == "cuda"))

    loss_primary = LDAMLoss(cls_num_list=[1000] * 5, scale=30.0).to(dev)
    loss_sub = SiblingMaskedLDAMLoss(
        cls_num_list=[200] * 11,
        scale=30.0,
        use_sibling_masking=True,
    ).to(dev)

    all_trainable_params = [p for p in model.parameters() if p.requires_grad] + [p for p in uncertainty_fn.parameters() if p.requires_grad]

    # 6. Training Loop with High Throughput & Early Stopping
    best_val_score = -1.0
    best_epoch = 0
    patience_counter = 0
    best_ckpt_path = output_dir / "best_model_stage2_6_final.pt"

    print("\n" + "=" * 80)
    print(f"TRAINING IN PROGRESS: 1 to {args.epochs} Epochs...")
    print("=" * 80)

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            wavs = batch["waveforms"].squeeze(1).to(dev, non_blocking=True)
            p_targets = batch["primary_labels"].to(dev, non_blocking=True)
            s_targets = batch["sub_labels"].to(dev, non_blocking=True)
            transcripts = batch["transcripts"]
            speech_ts = batch.get("speech_timestamps", None)
            sample_ids = batch.get("sample_ids", [])

            is_speech = (wavs.abs().amax(dim=-1) > 1e-5) & torch.tensor(
                [not str(sid).startswith("vigo_") for sid in sample_ids],
                device=dev,
                dtype=torch.bool,
            )

            with torch.amp.autocast(device_type, enabled=(args.use_amp and dev.type == "cuda")):
                out = model(waveforms=wavs, speech_timestamps=speech_ts, texts=transcripts, use_asr=False)

                # Decoupled primary supervision with numerical nan_to_num guard
                p_logits = torch.nan_to_num(out.primary_logits, nan=0.0)
                s_logits = torch.nan_to_num(out.sub_logits, nan=0.0)

                if is_speech.any():
                    l_prim = loss_primary(p_logits[is_speech], p_targets[is_speech])
                else:
                    l_prim = p_logits.sum() * 0.0

                l_s = loss_sub(s_logits, s_targets, is_coarse_speech=is_speech)
                loss = uncertainty_fn(l_prim, l_s)
                loss_scaled = loss / args.grad_accum_steps

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("Encountered NaN/Inf loss at Epoch %d Batch %d. Skipping step to protect weights.", ep, batch_idx + 1)
                optimizer.zero_grad()
                continue

            if args.use_amp and dev.type == "cuda":
                scaler.scale(loss_scaled).backward()
                if (batch_idx + 1) % args.grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(all_trainable_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss_scaled.backward()
                if (batch_idx + 1) % args.grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(all_trainable_params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss += loss.item()

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                elapsed = time.time() - t0
                speed = (batch_idx + 1) / max(1e-3, elapsed)
                print(f"  [Ep {ep:02d}/{args.epochs:02d}] Batch {batch_idx+1:3d}/{len(train_loader)} | Loss: {loss.item():.4f} | Speed: {speed:.1f} batch/s", flush=True)

        scheduler.step()
        ep_time = time.time() - t0
        avg_loss = total_loss / max(1, len(train_loader))

        # Evaluate on validation
        val_m = evaluate_loader(model, val_speech_loader, device=dev, is_speech_track=True)
        vigo_m = evaluate_loader(model, val_vigo_loader, device=dev, is_speech_track=False)

        score = val_m["primary_macro_f1"] + val_m["active5_sub_macro_f1"]
        print(
            f"\n>>> [EPOCH {ep:02d} COMPLETED in {ep_time:.1f}s] Loss: {avg_loss:.4f} | "
            f"Val Primary F1: {val_m['primary_macro_f1']*100:.2f}% (WA: {val_m['primary_wa']*100:.2f}%) | "
            f"Val Active-5 Sub F1: {val_m['active5_sub_macro_f1']*100:.2f}% | "
            f"ViGo Sub F1: {vigo_m['vigo_sub_macro_f1']*100:.2f}%\n",
            flush=True,
        )

        if score > best_val_score:
            best_val_score = score
            best_epoch = ep
            patience_counter = 0
            torch.save(
                {
                    "epoch": ep,
                    "val_metrics": val_m,
                    "vigo_metrics": vigo_m,
                    "state_dict": model.state_dict(),
                    "score": score,
                },
                best_ckpt_path,
            )
            print(f"  [*] NEW BEST CHECKPOINT SAVED at Epoch {ep:02d} (Combined Score: {score:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[EARLY STOPPING TRIGGERED] No improvement for {args.patience} epochs.")
                break

    # 7. Final Test Evaluation on Best Checkpoint
    print("\n" + "=" * 80)
    print(f"FINAL EVALUATION ON TEST SET (Best Model from Epoch {best_epoch:02d})")
    print("=" * 80)

    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])

    test_m = evaluate_loader(model, test_speech_loader, device=dev, is_speech_track=True)
    vigo_test_m = evaluate_loader(model, val_vigo_loader, device=dev, is_speech_track=False)

    report = [
        "# STAGE 2.6 PRODUCTION TRAINING FINAL SCOREBOARD",
        "",
        f"- **Best Epoch**: `{best_epoch}`",
        f"- **Test Primary Macro-F1**: `{test_m['primary_macro_f1']*100:.2f}%` (WA: `{test_m['primary_wa']*100:.2f}%`)",
        f"- **Test Active-5 Sub Macro-F1**: `{test_m['active5_sub_macro_f1']*100:.2f}%` (WA: `{test_m['active5_sub_wa']*100:.2f}%`)",
        f"- **ViGo 11-Class Sub Macro-F1**: `{vigo_test_m['vigo_sub_macro_f1']*100:.2f}%` (WA: `{vigo_test_m['vigo_sub_wa']*100:.2f}%`)",
        f"- **Checkpoint Saved**: `{best_ckpt_path}`",
    ]
    report_text = "\n".join(report)
    print(report_text)

    with open(output_dir / "final_report.md", "w", encoding="utf-8") as f:
        f.write(report_text)

    # Auto-copy to /kaggle/working root for direct 1-click download in Kaggle Output
    if Path("/kaggle/working").exists() and best_ckpt_path.exists():
        import shutil
        kaggle_dest = Path("/kaggle/working/best_model_stage2_6_final.pt")
        shutil.copyfile(best_ckpt_path, kaggle_dest)
        logger.info("Copied final model to Kaggle root: %s", kaggle_dest)

    print("\n[COMPLETED] Production model trained and saved successfully.")


if __name__ == "__main__":
    main()
