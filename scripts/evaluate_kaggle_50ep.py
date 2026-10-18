#!/usr/bin/env python3
"""
scripts/evaluate_kaggle_50ep.py
-------------------------------
Deep analysis and evaluation of the 50-epoch Kaggle trained checkpoint:
- Checkpoint: checkpoints/stage2_kaggle_50ep/best_model_stage2.pt
- Validates PhoBERT semantic diversity (z_diff)
- Checks Pause Projection weights (128D activation)
- Audits Gated Fusion audio-only routing
- Evaluates zero-shot transfer performance on Journal Crawled dataset (data/crawled)
"""

import sys
import logging
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn.functional as F
import numpy as np

from src.stage2.hierarchical_mer_stage2 import Stage2HierarchicalMERModel
from src.audio.audio_dataset import VoiceJournalDataset
from src.common.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("eval_kaggle_50ep")


def main():
    ckpt_path = Path("checkpoints/stage2_kaggle_50ep/best_model_stage2.pt")
    if not ckpt_path.exists():
        print(f"Error: {ckpt_path} not found!")
        return

    print("=" * 80)
    print("ANALYZING 50-EPOCH KAGGLE RETRAINED CHECKPOINT")
    print("=" * 80)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    epoch = ckpt.get("epoch")
    val_f1 = ckpt.get("val_primary_f1")
    val_metrics = ckpt.get("val_metrics", {})
    state_dict = ckpt.get("state_dict", {})

    print(f"[*] Checkpoint Origin: Epoch {epoch}")
    print(f"[*] Validation Primary Macro-F1: {val_f1:.4f} ({val_f1*100:.2f}%)")
    print(f"[*] Validation Primary WA:       {val_metrics.get('primary_wa', 0):.4f} ({val_metrics.get('primary_wa', 0)*100:.2f}%)")
    print(f"[*] Validation Sub Macro-F1:     {val_metrics.get('sub_f1', 0):.4f} ({val_metrics.get('sub_f1', 0)*100:.2f}%)")
    print(f"[*] Validation Sub WA:           {val_metrics.get('sub_wa', 0):.4f} ({val_metrics.get('sub_wa', 0)*100:.2f}%)")
    print(f"[*] Validation Loss:             {val_metrics.get('loss', 0):.4f}")

    # Build model and initialize backbones
    model = Stage2HierarchicalMERModel(device="cpu")
    model.load_pretrained()
    missing, unexpected = model.load_checkpoint_compatible(ckpt_path, map_location="cpu")
    print(f"[*] Loaded into Stage2HierarchicalMERModel: {len(missing)} missing, {len(unexpected)} unexpected.")
    model.eval()

    # 1. Audit Pause Projection Weights
    pause_w = model.fusion.pause_projection[0].weight.data  # shape (128, 4)
    pause_norm = torch.norm(pause_w).item()
    print(f"\n[1] PAUSE PROJECTION (128D) WEIGHT AUDIT:")
    print(f"    - Layer: fusion.pause_projection[0].weight, shape: {list(pause_w.shape)}")
    print(f"    - Weight Frobenius Norm: {pause_norm:.4f}")
    print(f"    - Mean magnitude per input feature [ratio, cpm, max_dur, mean_dur]:")
    col_norms = torch.norm(pause_w, dim=0).tolist()
    feat_names = ["pause_ratio", "count_per_min", "max_duration", "mean_duration"]
    for fn, cn in zip(feat_names, col_norms):
        print(f"      * {fn:16s}: {cn:.4f}")

    # 2. Audit PhoBERT Semantic Diversity
    print(f"\n[2] PHOBERT REPRESENTATION DIVERSITY AUDIT:")
    test_sentences = [
        "Hôm nay tôi thực sự vô cùng hạnh phúc và phấn khởi.",
        "Tôi đau khổ, suy sụp và tuyệt vọng tột cùng.",
        "Cơn giận dữ và phẫn uất bùng nổ không thể kiểm soát.",
        "Tôi thấy lo lắng, bồn chồn và bất an trong lòng.",
        "Một ngày bình thường, không có gì đáng chú ý.",
    ]
    with torch.no_grad():
        z_texts = model.text_encoder(test_sentences)

    dists = []
    for i in range(len(test_sentences)):
        for j in range(i + 1, len(test_sentences)):
            d = torch.norm(z_texts[i] - z_texts[j]).item()
            dists.append(d)
    min_d, max_d, mean_d = min(dists), max(dists), sum(dists) / len(dists)
    print(f"    - Evaluated across {len(test_sentences)} distinct emotional prompts.")
    print(f"    - Pairwise distance min={min_d:.4f}, max={max_d:.4f}, mean={mean_d:.4f}")
    if min_d > 0.30:
        print(f"    [PASSED] PhoBERT semantic space is highly differentiated (no collapse).")
    else:
        print(f"    [WARNING] PhoBERT representation diversity is low!")

    # 3. Audit Voice-Only Routing (RAVDESS English Blanking Mechanism)
    print(f"\n[3] GATED FUSION ROUTING AUDIT:")
    dummy_wav = torch.randn(2, 32000)
    with torch.no_grad():
        out_viet = model(waveforms=dummy_wav, texts=["Hôm nay tôi rất vui", "Tôi thấy buồn"])
        out_blank = model(waveforms=dummy_wav, texts=["", ""])

    print(f"    - Vietnamese Text inputs: gate_audio={out_viet.gate_audio.mean().item():.3f}, gate_text={out_viet.gate_text.mean().item():.3f}")
    print(f"    - Blank Text inputs (RAVDESS style): gate_audio={out_blank.gate_audio.mean().item():.3f}, gate_text={out_blank.gate_text.mean().item():.3f}")
    assert torch.allclose(out_blank.gate_audio, torch.ones_like(out_blank.gate_audio), atol=1e-3), "Voice-only routing failed!"
    print(f"    [PASSED] Blank text routes 100% to gate_audio (1.000).")

    # 4. Zero-shot Evaluation on Crawled Journal Dataset (if available)
    crawled_manifest = Path("data/crawled/metadata.csv")
    if crawled_manifest.exists():
        print(f"\n[4] EVALUATING ON LOCAL JOURNAL DATASET ({crawled_manifest}):")
        dataset = VoiceJournalDataset(data_dir=Path("data/crawled"), split="train", use_sub_labels=True)
        # Filter for samples that actually have audio files
        valid_indices = []
        for idx, s in enumerate(dataset.samples):
            if s.audio_path and Path(s.audio_path).is_file() and str(s.audio_path).lower().endswith(".wav"):
                valid_indices.append(idx)

        print(f"    - Found {len(valid_indices)} audio-paired journal samples.")
        if valid_indices:
            from torch.utils.data import Subset, DataLoader
            sub_ds = Subset(dataset, valid_indices)
            loader = DataLoader(sub_ds, batch_size=8, shuffle=False, collate_fn=VoiceJournalDataset.collate_fn)

            all_p_preds, all_p_targets = [], []
            all_s_preds, all_s_targets = [], []
            with torch.no_grad():
                for batch in loader:
                    wavs = batch["waveforms"].squeeze(1)
                    txts = batch["transcripts"]
                    sts = batch.get("speech_timestamps", None)
                    p_tgt = batch["primary_labels"]
                    s_tgt = batch["sub_labels"]

                    out = model(waveforms=wavs, speech_timestamps=sts, texts=txts, use_asr=False)
                    all_p_preds.append(out.primary_predictions())
                    all_p_targets.append(p_tgt)
                    all_s_preds.append(out.sub_predictions())
                    all_s_targets.append(s_tgt)

            p_pred = torch.cat(all_p_preds)
            p_true = torch.cat(all_p_targets)
            s_pred = torch.cat(all_s_preds)
            s_true = torch.cat(all_s_targets)

            p_metrics = compute_all_metrics(predictions=p_pred, targets=p_true, num_classes=5)
            s_metrics = compute_all_metrics(predictions=s_pred, targets=s_true, num_classes=11)

            print(f"    [*] Journal Test Primary Macro-F1: {p_metrics.get('macro_f1', 0):.4f} (WA: {p_metrics.get('wa', 0):.4f})")
            print(f"    [*] Journal Test Sub Macro-F1:     {s_metrics.get('macro_f1', 0):.4f} (WA: {s_metrics.get('wa', 0):.4f})")

            # Per-class Primary breakdown
            class_names = ["Joy", "Sadness", "Anxiety", "Anger", "Neutral"]
            print("\n    Per-Class Primary Emotion Performance on Journal:")
            for c_idx, c_name in enumerate(class_names):
                tp = ((p_pred == c_idx) & (p_true == c_idx)).sum().item()
                fp = ((p_pred == c_idx) & (p_true != c_idx)).sum().item()
                fn = ((p_pred != c_idx) & (p_true == c_idx)).sum().item()
                total_c = (p_true == c_idx).sum().item()
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1_c = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                print(f"      * {c_name:10s} (N={total_c:2d}): Prec={prec:.3f}, Rec={rec:.3f}, F1={f1_c:.3f}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
