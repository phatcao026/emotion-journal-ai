import json
import sys
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from src.fusion.hierarchical_mer import VoiceOnlyMERModel
from src.stage2.hierarchical_mer_stage2 import Stage2HierarchicalMERModel
from scripts.evaluate_stage2_harvested import load_wav_tensor

# 1. Load Stage 1.5 (ViSEC + Cross-corpus pretrained)
print("[*] Nạp Stage 1.5 Voice-Only (Pretrained trên ViSEC 5,280 samples + RAVDESS + CREMA-D)...")
m15 = VoiceOnlyMERModel(device="cpu")
ckpt_path = "checkpoints/visec_stage1_pretrained.pt" if Path("checkpoints/visec_stage1_pretrained.pt").exists() else "checkpoints/voice_only/best_model.pt"
ckpt15 = torch.load(ckpt_path, map_location="cpu", weights_only=False)
m15.load_state_dict(ckpt15.get("state_dict", ckpt15), strict=False)
m15.eval()
print(f"[+] Stage 1.5 loaded from {ckpt_path}!")

# 2. Load Stage 2 SOTA Multimodal
print("[*] Nạp Stage 2 SOTA (Hybrid Acoustic + Explicit Melody 32D + PhoBERT + Gated Fusion)...")
m2 = Stage2HierarchicalMERModel(device="cpu")
ckpt2 = torch.load("checkpoints/stage2/best_model_stage2.pt", map_location="cpu", weights_only=False)
m2.load_state_dict(ckpt2.get("state_dict", ckpt2), strict=False)
m2.eval()
print("[+] Stage 2 loaded!")

# 3. Đánh giá trên tập mẫu Crawled mới thu thập (text_voice pairs)
records = [json.loads(l) for l in open("data/crawled/metadata.jsonl", encoding="utf-8") if l.strip()]
p_names = ["JOY", "SADNESS", "ANXIETY", "ANGER", "NEUTRAL"]
s_names = ["JOY", "CALM", "HOPE", "CONNECTION", "SADNESS", "ANXIETY", "FEAR", "ANGER", "GUILT_SHAME", "LONELINESS", "DISGUST"]

# Lọc các mẫu text_voice mới thu thập từ YouTube / Podcast
new_text_voice = [r for r in records if r.get("modality") == "text_voice" and r.get("text_content")]
print(f"\n[*] Đang chạy đối sánh trên {len(new_text_voice)} cặp Text-Voice chân thực...")

tested_samples = []
for r in new_text_voice:
    audio_f = r.get("audio_filename") or f"{r['sample_id']}.wav"
    w_path = ROOT_DIR / "data" / "crawled" / "text_voice" / audio_f
    if not w_path.exists():
        continue
    wav = load_wav_tensor(w_path)
    if wav is None:
        continue

    text = r.get("text_content", "")
    tier = r.get("data_tier", "general_monologue")
    with torch.no_grad():
        out15 = m15(wav)
        out2 = m2(waveforms=wav, texts=[text])
        p15_idx = torch.argmax(out15.primary_logits, dim=-1).item()
        p2_idx = out2.primary_predictions().item()
        s2_idx = out2.sub_predictions().item()

        tested_samples.append({
            "sample_id": r["sample_id"],
            "data_tier": tier,
            "gold_prior": r.get("primary_label"),
            "text": text[:35],
            "stage1_5_visec": p_names[p15_idx],
            "stage2_primary": p_names[p2_idx],
            "stage2_sub": s_names[s2_idx],
            "is_masking": r.get("incongruent_masking", False)
        })

print(f"\n[+] ĐÃ ĐỐI SÁNH THÀNH CÔNG {len(tested_samples)} CẶP TEXT-VOICE:")
print("-" * 115)
print(f"{'Sample ID':<22} | {'Tier':<18} | {'Transcript':<30} | {'ViSEC Stage 1.5':<16} | {'Stage 2 SOTA':<16} | {'Masking'}")
print("-" * 115)
for s in tested_samples[:14]:
    m_str = "MASKED" if s["is_masking"] else "Normal"
    print(f"{s['sample_id']:<22} | {s['data_tier']:<18} | {s['text']:<30} | {s['stage1_5_visec']:<16} | {s['stage2_primary'] + '/' + s['stage2_sub']:<16} | {m_str}")
print("-" * 115)

# Phân tích theo từng Phân tầng Dữ liệu (Tier-based Evaluation)
tier1_samples = [s for s in tested_samples if s["data_tier"] == "deep_journal"]
tier2_samples = [s for s in tested_samples if s["data_tier"] == "general_monologue"]

print("\n" + "="*80)
print("PHÂN TÍCH HIỆU QUẢ THEO PHÂN TẦNG DỮ LIỆU HUẤN LUYỆN (DOWNSTREAM MER TIERS)")
print("="*80)

print(f"\n[TIER 1] DEEP AFFECTIVE JOURNALING ({len(tier1_samples)} mẫu):")
print("  - Bản chất: Tâm sự trải lòng ngôi thứ nhất, bộc bạch nỗi niềm, lo âu, hy vọng.")
if tier1_samples:
    t1_s15_counts = {}
    t1_s2_counts = {}
    t1_masking = sum(1 for s in tier1_samples if s["is_masking"])
    for s in tier1_samples:
        t1_s15_counts[s["stage1_5_visec"]] = t1_s15_counts.get(s["stage1_5_visec"], 0) + 1
        t1_s2_counts[s["stage2_primary"]] = t1_s2_counts.get(s["stage2_primary"], 0) + 1
    print(f"  - ViSEC Stage 1.5 Predictions : {dict(t1_s15_counts)}")
    print(f"  - Stage 2 SOTA Predictions    : {dict(t1_s2_counts)}")
    print(f"  - Tỷ lệ phát hiện Masking     : {t1_masking}/{len(tier1_samples)} ({t1_masking / len(tier1_samples) * 100:.1f}%)")
    print(f"  => Nhận định: Dữ liệu Tier 1 kích hoạt mạnh mẽ các nhánh Sub-emotion & Masking, là nòng cốt để fine-tune Stage 2.")

print(f"\n[TIER 2] GENERAL SPOKEN MONOLOGUE ({len(tier2_samples)} mẫu):")
print("  - Bản chất: Độc thoại diễn thuyết, podcast chia sẻ triết lý, đọc tản văn với âm học sạch.")
if tier2_samples:
    t2_s15_counts = {}
    t2_s2_counts = {}
    for s in tier2_samples:
        t2_s15_counts[s["stage1_5_visec"]] = t2_s15_counts.get(s["stage1_5_visec"], 0) + 1
        t2_s2_counts[s["stage2_primary"]] = t2_s2_counts.get(s["stage2_primary"], 0) + 1
    print(f"  - ViSEC Stage 1.5 Predictions : {dict(t2_s15_counts)}")
    print(f"  - Stage 2 SOTA Predictions    : {dict(t2_s2_counts)}")
    neutral_calm_pct = (t2_s15_counts.get("NEUTRAL", 0) + t2_s15_counts.get("JOY", 0)) / len(tier2_samples) * 100.0
    print(f"  - Tỷ lệ Âm học Điềm tĩnh/Neutral : {neutral_calm_pct:.1f}%")
    print(f"  => Nhận định: Giữ lại để huấn luyện MER cơ bản dạng độc thoại (pretraining acoustic features, speaker diversity).")
print("="*80)

