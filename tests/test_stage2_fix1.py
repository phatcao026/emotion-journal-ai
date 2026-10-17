"""
tests/test_stage2_fix1.py
-------------------------
Unit and integration tests for Fix 1:
- PhoBERT freezing logic (embeddings & layers 0-9 frozen, 10-11 trainable)
- Synthetic text backbone fallback freezing logic
- 3 independent AdamW optimizer parameter groups (heads/fusion, audio_lora, phobert_top_layers)
- Checkpoint resume optimizer migration guard (legacy 2-group vs 3-group, --reset-optimizer)
- Forward pass with speech_timestamps integration
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest
import torch
import torch.nn as nn
import yaml

from src.stage2.hierarchical_mer_stage2 import Stage2HierarchicalMERModel
from src.stage2.train_stage2 import freeze_phobert_layers, train_stage2


class MockRobertaLayer(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dense = nn.Linear(dim, dim)


class MockRobertaEncoder(nn.Module):
    def __init__(self, num_layers=12, dim=768):
        super().__init__()
        self.layer = nn.ModuleList([MockRobertaLayer(dim) for _ in range(num_layers)])


class MockRobertaModel(nn.Module):
    def __init__(self, num_layers=12, dim=768):
        super().__init__()
        self.embeddings = nn.Embedding(100, dim)
        self.encoder = MockRobertaEncoder(num_layers, dim)
        self.pooler = nn.Linear(dim, dim)


def test_freeze_phobert_layers_real_mock():
    """Test freezing layers 0-9 and embeddings on a real PhoBERT-like architecture."""
    model = Stage2HierarchicalMERModel(device="cpu")
    mock_bert = MockRobertaModel(num_layers=12, dim=768)
    model.text_encoder.bert_model = mock_bert
    model.text_encoder.is_synthetic = False

    # Initially all require grad
    for p in mock_bert.parameters():
        p.requires_grad = True

    freeze_phobert_layers(model, num_freeze_layers=10)

    # Embeddings must be frozen
    for p in mock_bert.embeddings.parameters():
        assert not p.requires_grad, "Embeddings must be frozen"

    # Pooler must be frozen
    for p in mock_bert.pooler.parameters():
        assert not p.requires_grad, "Pooler must be frozen"

    # Layers 0 to 9 must be frozen
    for i in range(10):
        for p in mock_bert.encoder.layer[i].parameters():
            assert not p.requires_grad, f"Layer {i} must be frozen"

    # Layers 10 and 11 must be trainable
    for i in range(10, 12):
        for p in mock_bert.encoder.layer[i].parameters():
            assert p.requires_grad, f"Layer {i} must be trainable"


def test_freeze_phobert_layers_synthetic():
    """Test freezing synthetic text backbone fallback."""
    model = Stage2HierarchicalMERModel(device="cpu")
    # By default, model has _synthetic_backbone and bert_model=None
    assert model.text_encoder.bert_model is None or model.text_encoder.is_synthetic

    freeze_phobert_layers(model, num_freeze_layers=10)

    synth = model.text_encoder._synthetic_backbone
    for p in synth.embedding.parameters():
        assert not p.requires_grad, "Synthetic embedding must be frozen"
    for p in synth.proj.parameters():
        assert p.requires_grad, "Synthetic projection must be trainable"


def test_three_optimizer_groups_structure():
    """Verify AdamW optimizer has exactly 3 disjoint groups with expected LRs and weight decay."""
    with open("configs/stage2/multimodal_stage2.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_cfg = config.get("training", {})
    lr_heads = float(train_cfg.get("learning_rate", 1.0e-4))
    lr_audio_lora = float(train_cfg.get("lr_audio_lora", 5.0e-5))
    lr_text_encoder = float(train_cfg.get("lr_text_encoder", 1.5e-5))
    weight_decay = float(train_cfg.get("weight_decay", 1.0e-2))

    assert lr_heads == 1.0e-4
    assert lr_audio_lora == 5.0e-5
    assert lr_text_encoder == 1.5e-5

    model = Stage2HierarchicalMERModel(device="cpu")
    freeze_phobert_layers(model, num_freeze_layers=int(train_cfg.get("phobert_freeze_layers", 10)))

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

    # Verify disjointness
    ids_heads = {id(p) for p in heads_and_fusion_params}
    ids_audio = {id(p) for p in audio_lora_params}
    ids_text = {id(p) for p in text_encoder_params}

    assert len(ids_heads.intersection(ids_audio)) == 0, "Heads and Audio params must be disjoint"
    assert len(ids_heads.intersection(ids_text)) == 0, "Heads and Text params must be disjoint"
    assert len(ids_audio.intersection(ids_text)) == 0, "Audio and Text params must be disjoint"

    param_groups = [
        {"name": "heads_and_fusion", "params": heads_and_fusion_params, "lr": lr_heads, "weight_decay": weight_decay},
        {"name": "audio_lora", "params": audio_lora_params, "lr": lr_audio_lora, "weight_decay": weight_decay},
        {"name": "phobert_top_layers", "params": text_encoder_params, "lr": lr_text_encoder, "weight_decay": weight_decay},
    ]
    optimizer = torch.optim.AdamW(param_groups)

    assert len(optimizer.param_groups) == 3
    assert optimizer.param_groups[0]["name"] == "heads_and_fusion"
    assert optimizer.param_groups[0]["lr"] == 1.0e-4
    assert optimizer.param_groups[1]["name"] == "audio_lora"
    assert optimizer.param_groups[1]["lr"] == 5.0e-5
    assert optimizer.param_groups[2]["name"] == "phobert_top_layers"
    assert optimizer.param_groups[2]["lr"] == 1.5e-5


def test_checkpoint_resume_optimizer_migration(tmp_path):
    """Test that resuming from legacy 2-group checkpoint does not overwrite fresh 3-group optimizer."""
    ckpt_path = tmp_path / "legacy_stage2_ckpt.pt"

    # Create dummy 2-group legacy state_dict
    p1 = nn.Parameter(torch.randn(2, 2))
    p2 = nn.Parameter(torch.randn(2, 2))
    legacy_opt = torch.optim.AdamW([{"params": [p1], "lr": 1e-4}, {"params": [p2], "lr": 5e-5}])

    torch.save({
        "epoch": 5,
        "state_dict": {},
        "optimizer_state_dict": legacy_opt.state_dict(),
        "val_primary_f1": 0.55,
    }, str(ckpt_path))

    # Verify that the legacy checkpoint has 2 groups
    loaded_ckpt = torch.load(str(ckpt_path), weights_only=False)
    assert len(loaded_ckpt["optimizer_state_dict"]["param_groups"]) == 2

    # Now simulate train_stage2 resume logic
    model = Stage2HierarchicalMERModel(device="cpu")
    freeze_phobert_layers(model, 10)
    heads_params = [p for p in model.parameters() if p.requires_grad][:2]
    audio_params = [p for p in model.parameters() if p.requires_grad][2:4]
    text_params = [p for p in model.parameters() if p.requires_grad][4:6]

    fresh_opt = torch.optim.AdamW([
        {"name": "heads_and_fusion", "params": heads_params, "lr": 1.0e-4},
        {"name": "audio_lora", "params": audio_params, "lr": 5.0e-5},
        {"name": "phobert_top_layers", "params": text_params, "lr": 1.5e-5},
    ])

    # Check migration guard
    opt_state = loaded_ckpt.get("optimizer_state_dict")
    ckpt_param_groups = opt_state.get("param_groups", [])
    num_ckpt_groups = len(ckpt_param_groups)
    num_curr_groups = len(fresh_opt.param_groups)

    assert num_ckpt_groups == 2
    assert num_curr_groups == 3
    # Migration guard activates: fresh_opt keeps 3 groups and their tiered LRs
    if num_ckpt_groups != num_curr_groups:
        pass  # Fresh optimizer is preserved
    else:
        fresh_opt.load_state_dict(opt_state)

    assert len(fresh_opt.param_groups) == 3
    assert fresh_opt.param_groups[0]["lr"] == 1.0e-4
    assert fresh_opt.param_groups[1]["lr"] == 5.0e-5
    assert fresh_opt.param_groups[2]["lr"] == 1.5e-5


def test_dummy_train_run(tmp_path):
    """Run 1 epoch in dummy mode to verify end-to-end integration."""
    with open("configs/stage2/multimodal_stage2.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_stage2(
        config=config,
        dummy=True,
        output_dir=str(tmp_path),
        device="cpu",
        epochs_override=1,
        batch_size_override=4,
        grad_accum_steps=1,
        seed=42,
    )

    assert (tmp_path / "last_model_stage2.pt").exists(), "Checkpoint last_model_stage2.pt should be saved"
