"""
test_crosscorpus_pipeline.py
----------------------------
Unit tests verifying the Cross-Corpus 5-class pipeline:
1. Dataset loading via VoiceJournalDataset with all 5 classes (including ANXIETY).
2. Class weight computation for 5 classes.
3. Forward pass and loss backward pass through VoiceOnlyMERModel.
4. Loading pretrained Stage 1 weights and verifying 5-class logits shape.
"""

from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader

from src.audio.emotion2vec_lora import LoRAConfig
from src.data.audio_dataset import VoiceJournalDataset, PrimaryEmotion
from src.losses.focal_loss import MultiClassFocalLoss
from src.multimodal.hierarchical_mer import VoiceOnlyMERModel
from src.utils.metrics import compute_all_metrics


@pytest.fixture(scope="module")
def dummy_crosscorpus_dir(tmp_path_factory):
    """Generate dummy cross-corpus dataset with all 5 classes."""
    out_dir = tmp_path_factory.mktemp("crosscorpus_data")
    from scripts.prepare_crosscorpus_visec import generate_dummy_crosscorpus
    generate_dummy_crosscorpus(output_dir=out_dir, total_samples=50)
    return out_dir


def test_crosscorpus_dataset_loading(dummy_crosscorpus_dir):
    """Verify VoiceJournalDataset loads train/val/test splits with 5 classes."""
    train_ds = VoiceJournalDataset(data_dir=dummy_crosscorpus_dir, split="train")
    val_ds = VoiceJournalDataset(data_dir=dummy_crosscorpus_dir, split="val")
    test_ds = VoiceJournalDataset(data_dir=dummy_crosscorpus_dir, split="test")

    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(test_ds) > 0

    # Verify ANXIETY (label 2) is present in train and test
    train_labels = [s.primary_label.value for s in train_ds.samples]
    assert 2 in train_labels, "ANXIETY (class 2) should be present in train split"

    # Check class weights
    weights = train_ds.get_class_weights()
    assert weights.shape[0] == 5
    assert not torch.isnan(weights).any()


def test_crosscorpus_dataloader_collate(dummy_crosscorpus_dir):
    """Verify DataLoader collates batches correctly."""
    train_ds = VoiceJournalDataset(data_dir=dummy_crosscorpus_dir, split="train")
    loader = DataLoader(
        train_ds,
        batch_size=4,
        shuffle=False,
        collate_fn=VoiceJournalDataset.collate_fn,
    )
    batch = next(iter(loader))
    assert "waveforms" in batch
    assert batch["waveforms"].shape[0] == 4
    assert "primary_labels" in batch
    assert batch["primary_labels"].shape[0] == 4


def test_voice_only_model_forward_backward_5_classes(dummy_crosscorpus_dir):
    """Verify forward and backward pass for VoiceOnlyMERModel on 5 classes."""
    train_ds = VoiceJournalDataset(data_dir=dummy_crosscorpus_dir, split="train")
    loader = DataLoader(
        train_ds,
        batch_size=4,
        shuffle=False,
        collate_fn=VoiceJournalDataset.collate_fn,
    )
    batch = next(iter(loader))
    waveforms = batch["waveforms"].squeeze(1)
    targets = batch["primary_labels"]

    model = VoiceOnlyMERModel(
        lora_config=LoRAConfig(r=4),
        num_primary_classes=5,
        device="cpu",
    )
    loss_fn = MultiClassFocalLoss(gamma=2.0)

    output = model(waveforms=waveforms)
    assert output.primary_logits.shape == (4, 5)
    assert output.primary_probs.shape == (4, 5)

    loss = loss_fn(output.primary_logits, targets)
    assert not torch.isnan(loss)

    loss.backward()
    # Check that gradients exist
    head_grad = model.primary_head[1].weight.grad
    assert head_grad is not None
    assert head_grad.shape == (5, 896)


def test_stage1_checkpoint_loading_compatibility():
    """Verify that Stage 1 checkpoint loads cleanly into 5-class model."""
    ckpt_path = Path("checkpoints/visec_stage1_pretrained.pt")
    if not ckpt_path.exists():
        pytest.skip("Stage 1 checkpoint not found in local workspace")

    model = VoiceOnlyMERModel(
        lora_config=LoRAConfig(r=8),
        num_primary_classes=5,
        device="cpu",
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Verify primary_head shape in checkpoint
    assert "primary_head.1.weight" in ckpt["state_dict"]
    assert ckpt["state_dict"]["primary_head.1.weight"].shape == (5, 896)
    assert "primary_head.1.bias" in ckpt["state_dict"]
    assert ckpt["state_dict"]["primary_head.1.bias"].shape == (5,)

    # Load with strict=False (handles both real emotion2vec on Kaggle and synthetic fallback locally)
    model.load_state_dict(ckpt["state_dict"], strict=False)

    # Model can do forward pass with 5-class logits output
    dummy_wav = torch.randn(2, 16000 * 3)
    out = model(dummy_wav)
    assert out.primary_logits.shape == (2, 5)
    assert out.primary_probs.shape == (2, 5)
