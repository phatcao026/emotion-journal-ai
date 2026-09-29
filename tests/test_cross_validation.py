"""
tests/test_cross_validation.py
------------------------------
Academic validation suite for Stage 2 MER:
1. 5-Fold Stratified Cross-Validation split integrity and class-stratification.
2. Leave-One-Corpus-Out (LOCO) evaluation across multi-corpus collections.
3. Class-imbalance handling: LDAM-DRW margin scaling verification.
4. Hierarchical logit masking: Guarantees 0% hierarchical constraint violations.
5. Acoustic melody biomarker dimension and range verification.
"""

from typing import List, Dict, Tuple
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.stage2.melody_extractor import ExplicitMelodyExtractor
from src.stage2.ldam_loss import LDAMLoss
from src.stage2.uncertainty_loss import MultiTaskUncertaintyLoss
from src.stage2.hierarchical_mer_stage2 import (
    TAXONOMY_MAP,
    TAXONOMY_MASK_MATRIX,
    apply_hierarchical_mask,
)


# ============================================================================
# 1. 5-Fold Stratified Cross-Validation Splitting
# ============================================================================

def create_stratified_kfold_splits(
    labels: List[int],
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Produce K Stratified splits ensuring balanced class representation in each fold.
    """
    rng = np.random.RandomState(seed)
    labels = np.array(labels)
    unique_classes, counts = np.unique(labels, return_counts=True)
    
    # Store indices per class
    class_indices = {cls: rng.permutation(np.where(labels == cls)[0]) for cls in unique_classes}
    
    folds = [[] for _ in range(n_splits)]
    for cls, indices in class_indices.items():
        split_chunks = np.array_split(indices, n_splits)
        for fold_idx in range(n_splits):
            folds[fold_idx].extend(split_chunks[fold_idx])
            
    cv_splits = []
    all_indices = set(range(len(labels)))
    for fold_idx in range(n_splits):
        val_idx = np.array(folds[fold_idx])
        train_idx = np.array(list(all_indices - set(val_idx)))
        cv_splits.append((train_idx, val_idx))
        
    return cv_splits


def test_stratified_kfold_integrity():
    """Verify 5-fold stratification covers all samples with zero overlap and preserves classes."""
    # Synthetic dataset with 5 classes and severe class imbalance (class 2 is minority)
    labels = (
        [0] * 100 +  # JOY
        [1] * 80 +   # SADNESS
        [2] * 20 +   # ANXIETY (minority ~6.6%)
        [3] * 60 +   # ANGER
        [4] * 40     # NEUTRAL
    )
    n_splits = 5
    splits = create_stratified_kfold_splits(labels, n_splits=n_splits)

    assert len(splits) == 5
    all_val_indices = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        # Disjoint sets: train and val must not overlap
        overlap = set(train_idx).intersection(set(val_idx))
        assert len(overlap) == 0, f"Fold {fold_idx} has data leakage!"
        assert len(train_idx) + len(val_idx) == len(labels)
        all_val_indices.extend(val_idx)

        # Every fold validation set must contain the minority class (ANXIETY)
        val_labels = [labels[i] for i in val_idx]
        assert 2 in val_labels, f"Minority class missing from fold {fold_idx} val set!"

    # Across all folds, each sample must be evaluated exactly once
    assert sorted(all_val_indices) == list(range(len(labels)))


# ============================================================================
# 2. Leave-One-Corpus-Out (LOCO) Cross-Corpus Partitioning
# ============================================================================

def test_leave_one_corpus_out_partitioning():
    """Verify LOCO partitioning isolates each corpus as a clean test target."""
    dataset = [
        {"id": f"sample_{i}", "corpus": "ViSEC", "label": i % 5} for i in range(100)
    ] + [
        {"id": f"sample_{i+100}", "corpus": "CREAM-D", "label": i % 5} for i in range(60)
    ] + [
        {"id": f"sample_{i+160}", "corpus": "RAVDESS", "label": i % 5} for i in range(40)
    ]

    unique_corpora = sorted(list({item["corpus"] for item in dataset}))
    assert len(unique_corpora) == 3

    for test_corpus in unique_corpora:
        train_data = [item for item in dataset if item["corpus"] != test_corpus]
        test_data = [item for item in dataset if item["corpus"] == test_corpus]

        # Verify strict domain isolation
        train_corpora = {item["corpus"] for item in train_data}
        test_corpora = {item["corpus"] for item in test_data}

        assert test_corpus not in train_corpora
        assert test_corpora == {test_corpus}
        assert len(train_data) + len(test_data) == len(dataset)


# ============================================================================
# 3. Class-Imbalance Margin Verification (LDAM-DRW)
# ============================================================================

def test_ldam_loss_margin_scaling():
    """
    Verify LDAM assigns larger margins Delta_j to rarer classes:
    Delta_j = C / (N_j ** 0.25).
    """
    class_counts = [500, 200, 20, 150, 100]  # Class 2 is rarest
    loss_fn = LDAMLoss(class_counts=class_counts, max_m=0.5, s=30.0)

    margins = loss_fn.m_list.cpu().numpy()
    # Minority class must have highest margin
    assert np.argmax(margins) == 2
    # Majority class must have lowest margin
    assert np.argmin(margins) == 0
    assert margins[2] > margins[0]

    # Verify forward computation works on synthetic logits
    logits = torch.randn(8, 5)
    targets = torch.tensor([0, 1, 2, 3, 4, 2, 0, 1])
    loss = loss_fn(logits, targets)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0


# ============================================================================
# 4. Multi-Task Uncertainty Loss Verification
# ============================================================================

def test_multitask_uncertainty_loss_gradients():
    """Verify Kendall uncertainty weighting smoothly backpropagates for 2 heads."""
    loss_fn = MultiTaskUncertaintyLoss(num_tasks=2)
    loss1 = torch.tensor(1.5, requires_grad=True)
    loss2 = torch.tensor(2.0, requires_grad=True)

    combined_loss = loss_fn(loss1, loss2)
    assert not torch.isnan(combined_loss)
    combined_loss.backward()

    # Log variance parameters must receive gradients
    assert loss_fn.log_vars.grad is not None
    assert loss_fn.log_vars.grad.shape == (2,)


# ============================================================================
# 5. Hierarchical Logit Masking (Zero-Violation Guarantee)
# ============================================================================

def test_hierarchical_logit_masking_strictness():
    """
    Verify that apply_hierarchical_mask sets logits of unpermitted sub-classes
    to -inf, guaranteeing 0% cognitive dissonance violations.
    """
    primary_logits = torch.tensor([
        [10.0, 0.0, 0.0, 0.0, 0.0],  # Argmax = 0 (JOY)
        [0.0, 0.0, 10.0, 0.0, 0.0],  # Argmax = 2 (ANXIETY)
    ])
    # Unconstrained random sub logits
    sub_logits = torch.randn(2, 11)

    masked_sub_logits = apply_hierarchical_mask(
        primary_logits=primary_logits,
        sub_logits=sub_logits,
        mask_matrix=TAXONOMY_MASK_MATRIX,
    )

    # For sample 0 (JOY): valid sub-emotions are TAXONOMY_MAP[0] = [0, 1] (HAPPINESS, PRIDE)
    joy_allowed = set(TAXONOMY_MAP[0])
    for sub_idx in range(11):
        val = masked_sub_logits[0, sub_idx].item()
        if sub_idx in joy_allowed:
            assert val != float("-inf"), f"Allowed sub-class {sub_idx} was masked!"
        else:
            assert val == float("-inf"), f"Forbidden sub-class {sub_idx} was not masked!"

    # Argmax under mask MUST be in joy_allowed
    pred_sub = torch.argmax(masked_sub_logits[0]).item()
    assert pred_sub in joy_allowed

    # For sample 1 (ANXIETY): valid sub-emotions are TAXONOMY_MAP[2] = [4, 5] (WORRY, PANIC)
    anxiety_allowed = set(TAXONOMY_MAP[2])
    for sub_idx in range(11):
        val = masked_sub_logits[1, sub_idx].item()
        if sub_idx in anxiety_allowed:
            assert val != float("-inf"), f"Allowed sub-class {sub_idx} was masked!"
        else:
            assert val == float("-inf"), f"Forbidden sub-class {sub_idx} was not masked!"

    pred_sub_anx = torch.argmax(masked_sub_logits[1]).item()
    assert pred_sub_anx in anxiety_allowed


# ============================================================================
# 6. Explicit Melody Extractor Biomarkers
# ============================================================================

def test_melody_extractor_dimension_and_bounds():
    """Verify ExplicitMelodyExtractor extracts 32-dim physical acoustic features."""
    extractor = ExplicitMelodyExtractor(sr=16000)
    assert extractor.num_features == 32

    # Synthetic waveform: 2 seconds of pure 220Hz sine wave (pitched)
    sr = 16000
    duration = 2.0
    t = torch.linspace(0, duration, int(sr * duration))
    waveform = (0.5 * torch.sin(2 * np.pi * 220.0 * t)).unsqueeze(0)  # Shape [1, 32000]

    features = extractor.extract(waveform)
    assert features.shape == (1, 32)
    assert not torch.isnan(features).any()
    assert not torch.isinf(features).any()

    # Batch extraction test
    batch_waveforms = torch.randn(4, 32000)
    batch_features = extractor.extract(batch_waveforms)
    assert batch_features.shape == (4, 32)
    assert not torch.isnan(batch_features).any()


# ============================================================================
# 7. Stage 2 Hierarchical MER Model End-to-End
# ============================================================================

def test_stage2_hierarchical_mer_end_to_end():
    """Verify Stage2HierarchicalMERModel runs full forward and backward pass."""
    from src.stage2.hierarchical_mer_stage2 import Stage2HierarchicalMERModel

    model = Stage2HierarchicalMERModel(
        num_primary_classes=5,
        num_sub_classes=11,
        use_melody=True,
        dim_melody=32,
        device="cpu",
    )
    # Put synthetic backbone in audio encoder if needed
    model.audio_encoder._synthetic = True

    dummy_waveforms = torch.randn(2, 32000)
    dummy_texts = ["Tôi cảm thấy rất lo lắng hôm nay", "Mọi chuyện vẫn bình thường"]

    output = model(waveforms=dummy_waveforms, texts=dummy_texts, use_asr=False)

    assert output.primary_logits.shape == (2, 5)
    assert output.sub_logits.shape == (2, 11)
    assert output.masked_sub_logits.shape == (2, 11)
    assert output.z_fused.shape == (2, 928)  # 768 + 128 + 32
    assert output.p_melody.shape == (2, 32)

    # Verify backward gradient flow
    loss = output.primary_logits.sum() + output.masked_sub_logits.sum()
    loss.backward()

    # Check gradients on primary and sub heads
    has_primary_grad = any(p.grad is not None for p in model.primary_head.parameters())
    has_sub_grad = any(p.grad is not None for p in model.sub_head.parameters())
    has_fusion_grad = any(p.grad is not None for p in model.fusion.parameters())

    assert has_primary_grad, "Primary head received no gradients!"
    assert has_sub_grad, "Sub head received no gradients!"
    assert has_fusion_grad, "Fusion layer received no gradients!"

