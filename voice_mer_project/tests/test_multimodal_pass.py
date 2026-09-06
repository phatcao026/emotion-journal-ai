"""
test_multimodal_pass.py

Unit and integration tests for the Multimodal Fusion subsystem.

Test coverage:
    1. CrossModalGatedFusion:
       - Input: Z_audio (B, 768), Z_semantic (B, 768), P_pause (B, 4)
       - Output: Z_fused (B, 896), gate_audio (B, 768), gate_text (B, 768)
       - Property: gate values in [0, 1] (sigmoid output)
       - Error handling: raises ValueError when output_dim != dim_audio + dim_pause
    2. Dual Classification Heads:
       - primary_head: Z_fused (B, 896) -> logits (B, 5)
       - sub_head: Z_semantic (B, 768) -> logits (B, 10)
    3. MultiClassFocalLoss:
       - Parameter validation
       - Scalar output with reduction='mean'
       - Output shape (B,) with reduction='none'
       - Gradient flow via loss.backward()
       - Equivalence to CrossEntropyLoss when gamma=0
    4. End-to-End HierarchicalMERModel:
       - Forward pass with dummy audio tensors (B=2, T=16000*15)
       - Gradient propagation through both heads simultaneously
"""

from __future__ import annotations

import logging
import os
import sys
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.losses.focal_loss import MultiClassFocalLoss
from src.multimodal.gated_fusion import CrossModalGatedFusion
from src.multimodal.hierarchical_mer import HierarchicalMERModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BATCH_SIZE = 4
DIM_AUDIO = 768
DIM_SEMANTIC = 768
DIM_PAUSE_RAW = 4
DIM_PAUSE_PROJ = 128
DIM_FUSED = 896      # = DIM_AUDIO + DIM_PAUSE_PROJ = 768 + 128
NUM_PRIMARY = 5
NUM_SUB = 10


class TestCrossModalGatedFusion(unittest.TestCase):
    """Tests for CrossModalGatedFusion."""

    def setUp(self) -> None:
        self.fusion = CrossModalGatedFusion(
            dim_audio=DIM_AUDIO,
            dim_semantic=DIM_SEMANTIC,
            dim_pause=DIM_PAUSE_PROJ,
            pause_raw_dim=DIM_PAUSE_RAW,
            output_dim=DIM_FUSED,
            dropout=0.0,
        )
        self.fusion.eval()

        self.z_audio = torch.randn(BATCH_SIZE, DIM_AUDIO)
        self.z_semantic = torch.randn(BATCH_SIZE, DIM_SEMANTIC)
        self.p_pause = torch.randn(BATCH_SIZE, DIM_PAUSE_RAW)

    def test_output_shape(self) -> None:
        """Verify Z_fused shape = (B, 896)."""
        with torch.no_grad():
            z_fused, _ = self.fusion(self.z_audio, self.z_semantic, self.p_pause)
            self.assertEqual(z_fused.shape, (BATCH_SIZE, DIM_FUSED))

    def test_gate_values_in_range(self) -> None:
        """Verify gate values are in [0, 1] (sigmoid output)."""
        with torch.no_grad():
            _, (g_a, g_t) = self.fusion(self.z_audio, self.z_semantic, self.p_pause)
            self.assertTrue((g_a >= 0.0).all().item() and (g_a <= 1.0).all().item())
            self.assertTrue((g_t >= 0.0).all().item() and (g_t <= 1.0).all().item())

    def test_gate_shape(self) -> None:
        """Verify gate tensor shapes = (B, 768)."""
        with torch.no_grad():
            _, (g_a, g_t) = self.fusion(self.z_audio, self.z_semantic, self.p_pause)
            self.assertEqual(g_a.shape, (BATCH_SIZE, DIM_AUDIO))
            self.assertEqual(g_t.shape, (BATCH_SIZE, DIM_SEMANTIC))

    def test_invalid_output_dim_raises(self) -> None:
        """Verify ValueError is raised when output_dim != dim_audio + dim_pause."""
        with self.assertRaises(ValueError):
            CrossModalGatedFusion(
                dim_audio=768,
                dim_semantic=768,
                dim_pause=128,
                pause_raw_dim=4,
                output_dim=512,  # Incorrect: must be 896
            )

    def test_fusion_integration(self) -> None:
        """Integration test: forward pass with dummy inputs."""
        with torch.no_grad():
            z_fused, (g_a, g_t) = self.fusion(
                self.z_audio, self.z_semantic, self.p_pause
            )
            self.assertEqual(z_fused.shape, (BATCH_SIZE, DIM_FUSED))
            self.assertEqual(g_a.shape, (BATCH_SIZE, DIM_AUDIO))
            self.assertEqual(g_t.shape, (BATCH_SIZE, DIM_SEMANTIC))
            logger.info("PASS CrossModalGatedFusion forward pass: Z_fused=%s", tuple(z_fused.shape))


class TestDualClassificationHeads(unittest.TestCase):
    """Tests for the two parallel classification heads."""

    def setUp(self) -> None:
        self.primary_head = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(DIM_FUSED, NUM_PRIMARY),
        )
        self.sub_head = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(DIM_SEMANTIC, NUM_SUB),
        )
        self.z_fused = torch.randn(BATCH_SIZE, DIM_FUSED)
        self.z_semantic = torch.randn(BATCH_SIZE, DIM_SEMANTIC)

    def test_primary_head_shape(self) -> None:
        """Verify primary_head output shape = (B, 5)."""
        with torch.no_grad():
            logits = self.primary_head(self.z_fused)
            self.assertEqual(logits.shape, (BATCH_SIZE, NUM_PRIMARY))

    def test_sub_head_shape(self) -> None:
        """Verify sub_head output shape = (B, 10)."""
        with torch.no_grad():
            logits = self.sub_head(self.z_semantic)
            self.assertEqual(logits.shape, (BATCH_SIZE, NUM_SUB))

    def test_both_heads_parallel(self) -> None:
        """Verify both heads run in parallel without interfering."""
        with torch.no_grad():
            primary_logits = self.primary_head(self.z_fused)
            sub_logits = self.sub_head(self.z_semantic)
            self.assertEqual(primary_logits.shape, (BATCH_SIZE, NUM_PRIMARY))
            self.assertEqual(sub_logits.shape, (BATCH_SIZE, NUM_SUB))


class TestMultiClassFocalLoss(unittest.TestCase):
    """Tests for MultiClassFocalLoss."""

    def setUp(self) -> None:
        self.logits = torch.randn(BATCH_SIZE, NUM_PRIMARY, requires_grad=True)
        self.targets = torch.randint(0, NUM_PRIMARY, (BATCH_SIZE,))

    def test_focal_loss_init_invalid_gamma(self) -> None:
        """Verify ValueError is raised when gamma < 0."""
        with self.assertRaises(ValueError):
            MultiClassFocalLoss(gamma=-1.0)

    def test_focal_loss_init_invalid_reduction(self) -> None:
        """Verify ValueError is raised for an invalid reduction strategy."""
        with self.assertRaises(ValueError):
            MultiClassFocalLoss(gamma=2.0, reduction="invalid")

    def test_focal_loss_scalar_output(self) -> None:
        """Verify the output is a scalar when reduction='mean' or 'sum'."""
        loss_fn = MultiClassFocalLoss(gamma=2.0, reduction="mean")
        loss = loss_fn(self.logits, self.targets)
        self.assertEqual(loss.ndim, 0)
        self.assertGreater(loss.item(), 0.0)

    def test_focal_loss_none_output_shape(self) -> None:
        """Verify output shape = (B,) when reduction='none'."""
        loss_fn = MultiClassFocalLoss(gamma=2.0, reduction="none")
        loss = loss_fn(self.logits, self.targets)
        self.assertEqual(loss.shape, (BATCH_SIZE,))

    def test_focal_loss_backward(self) -> None:
        """Verify gradient flow: loss.backward() completes without errors."""
        loss_fn = MultiClassFocalLoss(gamma=2.0, reduction="mean")
        loss = loss_fn(self.logits, self.targets)
        loss.backward()
        self.assertIsNotNone(self.logits.grad)

    def test_gamma_zero_equals_cross_entropy(self) -> None:
        """Verify FocalLoss(gamma=0) equals standard CrossEntropyLoss."""
        focal_fn = MultiClassFocalLoss(gamma=0.0, reduction="mean")
        focal_loss = focal_fn(self.logits.detach(), self.targets)
        ce_loss = F.cross_entropy(self.logits.detach(), self.targets)
        torch.testing.assert_close(focal_loss, ce_loss, rtol=1e-4, atol=1e-4)


class TestMultimodalEndToEnd(unittest.TestCase):
    """Integration tests for the full multimodal pipeline with dummy tensors."""

    def test_fusion_to_dual_head_pipeline(self) -> None:
        """Pipeline: [Z_audio, Z_semantic, P_pause] -> Z_fused -> [primary_logits, sub_logits]."""
        fusion = CrossModalGatedFusion(dropout=0.0)
        fusion.eval()

        primary_head = nn.Linear(DIM_FUSED, NUM_PRIMARY)
        sub_head = nn.Linear(DIM_SEMANTIC, NUM_SUB)

        z_audio = torch.randn(BATCH_SIZE, DIM_AUDIO)
        z_semantic = torch.randn(BATCH_SIZE, DIM_SEMANTIC)
        p_pause = torch.randn(BATCH_SIZE, DIM_PAUSE_RAW)

        with torch.no_grad():
            z_fused, _ = fusion(z_audio, z_semantic, p_pause)
            primary_logits = primary_head(z_fused)
            sub_logits = sub_head(z_semantic)

            self.assertEqual(z_fused.shape, (BATCH_SIZE, DIM_FUSED))
            self.assertEqual(primary_logits.shape, (BATCH_SIZE, NUM_PRIMARY))
            self.assertEqual(sub_logits.shape, (BATCH_SIZE, NUM_SUB))

    def test_combined_loss_backward(self) -> None:
        """Verify: total_loss = 0.7 * primary_loss + 0.3 * sub_loss -> backward()."""
        primary_head = nn.Linear(DIM_FUSED, NUM_PRIMARY)
        sub_head = nn.Linear(DIM_SEMANTIC, NUM_SUB)
        loss_fn = MultiClassFocalLoss(gamma=2.0, reduction="mean")

        z_fused = torch.randn(BATCH_SIZE, DIM_FUSED, requires_grad=True)
        z_semantic = torch.randn(BATCH_SIZE, DIM_SEMANTIC, requires_grad=True)
        primary_targets = torch.randint(0, NUM_PRIMARY, (BATCH_SIZE,))
        sub_targets = torch.randint(0, NUM_SUB, (BATCH_SIZE,))

        primary_logits = primary_head(z_fused)
        sub_logits = sub_head(z_semantic)
        primary_loss = loss_fn(primary_logits, primary_targets)
        sub_loss = loss_fn(sub_logits, sub_targets)
        total_loss = 0.7 * primary_loss + 0.3 * sub_loss
        total_loss.backward()

        self.assertIsNotNone(z_fused.grad)
        self.assertIsNotNone(z_semantic.grad)

    def test_hierarchical_mer_model_dummy_audio_backward(self) -> None:
        """SPEC TEST: Run HierarchicalMERModel on dummy audio (B=2, T=16000*15) with gradient check."""
        model = HierarchicalMERModel(
            num_primary_classes=NUM_PRIMARY,
            num_sub_classes=NUM_SUB,
            device="cpu",
        )
        model.train()

        # Dummy audio tensor (B=2, T=16000*15 = 240000)
        b = 2
        t = 16000 * 15
        dummy_audio = torch.randn(b, t, requires_grad=False)
        dummy_texts = [
            "Hôm nay tôi cảm thấy xúc động và muốn ghi lại nhật ký này.",
            "Một ngày đầy ắp những cảm xúc phức tạp và trăn trở.",
        ]

        primary_targets = torch.tensor([0, 1], dtype=torch.long)
        sub_targets = torch.tensor([2, 5], dtype=torch.long)

        loss_primary_fn = MultiClassFocalLoss(gamma=2.0)
        loss_sub_fn = MultiClassFocalLoss(gamma=2.0)

        output = model(waveforms=dummy_audio, texts=dummy_texts, use_asr=False)

        self.assertEqual(output.primary_logits.shape, (b, NUM_PRIMARY))
        self.assertEqual(output.sub_logits.shape, (b, NUM_SUB))
        self.assertEqual(output.z_fused.shape, (b, DIM_FUSED))

        # Check attention weights sum to 1.0
        attn_sums = output.attention_weights.sum(dim=-1)
        torch.testing.assert_close(attn_sums, torch.ones_like(attn_sums), rtol=1e-4, atol=1e-4)

        # Compute dual loss and backpropagate
        l_pri = loss_primary_fn(output.primary_logits, primary_targets)
        l_sub = loss_sub_fn(output.sub_logits, sub_targets)
        total_loss = 0.7 * l_pri + 0.3 * l_sub

        total_loss.backward()

        # Verify gradient flow into heads and fusion
        for param in model.primary_head.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
        for param in model.sub_head.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
        for param in model.fusion.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)

        logger.info("PASS HierarchicalMERModel end-to-end forward and backward with dummy audio: OK")


if __name__ == "__main__":
    print("=" * 60)
    print("  Multimodal Pipeline Tests - Voice MER Project")
    print("=" * 60)
    unittest.main(verbosity=2)
