"""
test_audio_pipeline.py

Unit and integration tests for the acoustic subsystem (Voice-Only pipeline).

Test coverage:
    1. TemporalAttentionPooling:
       - Output shapes (B, 768) and (B, N)
       - Attention weights sum to 1.0 along sequence dimension
       - Non-negative attention weights (softmax output)
       - Padding mask handling
       - Single-frame edge case (N=1)
       - Output projection when output_dim != input_dim
    2. PauseFeatures:
       - to_tensor() shape (4,), dtype float32, value correspondence
    3. VoiceJournalDataset (dummy mode):
       - len(dataset) matches dummy_size
       - Sample waveform shape (1, 96000)
       - Primary labels in [0, 4]
       - collate_fn padding and batching
    4. AudioPipelineIntegration:
       - End-to-end forward pass with TemporalAttentionPooling
       - DataLoader collation batching
"""

from __future__ import annotations

import logging
import os
import sys
import unittest
from typing import List

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.audio.attention_pooling import TemporalAttentionPooling
from src.audio.pause_analyzer import EmotionalPauseAnalyzer, PauseFeatures
from src.data.audio_dataset import PrimaryEmotion, VoiceJournalDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BATCH_SIZE = 4
SEQ_LEN = 20           # N = number of frames
FEATURE_DIM = 768      # d = emotion2vec feature dimensionality
SAMPLE_RATE = 16000
CHUNK_DURATION_S = 6.0
NUM_SAMPLES = int(CHUNK_DURATION_S * SAMPLE_RATE)   # 96000 samples


class TestTemporalAttentionPooling(unittest.TestCase):
    """Tests for TemporalAttentionPooling."""

    def setUp(self) -> None:
        self.pooling = TemporalAttentionPooling(
            input_dim=FEATURE_DIM,
            hidden_dim=256,
            output_dim=FEATURE_DIM,
            dropout=0.0,
        )
        self.pooling.eval()
        self.x = torch.randn(BATCH_SIZE, SEQ_LEN, FEATURE_DIM)

    def test_output_shape(self) -> None:
        """Verify output shapes: (B, 768) and (B, N)."""
        with torch.no_grad():
            z_audio, alpha = self.pooling(self.x)
            self.assertEqual(z_audio.shape, (BATCH_SIZE, FEATURE_DIM))
            self.assertEqual(alpha.shape, (BATCH_SIZE, SEQ_LEN))

    def test_attention_weights_sum_to_one(self) -> None:
        """Verify that alpha.sum(dim=1) ≈ 1.0 for every sample."""
        with torch.no_grad():
            _, alpha = self.pooling(self.x)
            sums = alpha.sum(dim=1)
            torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-5, atol=1e-5)

    def test_attention_weights_non_negative(self) -> None:
        """Verify that all attention weights are non-negative (softmax output)."""
        with torch.no_grad():
            _, alpha = self.pooling(self.x)
            self.assertTrue((alpha >= 0.0).all().item())

    def test_padding_mask(self) -> None:
        """Verify that the padding mask correctly drives masked positions to ~0."""
        mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.bool)
        mask[:, 10:] = False  # Mask out frames 10..19

        with torch.no_grad():
            _, alpha = self.pooling(self.x, mask=mask)
            # Masked positions should be very close to 0
            self.assertTrue((alpha[:, 10:] < 1e-4).all().item())
            # Valid positions should still sum to 1.0
            valid_sums = alpha[:, :10].sum(dim=1)
            torch.testing.assert_close(valid_sums, torch.ones_like(valid_sums), rtol=1e-4, atol=1e-4)

    def test_single_frame(self) -> None:
        """Verify the edge case of N=1 frame."""
        single_x = torch.randn(BATCH_SIZE, 1, FEATURE_DIM)
        with torch.no_grad():
            z_audio, alpha = self.pooling(single_x)
            self.assertEqual(z_audio.shape, (BATCH_SIZE, FEATURE_DIM))
            self.assertEqual(alpha.shape, (BATCH_SIZE, 1))
            torch.testing.assert_close(alpha, torch.ones_like(alpha))

    def test_output_dim_mismatch(self) -> None:
        """Verify the projection layer is active when output_dim != input_dim."""
        proj_pooling = TemporalAttentionPooling(
            input_dim=FEATURE_DIM,
            hidden_dim=256,
            output_dim=512,
            dropout=0.0,
        )
        proj_pooling.eval()
        with torch.no_grad():
            z_audio, alpha = proj_pooling(self.x)
            self.assertEqual(z_audio.shape, (BATCH_SIZE, 512))
            self.assertEqual(alpha.shape, (BATCH_SIZE, SEQ_LEN))


class TestPauseFeatureTensor(unittest.TestCase):
    """Tests for PauseFeatures.to_tensor()."""

    def setUp(self) -> None:
        self.feats = PauseFeatures(
            speech_pause_ratio=0.8,
            mean_pause_duration_s=0.6,
            pause_frequency=12.0,
            energy_drift_db=-2.5,
        )

    def test_tensor_shape(self) -> None:
        """Verify output shape = (4,)."""
        tensor = self.feats.to_tensor()
        self.assertEqual(tensor.shape, (4,))

    def test_tensor_dtype(self) -> None:
        """Verify output dtype = float32."""
        tensor = self.feats.to_tensor()
        self.assertEqual(tensor.dtype, torch.float32)

    def test_tensor_values(self) -> None:
        """Verify tensor values match the PauseFeatures attributes."""
        tensor = self.feats.to_tensor()
        expected = torch.tensor([0.8, 0.6, 12.0, -2.5], dtype=torch.float32)
        torch.testing.assert_close(tensor, expected)


class TestVoiceJournalDatasetDummy(unittest.TestCase):
    """Tests for VoiceJournalDataset in dummy mode."""

    def setUp(self) -> None:
        self.dummy_size = 20
        self.dataset = VoiceJournalDataset(
            dummy_mode=True,
            dummy_size=self.dummy_size,
            dummy_duration_s=CHUNK_DURATION_S,
        )

    def test_dataset_length(self) -> None:
        """Verify len(dataset) == dummy_size."""
        self.assertEqual(len(self.dataset), self.dummy_size)

    def test_sample_waveform_shape(self) -> None:
        """Verify waveform shape = (1, 96000)."""
        sample = self.dataset[0]
        self.assertEqual(sample["waveform"].shape, (1, NUM_SAMPLES))

    def test_sample_label_valid(self) -> None:
        """Verify labels are in [0, 4]."""
        for i in range(min(5, len(self.dataset))):
            label = self.dataset[i]["primary_label"].item()
            self.assertIn(label, list(range(len(PrimaryEmotion))))

    def test_collate_fn_padding(self) -> None:
        """Verify that collate_fn pads waveforms to the same length."""
        batch = [self.dataset[i] for i in range(4)]
        collated = VoiceJournalDataset.collate_fn(batch)
        self.assertEqual(collated["waveforms"].shape, (4, 1, NUM_SAMPLES))
        self.assertEqual(collated["primary_labels"].shape, (4,))


class TestAudioPipelineIntegration(unittest.TestCase):
    """Integration tests for the full audio pipeline."""

    def test_attention_pooling_full_forward(self) -> None:
        """Verify: dummy frame features (B, N, 768) -> Z_audio (B, 768)."""
        x = torch.randn(BATCH_SIZE, SEQ_LEN, FEATURE_DIM)
        pooling = TemporalAttentionPooling(
            input_dim=FEATURE_DIM, hidden_dim=256, output_dim=FEATURE_DIM, dropout=0.0
        )
        pooling.eval()
        with torch.no_grad():
            z_audio, alpha = pooling(x)
            self.assertEqual(z_audio.shape, (BATCH_SIZE, FEATURE_DIM))
            self.assertEqual(alpha.shape, (BATCH_SIZE, SEQ_LEN))
            logger.info("PASS Attention pooling forward pass: Z_audio=%s", tuple(z_audio.shape))

    def test_dummy_dataset_collate(self) -> None:
        """Verify: dummy dataset -> collated batch."""
        from torch.utils.data import DataLoader
        dataset = VoiceJournalDataset(dummy_mode=True, dummy_size=8, dummy_duration_s=6.0)
        loader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=VoiceJournalDataset.collate_fn,
        )
        batch = next(iter(loader))
        self.assertIn("waveforms", batch)
        self.assertIn("primary_labels", batch)
        self.assertEqual(batch["waveforms"].shape[0], 4)
        logger.info("PASS Dataset collate: waveforms shape=%s", tuple(batch["waveforms"].shape))


if __name__ == "__main__":
    print("=" * 60)
    print("  Audio Pipeline Tests - Voice MER Project")
    print("=" * 60)
    unittest.main(verbosity=2)
