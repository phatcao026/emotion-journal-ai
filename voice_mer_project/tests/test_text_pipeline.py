"""
test_text_pipeline.py

Unit and integration tests for the Text subsystem.

Test coverage:
    1. VietnameseTextPreprocessor:
       - Unicode NFC normalization
       - Artifact and whitespace cleaning
       - Sentence segmentation
       - Tokenization
    2. PhoBERTEncoder:
       - Output shape (B, 768) for batch input
       - Pooling strategies: 'cls', 'mean', 'max'
       - Gradient propagation through backward()
"""

from __future__ import annotations

import logging
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.text.text_encoder import PhoBERTEncoder
from src.text.text_preprocessor import VietnameseTextPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestVietnameseTextPreprocessor(unittest.TestCase):
    """Tests for VietnameseTextPreprocessor."""

    def setUp(self) -> None:
        self.preprocessor = VietnameseTextPreprocessor()

    def test_unicode_nfc_normalization(self) -> None:
        """Verify decomposition vs composition normalization."""
        # Decomposed 'à' (a + combining grave accent \u0300) -> Precomposed 'à' (\u00e0)
        decomposed = "a\u0300"
        normalized = self.preprocessor.normalize_unicode(decomposed)
        self.assertEqual(normalized, "\u00e0")

    def test_clean_text_noise_removal(self) -> None:
        """Verify removal of ASR noise tokens and multiple spaces."""
        raw = "Hôm nay [tiếng thở dài] tôi rất buồn...   nhiều chuyện."
        cleaned = self.preprocessor.clean_text(raw)
        self.assertNotIn("[tiếng thở dài]", cleaned)
        self.assertNotIn("   ", cleaned)

    def test_sentence_segmentation(self) -> None:
        """Verify splitting by sentence terminators."""
        text = "Tôi thấy vui. Ngày hôm nay rất tuyệt! Nhưng có một chút mệt."
        sentences = self.preprocessor.segment_sentences(text)
        self.assertEqual(len(sentences), 3)

    def test_tokenize(self) -> None:
        """Verify tokenization."""
        text = "Nhật ký hôm nay."
        tokens = self.preprocessor.tokenize(text)
        self.assertIn("Nhật", tokens)
        self.assertIn("ký", tokens)


class TestPhoBERTEncoder(unittest.TestCase):
    """Tests for PhoBERTEncoder."""

    def setUp(self) -> None:
        self.encoder = PhoBERTEncoder(pooling_strategy="cls")
        self.texts = [
            "Hôm nay tôi cảm thấy rất biết ơn và tự hào.",
            "Cảm giác thất vọng và cô đơn bủa vây.",
        ]

    def test_output_shape_cls(self) -> None:
        """Verify output shape is (B, 768)."""
        z_semantic = self.encoder(texts=self.texts)
        self.assertEqual(z_semantic.shape, (len(self.texts), 768))

    def test_output_shape_mean_pooling(self) -> None:
        """Verify mean pooling produces (B, 768)."""
        encoder_mean = PhoBERTEncoder(pooling_strategy="mean")
        z_semantic = encoder_mean(texts=self.texts)
        self.assertEqual(z_semantic.shape, (len(self.texts), 768))

    def test_output_shape_max_pooling(self) -> None:
        """Verify max pooling produces (B, 768)."""
        encoder_max = PhoBERTEncoder(pooling_strategy="max")
        z_semantic = encoder_max(texts=self.texts)
        self.assertEqual(z_semantic.shape, (len(self.texts), 768))

    def test_gradient_flow(self) -> None:
        """Verify loss.backward() flows gradients into trainable parameters."""
        z_semantic = self.encoder(texts=self.texts)
        loss = z_semantic.sum()
        loss.backward()

        has_grad = any(p.grad is not None for p in self.encoder.parameters())
        self.assertTrue(has_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("  Text Pipeline Tests - Voice MER Project")
    print("=" * 60)
    unittest.main(verbosity=2)
