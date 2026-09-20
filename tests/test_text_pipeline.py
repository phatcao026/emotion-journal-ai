"""
test_text_pipeline.py

Unit and integration tests for the Text subsystem.

Test coverage:
    1. VietnameseTextPreprocessor:
       - Unicode NFC normalization
       - Artifact and whitespace cleaning
       - Sentence segmentation
       - Tokenization
       - Teencode replacement
    2. PhoBERTEncoder & ViSoBERTEncoder:
       - Output shape (B, 768) for batch input
       - Pooling strategies: 'cls', 'mean', 'max'
       - Gradient propagation through backward()
    3. TextEmotionClassifier:
       - Multi-label classification head output shapes (B, 11) and (B, 5)
       - Probability range in [0, 1]
    4. TextJournalDataset & MultiLabelTextCollator:
       - Sample parsing, label vector generation, collator batching
    5. Multi-label Metrics:
       - Macro-F1, Micro-F1, Exact Match calculation, threshold tuning
    6. JournalEmotionDynamicsAnalyzer:
       - Russell circumplex (V, A) calculation, trajectory pattern detection, Peak-End rule
"""

from __future__ import annotations

import logging
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.common.metrics import compute_multilabel_metrics, tune_multilabel_thresholds
from src.text.dynamics import JournalEmotionDynamicsAnalyzer, RUSSELL_COORDINATES
from src.text.text_dataset import (
    TAXONOMY_11_LABELS,
    MultiLabelTextCollator,
    TextJournalDataset,
)
from src.text.text_encoder import (
    PhoBERTEncoder,
    TextEmotionClassifier,
    ViSoBERTEncoder,
    create_text_encoder,
)
from src.text.text_preprocessor import VietnameseTextPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestVietnameseTextPreprocessor(unittest.TestCase):
    """Tests for VietnameseTextPreprocessor."""

    def setUp(self) -> None:
        self.preprocessor = VietnameseTextPreprocessor(normalize_teencode=True)

    def test_unicode_nfc_normalization(self) -> None:
        """Verify decomposition vs composition normalization."""
        decomposed = "a\u0300"
        normalized = self.preprocessor.normalize_unicode(decomposed)
        self.assertEqual(normalized, "\u00e0")

    def test_clean_text_noise_removal(self) -> None:
        """Verify removal of ASR noise tokens and multiple spaces."""
        raw = "Hôm nay [tiếng thở dài] tôi rất buồn...   nhiều chuyện."
        cleaned = self.preprocessor.clean_text(raw)
        self.assertNotIn("[tiếng thở dài]", cleaned)
        self.assertNotIn("   ", cleaned)

    def test_teencode_replacement(self) -> None:
        """Verify normalization of Vietnamese teencode words."""
        raw = "hqua mik ko bt gì hết trơn á"
        cleaned = self.preprocessor.clean_text(raw)
        self.assertIn("hôm qua", cleaned)
        self.assertIn("mình", cleaned)
        self.assertIn("không", cleaned)
        self.assertIn("biết", cleaned)

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


class TestViSoBERTAndClassifier(unittest.TestCase):
    """Tests for ViSoBERTEncoder and TextEmotionClassifier."""

    def setUp(self) -> None:
        self.encoder = ViSoBERTEncoder(pooling_strategy="cls")
        self.classifier = TextEmotionClassifier(
            encoder=self.encoder,
            num_classes=11,
            num_primary_classes=5,
            dropout=0.2,
        )
        self.texts = [
            "Hôm nay tôi rất hạnh phúc và bình yên.",
            "Tôi cảm thấy lo lắng và bất an.",
        ]

    def test_visobert_shape(self) -> None:
        feat = self.encoder(texts=self.texts)
        self.assertEqual(feat.shape, (2, 768))

    def test_classifier_output(self) -> None:
        out = self.classifier(texts=self.texts)
        self.assertIn("logits", out)
        self.assertIn("probs", out)
        self.assertIn("primary_logits", out)

        self.assertEqual(out["logits"].shape, (2, 11))
        self.assertEqual(out["primary_logits"].shape, (2, 5))

        probs = out["probs"]
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())


class TestDatasetAndCollator(unittest.TestCase):
    """Tests for TextJournalDataset and MultiLabelTextCollator."""

    def test_dataset_loading(self) -> None:
        samples = [
            {"id": "1", "text": "Hôm nay tuyệt vời lắm!", "labels": ["JOY", "CONNECTION"]},
            {"id": "2", "text": "Buồn bã và cô đơn.", "labels": ["SADNESS", "LONELINESS"]},
        ]
        ds = TextJournalDataset(samples)
        self.assertEqual(len(ds), 2)

        item = ds[0]
        self.assertEqual(item["sub_targets"].shape, (11,))
        self.assertEqual(item["sub_targets"][0], 1.0)  # JOY is index 0

        collator = MultiLabelTextCollator()
        batch = collator([ds[0], ds[1]])
        self.assertEqual(batch["sub_targets"].shape, (2, 11))
        self.assertEqual(len(batch["texts"]), 2)


class TestMultilabelMetrics(unittest.TestCase):
    """Tests for multi-label metrics and threshold tuning."""

    def test_metrics_computation(self) -> None:
        y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        y_prob = np.array([[0.8, 0.1, 0.9], [0.2, 0.7, 0.1], [0.6, 0.6, 0.2]])

        report = compute_multilabel_metrics(y_true, y_prob, thresholds=0.5)
        self.assertIn("macro_f1", report)
        self.assertIn("micro_f1", report)
        self.assertIn("exact_match", report)
        self.assertGreaterEqual(report["macro_f1"], 0.5)

    def test_tune_thresholds(self) -> None:
        y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        y_prob = np.array([[0.7, 0.2], [0.8, 0.3], [0.1, 0.9], [0.2, 0.85]])
        thresh = tune_multilabel_thresholds(y_true, y_prob)
        self.assertEqual(len(thresh), 2)


class TestJournalEmotionDynamics(unittest.TestCase):
    """Tests for Russell's circumplex mapping and document dynamics."""

    def setUp(self) -> None:
        self.analyzer = JournalEmotionDynamicsAnalyzer()

    def test_sentence_va(self) -> None:
        v, a = self.analyzer.compute_sentence_va({"JOY": 0.9, "CALM": 0.1})
        self.assertGreater(v, 0.5)  # Joy has high valence

        v_sad, a_sad = self.analyzer.compute_sentence_va({"SADNESS": 0.9})
        self.assertLess(v_sad, -0.5)

    def test_document_dynamics(self) -> None:
        text = "Hôm nay bắt đầu thật tệ và buồn. Tôi cảm thấy rất cô đơn. Nhưng cuối ngày bạn bè đã đến động viên. Giờ tôi thấy ấm lòng và bình yên hơn nhiều."
        mock_probs = [
            {"SADNESS": 0.8},
            {"LONELINESS": 0.9},
            {"CONNECTION": 0.85, "HOPE": 0.6},
            {"CALM": 0.9, "JOY": 0.7},
        ]
        report = self.analyzer.analyze_document(text, sentence_probabilities=mock_probs)
        self.assertEqual(report.total_sentences, 4)
        self.assertIn("Rebound", report.trajectory_pattern)
        self.assertIsNotNone(report.empathic_reflection)


if __name__ == "__main__":
    print("=" * 60)
    print("  Text Pipeline Tests - Voice & Text MER Project")
    print("=" * 60)
    unittest.main(verbosity=2)
