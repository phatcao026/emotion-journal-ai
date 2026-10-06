"""
Tests for scripts/annotation_tool/annotation_manager.py
"""

import os
import shutil
import tempfile
import pytest
from scripts.annotation_tool.annotation_manager import AnnotationManager

@pytest.fixture
def temp_annot_dir():
    d = tempfile.mkdtemp(prefix="test_annot_")
    yield d
    shutil.rmtree(d, ignore_errors=True)

def test_annotation_manager_loading(temp_annot_dir):
    manager = AnnotationManager(
        crawled_meta_path="data/crawled/metadata.jsonl",
        annotation_dir=temp_annot_dir,
        taxonomy_path="data/taxonomy.json"
    )
    assert len(manager.samples) >= 100
    assert "JOY" in manager.taxonomy["primary_emotions"].values()

def test_annotation_manager_filtering(temp_annot_dir):
    manager = AnnotationManager(
        crawled_meta_path="data/crawled/metadata.jsonl",
        annotation_dir=temp_annot_dir,
        taxonomy_path="data/taxonomy.json"
    )
    text_samples = manager.get_filtered_samples(modality_filter="text")
    voice_samples = manager.get_filtered_samples(modality_filter="voice")
    assert len(text_samples) > 0
    assert len(voice_samples) > 0
    for s in text_samples:
        assert s["modality"] == "text"
    for s in voice_samples:
        assert s["modality"] == "voice"

def test_annotation_save_and_stats(temp_annot_dir):
    manager = AnnotationManager(
        crawled_meta_path="data/crawled/metadata.jsonl",
        annotation_dir=temp_annot_dir,
        taxonomy_path="data/taxonomy.json"
    )
    sample = manager.samples[0]
    sid = sample["sample_id"]

    rec = manager.save_annotation(
        sample_id=sid,
        annotator_id="test_user",
        primary_label="SADNESS",
        sub_label="LONELINESS",
        valence=-0.6,
        arousal=0.3,
        congruence="congruent",
        crisis_level=1,
        notes="Test note"
    )
    assert rec["sample_id"] == sid
    assert rec["primary_label"] == "SADNESS"

    # Verify retrieval
    ann = manager.get_existing_annotation(sid, "test_user")
    assert ann is not None
    assert ann["sub_label"] == "LONELINESS"

    # Verify stats
    stats = manager.get_statistics("test_user")
    assert stats["total_annotated"] == 1
    assert stats["primary_distribution"]["SADNESS"] == 1
    assert stats["avg_valence"] == -0.6
    assert os.path.exists(os.path.join(temp_annot_dir, "annotations.csv"))
