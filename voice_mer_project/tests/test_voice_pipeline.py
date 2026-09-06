"""
test_voice_pipeline.py

Alias and runner for the Voice pipeline tests.
Delegates to tests in test_audio_pipeline.py.
"""

import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test_audio_pipeline import (
    TestTemporalAttentionPooling,
    TestPauseFeatureTensor,
    TestVoiceJournalDatasetDummy,
    TestAudioPipelineIntegration,
)

if __name__ == "__main__":
    print("=" * 60)
    print("  Voice Pipeline Tests - Voice MER Project")
    print("=" * 60)
    unittest.main(verbosity=2)
