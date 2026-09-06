"""
voice_mer_project/src/audio/__init__.py

Acoustic processing subsystem (Voice-Only Pipeline).

Exports:
    VADPreprocessor          - VAD-based preprocessing and chunking
    EmotionalPauseAnalyzer   - Emotional pause feature extraction
    LoRAEmotion2Vec          - emotion2vec-base with LoRA adapter
    TemporalAttentionPooling - Self-attention pooling along the time axis
"""

from .vad_preprocessor import VADPreprocessor
from .pause_analyzer import EmotionalPauseAnalyzer
from .emotion2vec_lora import LoRAEmotion2Vec
from .attention_pooling import TemporalAttentionPooling

__all__ = [
    "VADPreprocessor",
    "EmotionalPauseAnalyzer",
    "LoRAEmotion2Vec",
    "TemporalAttentionPooling",
]
