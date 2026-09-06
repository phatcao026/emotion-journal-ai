"""
voice_mer_project/src/multimodal/__init__.py

Multimodal Fusion subsystem.

Exports:
    PhoWhisperTranscriber    - ASR: converts audio to Vietnamese text
    CrossModalGatedFusion    - Gated fusion combining three feature streams
    HierarchicalMERModel     - Full model with Dual Classification Heads
"""

from .asr_transcriber import PhoWhisperTranscriber
from .gated_fusion import CrossModalGatedFusion
from .hierarchical_mer import HierarchicalMERModel

__all__ = [
    "PhoWhisperTranscriber",
    "CrossModalGatedFusion",
    "HierarchicalMERModel",
]
