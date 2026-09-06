"""
voice_mer_project/src/text/__init__.py

Text processing subsystem (Text-Only Pipeline).
STUB INTERFACE – Reserved for teammate implementation.

Exports:
    VietnameseTextPreprocessor  - Vietnamese text normalization
    PhoBERTEncoder              - Extracts z_semantic (d=768) using PhoBERT
"""

from .text_preprocessor import VietnameseTextPreprocessor
from .text_encoder import PhoBERTEncoder

__all__ = [
    "VietnameseTextPreprocessor",
    "PhoBERTEncoder",
]
