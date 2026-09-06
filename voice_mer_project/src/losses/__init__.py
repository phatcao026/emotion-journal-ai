"""
voice_mer_project/src/losses/__init__.py

Loss functions for the Voice MER Pipeline.

Exports:
    MultiClassFocalLoss  - Focal Loss with gamma=2.0 for emotion classification
"""

from .focal_loss import MultiClassFocalLoss

__all__ = [
    "MultiClassFocalLoss",
]
