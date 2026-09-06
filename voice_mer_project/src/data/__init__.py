"""
voice_mer_project/src/data/__init__.py

Data management & DataLoader for the Voice MER Pipeline.

Exports:
    VoiceJournalDataset  - Dataset class supporting .wav files and dummy tensors
"""

from .audio_dataset import VoiceJournalDataset

__all__ = [
    "VoiceJournalDataset",
]
