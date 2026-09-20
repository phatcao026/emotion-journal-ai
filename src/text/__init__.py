"""
src/text

Text subsystem for Vietnamese Emotion Recognition in Personal Journals.
"""

from src.text.dynamics import (
    RUSSELL_COORDINATES,
    JournalDynamicsReport,
    JournalEmotionDynamicsAnalyzer,
    SentenceEmotionState,
)
from src.text.text_dataset import (
    SUB_TO_PRIMARY,
    TAXONOMY_5_PRIMARY,
    TAXONOMY_11_LABELS,
    MultiLabelTextCollator,
    TextJournalDataset,
    load_taxonomy,
)
from src.text.text_encoder import (
    PhoBERTEncoder,
    TextEmotionClassifier,
    ViSoBERTEncoder,
    create_text_encoder,
)
from src.text.text_preprocessor import (
    TextPreprocessingResult,
    VietnameseTextPreprocessor,
    make_preprocessor,
)

__all__ = [
    "VietnameseTextPreprocessor",
    "TextPreprocessingResult",
    "make_preprocessor",
    "PhoBERTEncoder",
    "ViSoBERTEncoder",
    "TextEmotionClassifier",
    "create_text_encoder",
    "TextJournalDataset",
    "MultiLabelTextCollator",
    "TAXONOMY_11_LABELS",
    "TAXONOMY_5_PRIMARY",
    "SUB_TO_PRIMARY",
    "load_taxonomy",
    "JournalEmotionDynamicsAnalyzer",
    "RUSSELL_COORDINATES",
    "JournalDynamicsReport",
    "SentenceEmotionState",
]
