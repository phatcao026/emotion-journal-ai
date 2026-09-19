"""
text_preprocessor.py

Vietnamese text preprocessing and normalization for the text subsystem.

Pipeline:
    Raw Text → Unicode Normalization (NFC) → Noise/Artifact Removal
    → Whitespace Normalization → Sentence Segmentation → Word/Syllable Tokenization
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TextPreprocessingResult:
    """Holds the result of the text preprocessing pipeline.

    Attributes:
        original_text (str): Raw input text before processing.
        normalized_text (str): Text after Unicode normalization (NFC).
        cleaned_text (str): Text after artifact and whitespace cleaning.
        sentences (List[str]): List of segmented sentences.
        tokens (List[str]): List of tokens.
        num_tokens (int): Total number of tokens.
    """

    original_text: str = ""
    normalized_text: str = ""
    cleaned_text: str = ""
    sentences: List[str] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)
    num_tokens: int = 0


class VietnameseTextPreprocessor:
    """Normalizes and preprocesses Vietnamese text.

    Supports:
        - Unicode NFC normalization for Vietnamese diacritics.
        - Cleaning ASR transcription artifacts and non-printable characters.
        - Sentence boundary detection and segmentation.
        - Whitespace and punctuation normalization.
        - Syllable/word tokenization.

    Args:
        lowercase (bool): Whether to convert text to lowercase. Default: False.
        remove_punctuation (bool): Whether to strip punctuation entirely. Default: False.
        max_length (Optional[int]): Maximum character length to truncate. Default: None.
    """

    def __init__(
        self,
        lowercase: bool = False,
        remove_punctuation: bool = False,
        max_length: Optional[int] = None,
    ) -> None:
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.max_length = max_length

        # Regex for multi-space and whitespace collapsing
        self._whitespace_re = re.compile(r"\s+")
        # Regex for common ASR noise tokens (e.g., [applause], <noise>, [laughter])
        self._noise_re = re.compile(r"\[.*?\]|<.*?>")
        # Regex for sentence splitting (. ! ? or newline)
        self._sentence_re = re.compile(r"(?<=[.!?\n])\s+")
        # Regex for basic word-level token splitting
        self._token_re = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def normalize_unicode(self, text: str) -> str:
        """Apply Unicode NFC normalization to text.

        Ensures unified representation of Vietnamese vowels with tone marks.

        Args:
            text (str): Input raw text.

        Returns:
            str: Normalized text in Unicode NFC form.
        """
        if not text:
            return ""
        return unicodedata.normalize("NFC", text)

    def clean_text(self, text: str) -> str:
        """Clean noise artifacts, control characters, and excess whitespace.

        Args:
            text (str): Text to clean.

        Returns:
            str: Cleaned text string.
        """
        if not text:
            return ""

        # Remove ASR tags/noise markers
        cleaned = self._noise_re.sub(" ", text)

        # Replace non-breaking spaces and other control chars
        cleaned = cleaned.replace("\u00a0", " ").replace("\ufeff", "")

        if self.remove_punctuation:
            cleaned = re.sub(r"[^\w\s]", " ", cleaned, flags=re.UNICODE)

        # Collapse excess whitespace
        cleaned = self._whitespace_re.sub(" ", cleaned).strip()

        if self.lowercase:
            cleaned = cleaned.lower()

        if self.max_length is not None and len(cleaned) > self.max_length:
            cleaned = cleaned[: self.max_length].strip()

        return cleaned

    def segment_sentences(self, text: str) -> List[str]:
        """Segment text into individual sentences.

        Args:
            text (str): Input text.

        Returns:
            List[str]: List of sentence strings.
        """
        if not text or not text.strip():
            return []

        raw_sentences = self._sentence_re.split(text.strip())
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences if sentences else [text.strip()]

    def tokenize(self, text: str) -> List[str]:
        """Tokenize Vietnamese text into tokens.

        Args:
            text (str): Input text to tokenize.

        Returns:
            List[str]: List of token strings.
        """
        if not text:
            return []
        return self._token_re.findall(text)

    def process(self, text: str) -> TextPreprocessingResult:
        """Execute the complete preprocessing pipeline on a single text string.

        Args:
            text (str): Raw input text.

        Returns:
            TextPreprocessingResult: Processed result containing tokens and sentences.
        """
        if not isinstance(text, str):
            text = str(text)

        normalized = self.normalize_unicode(text)
        cleaned = self.clean_text(normalized)
        sentences = self.segment_sentences(cleaned)
        tokens = self.tokenize(cleaned)

        return TextPreprocessingResult(
            original_text=text,
            normalized_text=normalized,
            cleaned_text=cleaned,
            sentences=sentences,
            tokens=tokens,
            num_tokens=len(tokens),
        )

    def batch_process(self, texts: List[str]) -> List[TextPreprocessingResult]:
        """Process a batch of text strings.

        Args:
            texts (List[str]): Batch of raw text strings.

        Returns:
            List[TextPreprocessingResult]: List of processed results.
        """
        return [self.process(t) for t in texts]
