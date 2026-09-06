"""
text_encoder.py

Extracts semantic features z_semantic in R^(B x 768) from Vietnamese text
using PhoBERT-base.

Architecture:
    Cleaned Text -> PhoBERT-base tokenizer -> PhoBERT-base encoder
    -> [CLS] token pooling / Mean pooling -> z_semantic in R^(B x 768)

Provides a synthetic fallback mode to ensure reliable execution in offline/CI
environments without requiring live Hugging Face model downloads.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SyntheticPhoBERTBackbone(nn.Module):
    """Lightweight synthetic text encoder used when pretrained weights are unavailable.

    Provides end-to-end differentiability and produces outputs of shape (B, 768).
    """

    def __init__(self, vocab_size: int = 10000, hidden_dim: int = 768) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling_strategy: str = "cls",
    ) -> torch.Tensor:
        """Forward pass through synthetic text backbone.

        Args:
            input_ids (torch.Tensor): (B, L) integer token IDs.
            attention_mask (Optional[torch.Tensor]): (B, L) binary mask.
            pooling_strategy (str): 'cls', 'mean', or 'max'.

        Returns:
            torch.Tensor: (B, hidden_dim).
        """
        x = self.embedding(input_ids)  # (B, L, hidden_dim)

        if pooling_strategy == "cls":
            pooled = x[:, 0, :]
        elif pooling_strategy == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = x.mean(dim=1)
        elif pooling_strategy == "max":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).bool()
                x_masked = x.masked_fill(~mask, -1e9)
                pooled = x_masked.max(dim=1).values
            else:
                pooled = x.max(dim=1).values
        else:
            pooled = x[:, 0, :]

        return self.proj(pooled)


class PhoBERTEncoder(nn.Module):
    """Extracts semantic features from Vietnamese text using PhoBERT.

    Loads PhoBERT-base (vinai/phobert-base-v2) and extracts a sentence-level
    vector z_semantic in R^768 representing the semantic content of the input.

    Args:
        model_id (str): HuggingFace model ID. Default: "vinai/phobert-base-v2".
        pooling_strategy (str): Pooling strategy ("cls", "mean", "max"). Default: "cls".
        max_length (int): Maximum sequence token length. Default: 256.
        freeze (bool): Freeze PhoBERT weights if True. Default: False.
        device (str): Device to run the model on. Default: "cpu".
        use_synthetic_fallback (bool): If True, permit synthetic backbone when
            offline or checkpoints are unavailable. Default: True.

    Outputs:
        z_semantic (torch.Tensor): shape (B, 768).
    """

    FEATURE_DIM: int = 768
    DEFAULT_MODEL_ID: str = "vinai/phobert-base-v2"

    def __init__(
        self,
        model_id: str = "vinai/phobert-base-v2",
        pooling_strategy: Literal["cls", "mean", "max"] = "cls",
        max_length: int = 256,
        freeze: bool = False,
        device: str = "cpu",
        use_synthetic_fallback: bool = True,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.pooling_strategy = pooling_strategy
        self.max_length = max_length
        self.freeze = freeze
        self.device_str = device
        self.feature_dim = self.FEATURE_DIM
        self.use_synthetic_fallback = use_synthetic_fallback

        self.tokenizer = None
        self.bert_model: Optional[nn.Module] = None
        self.is_synthetic: bool = False

        # Pre-initialize synthetic backbone so model is immediately usable in tests
        self._synthetic_backbone = SyntheticPhoBERTBackbone(
            vocab_size=10000, hidden_dim=self.FEATURE_DIM
        )

        logger.info(
            "PhoBERTEncoder initialized: model=%s, pooling=%s, device=%s",
            model_id,
            pooling_strategy,
            device,
        )

    def load_pretrained(self) -> None:
        """Load the PhoBERT tokenizer and model from HuggingFace Hub.

        Falls back gracefully to synthetic backbone if offline or unavailable.
        """
        try:
            from transformers import AutoModel, AutoTokenizer

            logger.info("Loading PhoBERT tokenizer from %s", self.model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            logger.info("Loading PhoBERT model from %s", self.model_id)
            self.bert_model = AutoModel.from_pretrained(self.model_id)

            if self.freeze:
                for param in self.bert_model.parameters():
                    param.requires_grad = False
                logger.info("PhoBERT weights frozen.")

            self.bert_model.to(torch.device(self.device_str))
            self.is_synthetic = False
            logger.info("PhoBERT pretrained model loaded successfully.")

        except Exception as exc:
            if self.use_synthetic_fallback:
                logger.warning(
                    "Could not load HuggingFace checkpoint '%s' (%s). "
                    "Operating in synthetic fallback mode.",
                    self.model_id,
                    exc,
                )
                self.is_synthetic = True
                self._synthetic_backbone.to(torch.device(self.device_str))
            else:
                raise RuntimeError(
                    f"Failed to load PhoBERT checkpoint '{self.model_id}': {exc}"
                ) from exc

    def _tokenize_strings(
        self, texts: List[str], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize list of text strings into input_ids and attention_mask."""
        if self.tokenizer is not None and not self.is_synthetic:
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)

        # Synthetic deterministic tokenization for test/offline environments
        batch_ids = []
        max_len = 0
        for text in texts:
            words = text.split() if text else ["<empty>"]
            # Hash words to integer token IDs in [1, 9999] (0 reserved for pad)
            ids = [
                (int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16) % 9998) + 1
                for w in words[: self.max_length]
            ]
            if not ids:
                ids = [1]
            batch_ids.append(ids)
            max_len = max(max_len, len(ids))

        b = len(texts)
        input_ids = torch.zeros((b, max_len), dtype=torch.long, device=device)
        attention_mask = torch.zeros((b, max_len), dtype=torch.long, device=device)
        for i, ids in enumerate(batch_ids):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            attention_mask[i, : len(ids)] = 1

        return input_ids, attention_mask

    def forward(
        self,
        texts: Optional[Union[str, List[str]]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract semantic vector z_semantic in R^(B x 768).

        Args:
            texts: Single text or batch of strings.
            input_ids: Pre-tokenized token IDs of shape (B, L).
            attention_mask: Attention mask of shape (B, L).

        Returns:
            torch.Tensor: Feature vector of shape (B, 768).
        """
        device = next(self.parameters()).device

        if texts is not None:
            if isinstance(texts, str):
                texts = [texts]
            input_ids, attention_mask = self._tokenize_strings(texts, device)
        elif input_ids is not None:
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            else:
                attention_mask = torch.ones_like(input_ids)
        else:
            raise ValueError("Either 'texts' or 'input_ids' must be provided.")

        # If real PhoBERT model is loaded
        if self.bert_model is not None and not self.is_synthetic:
            outputs = self.bert_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            hidden_states = outputs.last_hidden_state  # (B, L, 768)

            if self.pooling_strategy == "cls":
                z_semantic = hidden_states[:, 0, :]
            elif self.pooling_strategy == "mean":
                mask = attention_mask.unsqueeze(-1).float()
                z_semantic = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            elif self.pooling_strategy == "max":
                mask = attention_mask.unsqueeze(-1).bool()
                masked_hidden = hidden_states.masked_fill(~mask, -1e9)
                z_semantic = masked_hidden.max(dim=1).values
            else:
                z_semantic = hidden_states[:, 0, :]
            return z_semantic

        # Otherwise use synthetic backbone
        return self._synthetic_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pooling_strategy=self.pooling_strategy,
        )
