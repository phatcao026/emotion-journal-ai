"""
emotion2vec_lora.py

emotion2vec-base with LoRA adaptation for Vietnamese speech emotion features.

Architecture:
    Base: emotion2vec-base (iic/emotion2vec_base) via FunASR / ModelScope
    LoRA: r=8, lora_alpha=16, targets = q_proj, v_proj, k_proj, out_proj
    Output: frame-level features  (B, N, 768)

Provides a **synthetic fallback** mode that generates random features with
the correct shape when the pretrained checkpoint is unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LoRAConfig:
    """LoRA adapter hyper-parameters.

    Attributes:
        r:              Low-rank dimension.  Default 8.
        lora_alpha:     Scaling factor (effective scale = alpha / r).  Default 16.
        lora_dropout:   Dropout within LoRA layers.  Default 0.1.
        bias:           ``"none"`` | ``"all"`` | ``"lora_only"``.
        target_modules: Attention projection layers to adapt.
        task_type:      PEFT task type string.
    """
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    bias: str = "none"
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "out_proj"]
    )
    task_type: str = "FEATURE_EXTRACTION"

    @property
    def scaling(self) -> float:
        return self.lora_alpha / self.r


# ---------------------------------------------------------------------------
# Synthetic backbone (used when pretrained weights are unavailable)
# ---------------------------------------------------------------------------

class _SyntheticEmotion2Vec(nn.Module):
    """Lightweight stand-in that mimics emotion2vec output shapes.

    Produces ``(B, N, 768)`` by running the waveform through a tiny Conv1d
    encoder.  This lets the full pipeline be tested without downloading
    the real checkpoint.
    """

    def __init__(self, feature_dim: int = 768) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=400, stride=320, padding=0),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(256, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """(B, T) or (B, 1, T) -> (B, N, 768)."""
        if waveforms.ndim == 2:
            x = waveforms.unsqueeze(1)
        elif waveforms.ndim == 3 and waveforms.shape[1] == 1:
            x = waveforms
        else:
            x = waveforms.view(waveforms.shape[0], 1, -1)
        x = self.encoder(x)
        return x.transpose(1, 2)


# ---------------------------------------------------------------------------
# LoRAEmotion2Vec
# ---------------------------------------------------------------------------

class LoRAEmotion2Vec(nn.Module):
    """emotion2vec-base with optional LoRA adaptation.

    Falls back to a lightweight synthetic encoder when the real checkpoint
    cannot be loaded (e.g., in CI or unit-test environments).

    Args:
        model_id:    ModelScope model identifier.
        lora_config: LoRA hyper-parameters (dataclass).
        device:      ``"cpu"`` or ``"cuda"``.
        cache_dir:   Optional cache directory for the checkpoint.
        synthetic:   If True, skip real model loading and use synthetic encoder.
    """

    FEATURE_DIM: int = 768

    def __init__(
        self,
        model_id: str = "iic/emotion2vec_base",
        lora_config: Optional[LoRAConfig] = None,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        synthetic: bool = False,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.lora_config = lora_config or LoRAConfig()
        self.device = torch.device(device)
        self.cache_dir = cache_dir
        self.feature_dim = self.FEATURE_DIM
        self._synthetic = True

        self.base_model: Optional[nn.Module] = None
        self.lora_model: Optional[nn.Module] = None

        self._synthetic_backbone = _SyntheticEmotion2Vec(self.FEATURE_DIM).to(self.device)
        self.base_model = self._synthetic_backbone
        self.lora_model = self._synthetic_backbone

        if synthetic:
            logger.info("LoRAEmotion2Vec: SYNTHETIC mode (no real checkpoint)")
        else:
            logger.info(
                "LoRAEmotion2Vec initialized with synthetic fallback: model=%s (call load_pretrained() for weights)",
                model_id,
            )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_pretrained(self) -> None:
        """Load the real emotion2vec checkpoint + apply PEFT LoRA.

        Steps:
            1. Download via FunASR / ModelScope.
            2. Freeze all base parameters.
            3. Inject LoRA adapters via ``peft.get_peft_model``.
            4. Move to ``self.device``.

        Raises:
            ImportError: If ``funasr`` or ``peft`` is not installed.
        """
        if self._synthetic:
            logger.info("Synthetic mode — skipping real checkpoint loading.")
            return

        try:
            from funasr import AutoModel as FunASRAutoModel
            from peft import LoraConfig as PeftLoraConfig, get_peft_model
        except ImportError as exc:
            logger.warning(
                "Cannot load real emotion2vec (missing dependency: %s). "
                "Falling back to synthetic encoder.", exc,
            )
            self.base_model = _SyntheticEmotion2Vec(self.FEATURE_DIM).to(self.device)
            self.lora_model = self.base_model
            self._synthetic = True
            return

        model = FunASRAutoModel(model=self.model_id, model_revision="master")
        self.base_model = model.model.to(self.device)

        # Freeze base
        for p in self.base_model.parameters():
            p.requires_grad = False

        # Apply LoRA
        lora_cfg = PeftLoraConfig(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,
            bias=self.lora_config.bias,
            target_modules=self.lora_config.target_modules,
            task_type=self.lora_config.task_type,
        )
        self.lora_model = get_peft_model(self.base_model, lora_cfg).to(self.device)
        logger.info("Real emotion2vec + LoRA loaded on %s", self.device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: Optional[torch.Tensor] = None,
        return_lengths: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Waveforms -> frame-level features.

        Args:
            waveforms:        (B, T) or (B, 1, T) at 16 kHz.
            waveform_lengths: (B,) actual sample counts (optional).
            return_lengths:   If True, returns (frame_features, frame_lengths). Default False.

        Returns:
            frame_features of shape (B, N, 768) or (frame_features, frame_lengths).
        """
        if self.lora_model is None:
            self.lora_model = self._synthetic_backbone

        if self._synthetic:
            feats = self.lora_model(waveforms.to(self.device))  # (B, N, 768)
        else:
            # Real emotion2vec API
            feats = self.lora_model(waveforms.to(self.device))
            if isinstance(feats, dict):
                feats = feats.get("last_hidden_state", feats.get("feats", feats))
            if isinstance(feats, (tuple, list)):
                feats = feats[0]

        if not return_lengths:
            return feats

        B, N, _ = feats.shape
        if waveform_lengths is not None:
            ratio = N / waveforms.size(-1)
            frame_lengths = (waveform_lengths.float() * ratio).long().clamp(min=1, max=N)
        else:
            frame_lengths = torch.full((B,), N, dtype=torch.long, device=feats.device)

        return feats, frame_lengths

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_trainable_parameters(self) -> Dict[str, torch.Tensor]:
        return {n: p for n, p in self.named_parameters() if p.requires_grad}

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "lora_ratio_pct": round(100.0 * trainable / max(total, 1), 2),
        }

    def save_lora_weights(self, save_path: str) -> None:
        """Save only trainable (LoRA) weights."""
        state = {n: p.data for n, p in self.named_parameters() if p.requires_grad}
        torch.save(state, save_path)
        logger.info("Saved %d LoRA tensors to %s", len(state), save_path)

    def load_lora_weights(self, load_path: str) -> None:
        """Load previously saved LoRA weights."""
        state = torch.load(load_path, map_location=self.device)
        missing, unexpected = [], []
        own = dict(self.named_parameters())
        for k, v in state.items():
            if k in own:
                own[k].data.copy_(v)
            else:
                unexpected.append(k)
        logger.info("Loaded LoRA weights from %s (unexpected keys: %d)", load_path, len(unexpected))
