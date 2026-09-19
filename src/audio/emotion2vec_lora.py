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
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Suppress incompatible pre-installed torchao on Kaggle/Colab (< 0.16.0) to ensure peft loads cleanly
try:
    import torchao
    from packaging import version
    if hasattr(torchao, "__version__") and version.parse(torchao.__version__) < version.parse("0.16.0"):
        import sys
        sys.modules["torchao"] = None
except Exception:
    pass

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
        default_factory=lambda: ["qkv", "proj"]
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
        if x.shape[-1] < 400:
            x = nn.functional.pad(x, (0, 400 - x.shape[-1]))
        x = self.encoder(x)
        return x.transpose(1, 2)

    def extract_features(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Alias for forward extraction."""
        return self.forward(waveforms)


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
        self._synthetic = synthetic

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
            # Mask incompatible torchao (< 0.16.0) before importing peft
            try:
                import torchao
                from packaging import version
                if hasattr(torchao, "__version__") and version.parse(torchao.__version__) < version.parse("0.16.0"):
                    import sys
                    sys.modules["torchao"] = None
            except Exception:
                pass

            from funasr import AutoModel as FunASRAutoModel
            from peft import LoraConfig as PeftLoraConfig, get_peft_model

            # Vá lỗi FunASR Issue #2085: Lọc bỏ các tham số không hợp lệ truyền vào compute_mask_indices
            try:
                import funasr.models.emotion2vec.base as e2v_base
                if hasattr(e2v_base, "compute_mask_indices"):
                    _orig_cmi = e2v_base.compute_mask_indices

                    def _safe_compute_mask(*args, **kwargs):
                        for bad_arg in ["add_masks", "seed", "epoch", "indices"]:
                            kwargs.pop(bad_arg, None)
                        return _orig_cmi(*args, **kwargs)

                    e2v_base.compute_mask_indices = _safe_compute_mask
            except Exception as patch_exc:
                logger.debug("Could not patch e2v_base compute_mask_indices: %s", patch_exc)

        except ImportError as exc:
            logger.warning(
                "Cannot load real emotion2vec (missing dependency: %s). "
                "Falling back to synthetic encoder.", exc,
            )
            self.base_model = _SyntheticEmotion2Vec(self.FEATURE_DIM).to(self.device)
            self.lora_model = self.base_model
            self._synthetic = True
            return

        try:
            self.device = next(self.parameters()).device
        except StopIteration:
            pass

        try:
            model = FunASRAutoModel(model=self.model_id, model_revision="master")
            self.base_model = model.model.to(self.device)

            # Freeze base
            for p in self.base_model.parameters():
                p.requires_grad = False
            self.base_model.eval()

            # Apply LoRA
            lora_cfg = PeftLoraConfig(
                r=self.lora_config.r,
                lora_alpha=self.lora_config.lora_alpha,
                lora_dropout=self.lora_config.lora_dropout,
                bias=self.lora_config.bias,
                target_modules=self.lora_config.target_modules,
            )
            self.lora_model = get_peft_model(self.base_model, lora_cfg).to(self.device)
            self._synthetic = False
            logger.info("Real emotion2vec + LoRA loaded on %s", self.device)
        except Exception as exc:
            logger.warning(
                "Failed to load real emotion2vec checkpoint (%s). "
                "Falling back to synthetic encoder.", exc,
            )
            self.base_model = _SyntheticEmotion2Vec(self.FEATURE_DIM).to(self.device)
            self.lora_model = self.base_model
            self._synthetic = True

    # ------------------------------------------------------------------
    # Forward & Training Mode
    # ------------------------------------------------------------------

    def _apply(self, fn):
        """Synchronize self.device when model is moved to device (e.g., .to(device))."""
        super()._apply(fn)
        try:
            self.device = next(self.parameters()).device
        except StopIteration:
            pass
        return self

    def train(self, mode: bool = True) -> "LoRAEmotion2Vec":
        """Set module training mode, keeping base_model frozen in eval mode."""
        super().train(mode)
        if self.base_model is not None and not self._synthetic:
            self.base_model.eval()
        return self

    @staticmethod
    def _unpack_features(output: Any) -> Optional[torch.Tensor]:
        """Safely unpack tensor features from dictionary/tuple/list outputs."""
        if output is None:
            return None
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, dict):
            for key in ("feats", "last_hidden_state", "x"):
                val = output.get(key)
                if val is not None:
                    unpacked = LoRAEmotion2Vec._unpack_features(val)
                    if unpacked is not None:
                        return unpacked
            # Fallback: check any value in dict
            for val in output.values():
                unpacked = LoRAEmotion2Vec._unpack_features(val)
                if unpacked is not None:
                    return unpacked
            return None
        if isinstance(output, (tuple, list)):
            for item in output:
                unpacked = LoRAEmotion2Vec._unpack_features(item)
                if unpacked is not None:
                    return unpacked
            return None
        return None

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
        # Always keep base_model in eval mode for stable feature extraction
        if self.base_model is not None and not self._synthetic:
            self.base_model.eval()

        if self.lora_model is None:
            self.lora_model = self._synthetic_backbone

        # Ensure device matches underlying parameters
        try:
            target_device = next(self.parameters()).device
            self.device = target_device
        except StopIteration:
            target_device = self.device

        waveforms = waveforms.to(target_device)
        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(0)

        B = waveforms.shape[0]

        # 1. Squeeze 3D waveform inputs (B, 1, T) down to 2D (B, T) before passing to FunASR / LoRA
        if waveforms.ndim == 3 and waveforms.shape[1] == 1:
            wav_input = waveforms.squeeze(1)
        elif waveforms.ndim == 3 and waveforms.shape[-1] == 1:
            wav_input = waveforms.squeeze(-1)
        elif waveforms.ndim == 2:
            wav_input = waveforms
        else:
            wav_input = waveforms.view(B, -1)

        feats: Optional[torch.Tensor] = None

        if self._synthetic:
            raw_out = self.lora_model(wav_input)
            feats = self._unpack_features(raw_out)
        else:
            res = None

            # 2a. Check whether lora_model provides extract_features
            if hasattr(self.lora_model, "extract_features"):
                try:
                    res = self.lora_model.extract_features(wav_input)
                    if self._unpack_features(res) is None:
                        res = None
                except Exception as exc:
                    logger.debug("self.lora_model.extract_features raised: %s", exc)

            # 2b. Check whether base_model or underlying model provides extract_features
            if res is None and self.base_model is not None:
                underlying = getattr(self.base_model, "model", self.base_model)
                if hasattr(underlying, "extract_features"):
                    try:
                        res = underlying.extract_features(wav_input)
                        if self._unpack_features(res) is None:
                            res = None
                    except Exception as exc:
                        logger.debug("underlying.extract_features raised: %s", exc)

            # 2c. Try calling lora_model with features_only=True
            if res is None and self.lora_model is not None:
                try:
                    res = self.lora_model(wav_input, features_only=True)
                    if self._unpack_features(res) is None:
                        res = None
                except Exception as exc:
                    logger.debug("self.lora_model(wav_input, features_only=True) raised: %s", exc)

            # 2d. Fallback call to lora_model directly
            if res is None and self.lora_model is not None:
                try:
                    res = self.lora_model(wav_input)
                except Exception as exc:
                    logger.debug("self.lora_model forward call raised: %s", exc)

            # 3. Safely unpack dictionary/tuple returns (checking for keys "feats", "last_hidden_state", or "x")
            feats = self._unpack_features(res)

        # 4. Defensive fallback: if feats is None or not a torch.Tensor,
        #    log a warning and use self._synthetic_backbone(waveforms) so training never halts on NoneType.
        if feats is None or not isinstance(feats, torch.Tensor):
            logger.warning(
                "LoRAEmotion2Vec: extracted features is %s (not a Tensor); falling back to synthetic backbone.",
                type(feats).__name__ if feats is not None else "None",
            )
            feats = self._synthetic_backbone(waveforms)

        # 5. Verify that feats is always a 3D Tensor (B, N, 768) before returning
        if feats.ndim == 2:
            if feats.shape[0] == B and feats.shape[1] == self.feature_dim:
                feats = feats.unsqueeze(1)
            elif B == 1 and feats.shape[-1] == self.feature_dim:
                feats = feats.unsqueeze(0)
            elif feats.numel() == B * self.feature_dim:
                feats = feats.view(B, 1, self.feature_dim)
            else:
                logger.warning(
                    "LoRAEmotion2Vec: 2D features shape %s cannot be reshaped to (%d, N, %d); using synthetic fallback.",
                    tuple(feats.shape),
                    B,
                    self.feature_dim,
                )
                feats = self._synthetic_backbone(waveforms)

        if feats.ndim != 3 or feats.shape[0] != B or feats.shape[-1] != self.feature_dim:
            logger.warning(
                "LoRAEmotion2Vec: features shape %s does not match expected (%d, N, %d); using synthetic fallback.",
                tuple(feats.shape),
                B,
                self.feature_dim,
            )
            feats = self._synthetic_backbone(waveforms)

        feats = feats.float().to(target_device)

        if not return_lengths:
            return feats

        B_feats, N, _ = feats.shape
        if waveform_lengths is not None:
            ratio = N / waveforms.size(-1)
            frame_lengths = (waveform_lengths.float() * ratio).long().clamp(min=1, max=N)
        else:
            frame_lengths = torch.full((B_feats,), N, dtype=torch.long, device=feats.device)

        return feats, frame_lengths

    def extract_features(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: Optional[torch.Tensor] = None,
        return_lengths: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Extract frame-level features from waveforms (alias for forward)."""
        return self.forward(waveforms, waveform_lengths=waveform_lengths, return_lengths=return_lengths)

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
