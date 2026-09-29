"""
melody_extractor.py

Explicit Melody & Acoustic Biomarker Extractor for Speech Emotion Recognition.
Extracts 32-dimensional interpretable acoustic prosody and voice quality descriptors:
    - Pitch (F0) Dynamics: Mean, Std, Range, Slope, Curvature, Percentiles, Voicing Ratio (15 dims)
    - Laryngeal Micro-Instability & Voice Quality: Jitter (local, RAP), Shimmer (local, APQ3, APQ5), HNR (6 dims)
    - Energy Envelope Dynamics: RMS Mean, Std, Max, Slope, Crest Factor, Energy Entropy (6 dims)
    - Spectral & Formant Proxies: Centroid, Spread, Flux, High/Low Energy Ratio, Zero Crossing Rate (5 dims)

Mathematical foundations:
    - Boersma, P. (1993). Accurate short-term analysis of the fundamental frequency and the harmonics-to-noise ratio.
    - Eyben, F. et al. (2016). The Geneva Minimalistic Acoustic Parameter Set (eGeMAPS) for Voice Research.
    - Mundt, J. C. et al. (2012). Vocal acoustic biomarkers of depression severity and treatment response.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ExplicitMelodyExtractor(nn.Module):
    """Extracts 32 interpretable acoustic melody and voice quality features.

    Operates natively on PyTorch tensors on CPU or GPU.

    Input:
        waveform: (B, T) or (T,) at 16,000 Hz.
    Output:
        features: (B, 32) float32 tensor of normalized acoustic biomarkers.
    """

    FEATURE_DIM: int = 32

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length_ms: float = 40.0,
        hop_length_ms: float = 20.0,
        f0_min_hz: float = 65.0,
        f0_max_hz: float = 450.0,
        sr: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.sample_rate = sr if sr is not None else sample_rate
        self.frame_len = int(frame_length_ms * self.sample_rate / 1000.0)
        self.hop_len = int(hop_length_ms * self.sample_rate / 1000.0)
        self.min_lag = int(self.sample_rate / f0_max_hz)
        self.max_lag = int(self.sample_rate / f0_min_hz)

        # Batch projection to ensure smooth gradients if trained end-to-end
        self.norm = nn.LayerNorm(self.FEATURE_DIM)

    @property
    def num_features(self) -> int:
        """Dimensionality of extracted features (32)."""
        return self.FEATURE_DIM

    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        """Alias for forward extraction."""
        return self.forward(waveform)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract 32 melody features for a batch of waveforms.

        Args:
            waveform: Tensor of shape (B, T) or (T,) in float32.

        Returns:
            Tensor of shape (B, 32) in float32.
        """
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 3 and waveform.size(1) == 1:
            waveform = waveform.squeeze(1)

        batch_size = waveform.size(0)
        device = waveform.device

        features_list = []
        for b in range(batch_size):
            wav = waveform[b].float()
            feats = self._extract_single(wav, device)
            features_list.append(feats)

        batch_feats = torch.stack(features_list, dim=0)  # (B, 32)
        return self.norm(batch_feats)

    def _extract_single(self, wav: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Extract 32 features for a single 1D waveform."""
        T = wav.numel()
        if T < self.frame_len:
            wav = F.pad(wav, (0, self.frame_len - T))
            T = wav.numel()

        # Unfold into overlapping frames: (N, frame_len)
        frames = wav.unfold(0, self.frame_len, self.hop_len)
        num_frames = frames.size(0)

        # -------------------------------------------------------------
        # 1. Energy Envelope & RMS
        # -------------------------------------------------------------
        rms = torch.sqrt(torch.mean(frames ** 2, dim=-1) + 1e-9)  # (N,)
        rms_mean = rms.mean()
        rms_std = rms.std(unbiased=False) if num_frames > 1 else torch.tensor(0.0, device=device)
        rms_max = rms.max()
        crest_factor = rms_max / (rms_mean + 1e-7)

        # Linear slope of RMS (trend of energy over time)
        time_steps = torch.linspace(-1.0, 1.0, num_frames, device=device)
        rms_slope = torch.sum((rms - rms_mean) * time_steps) / (torch.sum(time_steps ** 2) + 1e-7)

        # Energy entropy
        p_energy = (rms ** 2) / (torch.sum(rms ** 2) + 1e-9)
        energy_entropy = -torch.sum(p_energy * torch.log(p_energy + 1e-9)) / math.log(max(num_frames, 2))

        # -------------------------------------------------------------
        # 2. Short-Term Autocorrelation for Pitch (F0) & Voicing
        # -------------------------------------------------------------
        # Center frames
        frames_centered = frames - frames.mean(dim=-1, keepdim=True)
        # Window with Hann
        hann = torch.hann_window(self.frame_len, device=device)
        frames_win = frames_centered * hann

        # FFT Autocorrelation
        n_fft = 2 ** math.ceil(math.log2(2 * self.frame_len))
        fft = torch.fft.rfft(frames_win, n=n_fft, dim=-1)
        corr = torch.fft.irfft(fft * torch.conj(fft), n=n_fft, dim=-1)[:, :self.frame_len]
        r0 = corr[:, 0:1].clamp(min=1e-9)
        norm_corr = corr / r0  # (N, frame_len)

        # Search peak within [min_lag, max_lag]
        valid_lags = norm_corr[:, self.min_lag:self.max_lag]
        max_corr, peak_idx = torch.max(valid_lags, dim=-1)
        best_lag = (peak_idx + self.min_lag).float()

        # Voiced decision: norm correlation peak > 0.3
        is_voiced = (max_corr > 0.3) & (rms > 0.01 * rms_max)
        num_voiced = is_voiced.sum()
        voicing_ratio = num_voiced.float() / max(num_frames, 1)

        f0_candidates = torch.where(is_voiced, self.sample_rate / best_lag, torch.tensor(0.0, device=device))
        voiced_f0 = f0_candidates[is_voiced]

        if num_voiced > 1:
            f0_mean = voiced_f0.mean()
            f0_std = voiced_f0.std(unbiased=False)
            f0_min = voiced_f0.min()
            f0_max = voiced_f0.max()
            f0_range = f0_max - f0_min

            # Percentiles
            sorted_f0, _ = torch.sort(voiced_f0)
            N_v = sorted_f0.numel()
            p10 = sorted_f0[int(0.10 * (N_v - 1))]
            p25 = sorted_f0[int(0.25 * (N_v - 1))]
            p50 = sorted_f0[int(0.50 * (N_v - 1))]
            p75 = sorted_f0[int(0.75 * (N_v - 1))]
            p90 = sorted_f0[int(0.90 * (N_v - 1))]

            # F0 slope & curvature across voiced frames
            v_idx = torch.nonzero(is_voiced, as_tuple=True)[0].float()
            v_t = (v_idx - v_idx.mean()) / (v_idx.std() + 1e-7) if v_idx.numel() > 1 else torch.zeros_like(v_idx)
            f0_norm = voiced_f0 - f0_mean
            f0_slope = torch.sum(f0_norm * v_t) / (torch.sum(v_t ** 2) + 1e-7)
            f0_curvature = torch.sum(f0_norm * (v_t ** 2 - 1.0)) / (torch.sum((v_t ** 2 - 1.0) ** 2) + 1e-7)

            # Jitter: Relative perturbation in pitch period T0 = 1/f0
            t0 = 1.0 / voiced_f0.clamp(min=50.0)
            t0_diff = torch.abs(t0[1:] - t0[:-1])
            jitter_local = t0_diff.mean() / (t0.mean() + 1e-7)

            # Jitter RAP (Relative Average Perturbation - 3-point moving average)
            if N_v >= 3:
                t0_rap = torch.abs(t0[1:-1] - (t0[:-2] + t0[1:-1] + t0[2:]) / 3.0)
                jitter_rap = t0_rap.mean() / (t0.mean() + 1e-7)
            else:
                jitter_rap = jitter_local

            # Shimmer: Relative perturbation in peak amplitude across voiced frames
            voiced_rms = rms[is_voiced]
            rms_diff = torch.abs(voiced_rms[1:] - voiced_rms[:-1])
            shimmer_local = rms_diff.mean() / (voiced_rms.mean() + 1e-7)

            # Shimmer APQ3 (3-point Amplitude Perturbation Quotient)
            if N_v >= 3:
                rms_apq3 = torch.abs(voiced_rms[1:-1] - (voiced_rms[:-2] + voiced_rms[1:-1] + voiced_rms[2:]) / 3.0)
                shimmer_apq3 = rms_apq3.mean() / (voiced_rms.mean() + 1e-7)
            else:
                shimmer_apq3 = shimmer_local

            # Shimmer APQ5 (5-point Amplitude Perturbation Quotient)
            if N_v >= 5:
                rms_apq5 = torch.abs(voiced_rms[2:-2] - (voiced_rms[:-4] + voiced_rms[1:-3] + voiced_rms[2:-2] + voiced_rms[3:-1] + voiced_rms[4:]) / 5.0)
                shimmer_apq5 = rms_apq5.mean() / (voiced_rms.mean() + 1e-7)
            else:
                shimmer_apq5 = shimmer_apq3

            # Harmonics-to-Noise Ratio (HNR in dB) proxy
            voiced_r = max_corr[is_voiced].clamp(min=1e-4, max=0.999)
            hnr_db = 10.0 * torch.log10(voiced_r / (1.0 - voiced_r + 1e-7)).mean()
        else:
            # Fallback for silent/unvoiced clips
            f0_mean = torch.tensor(120.0, device=device)
            f0_std = torch.tensor(0.0, device=device)
            f0_min = torch.tensor(120.0, device=device)
            f0_max = torch.tensor(120.0, device=device)
            f0_range = torch.tensor(0.0, device=device)
            p10 = p25 = p50 = p75 = p90 = torch.tensor(120.0, device=device)
            f0_slope = torch.tensor(0.0, device=device)
            f0_curvature = torch.tensor(0.0, device=device)
            jitter_local = torch.tensor(0.0, device=device)
            jitter_rap = torch.tensor(0.0, device=device)
            shimmer_local = torch.tensor(0.0, device=device)
            shimmer_apq3 = torch.tensor(0.0, device=device)
            shimmer_apq5 = torch.tensor(0.0, device=device)
            hnr_db = torch.tensor(0.0, device=device)

        # -------------------------------------------------------------
        # 3. Spectral Descriptors & Formant Proxies
        # -------------------------------------------------------------
        mag_spec = torch.abs(torch.fft.rfft(frames_win, n=512, dim=-1))  # (N, 257)
        freq_bins = torch.linspace(0.0, self.sample_rate / 2.0, 257, device=device)

        # Spectral Centroid (Brightness)
        spec_sum = mag_spec.sum(dim=-1).clamp(min=1e-7)
        centroid = torch.sum(mag_spec * freq_bins, dim=-1) / spec_sum
        spectral_centroid = centroid.mean()

        # Spectral Spread (Bandwidth)
        spread = torch.sqrt(torch.sum(mag_spec * ((freq_bins - centroid.unsqueeze(-1)) ** 2), dim=-1) / spec_sum + 1e-7)
        spectral_spread = spread.mean()

        # Spectral Flux (Rate of spectral variation between adjacent frames)
        if num_frames > 1:
            mag_norm = mag_spec / spec_sum.unsqueeze(-1)
            flux = torch.sqrt(torch.sum((mag_norm[1:] - mag_norm[:-1]) ** 2, dim=-1)).mean()
        else:
            flux = torch.tensor(0.0, device=device)

        # High/Low Spectral Energy Ratio (Proxy for spectral tilt / vocal roughness)
        # Low: 0 - 1000 Hz (bins 0-32), High: 1000 - 8000 Hz (bins 32-256)
        split_bin = int(1000.0 / (self.sample_rate / 512.0))
        e_low = mag_spec[:, :split_bin].sum().clamp(min=1e-7)
        e_high = mag_spec[:, split_bin:].sum().clamp(min=1e-7)
        high_low_ratio = e_high / e_low

        # Zero-Crossing Rate (ZCR)
        signs = torch.sign(frames)
        zcr = torch.mean(torch.abs(signs[:, 1:] - signs[:, :-1]) / 2.0, dim=-1).mean()

        # -------------------------------------------------------------
        # Assemble exactly 32 features
        # -------------------------------------------------------------
        features = torch.tensor([
            # 1-7: F0 Core Statistics
            f0_mean / 400.0,
            f0_std / 150.0,
            f0_min / 400.0,
            f0_max / 400.0,
            f0_range / 300.0,
            f0_slope.clamp(-100.0, 100.0) / 50.0,
            f0_curvature.clamp(-100.0, 100.0) / 50.0,
            # 8-15: F0 Percentiles & Voicing
            p10 / 400.0,
            p25 / 400.0,
            p50 / 400.0,
            p75 / 400.0,
            p90 / 400.0,
            voicing_ratio,
            num_voiced.float() / max(num_frames, 1),
            f0_std / (f0_mean + 1e-5),  # Pitch Coefficient of Variation
            # 16-21: Voice Quality & Micro-instability (Jitter, Shimmer, HNR)
            jitter_local.clamp(0.0, 1.0),
            jitter_rap.clamp(0.0, 1.0),
            shimmer_local.clamp(0.0, 1.0),
            shimmer_apq3.clamp(0.0, 1.0),
            shimmer_apq5.clamp(0.0, 1.0),
            hnr_db.clamp(-20.0, 40.0) / 30.0,
            # 22-27: Energy Envelope Dynamics
            rms_mean * 10.0,
            rms_std * 10.0,
            rms_max * 10.0,
            rms_slope.clamp(-5.0, 5.0),
            crest_factor.clamp(0.0, 20.0) / 10.0,
            energy_entropy,
            # 28-32: Spectral & Formant Proxies
            spectral_centroid / 4000.0,
            spectral_spread / 2000.0,
            flux * 10.0,
            high_low_ratio.clamp(0.0, 10.0) / 5.0,
            zcr * 5.0,
        ], dtype=torch.float32, device=device)

        return features
