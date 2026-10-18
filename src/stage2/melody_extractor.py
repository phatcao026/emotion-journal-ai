"""
src/stage2/melody_extractor.py

Explicit Melody & Acoustic Biomarker Extractor with Vietnamese Speaker-Relative
Semitone Normalization and Sentence-Final Particle Slope Extraction.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ExplicitMelodyExtractor(nn.Module):
    """Extracts 32 interpretable acoustic melody and voice quality features."""

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

        # Retained for state_dict backward compatibility; frozen so cross-feature LayerNorm does not distort units
        self.norm = nn.LayerNorm(self.FEATURE_DIM)
        for p in self.norm.parameters():
            p.requires_grad = False

    @property
    def num_features(self) -> int:
        return self.FEATURE_DIM

    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.forward(waveform)

    def extract_raw_biomarkers(self, waveform: torch.Tensor) -> dict[str, float]:
        if waveform.ndim == 1:
            wav = waveform.float()
        elif waveform.ndim == 2:
            wav = waveform[0].float()
        elif waveform.ndim == 3:
            wav = waveform[0, 0].float()
        else:
            wav = waveform.reshape(-1).float()

        device = wav.device
        _, raw_dict = self._extract_single_with_metrics(wav, device)
        return raw_dict

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 3 and waveform.size(1) == 1:
            waveform = waveform.squeeze(1)

        batch_size = waveform.size(0)
        device = waveform.device

        features_list = []
        for b in range(batch_size):
            wav = waveform[b].float()
            feats, _ = self._extract_single_with_metrics(wav, device)
            features_list.append(feats)

        # Return bounded physical/semitone features directly without cross-feature LayerNorm distortion
        return torch.stack(features_list, dim=0)

    def _extract_single(self, wav: torch.Tensor, device: torch.device) -> torch.Tensor:
        feats, _ = self._extract_single_with_metrics(wav, device)
        return feats

    def _extract_single_with_metrics(
        self, wav: torch.Tensor, device: torch.device
    ) -> tuple[torch.Tensor, dict[str, float]]:
        T = wav.numel()
        if T < self.frame_len or wav.abs().max() < 1e-5:
            zeros_feat = torch.zeros(self.FEATURE_DIM, dtype=torch.float32, device=device)
            empty_raw = {
                "f0_mean": 0.0,
                "f0_std": 0.0,
                "f0_slope": 0.0,
                "f0_curvature": 0.0,
                "voicing_ratio": 0.0,
                "jitter_local": 0.0,
                "jitter_rap": 0.0,
                "shimmer_local": 0.0,
                "hnr_db": 0.0,
                "rms_mean": 0.0,
                "spectral_centroid": 0.0,
            }
            return zeros_feat, empty_raw

        frames = wav.unfold(0, self.frame_len, self.hop_len)
        num_frames = frames.size(0)

        # 1. Energy Envelope & RMS (Utterance-relative normalization for journal vs podcast)
        rms = torch.sqrt(torch.mean(frames ** 2, dim=-1) + 1e-9)
        rms_mean = rms.mean()
        rms_std = rms.std(unbiased=False) if num_frames > 1 else torch.tensor(0.0, device=device)
        rms_max = rms.max()
        crest_factor = rms_max / (rms_mean + 1e-7)

        time_steps = torch.linspace(-1.0, 1.0, num_frames, device=device)
        rms_norm = rms / (rms_mean + 1e-7)
        rms_slope = torch.sum((rms_norm - 1.0) * time_steps) / (torch.sum(time_steps ** 2) + 1e-7)

        p_energy = (rms ** 2) / (torch.sum(rms ** 2) + 1e-9)
        energy_entropy = -torch.sum(p_energy * torch.log(p_energy + 1e-9)) / math.log(max(num_frames, 2))

        # 2. Short-Term Autocorrelation for Pitch (F0) & Voicing
        frames_centered = frames - frames.mean(dim=-1, keepdim=True)
        hann = torch.hann_window(self.frame_len, device=device)
        frames_win = frames_centered * hann

        n_fft = 2 ** math.ceil(math.log2(2 * self.frame_len))
        fft = torch.fft.rfft(frames_win, n=n_fft, dim=-1)
        corr = torch.fft.irfft(fft * torch.conj(fft), n=n_fft, dim=-1)[:, :self.frame_len]
        r0 = corr[:, 0:1].clamp(min=1e-9)
        norm_corr = corr / r0

        valid_lags = norm_corr[:, self.min_lag:self.max_lag]
        max_corr, peak_idx = torch.max(valid_lags, dim=-1)
        best_lag = (peak_idx + self.min_lag).float()

        is_voiced = (max_corr > 0.3) & (rms > 0.01 * rms_max)
        num_voiced = is_voiced.sum()
        voicing_ratio = num_voiced.float() / max(num_frames, 1)

        f0_candidates = torch.where(is_voiced, self.sample_rate / best_lag, torch.tensor(0.0, device=device))
        voiced_f0 = f0_candidates[is_voiced]

        if num_voiced > 5:
            k = min(15, num_voiced.item() if num_voiced.item() % 2 == 1 else num_voiced.item() - 1)
            if k >= 3:
                from scipy.signal import medfilt
                voiced_f0_np = voiced_f0.detach().cpu().numpy()
                voiced_f0 = torch.tensor(medfilt(voiced_f0_np, kernel_size=k), device=device, dtype=voiced_f0.dtype)

        if num_voiced > 1:
            f0_mean = voiced_f0.mean()
            f0_std = voiced_f0.std(unbiased=False)

            # Speaker-relative Semitones: 12 * log2(F0 / F0_mean) -> removes gender & mic bias
            f0_st = 12.0 * torch.log2((voiced_f0 / (f0_mean + 1e-7)).clamp(min=0.25, max=4.0))
            f0_min_st = f0_st.min()
            f0_max_st = f0_st.max()
            f0_range_st = f0_max_st - f0_min_st

            sorted_st, _ = torch.sort(f0_st)
            N_v = sorted_st.numel()
            p10_st = sorted_st[int(0.10 * (N_v - 1))]
            p25_st = sorted_st[int(0.25 * (N_v - 1))]
            p50_st = sorted_st[int(0.50 * (N_v - 1))]
            p75_st = sorted_st[int(0.75 * (N_v - 1))]
            p90_st = sorted_st[int(0.90 * (N_v - 1))]

            v_idx = torch.nonzero(is_voiced, as_tuple=True)[0].float()
            v_t = (v_idx - v_idx.mean()) / (v_idx.std() + 1e-7) if v_idx.numel() > 1 else torch.zeros_like(v_idx)
            f0_norm = voiced_f0 - f0_mean
            f0_slope = torch.sum(f0_norm * v_t) / (torch.sum(v_t ** 2) + 1e-7)

            # Vietnamese Sentence-Final Particle Slope (last 20% voiced frames)
            tail_len = max(2, int(0.2 * N_v))
            tail_st = f0_st[-tail_len:]
            tail_t = torch.linspace(-1.0, 1.0, tail_len, device=device)
            f0_final_slope_st = torch.sum((tail_st - tail_st.mean()) * tail_t) / (torch.sum(tail_t ** 2) + 1e-7)

            t0 = 1.0 / voiced_f0.clamp(min=50.0)
            t0_diff = torch.abs(t0[1:] - t0[:-1])
            jitter_local = t0_diff.mean() / (t0.mean() + 1e-7)

            if N_v >= 3:
                t0_rap = torch.abs(t0[1:-1] - (t0[:-2] + t0[1:-1] + t0[2:]) / 3.0)
                jitter_rap = t0_rap.mean() / (t0.mean() + 1e-7)
            else:
                jitter_rap = jitter_local

            voiced_rms = rms[is_voiced]
            rms_diff = torch.abs(voiced_rms[1:] - voiced_rms[:-1])
            shimmer_local = rms_diff.mean() / (voiced_rms.mean() + 1e-7)

            if N_v >= 3:
                rms_apq3 = torch.abs(voiced_rms[1:-1] - (voiced_rms[:-2] + voiced_rms[1:-1] + voiced_rms[2:]) / 3.0)
                shimmer_apq3 = rms_apq3.mean() / (voiced_rms.mean() + 1e-7)
            else:
                shimmer_apq3 = shimmer_local

            if N_v >= 5:
                rms_apq5 = torch.abs(
                    voiced_rms[2:-2]
                    - (voiced_rms[:-4] + voiced_rms[1:-3] + voiced_rms[2:-2] + voiced_rms[3:-1] + voiced_rms[4:]) / 5.0
                )
                shimmer_apq5 = rms_apq5.mean() / (voiced_rms.mean() + 1e-7)
            else:
                shimmer_apq5 = shimmer_apq3

            voiced_r = max_corr[is_voiced].clamp(min=1e-4, max=0.999)
            hnr_db = 10.0 * torch.log10(voiced_r / (1.0 - voiced_r + 1e-7)).mean()
        else:
            f0_mean = torch.tensor(0.0, device=device)
            f0_std = torch.tensor(0.0, device=device)
            f0_min_st = f0_max_st = f0_range_st = torch.tensor(0.0, device=device)
            p10_st = p25_st = p50_st = p75_st = p90_st = torch.tensor(0.0, device=device)
            f0_slope = torch.tensor(0.0, device=device)
            f0_final_slope_st = torch.tensor(0.0, device=device)
            jitter_local = jitter_rap = torch.tensor(0.0, device=device)
            shimmer_local = shimmer_apq3 = shimmer_apq5 = torch.tensor(0.0, device=device)
            hnr_db = torch.tensor(0.0, device=device)

        # 3. Spectral Descriptors & Formant Proxies
        mag_spec = torch.abs(torch.fft.rfft(frames_win, n=512, dim=-1))
        freq_bins = torch.linspace(0.0, self.sample_rate / 2.0, 257, device=device)

        spec_sum = mag_spec.sum(dim=-1).clamp(min=1e-7)
        centroid = torch.sum(mag_spec * freq_bins, dim=-1) / spec_sum
        spectral_centroid = centroid.mean()

        spread = torch.sqrt(torch.sum(mag_spec * ((freq_bins - centroid.unsqueeze(-1)) ** 2), dim=-1) / spec_sum + 1e-7)
        spectral_spread = spread.mean()

        if num_frames > 1:
            mag_norm = mag_spec / spec_sum.unsqueeze(-1)
            flux = torch.sqrt(torch.sum((mag_norm[1:] - mag_norm[:-1]) ** 2, dim=-1)).mean()
        else:
            flux = torch.tensor(0.0, device=device)

        split_bin = int(1000.0 / (self.sample_rate / 512.0))
        e_low = mag_spec[:, :split_bin].sum().clamp(min=1e-7)
        e_high = mag_spec[:, split_bin:].sum().clamp(min=1e-7)
        high_low_ratio = e_high / e_low

        signs = torch.sign(frames)
        zcr = torch.mean(torch.abs(signs[:, 1:] - signs[:, :-1]) / 2.0, dim=-1).mean()

        # Assemble 32 features (preserving exact indices 0, 1, 5, 15, 17, 20)
        features = torch.tensor([
            f0_mean / 300.0,                                 # [0] f0_mean
            f0_std / 80.0,                                   # [1] f0_std
            f0_min_st.clamp(-12.0, 12.0) / 12.0,             # [2] speaker-relative min semitone
            f0_max_st.clamp(-12.0, 12.0) / 12.0,             # [3] speaker-relative max semitone
            f0_range_st.clamp(0.0, 24.0) / 12.0,             # [4] semitone range
            f0_slope.clamp(-50.0, 50.0) / 25.0,              # [5] f0_slope
            f0_final_slope_st.clamp(-12.0, 12.0) / 12.0,     # [6] final 20% syllable slope (st)
            p10_st.clamp(-12.0, 12.0) / 12.0,                # [7]
            p25_st.clamp(-12.0, 12.0) / 12.0,                # [8]
            p50_st.clamp(-12.0, 12.0) / 12.0,                # [9]
            p75_st.clamp(-12.0, 12.0) / 12.0,                # [10]
            p90_st.clamp(-12.0, 12.0) / 12.0,                # [11]
            voicing_ratio,                                   # [12]
            num_voiced.float() / max(num_frames, 1),         # [13]
            f0_std / (f0_mean + 1e-5),                       # [14] CoV
            jitter_local.clamp(0.0, 1.0),                    # [15] jitter_local
            jitter_rap.clamp(0.0, 1.0),                      # [16]
            shimmer_local.clamp(0.0, 1.0),                   # [17] shimmer_local
            shimmer_apq3.clamp(0.0, 1.0),                    # [18]
            shimmer_apq5.clamp(0.0, 1.0),                    # [19]
            hnr_db.clamp(-20.0, 40.0) / 30.0,                # [20] hnr_db
            (rms_std / (rms_mean + 1e-7)).clamp(0.0, 5.0) / 2.5,  # [21] gain-invariant RMS CoV
            rms_std.clamp(0.0, 0.5) * 2.0,                   # [22]
            rms_max.clamp(0.0, 1.0),                         # [23]
            rms_slope.clamp(-5.0, 5.0) / 5.0,                # [24]
            crest_factor.clamp(0.0, 20.0) / 10.0,            # [25]
            energy_entropy,                                  # [26]
            spectral_centroid / 4000.0,                      # [27]
            spectral_spread / 2000.0,                        # [28]
            flux.clamp(0.0, 1.0) * 5.0,                      # [29]
            high_low_ratio.clamp(0.0, 10.0) / 5.0,           # [30]
            zcr * 5.0,                                       # [31]
        ], dtype=torch.float32, device=device)

        raw_dict = {
            "f0_mean": float(f0_mean.item()),
            "f0_std": float(f0_std.item()),
            "f0_slope": float(f0_slope.item()),
            "f0_curvature": float(f0_final_slope_st.item()),
            "voicing_ratio": float(voicing_ratio.item()),
            "jitter_local": float(jitter_local.item()),
            "jitter_rap": float(jitter_rap.item()),
            "shimmer_local": float(shimmer_local.item()),
            "hnr_db": float(hnr_db.item()),
            "rms_mean": float(rms_mean.item()),
            "spectral_centroid": float(spectral_centroid.item()),
        }

        return features, raw_dict
