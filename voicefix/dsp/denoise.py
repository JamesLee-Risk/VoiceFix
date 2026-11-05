"""Stereo-linked spectral-gate noise reduction with a learned noise profile.

One real-valued gain mask is computed from the channel-averaged power and
applied to every channel, so inter-channel timing and phase are untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter

@dataclass(frozen=True)
class DenoiseSettings:
    reduction_db: float  # maximum attenuation per pass
    over_subtraction: float  # how aggressively noise power is subtracted
    smoothing_frames: int = 3
    smoothing_bins: int = 3


def stft_size(sr: int) -> int:
    return 2048 if sr <= 50_000 else 4096


def _stft(audio: np.ndarray, sr: int) -> np.ndarray:
    n = stft_size(sr)
    _, _, spec = signal.stft(audio.T, fs=sr, window="hann", nperseg=n, noverlap=n * 3 // 4, boundary="even", padded=True)
    return spec  # (channels, bins, frames)


def _istft(spec: np.ndarray, sr: int, length: int) -> np.ndarray:
    n = stft_size(sr)
    _, out = signal.istft(spec, fs=sr, window="hann", nperseg=n, noverlap=n * 3 // 4, boundary=True)
    out = out.T[:length]
    if len(out) < length:
        out = np.pad(out, ((0, length - len(out)), (0, 0)))
    return out.astype(np.float32)


def learn_profile(noise: np.ndarray, sr: int) -> np.ndarray:
    """Mean linked noise power per frequency bin."""
    power = (np.abs(_stft(noise, sr)) ** 2).mean(axis=0)
    return power.mean(axis=1)


def gain_mask(power: np.ndarray, profile: np.ndarray, settings: DenoiseSettings) -> np.ndarray:
    floor = 10.0 ** (-settings.reduction_db / 20.0)
    noise = settings.over_subtraction * profile[:, None]
    wiener = np.sqrt(np.clip(1.0 - noise / np.maximum(power, 1e-20), 0.0, 1.0))
    smoothed = uniform_filter(wiener, size=(settings.smoothing_bins, settings.smoothing_frames), mode="nearest")
    return floor + (1.0 - floor) * smoothed


def reduce_noise(
    audio: np.ndarray, sr: int, profile: np.ndarray, settings: DenoiseSettings, offset: int = 0
) -> np.ndarray:
    """``offset`` is the absolute position of ``audio``; frames are aligned to a
    global hop grid so a preview chunk matches the full render."""
    if len(audio) < stft_size(sr):
        return np.asarray(audio, dtype=np.float32)
    lead = offset % (stft_size(sr) // 4)
    data = np.pad(np.asarray(audio, dtype=np.float64), ((lead, 0), (0, 0)))
    spec = _stft(data, sr)
    power = (np.abs(spec) ** 2).mean(axis=0)
    mask = gain_mask(power, profile, settings)
    return _istft(spec * mask[None, :, :], sr, len(data))[lead:]


def multi_pass_profiles(noise: np.ndarray, sr: int, settings: DenoiseSettings, passes: int) -> list[np.ndarray]:
    """Profiles for each pass, relearned from the noise sample after the previous pass."""
    profiles = []
    sample = np.asarray(noise, dtype=np.float32)
    for _ in range(passes):
        profile = learn_profile(sample, sr)
        profiles.append(profile)
        sample = reduce_noise(sample, sr, profile, settings)
    return profiles


def apply_reference_mask(
    audio: np.ndarray, sr: int, clean_mono: np.ndarray, reduction_db: float, offset: int = 0, smoothing: int = 3
) -> np.ndarray:
    """Apply, to every channel, the per-bin gain that turned the channel average
    into ``clean_mono``. The gain is real-valued and limited to ``reduction_db``."""
    lead = offset % (stft_size(sr) // 4)
    pad = ((lead, 0), (0, 0))
    data = np.pad(np.asarray(audio, dtype=np.float64), pad)
    spec = _stft(data, sr)
    mix = np.abs(_stft(data.mean(axis=1, keepdims=True), sr))[0]
    clean = np.abs(_stft(np.pad(np.asarray(clean_mono, dtype=np.float64)[:, None], pad), sr))[0]
    floor = 10.0 ** (-reduction_db / 20.0)
    gain = np.clip(clean / np.maximum(mix, 1e-12), floor, 1.0)
    gain = uniform_filter(gain, size=(smoothing, smoothing), mode="nearest")
    return _istft(spec * gain[None, :, :], sr, len(data))[lead:]
