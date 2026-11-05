"""Optional AI noise reduction (DeepFilterNet), applied as a stereo-linked mask.

DeepFilterNet cleans the channel average; the per-bin gain it achieved is then
applied to every channel with a real-valued mask. That keeps the guarantees of
the classic engine: no inter-channel change (binaural safe) and no phase shift.

Install with ``pip install -e ".[ai]"``. The model is downloaded once on first
use; audio never leaves the computer.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import types
from typing import Any, Optional

import numpy as np
import soxr

from voicefix.dsp import denoise

DF_SR = 48_000
DF_HOP = 480
INSTALL_HINT = '需要先安裝 AI 降噪元件：pip install -e ".[ai]"'

_lock = threading.Lock()
_model: Optional[tuple[Any, Any]] = None


class AIDenoiseUnavailable(RuntimeError):
    pass


def is_available() -> bool:
    return importlib.util.find_spec("df") is not None and importlib.util.find_spec("torch") is not None


def _install_torchaudio_shim() -> None:
    """DeepFilterNet 0.5.6 imports ``torchaudio.backend.common`` (removed in
    torchaudio 2.1) only for the type hints of file I/O helpers VoiceFix never
    calls. Provide a stand-in so the package imports on current torchaudio."""
    try:
        import torchaudio.backend.common  # noqa: F401
    except ImportError:
        common = types.ModuleType("torchaudio.backend.common")
        common.AudioMetaData = type("AudioMetaData", (), {})
        sys.modules.setdefault("torchaudio.backend", types.ModuleType("torchaudio.backend"))
        sys.modules["torchaudio.backend.common"] = common


def _load() -> tuple[Any, Any]:
    global _model
    if _model is None:
        if not is_available():
            raise AIDenoiseUnavailable(INSTALL_HINT)
        _install_torchaudio_shim()
        from df.enhance import init_df

        model, state, _ = init_df(log_level="ERROR")
        _model = (model, state)
    return _model


def enhance_mono(mono: np.ndarray, attenuation_db: float) -> np.ndarray:
    """DeepFilterNet on a 48 kHz mono signal; serialised because the model is shared."""
    with _lock:
        model, state = _load()  # installs the torchaudio shim before df.enhance is imported
        import torch
        from df.enhance import enhance

        with torch.no_grad():
            out = enhance(model, state, torch.from_numpy(np.ascontiguousarray(mono, dtype=np.float32)[None]),
                          atten_lim_db=attenuation_db)
    return out.numpy()[0].astype(np.float32)


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))


def reduce_noise_ai(audio: np.ndarray, sr: int, reduction_db: float, offset: int = 0) -> np.ndarray:
    """``offset`` aligns DeepFilterNet's hop grid to absolute positions (exact at
    48 kHz), so previews match the full render as closely as the model allows."""
    data = np.asarray(audio, dtype=np.float32)
    if len(data) < denoise.stft_size(sr):
        return data
    lead = offset % DF_HOP if sr == DF_SR else 0
    mono = np.pad(data.mean(axis=1), (lead, 0))
    mono48 = mono if sr == DF_SR else soxr.resample(mono, sr, DF_SR, quality="HQ")
    clean48 = enhance_mono(mono48, reduction_db)
    clean = _fit(clean48 if sr == DF_SR else soxr.resample(clean48, DF_SR, sr, quality="HQ"), len(mono))[lead:]
    return denoise.apply_reference_mask(data, sr, clean, reduction_db, offset)
