from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voicefix.core.cache import AudioCache, load_source
from voicefix.core.chain import StepChain
from voicefix.core.project import Flow, Project
from voicefix.steps.registry import build_steps

SR = 48_000


def speech_like(seconds: float, sr: int = SR, seed: int = 0, level: float = 0.3) -> np.ndarray:
    """Harmonic 'syllables' separated by pauses, mono (n,)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    f0 = 140 + 20 * np.sin(2 * np.pi * 0.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    voice = sum(np.sin(k * phase) / k for k in range(1, 12))
    envelope = np.zeros(n)
    pos = int(0.4 * sr)
    while pos < n - sr // 2:
        length = int(rng.uniform(0.6, 1.4) * sr)
        end = min(n, pos + length)
        envelope[pos:end] = np.hanning(end - pos)
        pos = end + int(rng.uniform(0.4, 0.8) * sr)
    return (level * voice * envelope / np.max(np.abs(voice))).astype(np.float32)


def noise(n: int, level_db: float, seed: int = 1, channels: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, channels)) * 10 ** (level_db / 20)).astype(np.float32)


def stereo(mono: np.ndarray) -> np.ndarray:
    return np.stack([mono, mono], axis=1)


@pytest.fixture
def make_session(tmp_path: Path):
    """Write ``audio`` to a WAV and return (project, chain) for the given flow."""

    def factory(audio: np.ndarray, sr: int = SR, flow: Flow = Flow.VOICE_WORK) -> tuple[Project, StepChain]:
        path = tmp_path / "take.wav"
        sf.write(path, audio, sr, subtype="FLOAT")
        cache = AudioCache(tmp_path / "cache")
        source, rate, key = load_source(path, cache)
        steps = build_steps(flow)
        project = Project.new(flow, str(path), [s.id for s in steps])
        return project, StepChain(steps, cache, source, rate, key)

    return factory


def index_of(chain: StepChain, code: str) -> int:
    return next(i for i, s in enumerate(chain.steps) if s.code == code)
