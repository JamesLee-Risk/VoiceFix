"""``VoiceFix.exe --self-test <report.txt>``: load every native component and run a
tiny render, so a packaged build can be checked on any computer.

Each check writes one line ("OK name" or "FAIL name: reason") to the report;
the exit code is 0 only when all of them pass.
"""

from __future__ import annotations

import tempfile
import traceback
from pathlib import Path
from typing import Callable

import numpy as np

SR = 48_000


def _tone(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    mono = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return np.stack([mono, mono], axis=1)


def _audio_files(work: Path) -> None:
    import soundfile as sf

    sf.write(work / "t.wav", _tone(0.5), SR)
    sf.write(work / "t.flac", _tone(0.5), SR)
    sf.write(work / "t.mp3", _tone(0.5), SR)
    data, sr = sf.read(work / "t.flac", always_2d=True)
    assert sr == SR and data.shape[1] == 2


def _playback(work: Path) -> None:
    import sounddevice as sd

    assert sd.get_portaudio_version()[0] > 0


def _resample(work: Path) -> None:
    import soxr

    assert len(soxr.resample(_tone(0.2), SR, 44_100)) > 0


def _rubberband(work: Path) -> None:
    import pedalboard

    out = pedalboard.time_stretch(_tone(0.5).T.copy(), SR, 1.0, 1.0, high_quality=False)
    assert out.shape[0] == 2


def _pitch(work: Path) -> None:
    import pyworld

    mono = _tone(0.5)[:, 0].astype(np.float64)
    f0, _ = pyworld.dio(mono, SR, frame_period=10)
    assert np.median(f0[f0 > 0]) > 200


def _interface(work: Path) -> None:
    import pyqtgraph  # noqa: F401
    from PySide6.QtWidgets import QWidget  # noqa: F401


def _render(work: Path) -> None:
    import soundfile as sf

    from voicefix.core.cache import AudioCache, load_source
    from voicefix.core.chain import StepChain
    from voicefix.core.project import Flow, Project, StepConfig
    from voicefix.steps.registry import build_steps

    sf.write(work / "take.wav", _tone(2.0), SR)
    cache = AudioCache(work / "cache")
    source, sr, key = load_source(work / "take.wav", cache)
    for flow in (Flow.VOICE_WORK, Flow.SINGING):
        steps = build_steps(flow)
        project = Project.new(flow, str(work / "take.wav"), [s.id for s in steps])
        for i, step in enumerate(steps):
            project = project.apply_step(i, StepConfig(strength=step.default_strength, option=step.default_option))
        out = StepChain(steps, cache, source, sr, key).render(project, len(steps) - 1)
        assert np.all(np.isfinite(out))


CHECKS: tuple[tuple[str, Callable[[Path], None]], ...] = (
    ("讀寫音檔（WAV／FLAC／MP3）", _audio_files),
    ("播放（PortAudio）", _playback),
    ("重取樣（soxr）", _resample),
    ("移調（Rubber Band）", _rubberband),
    ("音高偵測（WORLD）", _pitch),
    ("介面（Qt、pyqtgraph）", _interface),
    ("流程 A、B 全部步驟", _render),
)


def run(report: Path) -> int:
    lines, failed = [], False
    with tempfile.TemporaryDirectory(prefix="voicefix-selftest-") as tmp:
        for name, check in CHECKS:
            try:
                check(Path(tmp))
                lines.append(f"OK   {name}")
            except Exception:  # report every failure, keep checking the rest
                failed = True
                lines.append(f"FAIL {name}: {traceback.format_exc(limit=3)}")
    lines.append("結果：" + ("有項目失敗" if failed else "全部通過"))
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 1 if failed else 0
