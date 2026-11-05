from pathlib import Path

import numpy as np
import soundfile as sf

from tests.conftest import SR, speech_like, stereo
from voicefix.core.project import Flow, StepStatus
from voicefix.core.regions import Region
from voicefix.ui.peaks import PeakPyramid
from voicefix.ui.session import Session


def test_envelope_bounds_contain_signal():
    audio = stereo(speech_like(5.0))
    pyramid = PeakPyramid(audio)
    for start, end in [(0, len(audio)), (SR, SR + 2000), (100, 300)]:
        x, lo, hi = pyramid.envelope(start, end, 500)
        assert len(x) > 0 and np.all(lo <= hi)
        assert hi.max() <= audio[start:end].max() + 1e-6
        assert lo.min() >= audio[start:end].min() - 1e-6
    assert pyramid.envelope(0, len(audio), 500)[2].max() == audio.max()


def test_session_create_reopen_and_shared_settings(tmp_path: Path):
    src = tmp_path / "voice.wav"
    sf.write(src, stereo(speech_like(3.0)), SR)
    project_path = tmp_path / "voice.voicefix"
    session = Session.create(Flow.VOICE_WORK, src, project_path, tmp_path / "cache")
    session.update(session.project.apply_step(2, session.project.config(2)))
    session.set_shared("protected", (Region(0, 100),))
    assert session.project.status(2) == StepStatus.APPLIED  # low cut ignores protection

    reopened = Session.open(project_path, tmp_path / "cache")
    assert reopened.project == session.project
    assert reopened.chain.source_key == session.chain.source_key
