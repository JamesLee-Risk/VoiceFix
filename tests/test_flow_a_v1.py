"""A7 breaths and A9 room-tone fill."""

import numpy as np
import pytest

from tests.conftest import SR, index_of, noise, speech_like, stereo
from tests.singing import sung_take
from voicefix.core.project import StepConfig
from voicefix.core.regions import Region
from voicefix.dsp import roomtone


def rms_db(x: np.ndarray) -> float:
    return 10 * np.log10(np.mean(np.asarray(x, np.float64) ** 2) + 1e-20)


def test_a7_default_lowers_breaths_by_three_db(make_session):
    take = sung_take()
    project, chain = make_session(take.audio)
    i = index_of(chain, "A7")
    assert len(chain.events(project, i, StepConfig())) >= len(take.breaths) - 1
    project = project.apply_step(i, StepConfig(strength=chain.steps[i].default_strength))
    before, after = chain.render(project, i - 1), chain.render(project, i)
    lo, hi = take.breaths[3]
    mid = slice(lo + (hi - lo) // 3, hi - (hi - lo) // 3)
    assert rms_db(after[mid]) - rms_db(before[mid]) == pytest.approx(-3.0, abs=0.4)


@pytest.fixture
def muted(make_session):
    audio = stereo(speech_like(10.0)) + noise(10 * SR, -55)
    project, chain = make_session(audio)
    mute = Region(int(4.0 * SR), int(5.5 * SR))
    project = project.apply_step(index_of(chain, "A1"), StepConfig(settings={"mutes": [mute.to_json()]}))
    return project, chain, mute


def test_a9_fills_digital_silence_at_room_level(muted):
    project, chain, mute = muted
    i = index_of(chain, "A9")
    analysis = chain.analyze(project, i)
    assert analysis.recommend and any(m.overlaps(mute) for m in analysis.marks)
    project = project.apply_step(i, StepConfig(strength=50))
    before, after = chain.render(project, i - 1), chain.render(project, i)
    inner = slice(mute.start + SR // 10, mute.end - SR // 10)
    assert np.max(np.abs(before[inner])) == 0.0
    assert rms_db(after[inner]) == pytest.approx(-55.0, abs=2.0)  # same level as the room
    assert np.array_equal(after[: 3 * SR], before[: 3 * SR])  # elsewhere untouched


def test_a9_preview_equals_render_and_is_deterministic(muted):
    project, chain, mute = muted
    i = index_of(chain, "A9")
    cfg = StepConfig(strength=60)
    span = Region(mute.start - SR // 3, mute.start + SR // 2)
    preview = chain.preview(project, i, cfg, span)
    full = chain.render(project.apply_step(i, cfg), i)
    assert np.array_equal(preview.after, full[span.start : span.end])


def test_room_tone_level_is_steady():
    room = noise(3 * SR, -50)
    tone = roomtone.room_tone(room, SR, 12_345, 10 * SR)
    levels = [rms_db(tone[k * SR : (k + 1) * SR]) for k in range(10)]
    assert max(levels) - min(levels) < 1.0 and np.mean(levels) == pytest.approx(-50.0, abs=1.0)
    assert np.array_equal(roomtone.room_tone(room, SR, 5 * SR, SR), tone[5 * SR - 12_345 : 6 * SR - 12_345])


def test_room_tone_full_level_from_the_first_sample():
    room = noise(3 * SR, -50)
    tone = roomtone.room_tone(room, SR, 0, SR)
    assert rms_db(tone[: SR // 20]) == pytest.approx(rms_db(tone[SR // 2 :]), abs=1.5)
