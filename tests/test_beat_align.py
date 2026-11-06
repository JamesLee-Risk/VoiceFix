"""B7 / C7a: the vocal follows the accompaniment's beat."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tests.conftest import SR, noise
from tests.test_beats import BEAT, drum_loop
from voicefix.core.project import Flow, StepConfig, StepStatus
from voicefix.core.regions import Region
from voicefix.dsp import beats
from voicefix.ui.session import Session

EIGHTH = BEAT / 2
FIRST_BEAT = 0.3
# (grid index in eighths, error in seconds): two late, one early, the rest on the grid
SYLLABLES = [(4, 0.0), (6, 0.05), (8, 0.0), (10, -0.04), (12, 0.0), (16, 0.0), (18, 0.045), (20, 0.0)]


def sung_on_grid(seconds: float) -> np.ndarray:
    n = int(seconds * SR)
    mono = np.zeros(n, np.float32)
    for k, (slot, err) in enumerate(SYLLABLES):
        start = int((FIRST_BEAT + slot * EIGHTH + err) * SR)
        t = np.arange(int(0.25 * SR)) / SR
        hz = 220 * 2 ** ((k % 4) / 12)
        tone = sum(np.sin(2 * np.pi * hz * h * t) / h for h in range(1, 6))
        env = np.minimum(1, t / 0.004) * np.minimum(1, (0.25 - t) / 0.03)
        mono[start : start + len(t)] += (0.15 * tone / 2.3 * env).astype(np.float32)
    return np.stack([mono, mono], axis=1) + noise(n, -66)


def onset_errors(audio: np.ndarray) -> list[float]:
    """Distance (s) from each expected syllable's grid point to the detected onset."""
    found = beats.vocal_onsets(audio, SR) / SR
    out = []
    for slot, _ in SYLLABLES:
        grid = FIRST_BEAT + slot * EIGHTH
        out.append(float(found[np.argmin(np.abs(found - grid))] - grid))
    return out


@pytest.fixture
def session(tmp_path: Path) -> Session:
    vocal, backing = tmp_path / "vocal.wav", tmp_path / "backing.wav"
    sf.write(vocal, sung_on_grid(12.0), SR)
    sf.write(backing, drum_loop(12.0), SR)
    s = Session.create(Flow.SINGING, vocal, tmp_path / "song.voicefix", tmp_path / "cache")
    s.set_accompaniment(backing)
    return s


def b7(session: Session) -> int:
    return next(i for i, step in enumerate(session.steps) if step.code == "B7")


def test_lists_only_the_off_beat_syllables(session):
    i = b7(session)
    events = session.chain.events(session.project, i, StepConfig(option="eighth"))
    labels = [e.label.split()[0] for e in events]
    assert labels == ["慢", "搶拍", "慢"]
    assert float(events[0].label.split()[1]) == pytest.approx(50, abs=8)  # planted 50 ms late


def test_moves_late_and_early_syllables_onto_the_beat(session):
    i = b7(session)
    before = onset_errors(np.asarray(session.chain.render(session.project, i - 1)))
    project = session.project.apply_step(i, StepConfig(strength=100, option="eighth"))
    after = onset_errors(np.asarray(session.chain.render(project, i)))
    for (slot, err), b, a in zip(SYLLABLES, before, after):
        if err:
            assert abs(b) > 0.03 and abs(a) < 0.012, (slot, b, a)
        else:
            assert abs(a - b) < 0.011, (slot, b, a)  # on-beat syllables stay (within WSOLA and 5 ms detection precision)


def test_strength_keeps_part_of_the_feel(session):
    i = b7(session)
    project = session.project.apply_step(i, StepConfig(strength=50, option="eighth"))
    after = onset_errors(np.asarray(session.chain.render(project, i)))
    assert after[1] == pytest.approx(0.025, abs=0.012)  # half of the 50 ms


def test_unticked_syllable_stays(session):
    i = b7(session)
    cfg = StepConfig(strength=100, option="eighth")
    first = session.chain.events(session.project, i, cfg)[0].region
    from dataclasses import replace

    project = session.project.apply_step(i, replace(cfg, disabled_events=(first.start,)))
    after = onset_errors(np.asarray(session.chain.render(project, i)))
    assert after[1] == pytest.approx(0.05, abs=0.01)


def test_preview_equals_render(session):
    i = b7(session)
    cfg = StepConfig(strength=80, option="eighth")
    span = Region(int(2.0 * SR), int(5.0 * SR))
    preview = session.chain.preview(session.project, i, cfg, span)
    full = session.chain.render(session.project.apply_step(i, cfg), i)
    assert np.array_equal(preview.after, full[span.start : span.end])


def test_without_accompaniment_it_suggests_skipping(tmp_path: Path):
    vocal = tmp_path / "v.wav"
    sf.write(vocal, sung_on_grid(12.0), SR)
    s = Session.create(Flow.SINGING, vocal, tmp_path / "v.voicefix", tmp_path / "cache")
    i = b7(s)
    analysis = s.chain.analyze(s.project, i)
    assert not analysis.recommend and "伴奏" in analysis.summary[0]
    project = s.project.apply_step(i, StepConfig(strength=100))
    assert np.array_equal(s.chain.render(project, i), s.chain.render(project, i - 1))


def test_changing_accompaniment_makes_b7_stale(session, tmp_path: Path):
    i = b7(session)
    for j in range(i + 1):
        session.update(session.project.apply_step(j, StepConfig(strength=50)))
    other = tmp_path / "other.wav"
    sf.write(other, drum_loop(12.0, bpm=90.0), SR)
    session.set_accompaniment(other)
    assert session.project.status(i) == StepStatus.STALE
    assert session.project.status(i - 1) == StepStatus.APPLIED


def test_flow_c_beat_then_lead_then_harmony(tmp_path: Path):
    """C7a puts the lead on the beat; C7b makes the harmony follow that lead."""
    from voicefix.ui.multisession import MultiSession

    lead = sung_on_grid(12.0)
    harmony = np.roll(lead, int(0.03 * SR), axis=0) * 0.6  # 30 ms behind the (still off-beat) lead
    paths = [tmp_path / "lead.wav", tmp_path / "harmony.wav", tmp_path / "backing.wav"]
    for path, audio in zip(paths, (lead, harmony, drum_loop(12.0))):
        sf.write(path, audio, SR)
    s = MultiSession.create(["主唱", "和聲"], paths[:2], tmp_path / "c.voicefix", tmp_path / "cache")
    s.set_accompaniment(paths[2])
    a = next(i for i, step in enumerate(s.steps) if step.code == "C7a")
    b = a + 1
    for j in range(a):
        s.update(s.project.skip_step(j))
    for index, cfg in ((a, StepConfig(strength=100, option="eighth")), (b, StepConfig(strength=100))):
        candidate = s.project.apply_step(index, cfg)
        s.render_step(candidate, index, lambda m, f: None, None)
        s.update(candidate)
    harmony_c7a = s.chains[1].render(s.multi.project(1), a)
    assert np.array_equal(harmony_c7a, s.chains[1].render(s.multi.project(1), a - 1))  # C7a leaves harmonies
    lead_out = onset_errors(np.asarray(s.chains[0].render(s.multi.project(0), b)))
    harmony_out = onset_errors(np.asarray(s.chains[1].render(s.multi.project(1), b)))
    for (slot, err), lo, ho in zip(SYLLABLES, lead_out, harmony_out):
        assert abs(lo) < 0.012, ("lead", slot, lo)
        # C7b estimates the lead offset over 0.6 s windows, so single syllables the lead
        # moved in C7a are followed to within about 20 ms, not exactly
        assert abs(ho) < 0.02, ("harmony", slot, ho)


def test_gap_cleanup_before_b7_does_not_fake_onsets(session):
    """B2 lowers the gaps; the room tone returning at a gap's end must not look like a syllable."""
    gaps = next(i for i, step in enumerate(session.steps) if step.code == "B2")
    session.update(session.project.apply_step(gaps, StepConfig(strength=50, option="reduce")))
    i = b7(session)
    for j in range(gaps + 1, i):
        session.update(session.project.skip_step(j))
    for option in ("eighth", "sixteenth"):
        labels = [e.label.split()[0] for e in session.chain.events(session.project, i, StepConfig(option=option))]
        assert labels == ["慢", "搶拍", "慢"], option


def test_time_map_is_monotonic_and_moves_phrase_starts():
    from voicefix.steps.timing import MAX_SLOPE, beat_anchors

    seg = [Region(SR, 3 * SR), Region(int(3.3 * SR), 5 * SR)]
    onsets = np.array([SR, int(1.6 * SR), int(2.7 * SR), int(3.3 * SR)])
    moves = {SR: 0.06 * SR, int(2.7 * SR): 0.12 * SR, int(3.3 * SR): -0.06 * SR}  # late at a phrase start, etc.
    curve = beat_anchors(onsets, moves, seg, SR)
    t = np.arange(0, 6 * SR)
    offset = curve.at_samples(0, len(t))
    read = t + offset
    assert np.all(np.diff(read) > 0)  # never backwards, never a jump
    assert np.max(np.abs(np.diff(offset))) <= MAX_SLOPE + 1e-9
    assert offset[0] == 0 and offset[-1] == 0
    landed = np.interp(SR, read, t)  # where the first (late, phrase-start) onset ends up
    assert SR - landed == pytest.approx(0.06 * SR, rel=0.05)  # the full 60 ms move
