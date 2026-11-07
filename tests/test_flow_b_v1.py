"""B6 pitch correction and B11 space at chain level."""

from dataclasses import replace

import numpy as np
import pytest

from tests.conftest import SR, index_of
from tests.test_pitch import sung
from voicefix.core import loudness
from voicefix.core.exporter import ExportRequest, export
from voicefix.core.project import Flow, Project, StepConfig, StepStatus
from voicefix.core.regions import Region
from voicefix.dsp import pitch
from voicefix.steps.registry import build_steps

C_MAJOR = {"key": "C 大調"}


@pytest.fixture
def detuned(make_session):
    # A4 +40 (off), C5 in tune, E5 -35 (off), G5 +10 (in tune)
    audio = sung([(6940, 0.9), (7200, 0.9), (7565, 0.9), (7910, 0.9)])
    return make_session(audio, flow=Flow.SINGING)


def note_cents(audio: np.ndarray) -> list[float]:
    return [n.cents for n in pitch.segment_notes(pitch.track_pitch(audio, SR))]


def test_natural_mode_lists_only_clearly_off_notes(detuned):
    project, chain = detuned
    i = index_of(chain, "B6")
    events = chain.events(project, i, StepConfig(settings=C_MAJOR))
    assert len(events) == 2
    assert events[0].label.startswith("偏高 40") and "A4" in events[0].label
    assert events[1].label.startswith("偏低 35") and "E5" in events[1].label


def test_natural_mode_corrects_off_notes_and_leaves_the_rest(detuned):
    project, chain = detuned
    i = index_of(chain, "B6")
    before = np.array(chain.render(project, i - 1))
    project = project.apply_step(i, StepConfig(strength=100, settings=C_MAJOR))
    after = chain.render(project, i)
    got = note_cents(after)
    assert got[0] == pytest.approx(6900, abs=6) and got[2] == pytest.approx(7600, abs=6)
    in_tune = pitch.segment_notes(pitch.track_pitch(before, SR))[1].region
    assert np.array_equal(after[in_tune.start : in_tune.end], before[in_tune.start : in_tune.end])


def test_unticked_note_is_left_alone(detuned):
    project, chain = detuned
    i = index_of(chain, "B6")
    cfg = StepConfig(strength=100, settings=C_MAJOR)
    first = chain.events(project, i, cfg)[0].region
    project = project.apply_step(i, replace(cfg, disabled_events=(first.start,)))
    before, after = chain.render(project, i - 1), chain.render(project, i)
    assert np.array_equal(after[first.start : first.end], before[first.start : first.end])


def test_pitch_preview_equals_full_render(detuned):
    project, chain = detuned
    i = index_of(chain, "B6")
    cfg = StepConfig(strength=80, settings=C_MAJOR)
    span = Region(int(0.8 * SR), int(3.0 * SR))  # starts inside the first note
    preview = chain.preview(project, i, cfg, span)
    full = chain.render(project.apply_step(i, cfg), i)
    assert np.allclose(preview.after, full[span.start : span.end], atol=1e-6)


def test_space_adds_reverb_and_dry_export_leaves_it_out(detuned, tmp_path):
    import soundfile as sf

    project, chain = detuned
    i = index_of(chain, "B11")
    project = project.apply_step(i, StepConfig(strength=60, option="hall"))
    before, after = chain.render(project, i - 1), chain.render(project, i)
    tail = slice(int(4.72 * SR), int(4.9 * SR))  # silence after the last note
    assert loudness.linear_to_db(np.abs(after[tail]).max()) > loudness.linear_to_db(np.abs(before[tail]).max()) + 20
    written = export(chain, project, ExportRequest(tmp_path / "song", "wav24", dry=True))
    assert sorted(p.name for p in written) == ["song.wav", "song_dry.wav"]
    dry, _ = sf.read(tmp_path / "song_dry.wav", dtype="float32", always_2d=True)
    assert np.abs(dry[tail]).max() < np.abs(after[tail]).max() * 0.1


def test_space_chunk_equals_whole(detuned):
    project, chain = detuned
    i = index_of(chain, "B11")
    cfg = StepConfig(strength=60, option="hall")
    span = Region(2 * SR, 3 * SR)
    preview = chain.preview(project, i, cfg, span)
    full = chain.render(project.apply_step(i, cfg), i)
    assert np.allclose(preview.after, full[span.start : span.end], atol=1e-5)


def test_old_flow_b_projects_migrate():
    steps = build_steps(Flow.SINGING)
    old_ids = [s.id for s in steps if s.code not in ("B6", "B11")]
    old = Project.new(Flow.SINGING, "x.wav", old_ids)
    for i in range(old_ids.index("B8_eq") + 1):  # an old project finished up to B8
        old = old.apply_step(i, StepConfig(strength=40))
    old = replace(old, current_index=old_ids.index("B8_eq"))
    new = old.migrated([s.id for s in steps])
    assert new.step_ids[new.current_index] == "B8_eq"
    assert new.steps["B6_pitch"].status == StepStatus.SKIPPED  # inserted before finished work
    assert new.steps["B11_space"].status == StepStatus.NOT_STARTED  # after the finished work
    assert new.steps["B8_eq"].status == StepStatus.APPLIED
    assert new.is_unlocked(new.current_index)


def test_pitch_requires_cleanup_steps_first():
    steps = build_steps(Flow.SINGING)
    project = Project.new(Flow.SINGING, "x.wav", [s.id for s in steps])
    b6 = [s.code for s in steps].index("B6")
    assert [s.code for s in steps][b6 - 4 : b6] == ["B2", "B3", "B4", "B5"]
    for i in range(b6 - 1):
        project = project.skip_step(i)
    assert not project.is_unlocked(b6)
    assert project.skip_step(b6 - 1).is_unlocked(b6)


def test_measure_not_repeated_when_toggling_notes(detuned, monkeypatch):
    project, chain = detuned
    i = index_of(chain, "B6")
    step = chain.steps[i]
    calls = []
    original = step.measure
    monkeypatch.setattr(step, "measure", lambda *a, **k: calls.append(1) or original(*a, **k))
    cfg = StepConfig(strength=80, settings=C_MAJOR)
    first = chain.events(project, i, cfg)[0].region
    chain.events(project, i, replace(cfg, disabled_events=(first.start,)))
    chain.events(project, i, replace(cfg, settings={"key": "D 大調"}))
    chain.preview(project, i, cfg, Region(SR, 2 * SR))
    assert len(calls) == 1


def test_effect_mode_flattens_vibrato_onto_the_scale(make_session):
    audio = sung([(6930, 1.2), (7180, 1.2)], vibrato_cents=40)
    project, chain = make_session(audio, flow=Flow.SINGING)
    i = index_of(chain, "B6")
    before = pitch.track_pitch(np.array(chain.render(project, i - 1)), SR)
    project = project.apply_step(i, StepConfig(strength=100, option="effect", settings=C_MAJOR))
    after = pitch.track_pitch(np.array(chain.render(project, i)), SR)
    notes = pitch.segment_notes(before)
    assert len(pitch.segment_notes(after)) == len(notes) == 2
    for note, target in zip(notes, (6900, 7200)):
        a, b = note.frames
        inner = slice(a + 10, b - 10)
        assert np.nanstd(after.cents[inner]) < np.nanstd(before.cents[inner]) * 0.5
        assert np.nanmedian(after.cents[inner]) == pytest.approx(target, abs=5)


def test_effect_preview_equals_full_render(make_session):
    audio = sung([(6930, 1.2), (7180, 1.2)], vibrato_cents=40)
    project, chain = make_session(audio, flow=Flow.SINGING)
    i = index_of(chain, "B6")
    cfg = StepConfig(strength=80, option="effect", settings=C_MAJOR)
    span = Region(int(0.9 * SR), int(2.2 * SR))
    preview = chain.preview(project, i, cfg, span)
    full = chain.render(project.apply_step(i, cfg), i)
    assert np.array_equal(preview.after, full[span.start : span.end])


@pytest.fixture
def way_off(make_session):
    # E4 sung 80 cents sharp: nearer to F4, so it looks like an in-tune F4
    return make_session(sung([(6400 + 80, 0.9), (7200, 0.9)]), flow=Flow.SINGING)


def test_far_off_note_is_mistaken_for_its_neighbour(way_off):
    project, chain = way_off
    i = index_of(chain, "B6")
    assert chain.events(project, i, StepConfig(settings=C_MAJOR)) == []
    notes = chain.analyze(project, i, config=StepConfig(settings=C_MAJOR)).data["notes"]
    assert note_name_of(notes[0]["target"]) == "F4" and not notes[0]["manual"]


def note_name_of(cents: float) -> str:
    from voicefix.dsp.key import note_name

    return note_name(cents)


def test_hand_picked_target_fixes_a_far_off_note(way_off):
    project, chain = way_off
    i = index_of(chain, "B6")
    first = pitch.segment_notes(pitch.track_pitch(np.array(chain.render(project, i - 1)), SR))[0]
    cfg = StepConfig(strength=100, settings=C_MAJOR, note_targets=((first.region.start, 6400),))
    events = chain.events(project, i, cfg)
    assert len(events) == 1 and events[0].label.startswith("偏高 80") and "E4，手動" in events[0].label
    got = note_cents(chain.render(project.apply_step(i, cfg), i))
    assert got[0] == pytest.approx(6400, abs=8)
    assert got[1] == pytest.approx(7200, abs=6)


def test_note_targets_saved_and_per_track(tmp_path):
    import soundfile as sf

    from voicefix.core.project import load_project, save_project
    from voicefix.ui.multisession import MultiSession

    steps_project = Project.new(Flow.SINGING, "x.wav", [s.id for s in build_steps(Flow.SINGING)])
    cfg = StepConfig(note_targets=((48_000, 6400),))
    steps_project = steps_project.with_config(5, cfg)
    save_project(steps_project, tmp_path / "p.voicefix")
    assert load_project(tmp_path / "p.voicefix").config(5).note_targets == ((48_000, 6400),)

    paths = [tmp_path / "a.wav", tmp_path / "b.wav"]
    for p in paths:
        sf.write(p, sung([(6400, 0.5)]), SR)
    session = MultiSession.create(["主唱", "和聲"], paths, tmp_path / "c.voicefix", tmp_path / "cache")
    c6 = next(i for i, s in enumerate(session.steps) if s.code == "C6")
    session.update(session.project.with_config(c6, replace(session.project.config(c6), note_targets=((100, 6500),))))
    assert session.multi.project(1).config(c6).note_targets == ()


def test_target_too_far_from_the_sung_note_is_ignored(way_off):
    project, chain = way_off
    i = index_of(chain, "B6")
    first = pitch.segment_notes(pitch.track_pitch(np.array(chain.render(project, i - 1)), SR))[0]
    cfg = StepConfig(strength=100, settings=C_MAJOR, note_targets=((first.region.start, 6480 - 1200),))  # an octave
    assert chain.events(project, i, cfg) == []


def test_nearest_target_wins_for_close_notes():
    from voicefix.steps.base import StepContext

    ctx = StepContext(SR, 10 * SR, (), (), None, False, (), {}, note_targets=((1000, 6400), (2400, 6500)))
    assert ctx.target_for(Region(2000, 3000)) == 6500  # 400 samples from the second, 1000 from the first
    assert ctx.target_for(Region(1100, 2000)) == 6400
