"""Flow C: multi-track singing (lead + harmonies)."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tests.conftest import SR
from tests.test_align import ONSETS, SHIFTS, phrase_track
from voicefix.core import loudness
from voicefix.core.exporter import ExportRequest
from voicefix.core.multitrack import MAX_TRACKS, MultiProject, load_multiproject
from voicefix.core.project import StepConfig, StepStatus
from voicefix.dsp import align
from voicefix.ui.multisession import MultiSession


@pytest.fixture
def session(tmp_path: Path) -> MultiSession:
    lead = phrase_track(ONSETS, 6900, [0.0] * 12)
    harmony = phrase_track(ONSETS, 7300, SHIFTS) * 0.5
    paths = [tmp_path / "lead.wav", tmp_path / "harmony.wav"]
    for path, audio in zip(paths, (lead, harmony)):
        sf.write(path, audio, SR)
    return MultiSession.create(["主唱", "和聲 1"], paths, tmp_path / "song.voicefix", tmp_path / "cache")


def index_of(session: MultiSession, code: str) -> int:
    return next(i for i, s in enumerate(session.steps) if s.code == code)


def apply(session: MultiSession, index: int, config: StepConfig) -> None:
    candidate = session.project.apply_step(index, config)
    session.render_step(candidate, index, lambda m, f: None, None)
    session.update(candidate)


def test_track_limit():
    with pytest.raises(ValueError):
        MultiProject.new([f"t{i}" for i in range(MAX_TRACKS + 1)], ["x.wav"] * (MAX_TRACKS + 1), ["C0_import"])


def test_parameters_shared_unless_unlinked(session):
    i = index_of(session, "C3")
    session.update(session.project.apply_step(i, StepConfig(strength=40)))
    assert session.multi.project(1).config(i).strength == 40
    assert session.multi.project(1).status(i) == StepStatus.APPLIED

    session.current_track = 1
    session.set_linked(session.steps[i].id, False)
    session.update(session.project.apply_step(i, StepConfig(strength=10)))
    assert session.multi.project(0).config(i).strength == 40  # lead keeps the shared value
    assert session.multi.project(1).config(i).strength == 10

    session.set_linked(session.steps[i].id, True)  # re-joining takes the shared value back
    assert session.multi.project(1).config(i).strength == 40


def test_events_stay_per_track(session):
    i = index_of(session, "C4")
    session.update(session.project.apply_step(i, StepConfig(strength=30, disabled_events=(12345,))))
    assert session.multi.project(1).config(i).disabled_events == ()


def test_c7_aligns_harmony_and_leaves_lead_alone(session):
    i = index_of(session, "C7b")
    for j in range(i):
        session.update(session.project.skip_step(j))
    apply(session, i, StepConfig(strength=100))
    lead_in = session.chains[0].render(session.multi.project(0), i - 1)
    lead_out = session.chains[0].render(session.multi.project(0), i)
    assert np.array_equal(lead_in, lead_out)
    harmony_out = session.chains[1].render(session.multi.project(1), i)
    after = align.estimate_offsets(np.asarray(lead_out), np.asarray(harmony_out), SR, 0.15)
    for onset in (ONSETS[2], ONSETS[9]):
        assert abs(after.at_samples(int(onset * SR), 1)[0] / SR) < 0.008


def test_c7_preview_equals_render(session):
    i = index_of(session, "C7b")
    for j in range(i):
        session.update(session.project.skip_step(j))
    session.current_track = 1
    cfg = StepConfig(strength=80)
    from voicefix.core.regions import Region

    span = Region(int(5.0 * SR), int(8.0 * SR))
    preview = session.chain.preview(session.project, i, cfg, span)
    apply(session, i, cfg)
    full = session.chain.render(session.project, i)
    assert np.array_equal(preview.after, full[span.start : span.end])


def test_c12_one_gain_for_all_tracks_keeps_balance(session):
    i = index_of(session, "C12")
    for j in range(i):
        session.update(session.project.skip_step(j))
    before = [np.array(c.render(session.multi.project(t), i - 1)) for t, c in enumerate(session.chains)]
    apply(session, i, StepConfig(option="stem"))
    after = [np.array(c.render(session.multi.project(t), i)) for t, c in enumerate(session.chains)]
    assert loudness.integrated_loudness(after[0], SR) == pytest.approx(-20.0, abs=0.5)
    gain_lead = loudness.integrated_loudness(after[0], SR) - loudness.integrated_loudness(before[0], SR)
    gain_harmony = loudness.integrated_loudness(after[1], SR) - loudness.integrated_loudness(before[1], SR)
    assert gain_harmony == pytest.approx(gain_lead, abs=0.3)


def test_export_stems_and_mix(session, tmp_path: Path):
    for j in range(len(session.steps) - 1):
        session.update(session.project.skip_step(j))
    last = len(session.steps) - 1
    project = session.project.apply_step(last, StepConfig(option="stem"))
    written = session.export(project, ExportRequest(tmp_path / "out", "wav16", mix=True), lambda m, f: None, None)
    assert [p.name for p in written] == ["out_1_主唱.wav", "out_2_和聲_1.wav", "out_mix.wav"]
    mix, sr = sf.read(written[-1], always_2d=True)
    assert sr == SR and len(mix) == max(len(c.source) for c in session.chains)


def test_project_file_round_trip(session):
    session.current_track = 1
    session.set_linked(session.steps[3].id, False)
    reopened = MultiSession.open(session.project_path, session.chains[0].cache.directory.parent.parent)
    assert reopened.multi == load_multiproject(session.project_path)
    assert reopened.current_track == 1 and not reopened.is_linked(session.steps[3].id)


def test_mismatched_sample_rates_rejected(tmp_path: Path):
    paths = [tmp_path / "a.wav", tmp_path / "b.wav"]
    sf.write(paths[0], np.zeros((SR, 2), np.float32), SR)
    sf.write(paths[1], np.zeros((44_100, 2), np.float32), 44_100)
    with pytest.raises(ValueError):
        MultiSession.create(["a", "b"], paths, tmp_path / "x.voicefix", tmp_path / "cache")


def test_render_follows_the_candidate_lead(session, monkeypatch):
    """A harmony applied together with a lead change must use the new lead."""
    i = index_of(session, "C7b")
    for j in range(i):
        session.update(session.project.skip_step(j))
    apply(session, i, StepConfig(strength=100))
    seen = []
    original = session._reference

    def spy(index):
        key, loader = original(index)
        seen.append(key)
        return key, loader

    monkeypatch.setattr(session.chains[1], "reference", spy)
    c3 = index_of(session, "C3")
    candidate = session.project.apply_step(c3, StepConfig(strength=60))  # changes the lead's earlier step
    assert candidate.status(i) == StepStatus.STALE  # so C7 is recomputed and needs the lead
    session.render_step(candidate, i, lambda m, f: None, None)
    lead_candidate = session.multi.propagate(session.current_track, candidate).project(0)
    assert session.chains[0].output_key(lead_candidate, i - 1) in seen


def test_others_mix_excludes_the_given_track(session):
    mixed = session.others_mix(1, exclude=0)
    harmony = session.chains[1].render(session.multi.project(1), 0)
    assert np.allclose(mixed[: len(harmony)], np.asarray(harmony) * 10 ** (-3 / 20), atol=1e-6)


def test_long_pieces_are_split_at_quiet_points():
    from voicefix.core.regions import Region
    from voicefix.steps.timing import MAX_PIECE_SECONDS, split_long

    audio = (np.random.default_rng(0).standard_normal((40 * SR, 2)) * 0.1).astype(np.float32)
    audio[int(13.5 * SR) : int(13.6 * SR)] = 0.0  # a quiet moment just before the 15 s limit
    parts = split_long(audio, SR, Region(0, len(audio)))
    assert all(p.length <= MAX_PIECE_SECONDS * SR for p in parts)
    assert parts[0].end / SR == pytest.approx(13.55, abs=0.06)
    assert parts[0].start == 0 and parts[-1].end == len(audio)
    assert all(a.end == b.start for a, b in zip(parts, parts[1:]))


def test_offsets_on_empty_audio():
    curve = align.estimate_offsets(np.zeros((10, 2), np.float32), np.zeros((10, 2), np.float32), SR, 0.15)
    assert curve.at_samples(0, 5).tolist() == [0.0] * 5
