"""Flow B (singing) steps on synthetic sung phrases."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tests.conftest import SR, index_of
from tests.singing import sung_take
from voicefix.core import loudness
from voicefix.core.exporter import ExportRequest, export, mixed_blocks
from voicefix.core.project import Flow, StepConfig
from voicefix.core.regions import Region


def rms_db(x: np.ndarray) -> float:
    return 10 * np.log10(np.mean(np.asarray(x, np.float64) ** 2) + 1e-20)


@pytest.fixture
def take():
    return sung_take()


@pytest.fixture
def session(make_session, take):
    return make_session(take.audio, flow=Flow.SINGING)


def apply_defaults(project, chain):
    for i, step in enumerate(chain.steps):
        project = project.apply_step(i, StepConfig(strength=step.default_strength, option=step.default_option))
    return project


def test_gap_cleanup_lowers_silence_but_keeps_notes_and_breaths(session, take):
    project, chain = session
    i = index_of(chain, "B2")
    project = project.apply_step(i, StepConfig(strength=50, option="reduce"))
    before, after = chain.render(project, i - 1), chain.render(project, i)
    silent = slice(int(0.05 * SR), int(0.25 * SR))  # leading silence
    assert rms_db(after[silent]) < rms_db(before[silent]) - 15
    lo, hi = take.notes[3]
    assert np.allclose(after[lo:hi], before[lo:hi])
    b_lo, b_hi = take.breaths[3]
    core = slice(b_lo + (b_hi - b_lo) // 4, b_hi - (b_hi - b_lo) // 4)
    assert abs(rms_db(after[core]) - rms_db(before[core])) < 1.0


def test_breaths_reduced_by_default_six_db(session, take):
    project, chain = session
    i = index_of(chain, "B5")
    events = chain.events(project, i, StepConfig())
    assert len(events) >= len(take.breaths) - 1
    project = project.apply_step(i, StepConfig(strength=30))
    before, after = chain.render(project, i - 1), chain.render(project, i)
    lo, hi = take.breaths[2]
    mid = slice(lo + (hi - lo) // 3, hi - (hi - lo) // 3)
    assert rms_db(after[mid]) - rms_db(before[mid]) == pytest.approx(-6.0, abs=0.5)
    n_lo, n_hi = take.notes[2]
    assert np.allclose(after[n_lo + SR // 10 : n_hi], before[n_lo + SR // 10 : n_hi])


def test_combined_clicks_and_pops_labels(session):
    project, chain = session
    audio = np.array(chain.source)
    step = chain.steps[index_of(chain, "B4")]
    assert step.kind_label(Region(0, 300), SR) == "口水音"
    assert step.kind_label(Region(0, 3000), SR) == "噴麥"
    assert all(e.label.split()[0] in ("口水音", "噴麥") for e in step.detect_events(audio, SR, step.resolve_params(60, None, {})))


@pytest.mark.parametrize("code", ["B8", "B9", "B10"])
def test_tone_steps_preview_matches_render(session, code):
    project, chain = session
    i = index_of(chain, code)
    cfg = StepConfig(strength=70, option=chain.steps[i].default_option)
    span = Region(3 * SR, 6 * SR)
    preview = chain.preview(project, i, cfg, span)
    full = chain.render(project.apply_step(i, cfg), i)
    assert np.allclose(preview.after, full[span.start : span.end], atol=1e-4)
    assert not np.allclose(preview.after, preview.before, atol=1e-4)  # it actually did something


def test_b12_hits_vocal_target_and_ceiling(session):
    project, chain = session
    i = index_of(chain, "B12")
    project = project.apply_step(i, StepConfig(option="stem"))
    out = chain.render(project, i)
    assert loudness.integrated_loudness(out, SR) == pytest.approx(-20.0, abs=0.5)
    assert loudness.linear_to_db(loudness.true_peak(out)) <= -1.0


def test_export_with_mix_and_untouched_accompaniment(session, tmp_path: Path):
    project, chain = session
    project = apply_defaults(project, chain).skip_step(index_of(chain, "B11"))
    t = np.arange(len(chain.source)) / SR
    backing = np.stack([0.2 * np.sin(2 * np.pi * 110 * t)] * 2, axis=1).astype(np.float32)
    written = export(chain, project, ExportRequest(tmp_path / "song", "wav24", dry=True, mix=True), backing=backing)
    names = sorted(p.name for p in written)
    assert names == ["song.wav", "song_mix.wav"]  # no spatial effect applied, so no separate dry file
    vocal, _ = sf.read(tmp_path / "song.wav", dtype="float32", always_2d=True)
    mix, _ = sf.read(tmp_path / "song_mix.wav", dtype="float32", always_2d=True)
    assert len(mix) == len(vocal)
    assert loudness.linear_to_db(loudness.true_peak(mix)) <= -0.9
    quiet = slice(int(0.05 * SR), int(0.25 * SR))  # vocal is silent here: the mix is the backing itself
    assert np.allclose(mix[quiet], backing[quiet], atol=2e-3)


def test_mix_blocks_cover_whole_length():
    vocal = np.zeros((int(70.5 * SR), 2), np.float32)
    backing = np.full_like(vocal, 0.1)
    total = sum(len(b) for b in mixed_blocks(vocal, backing, SR))
    assert total == len(vocal)


def test_mix_export_without_backing_is_rejected(session, tmp_path: Path):
    project, chain = session
    with pytest.raises(ValueError):
        export(chain, project, ExportRequest(tmp_path / "x", "wav16", mix=True))


def test_dry_export_leaves_out_wet_effects(session, tmp_path: Path):
    project, chain = session
    project = apply_defaults(project, chain)  # includes B11 (space)
    written = export(chain, project, ExportRequest(tmp_path / "song", "wav24", dry=True))
    assert sorted(p.name for p in written) == ["song.wav", "song_dry.wav"]
    dry, _ = sf.read(tmp_path / "song_dry.wav", dtype="float32", always_2d=True)
    without_space = chain.render(project.skip_step(index_of(chain, "B11")), len(chain.steps) - 1)
    assert np.max(np.abs(dry - np.clip(without_space, -1, 1))) < 2.0**-22


def test_streamed_resample_matches_whole_file(tmp_path: Path):
    import soxr

    from voicefix.core.audio_io import iter_resampled

    rng = np.random.default_rng(3)
    src = (rng.standard_normal(int(47.3 * 44_100)) * 0.1).astype(np.float32)
    sf.write(tmp_path / "acc.wav", src, 44_100, subtype="FLOAT")
    frames = int(47.3 * SR)
    streamed = np.concatenate(list(iter_resampled(tmp_path / "acc.wav", SR, frames, 2)))
    whole = soxr.resample(src, 44_100, SR, quality="VHQ")[:frames]
    assert streamed.shape == (frames, 2)
    core = slice(SR, len(whole) - SR)  # the very ends see different edge padding
    assert np.allclose(streamed[core, 0], whole[core], atol=1e-5)
    assert np.array_equal(streamed[:, 0], streamed[:, 1])


def test_export_never_overwrites_the_accompaniment(session, tmp_path: Path):
    from dataclasses import replace as dc_replace

    project, chain = session
    accompaniment = tmp_path / "song_mix.wav"
    accompaniment.write_bytes(b"precious")
    project = dc_replace(project, accompaniment_path=str(accompaniment))
    backing = np.zeros((len(chain.source), 2), np.float32)
    with pytest.raises(ValueError):
        export(chain, project, ExportRequest(tmp_path / "song", "wav16", mix=True), backing=backing)
    assert accompaniment.read_bytes() == b"precious"


def test_breath_preview_includes_events_starting_before_span(session, take):
    project, chain = session
    i = index_of(chain, "B5")
    cfg = StepConfig(strength=60)
    lo, hi = take.breaths[4]
    span = Region((lo + hi) // 2, (lo + hi) // 2 + 2 * SR)  # starts in the middle of a breath
    preview = chain.preview(project, i, cfg, span)
    full = chain.render(project.apply_step(i, cfg), i)
    assert np.allclose(preview.after, full[span.start : span.end], atol=1e-5)
