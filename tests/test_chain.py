"""Chain-level tests mirroring the PRD acceptance criteria."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tests.conftest import SR, index_of, noise, speech_like, stereo
from voicefix.core import loudness
from voicefix.core.cache import AudioCache, load_source
from voicefix.core.chain import StepChain
from voicefix.core.exporter import ExportRequest, export
from voicefix.core.project import Flow, Project, StepConfig, StepStatus, load_project, save_project
from voicefix.core.regions import Region, ScopeMode
from voicefix.steps.registry import build_steps, first_affected


def noisy_take(seconds: float = 8.0) -> np.ndarray:
    audio = stereo(speech_like(seconds)) + noise(int(seconds * SR), -48)
    t = np.arange(len(audio)) / SR
    audio += (0.01 * np.sin(2 * np.pi * 60 * t)).astype(np.float32)[:, None]  # hum
    return audio


def apply_all(project: Project, chain: StepChain, overrides: dict[str, StepConfig] | None = None) -> Project:
    overrides = overrides or {}
    for i, step in enumerate(chain.steps):
        cfg = overrides.get(step.code, StepConfig(strength=step.default_strength, option=step.default_option))
        project = project.apply_step(i, cfg)
    return project


def test_skipped_step_is_sample_identical(make_session):
    project, chain = make_session(noisy_take())
    i = index_of(chain, "A4")
    project = project.skip_step(i)
    assert np.array_equal(chain.render(project, i), chain.render(project, i - 1))


@pytest.mark.parametrize("code", ["A4", "A5", "A6", "A8"])
def test_protected_regions_untouched_in_a4_to_a8(make_session, code):
    project, chain = make_session(noisy_take())
    protect = Region(2 * SR, 4 * SR)
    project = project.with_shared(0, protected=(protect,))
    i = index_of(chain, code)
    step = chain.steps[i]
    project = project.apply_step(i, StepConfig(strength=90, option=step.default_option))
    before, after = chain.render(project, i - 1), chain.render(project, i)
    assert np.array_equal(after[protect.start : protect.end], before[protect.start : protect.end])


def test_binaural_channels_keep_timing(make_session):
    mono = speech_like(8.0)
    delay = 24  # 0.5 ms inter-aural delay
    audio = np.stack([mono, np.roll(mono, delay)], axis=1) + noise(len(mono), -50)
    project, chain = make_session(audio)
    project = apply_all(replace(project, binaural=True), chain)
    out = chain.render(project, len(chain.steps) - 1)

    def lag(x: np.ndarray) -> int:
        window = slice(SR, 7 * SR)
        corr = np.correlate(x[window, 1], x[window, 0][100:-100], mode="valid")
        return int(np.argmax(corr)) - 100

    assert lag(out) == lag(audio) == delay


def test_changing_a_step_marks_later_steps_stale(make_session):
    project, chain = make_session(noisy_take())
    project = apply_all(project, chain)
    i = index_of(chain, "A3")
    project = project.apply_step(i, StepConfig(strength=80, option="auto"))
    assert all(project.status(j) == StepStatus.STALE for j in range(i + 1, len(chain.steps)))
    assert project.status(i - 1) == StepStatus.APPLIED


def test_export_matches_fresh_step_by_step_render(make_session, tmp_path: Path):
    project, chain = make_session(noisy_take())
    project = apply_all(project, chain)
    project = project.apply_step(index_of(chain, "A2"), StepConfig(strength=70))  # make later steps stale
    written = export(chain, project, ExportRequest(tmp_path / "out", "wav24"))
    exported, sr = sf.read(written[0], dtype="float32", always_2d=True)

    fresh_cache = AudioCache(tmp_path / "fresh")
    source, rate, key = load_source(project.source_path, fresh_cache)
    fresh = StepChain(build_steps(project.flow), fresh_cache, source, rate, key)
    reference = np.asarray(fresh.render(project, len(fresh.steps) - 1))
    assert sr == SR and len(exported) == len(reference)
    assert np.max(np.abs(exported - np.clip(reference, -1, 1))) < 2.0 ** -22  # 24-bit quantisation only


def test_rendering_is_deterministic(make_session, tmp_path: Path):
    project, chain = make_session(noisy_take())
    project = apply_all(project, chain)
    first = np.array(chain.render(project, len(chain.steps) - 1))
    other = StepChain(build_steps(project.flow), AudioCache(tmp_path / "again"), chain.source, chain.sr, chain.source_key)
    assert np.array_equal(first, other.render(project, len(chain.steps) - 1))


def test_deletions_shorten_and_regions_follow(make_session):
    project, chain = make_session(noisy_take())
    a1 = index_of(chain, "A1")
    project = project.apply_step(a1, StepConfig(settings={"deletions": [[SR, 2 * SR]]}))
    assert len(chain.render(project, a1)) == len(chain.source) - SR
    project = project.with_shared(0, protected=(Region(3 * SR, 4 * SR),))
    ctx = chain.context(project, a1 + 1)
    assert ctx.protected == (Region(2 * SR, 3 * SR),)


def test_preview_matches_full_render_inside_span(make_session):
    project, chain = make_session(noisy_take())
    i = index_of(chain, "A4")
    cfg = StepConfig(strength=40, option="2")
    span = Region(3 * SR, 5 * SR)
    preview = chain.preview(project, i, cfg, span)
    full = chain.render(project.apply_step(i, cfg), i)
    assert np.allclose(preview.after, full[span.start : span.end], atol=1e-4)
    assert np.array_equal(preview.before, chain.render(project, i - 1)[span.start : span.end])


def test_a10_reaches_target_loudness_and_ceiling(make_session):
    project, chain = make_session(noisy_take() * 0.3)
    i = index_of(chain, "A10")
    project = project.apply_step(i, StepConfig(option="standard"))
    out = chain.render(project, i)
    assert loudness.integrated_loudness(out, SR) == pytest.approx(-18.0, abs=0.5)
    assert loudness.linear_to_db(loudness.true_peak(out)) <= -1.0


def test_leveler_reduces_loudness_spread(make_session):
    mono = speech_like(10.0)
    mono[: 5 * SR] *= 0.4  # first half much quieter
    project, chain = make_session(stereo(mono) + noise(len(mono), -70))
    i = index_of(chain, "A8")
    step = chain.steps[i]
    before = chain.render(project, i - 1)
    after = chain.render(project.apply_step(i, StepConfig(strength=100, option="together")), i)
    levels = step.measure(before, SR, chain.context(project, i))

    def spread(audio):
        values = [loudness.segment_loudness(np.asarray(audio[s.start : s.end]), SR) for s in levels.segments]
        return max(values) - min(values)

    assert spread(after) < spread(before) * 0.5


def test_dehum_analysis_finds_60hz(make_session):
    project, chain = make_session(noisy_take())
    analysis = chain.analyze(project, index_of(chain, "A3"))
    assert analysis.recommend and analysis.data["mains_hz"] == 60.0


def test_selection_scope_leaves_rest_untouched(make_session):
    project, chain = make_session(noisy_take())
    i = index_of(chain, "A2")
    sel = Region(2 * SR, 3 * SR)
    project = project.apply_step(i, StepConfig(strength=100, scope=ScopeMode.SELECTION, selections=(sel,)))
    before, after = chain.render(project, i - 1), chain.render(project, i)
    assert np.array_equal(after[: sel.start], before[: sel.start])
    assert np.array_equal(after[sel.end :], before[sel.end :])
    assert not np.array_equal(after[sel.start : sel.end], before[sel.start : sel.end])


def test_disabled_events_are_not_repaired(make_session):
    audio = stereo(speech_like(4.0)) + noise(4 * SR, -60)
    p = int(1.9 * SR)
    audio[p : p + 24] += 0.4 * np.hanning(24)[:, None].astype(np.float32)
    project, chain = make_session(audio)
    i = index_of(chain, "A5")
    cfg = StepConfig(strength=60)
    events = chain.events(project, i, cfg)
    target = next(e for e in events if e.region.start <= p + 12 < e.region.end)
    project = project.apply_step(i, replace(cfg, disabled_events=(target.region.start,)))
    before, after = chain.render(project, i - 1), chain.render(project, i)
    r = target.region
    assert np.array_equal(after[r.start : r.end], before[r.start : r.end])


def test_project_round_trip(tmp_path: Path):
    steps = build_steps(Flow.VOICE_WORK)
    project = Project.new(Flow.VOICE_WORK, "x.wav", [s.id for s in steps])
    project = project.apply_step(2, StepConfig(strength=42, selections=(Region(1, 9),), advanced={"cutoff_hz": 70.0}))
    project = project.with_shared(first_affected(steps, "protected"), protected=(Region(5, 50),), binaural=True)
    path = tmp_path / "p.voicefix"
    save_project(project, path)
    assert load_project(path) == project


def test_concurrent_renders_of_same_output_are_safe(make_session):
    import threading as th

    project, chain = make_session(noisy_take())
    i = index_of(chain, "A4")
    project = project.apply_step(i, StepConfig(strength=40))
    results, errors = [], []

    def worker():
        try:
            results.append(np.array(chain.render(project, i)))
        except Exception as exc:  # collected and asserted below
            errors.append(exc)

    threads = [th.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert all(np.array_equal(r, results[0]) for r in results)
    assert not list(chain.cache.directory.glob("*.partial"))


def test_leftover_partial_files_are_cleaned_on_open(tmp_path: Path):
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "abc.123.partial").write_bytes(b"junk")
    AudioCache(tmp_path / "c")
    assert not list((tmp_path / "c").glob("*.partial"))
