"""End-to-end: drive the main window through flow A without a display."""

import os
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QThreadPool  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from tests.conftest import SR, noise, speech_like, stereo  # noqa: E402
from voicefix.core.project import Flow, StepStatus  # noqa: E402
from voicefix.core.regions import Region  # noqa: E402
from voicefix.ui.main_window import MainWindow  # noqa: E402
from voicefix.ui.session import Session  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def settle(app: QApplication, timeout: float = 30.0) -> None:
    """Pump events until background work and debounced timers are done."""
    deadline = time.time() + timeout
    idle_rounds = 0
    while time.time() < deadline and idle_rounds < 5:
        app.processEvents()
        busy = QThreadPool.globalInstance().activeThreadCount() > 0
        idle_rounds = 0 if busy else idle_rounds + 1
        time.sleep(0.06)  # longer than the preview debounce granularity
    app.processEvents()


@pytest.fixture
def window(app, tmp_path: Path, monkeypatch):
    errors: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: errors.append(a[2]))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    audio = stereo(speech_like(12.0)) + noise(12 * SR, -50)
    for pos in (int(3.2 * SR), int(8.7 * SR)):  # two mouth clicks for the event steps
        audio[pos : pos + 24] += 0.4 * np.hanning(24)[:, None].astype(np.float32)
    src = tmp_path / "voice.wav"
    sf.write(src, audio, SR)
    session = Session.create(Flow.VOICE_WORK, src, tmp_path / "voice.voicefix", tmp_path / "cache")
    win = MainWindow(session)
    win.errors = errors
    settle(app)
    yield win
    win.close()
    settle(app)


def test_walk_flow_a_with_suggestions_and_export(app, window, tmp_path: Path, monkeypatch):
    steps = window.session.steps
    assert window.index == 0
    assert window.project.noise_sample is not None  # A0 proposed the quietest 3 s
    for i in range(len(steps) - 1):
        assert window.index == i, f"expected step {steps[i].code}, at {steps[window.index].code}"
        assert window.audition.mixer.length > 0, f"{steps[i].code}: no preview"
        window._apply()
        settle(app)
    assert all(window.project.status(i) == StepStatus.APPLIED for i in range(len(steps) - 1))
    # A0 shows the noise sample as a selection; it must not leak into later steps
    assert all(window.project.config(i).selections == () for i in range(1, len(steps) - 1))

    out = tmp_path / "exported"
    monkeypatch.setattr("voicefix.ui.main_window.QFileDialog.getSaveFileName", lambda *a, **k: (str(out), ""))
    window._export("wav16", False)
    settle(app)
    data, sr = sf.read(out.with_suffix(".wav"))
    assert sr == SR and len(data) > 0
    assert window.errors == []


def test_changing_an_earlier_step_marks_later_ones_stale(app, window):
    for _ in range(4):
        window._apply()
        settle(app)
    window.enter_step(2)
    settle(app)
    window.controls.slider.setValue(80)
    window._apply()
    settle(app)
    assert window.project.status(3) == StepStatus.STALE
    window.enter_step(3)  # entering recomputes the stale input
    settle(app)
    assert window.project.status(2) == StepStatus.APPLIED
    assert window.errors == []


def test_protecting_a_selection_shows_and_persists(app, window):
    window.waveform.set_selections([Region(SR, 2 * SR)])
    window._protect_selection()
    settle(app)
    assert window.project.protected == (Region(SR, 2 * SR),)
    window.session.save()
    assert Session.open(window.session.project_path, window.session.cache.directory.parent).project.protected


def test_skip_moves_on_and_preview_switches_modes(app, window):
    window._skip()
    settle(app)
    assert window.index == 1 and window.project.status(0) == StepStatus.SKIPPED
    window.audition.toggle_removed()
    block, _ = window.audition.mixer.render(1024)
    assert np.all(np.isfinite(block))


def test_events_in_protected_region_come_back_after_unprotect(app, window):
    for _ in range(5):  # A0..A4 -> now on A5
        window._apply()
        settle(app)
    assert window.step.code == "A5"
    assert window.events, "the planted clicks should be detected"
    first = window.events[0].region
    window.waveform.set_selections([Region(max(0, first.start - SR // 10), first.end + SR // 10)])
    window._protect_selection()
    settle(app)
    assert window.event_list.enabled()[0] is False
    window._unprotect()
    settle(app)
    assert window.event_list.enabled()[0] is True
    assert window._draft().disabled_events == ()


def test_editing_disabled_until_input_ready(app, window):
    window.enter_step(0)
    window._set_editing(False)
    assert not window.protect_button.isEnabled()
    window._on_input_ready()
    assert window.protect_button.isEnabled()


@pytest.fixture
def singing_window(app, tmp_path: Path, monkeypatch):
    from tests.singing import sung_take

    errors: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: errors.append(a[2]))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    take = sung_take()
    src, backing = tmp_path / "vocal.wav", tmp_path / "backing.wav"
    sf.write(src, take.audio, SR)
    t = np.arange(len(take.audio)) / 44_100
    sf.write(backing, (0.1 * np.sin(2 * np.pi * 110 * t)).astype(np.float32), 44_100)  # other rate, mono
    session = Session.create(Flow.SINGING, src, tmp_path / "song.voicefix", tmp_path / "cache")
    win = MainWindow(session)
    win.errors = errors
    win.backing_path = backing
    settle(app)
    yield win
    win.close()
    settle(app)


def test_walk_flow_b_with_accompaniment_and_mix_export(app, singing_window, tmp_path: Path, monkeypatch):
    win = singing_window
    assert win.step.code == "B0" and not win.import_panel.song_box.isHidden()
    monkeypatch.setattr(
        "voicefix.ui.main_window.QFileDialog.getOpenFileName", lambda *a, **k: (str(win.backing_path), "")
    )
    win._choose_accompaniment()
    settle(app)  # the accompaniment is decoded on a worker thread
    assert win.session.backing is not None and win.session.backing.shape == (len(win.chain.source), 2)
    win._set_key("A 小調")
    while win.step.code != "B12":
        win._apply()
        settle(app)
        expected = win.step.audition_with_backing
        assert win.transport.backing.isChecked() == expected, win.step.code
        assert win.audition.mixer.backing_enabled == expected
    assert win.project.config(0).settings["key"] == "A 小調"

    out = tmp_path / "final"
    monkeypatch.setattr("voicefix.ui.main_window.QFileDialog.getSaveFileName", lambda *a, **k: (str(out), ""))
    win.export_panel.mix.setChecked(True)
    win.export_panel.export.click()
    settle(app)
    assert out.with_suffix(".wav").exists() and (tmp_path / "final_mix.wav").exists()
    assert all(win.project.status(i) == StepStatus.APPLIED for i in range(len(win.session.steps)))
    assert win.errors == []

    reopened = Session.open(win.session.project_path, win.session.cache.directory.parent)
    assert reopened.backing is not None


def test_key_saved_immediately_and_export_options_not_in_settings(app, singing_window, tmp_path: Path, monkeypatch):
    win = singing_window
    win._set_key("D 大調")
    assert win.project.config(0).settings["key"] == "D 大調"
    assert win.project.status(0) == StepStatus.NOT_STARTED
    win.enter_step(0)
    settle(app)
    assert win.import_panel.key.currentData() == "D 大調"
    while win.step.code != "B12":
        win._apply()
        settle(app)
    monkeypatch.setattr("voicefix.ui.main_window.QFileDialog.getSaveFileName", lambda *a, **k: (str(tmp_path / "o"), ""))
    win._export("wav16", False, False, False)
    settle(app)
    key_before = win.chain.output_key(win.project, win.index)
    win._export("flac", False, True, False)
    settle(app)
    assert win.chain.output_key(win.project, win.index) == key_before
    assert "format" not in win.project.config(win.index).settings


def test_broken_accompaniment_does_not_block_opening(app, tmp_path: Path):
    from tests.singing import sung_take

    src = tmp_path / "v.wav"
    sf.write(src, sung_take(phrases=2).audio, SR)
    session = Session.create(Flow.SINGING, src, tmp_path / "v.voicefix", tmp_path / "cache")
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"RIFF\x00\x00\x00\x00WAVEjunk")
    from dataclasses import replace as dc_replace

    session.update(dc_replace(session.project, accompaniment_path=str(broken)))
    reopened = Session.open(tmp_path / "v.voicefix", tmp_path / "cache")
    assert reopened.backing is None and reopened.backing_error


@pytest.fixture
def multitrack_window(app, tmp_path: Path, monkeypatch):
    from tests.test_align import ONSETS, SHIFTS, phrase_track
    from voicefix.ui.multisession import MultiSession

    errors: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: errors.append(a[2]))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    paths = [tmp_path / "lead.wav", tmp_path / "harmony.wav"]
    sf.write(paths[0], phrase_track(ONSETS, 6900, [0.0] * 12), SR)
    sf.write(paths[1], phrase_track(ONSETS, 7300, SHIFTS) * 0.5, SR)
    session = MultiSession.create(["主唱", "和聲"], paths, tmp_path / "song.voicefix", tmp_path / "cache")
    win = MainWindow(session)
    win.errors = errors
    settle(app)
    yield win
    win.close()
    settle(app)


def test_walk_flow_c_switch_tracks_and_export(app, multitrack_window, tmp_path: Path, monkeypatch):
    win = multitrack_window
    assert not win.track_bar.isHidden() and win.track_bar.tracks.count() == 2
    while win.step.code != "C3":
        win._apply()
        settle(app)
    win.track_bar.tracks.setCurrentIndex(1)  # look at the harmony
    settle(app)
    assert win.session.current_track == 1 and win.step.code == "C3"
    win.track_bar.own.setChecked(True)  # this harmony gets its own noise-reduction strength
    settle(app)
    win.controls.slider.setValue(10)
    win._apply()
    settle(app)
    assert win.session.multi.project(1).config(2).strength == 10
    assert win.session.multi.project(0).config(2).strength != 10
    while win.step.code != "C12":
        assert win.transport.others.isChecked() == win.step.audition_with_backing
        win._apply()
        settle(app)
    assert win.transport.others.isChecked() and win._others is not None  # other tracks play along
    out = tmp_path / "final"
    monkeypatch.setattr("voicefix.ui.main_window.QFileDialog.getSaveFileName", lambda *a, **k: (str(out), ""))
    win.export_panel.mix.setChecked(True)
    win.export_panel.export.click()
    settle(app)
    assert (tmp_path / "final_1_主唱.wav").exists() and (tmp_path / "final_2_和聲.wav").exists()
    assert (tmp_path / "final_mix.wav").exists()
    assert all(win.session.multi.project(t).status(i) == StepStatus.APPLIED for t in range(2) for i in range(12))
    assert win.errors == []


def test_pick_target_for_far_off_note_on_pitch_plot(app, tmp_path: Path, monkeypatch):
    from tests.test_pitch import sung
    from voicefix.dsp import pitch

    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: pytest.fail(a[2]))
    src = tmp_path / "v.wav"
    sf.write(src, sung([(6400 + 80, 0.9), (7200, 0.9)]), SR)  # E4 sung 80 cents sharp
    session = Session.create(Flow.SINGING, src, tmp_path / "v.voicefix", tmp_path / "cache")
    win = MainWindow(session)
    settle(app)
    win._set_key("C 大調")
    while win.step.code != "B6":
        win._skip()
        settle(app)
    assert win.events == []  # looks like an in-tune F4
    first = pitch.segment_notes(pitch.track_pitch(np.array(win.chain.render(win.project, win.index - 1)), SR))[0]
    win.waveform.pitch_clicked.emit(first.region.start + SR // 10)  # click the note on the pitch plot
    assert "F4" in win.pitch_panel.note_label.text() and win.pitch_panel.down.isEnabled()
    win.pitch_panel.down.click()  # it should be E4
    settle(app)
    assert "E4" in win.pitch_panel.note_label.text() and "你指定的" in win.pitch_panel.note_label.text()
    assert len(win.events) == 1 and "手動" in win.events[0].label
    win.controls.slider.setValue(100)
    win._apply()
    settle(app)
    b6 = win.index - 1
    assert win.project.config(b6).note_targets
    fixed = pitch.segment_notes(pitch.track_pitch(np.array(win.chain.render(win.project, b6)), SR))
    assert fixed[0].cents == pytest.approx(6400, abs=8)
    win.close()
    settle(app)


def test_quick_presses_add_up_and_return_to_auto(app, tmp_path: Path, monkeypatch):
    from tests.test_pitch import sung
    from voicefix.dsp import pitch

    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: pytest.fail(a[2]))
    src = tmp_path / "v.wav"
    sf.write(src, sung([(6400 + 80, 0.9), (7200, 0.9)]), SR)
    win = MainWindow(Session.create(Flow.SINGING, src, tmp_path / "v.voicefix", tmp_path / "cache"))
    settle(app)
    win._set_key("C 大調")
    while win.step.code != "B6":
        win._skip()
        settle(app)
    first = pitch.segment_notes(pitch.track_pitch(np.array(win.chain.render(win.project, win.index - 1)), SR))[0]
    win.waveform.pitch_clicked.emit(first.region.start + SR // 10)
    win.pitch_panel.down.click()
    win.pitch_panel.down.click()  # before the re-analysis comes back
    assert list(win.note_targets.targets()) == [(win.note_targets.targets()[0][0], 6300)]  # F4 -> E4 -> Eb4
    win.pitch_panel.up.click()
    win.pitch_panel.up.click()  # back to F4, which is the automatic target
    settle(app)
    assert win.note_targets.targets() == ()
    win.close()
    settle(app)
