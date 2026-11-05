"""Pitch tracking, note segmentation and shifting (B6 building blocks)."""

import numpy as np
import pytest

from tests.conftest import SR, noise
from voicefix.core.regions import Region
from voicefix.dsp import pitch
from voicefix.dsp.key import CHROMATIC, note_name, scale_pitch_classes


def sung(notes: list[tuple[float, float]], vibrato_cents: float = 0.0, gap: float = 0.25) -> np.ndarray:
    """(midi_cents, seconds) notes with harmonics, separated by short silences; stereo."""
    pieces = [np.zeros(int(0.3 * SR), np.float32)]
    for cents, seconds in notes:
        t = np.arange(int(seconds * SR)) / SR
        hz = 440.0 * 2 ** ((cents - 6900 + vibrato_cents * np.sin(2 * np.pi * 5.5 * t)) / 1200.0)
        phase = 2 * np.pi * np.cumsum(hz) / SR
        tone = sum(np.sin(k * phase) / k for k in range(1, 12))
        env = np.minimum(1, np.minimum(t / 0.03, (seconds - t) / 0.05))
        pieces += [(0.25 * tone / np.abs(tone).max() * env).astype(np.float32), np.zeros(int(gap * SR), np.float32)]
    mono = np.concatenate(pieces)
    return np.stack([mono, mono], axis=1) + noise(len(mono), -70)


def test_scale_and_note_helpers():
    assert scale_pitch_classes("C 大調") == (0, 2, 4, 5, 7, 9, 11)
    assert 11 in scale_pitch_classes("A 小調")  # raised 7th (G#)
    assert scale_pitch_classes(CHROMATIC) == tuple(range(12))
    assert note_name(6900) == "A4"
    assert pitch.nearest_in_scale(6935.0, scale_pitch_classes("C 大調")) == pytest.approx(6900.0)
    assert pitch.nearest_in_scale(6155.0, scale_pitch_classes("C 大調")) == pytest.approx(6200.0)  # C#4+55 -> D4


def test_notes_found_with_their_detuning():
    audio = sung([(6900 + 40, 0.8), (7100, 0.8), (7200 - 35, 0.8)])
    notes = pitch.segment_notes(pitch.track_pitch(audio, SR))
    assert len(notes) == 3
    assert [round(n.cents) for n in notes] == pytest.approx([6940, 7100, 7165], abs=5)


def test_vibrato_stays_one_note():
    audio = sung([(6900, 1.5)], vibrato_cents=60)
    notes = pitch.segment_notes(pitch.track_pitch(audio, SR))
    assert len(notes) == 1 and notes[0].cents == pytest.approx(6900, abs=10)


def test_shift_segment_hits_target_and_keeps_timing():
    audio = sung([(6900 + 40, 1.0)])
    out = pitch.shift_segment(audio, SR, -0.4)
    assert out.shape == audio.shape
    notes = pitch.segment_notes(pitch.track_pitch(out, SR))
    assert notes[0].cents == pytest.approx(6900, abs=4)
    before = pitch.segment_notes(pitch.track_pitch(audio, SR))[0].region
    assert abs(notes[0].region.start - before.start) <= int(0.02 * SR)


def test_shift_is_deterministic():
    audio = sung([(6900 + 40, 0.6)])
    assert np.array_equal(pitch.shift_segment(audio, SR, -0.4), pitch.shift_segment(audio, SR, -0.4))


def test_psola_identity_is_exact():
    from voicefix.dsp.psola import psola

    audio = sung([(6930, 0.8)], vibrato_cents=30)
    track = pitch.track_pitch(audio, SR)
    hz = np.where(np.isnan(track.cents), 0.0, 440 * 2 ** ((track.cents - 6900) / 1200))
    f0 = hz[np.minimum(np.arange(len(audio)) // track.frame, len(hz) - 1)]
    assert np.array_equal(psola(audio, SR, f0, np.ones(len(audio))), audio.astype(np.float32))
