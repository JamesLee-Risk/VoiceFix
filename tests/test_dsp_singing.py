import numpy as np
import pytest

from tests.conftest import SR, noise, stereo
from tests.singing import _note, sung_take
from voicefix.core import loudness
from voicefix.core.regions import Region
from voicefix.dsp.breaths import BreathSettings, detect_breaths
from voicefix.dsp.dynamics import compress, deess, sibilance_levels
from voicefix.dsp.key import estimate_key
from voicefix.dsp.tone import analyse_tone


def overlaps(region: Region, span: tuple[int, int]) -> bool:
    return region.start < span[1] and span[0] < region.end


def rms_db(x: np.ndarray) -> float:
    return 10 * np.log10(np.mean(np.asarray(x, np.float64) ** 2) + 1e-20)


def test_breaths_found_and_notes_left_alone():
    take = sung_take()
    events = detect_breaths(take.audio, SR, BreathSettings())
    found = sum(any(overlaps(e, b) for e in events) for b in take.breaths)
    assert found >= len(take.breaths) - 1
    for e in events:
        assert not any(overlaps(e, (lo + SR // 10, hi - SR // 10)) for lo, hi in take.notes), "breath inside a note"


def test_breath_detection_chunk_equals_whole():
    take = sung_take()
    whole = detect_breaths(take.audio, SR, BreathSettings())
    lo, hi = 3 * SR + 17, 9 * SR + 5
    pad = int(2.0 * SR)
    part = detect_breaths(take.audio[lo - pad : hi + pad], SR, BreathSettings(), offset=lo - pad)
    inside = lambda events: [e for e in events if lo + SR // 2 <= e.start < hi - SR // 2]  # noqa: E731
    assert inside(part) == inside(whole)


def test_compressor_narrows_level_differences():
    loud = _note(220, 2.0, 0.5)
    quiet = _note(220, 2.0, 0.08)
    audio = stereo(np.concatenate([loud, quiet]))
    out = compress(audio, SR, threshold_db=-30, ratio=4.0, makeup_db=0.0)
    before = rms_db(audio[: 2 * SR]) - rms_db(audio[2 * SR :])
    after = rms_db(out[: 2 * SR]) - rms_db(out[2 * SR :])
    assert after < before - 4


def test_compressor_chunk_equals_whole():
    audio = sung_take().audio
    whole = compress(audio, SR, -30, 3.0, 2.0)
    lo, hi, pad = 2 * SR + 123, 5 * SR, SR // 2
    part = compress(audio[lo - pad : hi + pad], SR, -30, 3.0, 2.0, offset=lo - pad)[pad:-pad]
    assert np.allclose(part, whole[lo:hi], atol=1e-6)


def test_deesser_cuts_sibilants_but_not_vowels():
    take = sung_take(ess_level=0.15)
    hf_db, _, _, _ = sibilance_levels(take.audio, SR)
    threshold = float(np.percentile(hf_db, 97)) - 6
    out = deess(take.audio, SR, threshold, 10.0)
    ess = slice(*take.sibilants[2])
    lo, hi = take.notes[2]
    vowel = slice(lo + SR // 4, hi - SR // 4)
    assert rms_db(out[ess]) < rms_db(take.audio[ess]) - 3
    assert abs(rms_db(out[vowel]) - rms_db(take.audio[vowel])) < 0.5


def test_tone_analysis_finds_planted_resonance():
    from voicefix.dsp.filters import peaking_sos, zero_phase

    audio = sung_take().audio
    resonant = zero_phase(peaking_sos(SR, 3150.0, 5.0, 6.0), audio)  # +10 dB after zero-phase
    profile = analyse_tone(resonant, SR)
    assert any(abs(np.log2(hz / 3150.0)) < 0.15 for hz, _ in profile.resonances)


def test_tone_analysis_quiet_on_a_plain_take():
    assert analyse_tone(sung_take().audio, SR).resonances == ()


def test_key_estimate_for_a_major_scale():
    scale = [261.63, 293.66, 329.63, 349.23, 392.0, 440.0, 493.88, 523.25]  # C major
    melody = np.concatenate([_note(f, 0.6, 0.3) for f in scale * 3] + [_note(261.63, 1.5, 0.3)])
    key, confidence = estimate_key(stereo(melody) + noise(len(melody), -70), SR)
    assert key in ("C 大調", "A 小調")
    assert confidence > 0.5


def test_true_peak_unchanged_by_quiet_compression():
    audio = stereo(_note(220, 1.0, 0.05))
    out = compress(audio, SR, threshold_db=-10, ratio=3.0, makeup_db=0.0)
    assert loudness.true_peak(out) == pytest.approx(loudness.true_peak(audio), rel=1e-3)
