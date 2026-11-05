import numpy as np
import pyloudnorm
import pytest

from tests.conftest import SR, noise, speech_like, stereo
from voicefix.core import loudness
from voicefix.core.regions import Region
from voicefix.dsp import denoise, transients
from voicefix.dsp.dynamics import limit
from voicefix.dsp.filters import highpass_sos, zero_phase


def tone(freq: float, seconds: float, level: float = 0.5, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (level * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def rms_db(x: np.ndarray) -> float:
    return 10 * np.log10(np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-20)


@pytest.mark.parametrize("sr", [44_100, 48_000, 96_000])
def test_integrated_loudness_matches_pyloudnorm(sr):
    audio = stereo(speech_like(8.0, sr)) + noise(int(8.0 * sr), -50)
    expected = pyloudnorm.Meter(sr).integrated_loudness(audio.astype(np.float64))
    assert loudness.integrated_loudness(audio, sr) == pytest.approx(expected, abs=0.1)


def test_true_peak_catches_inter_sample_peaks():
    # A tone at fs/4 sampled at 45 degrees never hits its real peak on a sample.
    t = np.arange(SR)
    x = (np.sin(2 * np.pi * t / 4 + np.pi / 4) * 0.9).astype(np.float32)[:, None]
    assert np.abs(x).max() < 0.65
    assert loudness.true_peak(x) == pytest.approx(0.9, abs=0.02)


def test_zero_phase_highpass_removes_rumble_and_keeps_timing():
    audio = stereo(tone(25, 2.0) + tone(1000, 2.0, 0.2))
    out = zero_phase(highpass_sos(SR, 80, 2), audio)
    before = np.abs(np.fft.rfft(audio[SR // 2 : SR + SR // 2, 0]))
    after = np.abs(np.fft.rfft(out[SR // 2 : SR + SR // 2, 0]))
    assert after[25] < before[25] * 0.02  # at least -34 dB
    assert after[1000] == pytest.approx(before[1000], rel=0.01)
    impulse = np.zeros((SR, 1), dtype=np.float32)
    impulse[SR // 2] = 1.0
    assert int(np.argmax(np.abs(zero_phase(highpass_sos(SR, 80, 2), impulse)))) == SR // 2


def test_denoise_lowers_noise_in_pauses_and_keeps_voice():
    clean = stereo(speech_like(6.0))
    hiss = noise(len(clean), -45)
    noisy = clean + hiss
    profile = denoise.learn_profile(hiss[: SR * 2], SR)
    out = denoise.reduce_noise(noisy, SR, profile, denoise.DenoiseSettings(18.0, 1.5))
    pause = slice(0, int(0.35 * SR))
    assert rms_db(out[pause]) < rms_db(noisy[pause]) - 10
    loud = np.abs(clean[:, 0]) > 0.1
    assert abs(rms_db(out[loud]) - rms_db(noisy[loud])) < 1.5


def test_clicks_are_detected_and_softened():
    audio = stereo(speech_like(4.0)) + noise(4 * SR, -60)
    positions = [int(0.2 * SR), int(1.9 * SR), int(3.1 * SR)]
    for p in positions:
        audio[p : p + 24] += 0.4 * np.hanning(24)[:, None].astype(np.float32) * np.sign(np.arange(24) % 2 - 0.5)[:, None]
    settings = transients.ClickSettings(threshold_db=20.0)
    events = transients.detect_clicks(audio, SR, settings)
    for p in positions:
        assert any(e.start <= p + 12 < e.end for e in events), f"missed click at {p}"
    assert len(events) <= len(positions) + 2
    repaired = transients.repair_clicks(audio, SR, events, settings)
    for p in positions:
        assert np.abs(np.diff(repaired[p : p + 24, 0])).max() < np.abs(np.diff(audio[p : p + 24, 0])).max() * 0.5


def test_plosive_burst_is_detected():
    audio = stereo(speech_like(3.0, level=0.1)) + noise(3 * SR, -60)
    start = int(1.5 * SR)
    burst = tone(40, 0.05, 0.6) * np.hanning(int(0.05 * SR)).astype(np.float32)
    audio[start : start + len(burst)] += burst[:, None]
    events = transients.detect_plosives(audio, SR, transients.PlosiveSettings(threshold_db=12.0, cutoff_hz=150.0))
    assert any(e.start <= start + len(burst) // 2 < e.end for e in events)


def test_limiter_respects_true_peak_ceiling():
    audio = stereo(speech_like(3.0, level=0.95)) * 1.6
    out = limit(audio, SR, -1.0)
    assert loudness.linear_to_db(loudness.true_peak(out)) <= -1.0
    quiet = np.abs(audio[:, 0]) < 0.3
    assert np.allclose(out[quiet][:100], audio[quiet][:100], atol=0.2)


def test_limiter_is_identity_below_ceiling():
    audio = stereo(speech_like(1.0, level=0.2))
    assert np.array_equal(limit(audio, SR, -1.0), audio)


def test_limiter_chunk_equals_whole():
    audio = stereo(speech_like(3.0, level=0.95)) * 1.6
    whole = limit(audio, SR, -1.0)
    pad = int(0.2 * SR)
    core = Region(SR, 2 * SR)
    part = limit(audio[core.start - pad : core.end + pad], SR, -1.0)[pad:-pad]
    assert np.allclose(part, whole[core.start : core.end], atol=1e-5)


def test_segment_loudness_streams_long_segments():
    audio = stereo(speech_like(70.0)) + noise(70 * SR, -50)  # longer than one 30 s stream chunk
    from scipy import signal as sg

    weighted = sg.sosfilt(loudness.k_weighting_sos(SR), audio.astype(np.float64), axis=0)
    expected = loudness.power_to_lufs(float((weighted**2).mean(axis=0).sum()))
    assert loudness.segment_loudness(audio, SR) == pytest.approx(expected, abs=1e-6)
