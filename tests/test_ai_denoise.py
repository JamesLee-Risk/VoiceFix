"""AI noise reduction (DeepFilterNet). Skipped when the optional packages are absent."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tests.conftest import SR, index_of, noise, speech_like, stereo
from voicefix.core.project import StepConfig
from voicefix.core.regions import Region
from voicefix.dsp import ai_denoise, denoise

needs_ai = pytest.mark.skipif(not ai_denoise.is_available(), reason="AI extras not installed")


TTS_SCRIPT = r"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(48000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$s.SetOutputToWaveFile($env:VOICEFIX_TTS_OUT, $fmt)
$s.Speak("Welcome back. Tonight we will relax together. Take a slow breath in, and let it go. The rain is falling softly outside the window. You are safe, warm, and calm.")
$s.Dispose()
"""


@pytest.fixture(scope="module")
def real_speech(tmp_path_factory) -> np.ndarray:
    """Real speech from the Windows speech synthesiser, generated at test time
    (synthesised voices are not committed to the repository)."""
    if sys.platform != "win32":
        pytest.skip("needs the Windows speech synthesiser")
    out = tmp_path_factory.mktemp("tts") / "speech.wav"
    env = {**os.environ, "VOICEFIX_TTS_OUT": str(out)}
    result = subprocess.run(["powershell", "-NoProfile", "-Command", TTS_SCRIPT], env=env, capture_output=True, timeout=120)
    if result.returncode != 0 or not out.exists():
        pytest.skip("speech synthesiser unavailable")
    speech, sr = sf.read(out, dtype="float32")
    assert sr == SR
    return speech


def rms_db(x: np.ndarray) -> float:
    return 10 * np.log10(np.mean(np.asarray(x, np.float64) ** 2) + 1e-20)


def test_reference_mask_is_identity_for_a_clean_reference():
    audio = stereo(speech_like(3.0)) + noise(3 * SR, -50)
    out = denoise.apply_reference_mask(audio, SR, audio.mean(axis=1), 20.0)
    assert np.allclose(out, audio, atol=1e-4)


def test_reference_mask_respects_reduction_limit():
    audio = noise(2 * SR, -30)
    out = denoise.apply_reference_mask(audio, SR, np.zeros(len(audio), np.float32), 12.0)
    assert rms_db(out) - rms_db(audio) == pytest.approx(-12.0, abs=0.5)


@needs_ai
def test_ai_reduces_noise_and_keeps_binaural_timing():
    mono = speech_like(6.0)
    delay = 24
    audio = np.stack([mono, np.roll(mono, delay)], axis=1) + noise(len(mono), -45)
    out = ai_denoise.reduce_noise_ai(audio, SR, 20.0)
    pause = slice(0, int(0.35 * SR))
    assert rms_db(out[pause]) < rms_db(audio[pause]) - 10
    window = slice(SR, 5 * SR)
    corr = np.correlate(out[window, 1], out[window, 0][100:-100], mode="valid")
    assert int(np.argmax(corr)) - 100 == delay


@needs_ai
def test_ai_keeps_real_speech_and_removes_noise(real_speech):
    audio = stereo(real_speech * 0.5) + noise(len(real_speech), -45)
    out = ai_denoise.reduce_noise_ai(audio, SR, 20.0)
    level = np.abs(real_speech) > 0.05
    assert abs(rms_db(out[level]) - rms_db(audio[level])) < 1.5  # speech kept
    gaps = np.convolve(np.abs(real_speech), np.ones(SR // 10), "same") < 1e-4
    assert rms_db(out[gaps]) < rms_db(audio[gaps]) - 12  # hiss removed between words


@needs_ai
def test_ai_option_offered_and_preview_close_to_render(make_session, real_speech):
    project, chain = make_session(stereo(real_speech * 0.5) + noise(len(real_speech), -45))
    i = index_of(chain, "A4")
    assert "ai" in [o.key for o in chain.steps[i].options]
    cfg = StepConfig(strength=50, option="ai")
    span = Region(3 * SR, 5 * SR)
    preview = chain.preview(project, i, cfg, span)
    full = chain.render(project.apply_step(i, cfg), i)
    diff_db = rms_db(preview.after - full[span.start : span.end]) - rms_db(full[span.start : span.end])
    assert diff_db < -40  # recurrent model: not bit-identical, but inaudibly close


def test_ai_option_hidden_and_explained_when_not_installed(monkeypatch):
    from voicefix.steps.flow_a.cleanup import AI_WARNING, Denoise

    step = Denoise()
    assert step.warning(30, "ai") == AI_WARNING
    assert step.warning(30, "1") is None
    monkeypatch.setattr(ai_denoise, "is_available", lambda: False)
    assert "ai" not in [o.key for o in step.options]
    with pytest.raises(ai_denoise.AIDenoiseUnavailable):
        step.prepare(None, SR, step.resolve_params(30, "ai", {}), None, None, None)
