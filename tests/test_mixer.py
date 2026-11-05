import numpy as np
import pytest

from tests.conftest import SR, noise, speech_like, stereo
from voicefix.core import loudness
from voicefix.playback.mixer import CROSSFADE_SECONDS, AuditionMixer, AuditionMode


def make_mixer(after_gain: float = 0.5) -> tuple[AuditionMixer, np.ndarray, np.ndarray]:
    before = stereo(speech_like(4.0)) + noise(4 * SR, -40)
    after = (before * after_gain).astype(np.float32)
    mixer = AuditionMixer(SR)
    mixer.set_buffers(before, after, offset=0)
    return mixer, before, after


def test_gain_match_aligns_loudness_within_half_lu():
    mixer, before, after = make_mixer(0.3)
    mixer.set_mode(AuditionMode.AFTER)
    mixer.loop = False
    rendered, _ = mixer.render(len(before))
    diff = loudness.integrated_loudness(rendered, SR) - loudness.integrated_loudness(before, SR)
    assert abs(diff) <= 0.5


def test_switching_mode_keeps_position_and_crossfades():
    mixer, before, after = make_mixer()
    mixer.gain_match_enabled = False
    mixer.render(1000)
    mixer.set_mode(AuditionMode.BEFORE)
    assert mixer.position == 1000
    block, _ = mixer.render(2000)
    fade = int(CROSSFADE_SECONDS * SR)
    assert np.allclose(block[fade + 10 :], before[1000 + fade + 10 : 3000])
    assert np.max(np.abs(np.diff(block[:, 0]))) < np.max(np.abs(np.diff(before[:3000, 0]))) * 1.5


def test_removed_is_before_minus_after():
    mixer, before, after = make_mixer()
    mixer.set_mode(AuditionMode.REMOVED)
    mixer.render(mixer._fade_len)  # finish the crossfade
    mixer.seek(5000)
    block, _ = mixer.render(1000)
    assert np.allclose(block, (before - after)[5000:6000], atol=1e-6)


def test_loop_wraps_and_non_loop_finishes():
    mixer, before, _ = make_mixer()
    mixer.seek(len(before) - 100)
    _, finished = mixer.render(200)
    assert not finished and mixer.position == 100
    mixer.loop = False
    mixer.seek(len(before) - 100)
    _, finished = mixer.render(200)
    assert finished


def test_new_buffers_keep_absolute_position():
    mixer, before, after = make_mixer()
    mixer.render(SR)  # play head at 1 s
    mixer.set_buffers(before[SR // 2 :], after[SR // 2 :], offset=SR // 2)
    assert mixer.position + mixer.offset == SR


def test_mismatched_buffers_rejected():
    mixer = AuditionMixer(SR)
    with pytest.raises(ValueError):
        mixer.set_buffers(np.zeros((10, 2), np.float32), np.zeros((11, 2), np.float32), 0)


def test_backing_plays_with_before_and_after_but_not_removed():
    mixer, before, after = make_mixer()
    mixer.gain_match_enabled = False
    backing = np.full((len(before) + 1000, 2), 0.1, dtype=np.float32)
    mixer.set_backing(backing)
    mixer.backing_enabled = True
    mixer.seek(1000)
    block, _ = mixer.render(500)
    assert np.allclose(block, np.clip(after[1000:1500] + 0.1, -1, 1), atol=1e-6)
    mixer.set_mode(AuditionMode.REMOVED)
    mixer.render(mixer._fade_len)
    block, _ = mixer.render(500)
    start = 1500 + mixer._fade_len
    assert np.allclose(block, (before - after)[start : start + 500], atol=1e-6)
    assert np.array_equal(mixer.signal(AuditionMode.BEFORE), before)  # spectrogram shows the vocal only
