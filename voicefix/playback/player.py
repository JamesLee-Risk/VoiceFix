"""Sound-card output for the audition mixer (PortAudio via sounddevice)."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import sounddevice as sd

from voicefix.playback.mixer import OUTPUT_CHANNELS, AuditionMixer

log = logging.getLogger(__name__)
BLOCK_FRAMES = 512


class PlaybackError(RuntimeError):
    pass


class Player:
    def __init__(self, mixer: AuditionMixer) -> None:
        self.mixer = mixer
        self._stream: Optional[sd.OutputStream] = None

    @property
    def is_playing(self) -> bool:
        return self._stream is not None and self._stream.active

    def _callback(self, outdata: np.ndarray, frames: int, time, status) -> None:  # noqa: ANN001 - PortAudio signature
        if status:
            log.debug("Playback status: %s", status)
        block, finished = self.mixer.render(frames)
        outdata[:] = block
        if finished:
            raise sd.CallbackStop()

    def play(self) -> None:
        self.stop()
        try:
            self._stream = sd.OutputStream(
                samplerate=self.mixer.sr,
                channels=OUTPUT_CHANNELS,
                dtype="float32",
                blocksize=BLOCK_FRAMES,
                callback=self._callback,
            )
            self._stream.start()
        except sd.PortAudioError as exc:
            self._stream = None
            raise PlaybackError(f"無法開啟音訊輸出裝置：{exc}") from exc

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
