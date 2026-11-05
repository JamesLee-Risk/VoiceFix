"""A7 breath adjustment and A9 room-tone fill (flow A, v1)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from voicefix.core import analysis as an
from voicefix.core.regions import Region, ScopeMode, region_mask
from voicefix.dsp import roomtone
from voicefix.dsp.breaths import BreathSettings
from voicefix.steps.base import Analysis, Params, ParamSpec, Step, StepContext
from voicefix.steps.flow_b.cleanup import Breaths

FALLBACK_SAMPLE_SECONDS = 0.5  # short enough to fit between words when A0 set no sample
FILL_FADE_SECONDS = 0.02


class VoiceBreaths(Breaths):
    id = "A7_breaths"
    code = "A7"
    title = "調整呼吸聲"
    description = "把太明顯的呼吸聲調小一點。呼吸也是表演的一部分，預設只降一點點；刻意的喘息請標成保護區段。"
    strength_label = "降低量"
    default_strength = 25  # = -3 dB, the PRD default (range 0 to -12 dB)
    event_label = "呼吸"
    advanced = (
        ParamSpec("reduction_db", "降低量", "dB", 0.0, 12.0, 1),
        ParamSpec("flatness", "偵測門檻（噪音感）", "", 0.05, 0.6, 2),
    )

    def map_strength(self, strength: int, option: Optional[str]) -> dict[str, float]:
        return {"reduction_db": 0.12 * strength, "flatness": BreathSettings().flatness}


class RoomToneFill(Step):
    id = "A9_roomtone"
    code = "A9"
    title = "讓空白處自然"
    description = "剪輯或靜音後的空白會變成「完全無聲」，聽起來像斷掉。這一步用你錄音裡的背景聲把它補起來。"
    term = "Room tone fill"
    strength_label = "音量"
    default_strength = 50  # same level as the room
    scopes = (ScopeMode.WHOLE,)
    respects_protection = False  # PRD: protected regions are exempt in A4-A8 only
    shared_keys = ("noise_sample",)
    pad_seconds = 0.0
    advanced = (ParamSpec("gain_db", "底噪音量（相對原本）", "dB", -24.0, 12.0, 1),)

    def map_strength(self, strength: int, option: Optional[str]) -> dict[str, float]:
        return {"gain_db": (strength - 50) * 0.24}  # 0..100 -> -12..+12 dB

    def _noise_region(self, audio: np.ndarray, sr: int, ctx: StepContext) -> Optional[Region]:
        if ctx.noise_sample and ctx.noise_sample.length > int(roomtone.GRAIN_SECONDS * sr) * 2:
            return ctx.noise_sample
        frame_db, frame = an.frame_power_db(audio, sr, 0.05)
        return an.quietest_window(frame_db, frame, int(FALLBACK_SAMPLE_SECONDS / 0.05))  # skips digital silence

    def measure(self, audio: np.ndarray, sr: int, ctx: StepContext) -> dict[str, Any]:
        region = self._noise_region(audio, sr, ctx)
        if region is None:
            return {"noise": None, "silences": []}
        noise = np.array(audio[region.start : region.end], dtype=np.float32)
        floor = roomtone.room_floor_db(noise, sr)
        return {"noise": noise, "silences": roomtone.silent_regions(audio, sr, floor), "region": region}

    def measure_fingerprint(self, ctx: StepContext) -> Any:
        return ctx.noise_sample.to_json() if ctx.noise_sample else None

    def analyze(self, audio: np.ndarray, sr: int, ctx: StepContext) -> Analysis:
        measured = self.measure(audio, sr, ctx)
        silences = measured["silences"]
        if measured["noise"] is None:
            return Analysis(("找不到可以當作背景聲的片段，無法補底噪。",), False, self.default_strength)
        seconds = sum(r.length for r in silences) / sr
        if not silences:
            return Analysis(("沒有「完全無聲」的空白，不需要補底噪。",), False, self.default_strength)
        return Analysis(
            (f"找到 {len(silences)} 段完全無聲的空白（共 {seconds:.1f} 秒），會用背景聲補起來。",),
            True,
            self.default_strength,
            marks=tuple(silences),
        )

    def process(self, chunk: np.ndarray, sr: int, params: Params, ctx: StepContext, plan: Any, offset: int) -> np.ndarray:
        if plan["noise"] is None or not plan["silences"]:
            return chunk
        end = offset + len(chunk)
        mask = region_mask(len(chunk), plan["silences"], max(1, int(FILL_FADE_SECONDS * sr)), fades_inside=True, offset=offset)
        if not np.any(mask):
            return chunk
        fill = roomtone.room_tone(plan["noise"], sr, offset, end - offset) * np.float32(10.0 ** (params["gain_db"] / 20.0))
        return (chunk + fill * mask[:, None]).astype(np.float32)
