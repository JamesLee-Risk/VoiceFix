"""B3 noise reduction, B4 mouth clicks and pops, B5 breaths."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from voicefix.core.regions import Region
from voicefix.dsp import transients
from voicefix.dsp.breaths import BreathSettings, detect_breaths
from voicefix.dsp.dynamics import segment_gain_envelope
from voicefix.steps.base import Analysis, EventStep, Params, ParamSpec, StepContext
from voicefix.steps.flow_a.cleanup import Denoise

SINGING_DENOISE_SCALE = 0.7  # PRD: "same as A4, lighter defaults"
CLICK_MAX_SECONDS = 0.02  # clicks are <= 10 ms incl. padding; plosives >= 30 ms
BREATH_FADE_SECONDS = 0.015


class SingingDenoise(Denoise):
    id = "B3_denoise"
    code = "B3"
    title = "去除背景雜音"
    description = "讓樂句之間和尾音不再有嘶嘶聲。用 B0 選的底噪樣本學習噪音，預設比音聲作品更輕。"
    default_strength = 20

    def analyze(self, audio: np.ndarray, sr: int, ctx: StepContext) -> Analysis:
        base = super().analyze(audio, sr, ctx)
        lighter = int(round((base.suggested_strength or self.default_strength) * SINGING_DENOISE_SCALE))
        return replace(base, suggested_strength=lighter)


class ClicksAndPlosives(EventStep):
    id = "B4_clicks_pops"
    code = "B4"
    title = "去除口水聲與噴麥"
    description = "找出短促的口水音，以及ㄅ、ㄆ打到麥克風的「噗」聲，逐筆確認後修掉。"
    term = "De-click / De-plosive"
    strength_label = "敏感度"
    default_strength = 30
    event_label = "口水音或噴麥"
    pad_seconds = 0.5
    advanced = (
        ParamSpec("threshold_db", "口水音偵測門檻", "dB", 6.0, 40.0, 1),
        ParamSpec("plosive_threshold_db", "噴麥偵測門檻", "dB", 6.0, 30.0, 1),
        ParamSpec("cutoff_hz", "噴麥修補的低切頻率", "Hz", 60.0, 300.0, 0),
    )

    def map_strength(self, strength: int, option: Optional[str]) -> dict[str, float]:
        return {
            "threshold_db": 30.0 - 0.2 * strength,
            "plosive_threshold_db": 18.0 - 0.06 * strength,
            "cutoff_hz": 150.0,
        }

    @staticmethod
    def _is_click(region: Region, sr: int) -> bool:
        return region.length <= CLICK_MAX_SECONDS * sr

    def kind_label(self, region: Region, sr: int) -> str:
        return "口水音" if self._is_click(region, sr) else "噴麥"

    def detect(self, chunk: np.ndarray, sr: int, params: Params, offset: int) -> list[Region]:
        clicks = transients.detect_clicks(chunk, sr, transients.ClickSettings(params["threshold_db"]), offset)
        pops = transients.detect_plosives(
            chunk, sr, transients.PlosiveSettings(params["plosive_threshold_db"], params["cutoff_hz"]), offset
        )
        return sorted(clicks + pops)

    def repair(self, chunk: np.ndarray, sr: int, params: Params, events: list[Region], offset: int) -> np.ndarray:
        clicks = [e for e in events if self._is_click(e, sr)]
        pops = [e for e in events if not self._is_click(e, sr)]
        out = transients.repair_clicks(chunk, sr, clicks, transients.ClickSettings(params["threshold_db"]), offset)
        settings = transients.PlosiveSettings(params["plosive_threshold_db"], params["cutoff_hz"])
        return transients.repair_plosives(out, sr, pops, settings, offset)


class Breaths(EventStep):
    id = "B5_breaths"
    code = "B5"
    title = "調整換氣聲"
    description = "把太明顯的換氣聲調小（不是刪掉），保留自然的呼吸感。"
    term = "Breath control"
    strength_label = "降低量"
    default_strength = 30  # = -6 dB, the PRD default
    event_label = "換氣"
    pad_seconds = 0.1
    detect_pad_seconds = 2.0  # thresholds look at +-1.5 s of context
    max_event_seconds = 1.1
    advanced = (
        ParamSpec("reduction_db", "降低量", "dB", 0.0, 24.0, 1),
        ParamSpec("flatness", "偵測門檻（噪音感）", "", 0.05, 0.6, 2),
    )

    def map_strength(self, strength: int, option: Optional[str]) -> dict[str, float]:
        return {"reduction_db": 0.2 * strength, "flatness": BreathSettings().flatness}

    def detect(self, chunk: np.ndarray, sr: int, params: Params, offset: int) -> list[Region]:
        return detect_breaths(chunk, sr, BreathSettings(flatness=params["flatness"]), offset)

    def repair(self, chunk: np.ndarray, sr: int, params: Params, events: list[Region], offset: int) -> np.ndarray:
        if not events:
            return chunk
        fade = max(1, int(sr * BREATH_FADE_SECONDS))
        gains = [(e, -params["reduction_db"]) for e in events]
        return (chunk * segment_gain_envelope(len(chunk), offset, gains, fade)[:, None]).astype(np.float32)
