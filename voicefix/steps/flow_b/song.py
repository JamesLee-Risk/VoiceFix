"""B0 import (vocal, accompaniment, key), B2 gap clean-up, B12 loudness and export."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

import numpy as np

from voicefix.core import analysis as an
from voicefix.core.regions import Region
from voicefix.dsp.dynamics import segment_gain_envelope
from voicefix.dsp.key import estimate_key
from voicefix.steps.base import Analysis, Option, Params, ParamSpec, Step, StepContext
from voicefix.steps.flow_a.inspect_edit import ImportCheck
from voicefix.steps.flow_a.level_export import LoudnessPeak

KEY_CONFIDENCE_OK = 0.6
GAP_FADE_SECONDS = 0.03
MUTE_DB = 120.0


class SongImport(ImportCheck):
    """Settings: {"key": "<key name>" | "auto"}. The accompaniment path is project-wide."""

    id = "B0_import"
    code = "B0"
    title = "放入你的歌聲"
    description = "檢查主唱錄音，可以放入伴奏（只用來試聽，不會被處理），並確認歌曲調性。"

    def analyze(self, audio: np.ndarray, sr: int, ctx: StepContext) -> Analysis:
        base = super().analyze(audio, sr, ctx)
        key, confidence = estimate_key(audio, sr)
        hint = "" if confidence >= KEY_CONFIDENCE_OK else "（把握不高，請確認）"
        return replace(
            base,
            summary=(*base.summary, f"自動偵測調性：{key}{hint}。調性會用在之後的音準修正。"),
            data={**base.data, "key": key, "key_confidence": confidence},
        )


class GapCleanup(Step):
    id = "B2_gaps"
    code = "B2"
    title = "清掉句子間的雜音"
    description = "把樂句之間只剩背景聲的空白壓低，去掉耳機漏音和環境聲。換氣聲會留給下一步處理。"
    term = "Gap clean-up / Downward expansion"
    strength_label = "空白降低量"
    default_strength = 50
    options = (Option("reduce", "降低音量"), Option("mute", "完全靜音"))
    option_label = "空白處理方式"
    default_option = "reduce"
    pad_seconds = 0.0
    advanced = (
        ParamSpec("reduction_db", "降低量", "dB", 0.0, 60.0, 1),
        ParamSpec("above_floor_db", "視為空白的音量（高於底噪）", "dB", 2.0, 15.0, 1),
        ParamSpec("min_gap_seconds", "最短空白長度", "秒", 0.1, 2.0, 2),
    )

    def map_strength(self, strength: int, option: Optional[str]) -> dict[str, float]:
        mute = (option or self.default_option) == "mute"
        return {
            "reduction_db": MUTE_DB if mute else 0.4 * strength,
            "above_floor_db": 6.0,
            "min_gap_seconds": 0.3,
        }

    def measure(self, audio: np.ndarray, sr: int, ctx: StepContext) -> tuple[np.ndarray, int]:
        return an.frame_power_db(audio, sr, 0.01)

    def _gaps(self, measured: tuple[np.ndarray, int], sr: int, params: Params) -> list[Region]:
        frame_db, frame = measured
        return an.quiet_regions(frame_db, frame, params["above_floor_db"], params["min_gap_seconds"], sr)

    def analyze(self, audio: np.ndarray, sr: int, ctx: StepContext) -> Analysis:
        measured = self.measure(audio, sr, ctx)
        gaps = self._gaps(measured, sr, self.resolve_params(None, None, {}))
        seconds = sum(g.length for g in gaps) / sr
        share = seconds / max(len(audio) / sr, 1e-9)
        lines = (f"找到 {len(gaps)} 段樂句間的空白，共 {seconds:.1f} 秒（佔 {share:.0%}）。",)
        if share < 0.05:
            return Analysis((*lines, "空白很少，可以略過。"), False, self.default_strength, "reduce", marks=tuple(gaps))
        return Analysis((*lines, "建議把空白壓低，讓混音時更乾淨。"), True, self.default_strength, "reduce", marks=tuple(gaps))

    def prepare(self, audio, sr, params, ctx, span, measured) -> list[tuple[Region, float]]:
        return [(gap, -params["reduction_db"]) for gap in self._gaps(measured, sr, params)]

    def process(self, chunk: np.ndarray, sr: int, params: Params, ctx: StepContext, plan: Any, offset: int) -> np.ndarray:
        fade = max(1, int(sr * GAP_FADE_SECONDS))
        return (chunk * segment_gain_envelope(len(chunk), offset, plan, fade)[:, None]).astype(np.float32)


# Vocal-stem targets are placeholders until measured against real deliveries.
VOCAL_LOUDNESS_PRESETS = {"stem": -20.0, "release": -16.0}


class SingingLoudnessExport(LoudnessPeak):
    id = "B12_export"
    code = "B12"
    title = "匯出"
    description = "調整整體音量並保護峰值，然後匯出人聲。也可以另外匯出乾聲，和與伴奏混合的試聽版。"
    term = "Loudness / True-peak limiter / Export"
    options = (Option("stem", "交給混音師（保留空間）"), Option("release", "自己使用（較大聲）"))
    default_option = "stem"
    panel = "export"
    audition_with_backing = True

    def map_strength(self, strength: int, option: Optional[str]) -> dict[str, float]:
        params = super().map_strength(strength, "quiet")
        return {**params, "target_lufs": VOCAL_LOUDNESS_PRESETS[option or self.default_option]}
