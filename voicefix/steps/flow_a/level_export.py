"""A8 per-segment levelling, A10 loudness and peak protection, A11 export."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from voicefix.core import analysis as an
from voicefix.core import loudness
from voicefix.core.regions import Region, ScopeMode
from voicefix.dsp.dynamics import limit, segment_gain_envelope
from voicefix.steps.base import Analysis, Option, Params, ParamSpec, PassthroughStep, Step, StepContext

WHISPER_BELOW_DB = 10.0  # segments this far under the median count as whispers
LEVEL_FADE_SECONDS = 0.02
MIN_SPREAD_DB = 3.0


@dataclass(frozen=True)
class SegmentLevels:
    segments: tuple[Region, ...]
    loudness: tuple[float, ...]
    whisper: tuple[bool, ...]

    def median(self, whisper: bool) -> Optional[float]:
        values = [l for l, w in zip(self.loudness, self.whisper) if w == whisper]
        return float(np.median(values)) if values else None


def measure_segments(audio: np.ndarray, sr: int) -> SegmentLevels:
    segments, _, _ = an.speech_segments(audio, sr)
    levels = [loudness.segment_loudness(np.asarray(audio[s.start : s.end]), sr) for s in segments]
    if not levels:
        return SegmentLevels((), (), ())
    median = float(np.median(levels))
    return SegmentLevels(tuple(segments), tuple(levels), tuple(l < median - WHISPER_BELOW_DB for l in levels))


class Leveler(Step):
    id = "A8_level"
    code = "A8"
    title = "讓音量一致"
    description = "把太小聲或太大聲的台詞拉到差不多的音量。只調整有聲音的段落，不會把空白處的底噪拉高。"
    term = "Clip gain levelling"
    strength_label = "平整程度"
    default_strength = 50
    options = (Option("separate", "耳語保持較小聲"), Option("together", "耳語也拉到一樣大"))
    option_label = "耳語段落"
    default_option = "separate"
    pad_seconds = 0.0
    advanced = (ParamSpec("max_gain_db", "單段最大調整量", "dB", 3.0, 24.0, 1),)

    def map_strength(self, strength: int, option: Optional[str]) -> dict[str, float]:
        return {
            "amount": strength / 100.0,
            "max_gain_db": 12.0,
            "separate_whisper": 1.0 if (option or self.default_option) == "separate" else 0.0,
        }

    def measure(self, audio: np.ndarray, sr: int, ctx: StepContext) -> SegmentLevels:
        return measure_segments(audio, sr)

    def analyze(self, audio: np.ndarray, sr: int, ctx: StepContext) -> Analysis:
        levels = measure_segments(audio, sr)
        if len(levels.segments) < 2:
            return Analysis(("台詞段落太少，不需要平整。",), False, self.default_strength)
        normal = [l for l, w in zip(levels.loudness, levels.whisper) if not w] or list(levels.loudness)
        spread = float(np.percentile(normal, 90) - np.percentile(normal, 10))
        whispers = sum(levels.whisper)
        lines = [f"共 {len(levels.segments)} 段台詞，一般段落的音量落差約 {spread:.1f} dB。"]
        if whispers:
            lines.append(f"其中 {whispers} 段判斷為耳語，預設會保持比一般台詞小聲。")
        recommend = spread > MIN_SPREAD_DB or whispers > 0
        strength = int(np.clip(30 + spread * 4, 30, 80))
        return Analysis(tuple(lines), recommend, strength, "separate", marks=levels.segments)

    def prepare(self, audio, sr, params, ctx, span, measured: SegmentLevels) -> list[tuple[Region, float]]:
        separate = params["separate_whisper"] >= 0.5
        main_target = measured.median(False) if measured.median(False) is not None else measured.median(True)
        gains = []
        for seg, level, is_whisper in zip(measured.segments, measured.loudness, measured.whisper):
            target = measured.median(True) if (separate and is_whisper) else main_target
            if target is None or level <= loudness.SILENCE_LUFS:
                continue
            gain = np.clip(params["amount"] * (target - level), -params["max_gain_db"], params["max_gain_db"])
            gains.append((seg, float(gain)))
        return gains

    def process(self, chunk: np.ndarray, sr: int, params: Params, ctx: StepContext, plan: Any, offset: int) -> np.ndarray:
        fade = max(1, int(sr * LEVEL_FADE_SECONDS))
        return (chunk * segment_gain_envelope(len(chunk), offset, plan, fade)[:, None]).astype(np.float32)


# Target loudness for the presets is an open item in the PRD (to be decided after
# measuring published works); these values are placeholders.
LOUDNESS_PRESETS = {"quiet": -24.0, "standard": -18.0}
DEFAULT_CEILING_DBTP = -1.0
MAX_LOUDNESS_GAIN_DB = 30.0


class LoudnessPeak(Step):
    id = "A10_loudness"
    code = "A10"
    title = "最後的音量保護"
    description = "把整體音量調到合適的大小，並確保沒有任何聲音超過上限而刺耳或破音。保護區段也會套用。"
    term = "Loudness normalisation / True-peak limiter"
    has_strength = False
    options = (Option("quiet", "安靜（適合助眠、耳語作品）"), Option("standard", "標準"))
    option_label = "目標響度"
    default_option = "quiet"
    scopes = (ScopeMode.WHOLE,)
    respects_protection = False
    shared_keys = ()
    pad_seconds = 0.2
    advanced = (
        ParamSpec("target_lufs", "目標響度", "LUFS", -36.0, -10.0, 1),
        ParamSpec("ceiling_dbtp", "峰值上限", "dBTP", -6.0, -0.1, 1),
    )

    def map_strength(self, strength: int, option: Optional[str]) -> dict[str, float]:
        return {"target_lufs": LOUDNESS_PRESETS[option or self.default_option], "ceiling_dbtp": DEFAULT_CEILING_DBTP}

    def measure(self, audio: np.ndarray, sr: int, ctx: StepContext) -> dict[str, float]:
        return {
            "integrated": loudness.integrated_loudness(audio, sr),
            "true_peak_db": loudness.linear_to_db(loudness.true_peak(audio)),
        }

    def analyze(self, audio: np.ndarray, sr: int, ctx: StepContext) -> Analysis:
        m = self.measure(audio, sr, ctx)
        return Analysis(
            (f"目前整體響度 {m['integrated']:.1f} LUFS，最大峰值 {m['true_peak_db']:.1f} dBTP。",),
            True,
            suggested_option=self.default_option,
            data=m,
        )

    def prepare(self, audio, sr, params, ctx, span, measured: dict[str, float]) -> float:
        if measured["integrated"] <= loudness.SILENCE_LUFS:
            return 1.0
        gain_db = np.clip(params["target_lufs"] - measured["integrated"], -MAX_LOUDNESS_GAIN_DB, MAX_LOUDNESS_GAIN_DB)
        return loudness.db_to_linear(float(gain_db))

    def process(self, chunk: np.ndarray, sr: int, params: Params, ctx: StepContext, plan: Any, offset: int) -> np.ndarray:
        return limit(chunk * np.float32(plan), sr, params["ceiling_dbtp"])


class Export(PassthroughStep):
    """Settings: {"format": "wav24" | "wav16" | "flac" | "mp3", "split": bool}."""

    id = "A11_export"
    code = "A11"
    title = "匯出"
    description = "選擇格式並匯出成品。原始檔不會被覆寫。"
    term = "Export"
    panel = "export"

    def analyze(self, audio: np.ndarray, sr: int, ctx: StepContext) -> Analysis:
        return Analysis(("選好格式後按「匯出」。可以依台詞段落分成多個檔案。",), True)
