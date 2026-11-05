"""A2 low cut, A3 de-hum, A4 noise reduction, A5 mouth clicks, A6 plosives."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from voicefix.core import analysis as an
from voicefix.core.regions import Region
from voicefix.dsp import ai_denoise, denoise, transients
from voicefix.dsp.filters import cascade, highpass_sos, peaking_sos, zero_phase
from voicefix.steps.base import Analysis, EventStep, Option, Params, ParamSpec, Step, StepContext

NOISE_SAMPLE_SECONDS = 3.0
CLASSIC_OPTIONS = (Option("1", "傳統 1 次"), Option("2", "傳統 2 次（較自然）"), Option("3", "傳統 3 次（最細緻）"))
AI_OPTION = Option("ai", "AI 降噪（較強）")
AI_WARNING = "AI 降噪是用說話聲訓練的，可能把舔耳、口腔音等表演音或歌聲當成噪音刪掉。請切到「被移除的聲音」確認，必要時把表演音標成保護區段。"


class LowCut(Step):
    id = "A2_lowcut"
    code = "A2"
    title = "去除低頻震動聲"
    description = "濾掉桌子震動、冷氣低鳴、麥克風架碰撞這類人耳不需要的低頻。"
    term = "High-pass filter / Low cut"
    default_strength = 30
    respects_protection = False
    shared_keys = ()
    pad_seconds = 1.0
    advanced = (ParamSpec("cutoff_hz", "截止頻率", "Hz", 20.0, 150.0, 0),)

    def map_strength(self, strength: int, option: Optional[str]) -> dict[str, float]:
        return {"cutoff_hz": 40.0 + 0.6 * strength}

    def analyze(self, audio: np.ndarray, sr: int, ctx: StepContext) -> Analysis:
        freqs, power = an.spectrum_excerpts(audio, sr, nperseg=8192)
        low = power[freqs < 80].sum() / max(power.sum(), 1e-20)
        low_db = 10 * np.log10(max(low, 1e-12))
        if low_db > -15:
            return Analysis((f"80 Hz 以下的能量偏高（佔 {low_db:.0f} dB），建議去除。",), True, 60)
        if low_db > -30:
            return Analysis((f"有一些低頻聲（佔 {low_db:.0f} dB），建議輕度去除。",), True, 40)
        return Analysis((f"低頻很少（佔 {low_db:.0f} dB），可以略過。",), False, 30)

    def process(self, chunk: np.ndarray, sr: int, params: Params, ctx: StepContext, plan: Any, offset: int) -> np.ndarray:
        return zero_phase(highpass_sos(sr, params["cutoff_hz"], 2), chunk)


MAINS_CANDIDATES = (50.0, 60.0)
MAX_HARMONICS = 8
HUM_DETECT_DB = 8.0


def hum_prominence(freqs: np.ndarray, power: np.ndarray, f0: float, harmonics: int) -> list[float]:
    """Peak-over-neighbourhood level (dB) at each harmonic of f0."""
    result = []
    db = 10 * np.log10(np.maximum(power, 1e-20))
    for k in range(1, harmonics + 1):
        f = k * f0
        near = np.abs(freqs - f)
        peak = db[near <= 1.5].max(initial=-200.0)
        ring = db[(near > 3.0) & (near <= 12.0)]
        result.append(float(peak - np.median(ring)) if len(ring) else 0.0)
    return result


class DeHum(Step):
    id = "A3_dehum"
    code = "A3"
    title = "去除嗡嗡聲"
    description = "去掉電源干擾造成的「嗡——」低鳴（50 或 60 Hz 與它的倍頻）。"
    term = "De-hum / Notch filter"
    default_strength = 40
    respects_protection = False
    shared_keys = ()
    pad_seconds = 2.0
    options = (Option("auto", "自動判斷"), Option("50", "50 Hz"), Option("60", "60 Hz"))
    option_label = "電源頻率"
    default_option = "auto"
    advanced = (
        ParamSpec("harmonics", "處理的倍頻數", "個", 1, MAX_HARMONICS, 0),
        ParamSpec("depth_db", "削減深度", "dB", 6.0, 48.0, 0),
        ParamSpec("q", "頻寬（Q 值）", "", 5.0, 80.0, 0),
    )

    def map_strength(self, strength: int, option: Optional[str]) -> dict[str, float]:
        mains = {"50": 50.0, "60": 60.0}.get(option or "auto", 0.0)
        return {
            "harmonics": float(round(2 + 6 * strength / 100)),
            "depth_db": 12.0 + 0.28 * strength,
            "q": 25.0,
            "mains_hz": mains,
        }

    def measure(self, audio: np.ndarray, sr: int, ctx: StepContext) -> dict[str, Any]:
        freqs, power = an.spectrum_excerpts(audio, sr, nperseg=1 << int(np.ceil(np.log2(sr))))
        scores = {f0: hum_prominence(freqs, power, f0, MAX_HARMONICS) for f0 in MAINS_CANDIDATES}
        best = max(scores, key=lambda f0: sum(max(p, 0.0) for p in scores[f0]))
        return {"mains_hz": best, "prominence": scores[best]}

    def analyze(self, audio: np.ndarray, sr: int, ctx: StepContext) -> Analysis:
        found = self.measure(audio, sr, ctx)
        strongest = max(found["prominence"])
        detected = sum(p > HUM_DETECT_DB / 2 for p in found["prominence"])
        if strongest < HUM_DETECT_DB:
            return Analysis(("沒有偵測到明顯的電源嗡嗡聲，建議略過。",), False, self.default_strength, "auto")
        strength = int(np.clip(10 + detected * 10, 20, 90))
        return Analysis(
            (f"偵測到 {found['mains_hz']:.0f} Hz 嗡嗡聲，最明顯處高出周圍 {strongest:.0f} dB，共 {detected} 個倍頻。",),
            True,
            strength,
            "auto",
            data=found,
        )

    def prepare(self, audio, sr, params, ctx, span, measured) -> np.ndarray:
        mains = params["mains_hz"] or measured["mains_hz"]
        count = int(params["harmonics"])
        sections = [
            peaking_sos(sr, k * mains, -params["depth_db"] / 2.0, params["q"])  # filtfilt doubles the cut
            for k in range(1, count + 1)
            if k * mains < sr * 0.45
        ]
        return cascade(sections)

    def process(self, chunk: np.ndarray, sr: int, params: Params, ctx: StepContext, plan: Any, offset: int) -> np.ndarray:
        return zero_phase(plan, chunk)


class Denoise(Step):
    id = "A4_denoise"
    code = "A4"
    title = "去除背景沙沙聲"
    description = "讓安靜段落不再有嘶嘶聲。先用 A0 選的底噪樣本學習噪音長相，再把它減掉。"
    term = "Noise reduction / Spectral gating"
    default_strength = 30
    warn_above = 70
    warn_text = "強度超過 70 可能產生水聲或金屬感。建議降低強度，改用 2～3 次輕降。"
    option_label = "處理方式"
    default_option = "1"
    shared_keys = ("protected", "noise_sample")
    pad_seconds = 2.0  # also warms up the AI model's recurrent state
    advanced = (
        ParamSpec("reduction_db", "最大降噪量", "dB", 3.0, 40.0, 1),
        ParamSpec("over_subtraction", "噪音扣除倍率", "x", 0.5, 3.0, 2),
    )

    def warning(self, strength: int, option: Optional[str] = None) -> Optional[str]:
        if option == AI_OPTION.key:
            return AI_WARNING
        return super().warning(strength, option)

    @property
    def options(self) -> tuple[Option, ...]:  # type: ignore[override]
        """The AI engine is offered only when its optional packages are installed."""
        return CLASSIC_OPTIONS + ((AI_OPTION,) if ai_denoise.is_available() else ())

    def map_strength(self, strength: int, option: Optional[str]) -> dict[str, float]:
        ai = (option or self.default_option) == AI_OPTION.key
        return {
            "reduction_db": 4.0 + 0.26 * strength,
            "over_subtraction": 1.0 + strength / 100.0,
            "passes": 1.0 if ai else float(int(option or "1")),
            "ai": 1.0 if ai else 0.0,
        }

    def _noise_region(self, audio: np.ndarray, sr: int, ctx: StepContext) -> Optional[Region]:
        if ctx.noise_sample and ctx.noise_sample.length >= denoise.stft_size(sr):
            return ctx.noise_sample
        frame_db, frame = an.frame_power_db(audio, sr, 0.05)
        return an.quietest_window(frame_db, frame, int(NOISE_SAMPLE_SECONDS / 0.05))

    def measure(self, audio: np.ndarray, sr: int, ctx: StepContext) -> Optional[Region]:
        return self._noise_region(audio, sr, ctx)

    def measure_fingerprint(self, ctx: StepContext) -> Any:
        return ctx.noise_sample.to_json() if ctx.noise_sample else None

    def analyze(self, audio: np.ndarray, sr: int, ctx: StepContext) -> Analysis:
        region = self._noise_region(audio, sr, ctx)
        if region is None:
            return Analysis(("錄音太短，無法學習噪音。",), False, self.default_strength, "1")
        noise_db = an.noise_floor_db(an.frame_power_db(np.asarray(audio[region.start : region.end]), sr, 0.05)[0])
        frame_db, _ = an.frame_power_db(audio, sr, 0.05)
        snr = float(np.percentile(frame_db, 95)) - noise_db
        source = "A0 選的底噪樣本" if ctx.noise_sample else "自動找到的最安靜片段"
        lines = [f"使用{source}學習噪音。估計訊噪比約 {snr:.0f} dB。"]
        if snr > 55:
            return Analysis((*lines, "錄音已經很乾淨，可以略過。"), False, 20, "1", marks=(region,))
        if snr > 40:
            return Analysis((*lines, "建議輕度降噪。"), True, 25, "1", marks=(region,))
        if snr > 25:
            return Analysis((*lines, "建議中度降噪。"), True, 35, "2", marks=(region,))
        return Analysis((*lines, "噪音明顯，建議分 2～3 次輕降。"), True, 45, "3", marks=(region,))

    @staticmethod
    def _settings(params: Params) -> denoise.DenoiseSettings:
        passes = max(1, int(params["passes"]))
        return denoise.DenoiseSettings(
            reduction_db=params["reduction_db"] / passes,
            over_subtraction=1.0 + (params["over_subtraction"] - 1.0) / passes,
        )

    def prepare(self, audio, sr, params, ctx, span, measured) -> list[np.ndarray]:
        if params["ai"]:
            if not ai_denoise.is_available():
                raise ai_denoise.AIDenoiseUnavailable(f"這個步驟設定為 AI 降噪。{ai_denoise.INSTALL_HINT}")
            return []  # the AI engine needs no noise profile
        if measured is None:
            return []
        noise = np.asarray(audio[measured.start : measured.end], dtype=np.float32)
        return denoise.multi_pass_profiles(noise, sr, self._settings(params), int(params["passes"]))

    def process(self, chunk: np.ndarray, sr: int, params: Params, ctx: StepContext, plan: Any, offset: int) -> np.ndarray:
        if params["ai"]:
            return ai_denoise.reduce_noise_ai(chunk, sr, params["reduction_db"], offset)
        settings = self._settings(params)
        out = chunk
        for profile in plan:
            out = denoise.reduce_noise(out, sr, profile, settings, offset)
        return out


class MouthClicks(EventStep):
    id = "A5_clicks"
    code = "A5"
    title = "去除口水聲、黏嘴聲"
    description = "找出短促的「嘖」「啪」口水音並修掉。舔耳等刻意的表演音請先標成保護區段。"
    term = "De-click / Mouth de-click"
    strength_label = "敏感度"
    default_strength = 30
    event_label = "口水音"
    pad_seconds = 0.1
    advanced = (
        ParamSpec("threshold_db", "偵測門檻（高出周圍）", "dB", 6.0, 40.0, 1),
        ParamSpec("repair_lowpass_hz", "修補時保留的頻率上限", "Hz", 800.0, 6000.0, 0),
    )

    def map_strength(self, strength: int, option: Optional[str]) -> dict[str, float]:
        return {"threshold_db": 30.0 - 0.2 * strength, "repair_lowpass_hz": 2000.0}

    @staticmethod
    def _settings(params: Params) -> transients.ClickSettings:
        return transients.ClickSettings(params["threshold_db"], repair_lowpass_hz=params["repair_lowpass_hz"])

    def detect(self, chunk: np.ndarray, sr: int, params: Params, offset: int) -> list[Region]:
        return transients.detect_clicks(chunk, sr, self._settings(params), offset)

    def repair(self, chunk: np.ndarray, sr: int, params: Params, events: list[Region], offset: int) -> np.ndarray:
        return transients.repair_clicks(chunk, sr, events, self._settings(params), offset)


class Plosives(EventStep):
    id = "A6_plosives"
    code = "A6"
    title = "去除噴麥聲"
    description = "修掉ㄅ、ㄆ這類氣流直接打到麥克風的「噗」聲。"
    term = "De-plosive / Pop removal"
    default_strength = 50
    event_label = "噴麥"
    pad_seconds = 0.5
    advanced = (
        ParamSpec("threshold_db", "偵測門檻（低頻突增）", "dB", 6.0, 30.0, 1),
        ParamSpec("cutoff_hz", "修補的低切頻率", "Hz", 60.0, 300.0, 0),
    )

    def map_strength(self, strength: int, option: Optional[str]) -> dict[str, float]:
        return {"threshold_db": 15.0, "cutoff_hz": 80.0 + 1.7 * strength}

    @staticmethod
    def _settings(params: Params) -> transients.PlosiveSettings:
        return transients.PlosiveSettings(params["threshold_db"], params["cutoff_hz"])

    def detect(self, chunk: np.ndarray, sr: int, params: Params, offset: int) -> list[Region]:
        return transients.detect_plosives(chunk, sr, self._settings(params), offset)

    def repair(self, chunk: np.ndarray, sr: int, params: Params, events: list[Region], offset: int) -> np.ndarray:
        return transients.repair_plosives(chunk, sr, events, self._settings(params), offset)
