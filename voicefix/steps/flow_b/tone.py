"""B8 tone EQ, B9 compression, B10 de-essing. These are judged inside the
accompaniment, so they audition with it by default."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from voicefix.core import analysis as an
from voicefix.core.chunking import CHUNK_SECONDS
from voicefix.dsp.dynamics import HF_DOMINANCE_DB, compress, deess, sibilance_levels
from voicefix.dsp.filters import cascade, highpass_sos, peaking_sos, shelf_sos, zero_phase
from voicefix.dsp.tone import ToneProfile, analyse_tone
from voicefix.steps.base import Analysis, Option, Params, ParamSpec, Step, StepContext

STYLE_CODES = {"warm": 0.0, "clear": 1.0, "bright": 2.0}
ACTIVE_ABOVE_FLOOR_DB = 15.0


class ToneEQ(Step):
    id = "B8_eq"
    code = "B8"
    title = "調整音色"
    description = "依喜歡的風格調整音色，並自動壓下明顯刺耳的窄共振。已包含低切。"
    term = "EQ (low cut, resonance cuts, style curve)"
    default_strength = 40
    options = (Option("warm", "溫暖"), Option("clear", "清晰"), Option("bright", "明亮"))
    option_label = "風格"
    default_option = "clear"
    audition_with_backing = True
    pad_seconds = 1.0
    advanced = (
        ParamSpec("lowcut_hz", "低切頻率", "Hz", 40.0, 160.0, 0),
        ParamSpec("max_cut_db", "共振最大削減", "dB", 0.0, 12.0, 1),
    )

    def map_strength(self, strength: int, option: Optional[str]) -> dict[str, float]:
        return {
            "amount": strength / 100.0,
            "style": STYLE_CODES[option or self.default_option],
            "lowcut_hz": 80.0,
            "max_cut_db": 6.0,
        }

    def measure(self, audio: np.ndarray, sr: int, ctx: StepContext) -> ToneProfile:
        return analyse_tone(audio, sr)

    def analyze(self, audio: np.ndarray, sr: int, ctx: StepContext) -> Analysis:
        profile = analyse_tone(audio, sr)
        if profile.resonances:
            listed = "、".join(f"{hz:.0f} Hz（+{db:.1f} dB）" for hz, db in profile.resonances)
            lines = [f"找到明顯的窄共振：{listed}，會自動輕輕壓下。"]
        else:
            lines = ["沒有明顯刺耳的共振，只需要依喜好選風格。"]
        lines.append("「清晰」會稍微減少悶悶的中低頻；音色看個人喜好，建議在伴奏中聽，不需要可以略過。")
        return Analysis(tuple(lines), True, self.default_strength, self.default_option, data={"profile": profile})

    def prepare(self, audio, sr, params, ctx, span, measured: ToneProfile) -> np.ndarray:
        # every gain is halved because zero-phase filtering applies the filter twice
        a, style, cap = params["amount"], params["style"], params["max_cut_db"]
        sections = [highpass_sos(sr, params["lowcut_hz"], 2)]
        for hz, excess in measured.resonances:
            if hz < sr * 0.45:
                sections.append(peaking_sos(sr, hz, -min(excess, cap) * a / 2, 5.0))
        if style == STYLE_CODES["warm"]:
            sections += [shelf_sos(sr, 250.0, 2.5 * a / 2, high=False), shelf_sos(sr, 7000.0, -1.5 * a / 2, high=True)]
        elif style == STYLE_CODES["clear"]:
            sections += [peaking_sos(sr, 300.0, -2.0 * a / 2, 1.0), peaking_sos(sr, 3000.0, 2.0 * a / 2, 0.8)]
        else:
            sections += [shelf_sos(sr, 6000.0, 3.0 * a / 2, high=True), peaking_sos(sr, 300.0, -1.0 * a / 2, 1.0)]
        return cascade(sections)

    def process(self, chunk: np.ndarray, sr: int, params: Params, ctx: StepContext, plan: Any, offset: int) -> np.ndarray:
        return zero_phase(plan, chunk)


def active_level_db(audio: np.ndarray, sr: int) -> tuple[float, float]:
    """(typical level, dynamic range) of the sung parts, from 20 ms frames."""
    frame_db, _ = an.frame_power_db(audio, sr, 0.02)
    if len(frame_db) == 0:
        return -60.0, 0.0
    active = frame_db[frame_db > an.noise_floor_db(frame_db) + ACTIVE_ABOVE_FLOOR_DB]
    if len(active) < 10:
        return float(np.percentile(frame_db, 75)), 0.0
    return float(np.percentile(active, 75)), float(np.percentile(active, 95) - np.percentile(active, 10))


class Compressor(Step):
    id = "B9_compress"
    code = "B9"
    title = "讓音量更穩定"
    description = "把忽大忽小的歌聲收得穩一點，讓每個字在伴奏中都聽得清楚。太強會讓歌聲失去起伏。"
    term = "Compressor"
    strength_label = "穩定度"
    default_strength = 40
    warn_above = 75
    warn_text = "穩定度太高會讓歌聲變扁、失去情緒起伏。"
    audition_with_backing = True
    pad_seconds = 0.5
    advanced = (
        ParamSpec("ratio", "壓縮比", ":1", 1.0, 10.0, 1),
        ParamSpec("threshold_below_db", "門檻（低於人聲平均）", "dB", 0.0, 24.0, 1),
    )

    def map_strength(self, strength: int, option: Optional[str]) -> dict[str, float]:
        return {"ratio": 1.5 + 2.5 * strength / 100.0, "threshold_below_db": 2.0 + 10.0 * strength / 100.0}

    def measure(self, audio: np.ndarray, sr: int, ctx: StepContext) -> tuple[float, float]:
        return active_level_db(audio, sr)

    def analyze(self, audio: np.ndarray, sr: int, ctx: StepContext) -> Analysis:
        _, spread = active_level_db(audio, sr)
        strength = int(np.clip(20 + (spread - 6.0) * 4.0, 20, 70))
        lines = (f"歌聲的音量起伏約 {spread:.0f} dB。",)
        if spread < 8:
            return Analysis((*lines, "已經相當穩定，只需要輕微壓縮或略過。"), False, strength)
        return Analysis((*lines, "建議壓縮，讓歌聲在伴奏中更穩。"), True, strength)

    def prepare(self, audio, sr, params, ctx, span, measured: tuple[float, float]) -> dict[str, float]:
        level, _ = measured
        threshold = level - params["threshold_below_db"]
        makeup = 0.5 * params["threshold_below_db"] * (1.0 - 1.0 / params["ratio"])
        return {"threshold_db": threshold, "makeup_db": makeup}

    def process(self, chunk: np.ndarray, sr: int, params: Params, ctx: StepContext, plan: Any, offset: int) -> np.ndarray:
        return compress(chunk, sr, plan["threshold_db"], params["ratio"], plan["makeup_db"], offset)


SIBILANT_PERCENTILE = 97.0


def sibilant_reference_db(audio: np.ndarray, sr: int) -> tuple[float, int]:
    """(level the loudest 3 % of HF-dominant frames reach, rough count of 's' sounds).

    Frame levels are streamed per chunk but judged against whole-file
    statistics, so the result does not depend on where chunks start.
    """
    hf_parts, full_parts = [], []
    chunk = int(CHUNK_SECONDS * sr)
    for start in range(0, len(audio), chunk):
        hf_db, full_db, _, _ = sibilance_levels(np.asarray(audio[start : start + chunk]), sr)
        hf_parts.append(hf_db.astype(np.float32))
        full_parts.append(full_db.astype(np.float32))
    if not hf_parts:
        return -30.0, 0
    hf_db, full_db = np.concatenate(hf_parts), np.concatenate(full_parts)
    dominant = hf_db - full_db > HF_DOMINANCE_DB
    values = hf_db[dominant & (full_db > an.noise_floor_db(full_db) + ACTIVE_ABOVE_FLOOR_DB)]
    if len(values) < 10:
        return -30.0, 0
    count = len(an.runs_above(dominant & (hf_db > np.percentile(hf_db, 90))))
    return float(np.percentile(values, SIBILANT_PERCENTILE)), count


class DeEsser(Step):
    id = "B10_deess"
    code = "B10"
    title = "減少刺耳的ㄙ、ㄘ聲"
    description = "只在齒音出現的瞬間把高頻壓下來，其他時候不影響音色。"
    term = "De-esser (split band)"
    default_strength = 40
    warn_above = 80
    warn_text = "太強會讓ㄙ、ㄘ聽起來像口齒不清。"
    audition_with_backing = True
    pad_seconds = 0.3
    advanced = (
        ParamSpec("threshold_below_db", "門檻（低於齒音高峰）", "dB", 0.0, 24.0, 1),
        ParamSpec("max_reduction_db", "最大削減", "dB", 0.0, 18.0, 1),
    )

    def map_strength(self, strength: int, option: Optional[str]) -> dict[str, float]:
        return {"threshold_below_db": 12.0 * strength / 100.0, "max_reduction_db": 3.0 + 9.0 * strength / 100.0}

    def measure(self, audio: np.ndarray, sr: int, ctx: StepContext) -> tuple[float, int]:
        return sibilant_reference_db(audio, sr)

    def analyze(self, audio: np.ndarray, sr: int, ctx: StepContext) -> Analysis:
        _, count = sibilant_reference_db(audio, sr)
        minutes = max(len(audio) / sr / 60.0, 1e-9)
        if count == 0:
            return Analysis(("沒有明顯的齒音，可以略過。",), False, self.default_strength)
        return Analysis((f"偵測到約 {count} 處齒音（每分鐘約 {count / minutes:.0f} 處）。",), True, self.default_strength)

    def prepare(self, audio, sr, params, ctx, span, measured: tuple[float, int]) -> float:
        reference, _ = measured
        return reference - params["threshold_below_db"]

    def process(self, chunk: np.ndarray, sr: int, params: Params, ctx: StepContext, plan: Any, offset: int) -> np.ndarray:
        return deess(chunk, sr, plan, params["max_reduction_db"], offset)
