"""Flow C: multi-track singing (1 lead + up to 4 harmonies / doubles).

Every track runs the same steps as flow B. Timing goes beat -> lead -> harmony:
C7a moves the lead onto the accompaniment's beat, C7b moves each harmony onto
the (already aligned) lead, and C12 uses the lead's loudness for one common
gain, so the balance between the parts is kept.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from voicefix.core import analysis as an
from voicefix.dsp import align
from voicefix.steps.base import Analysis, Params, Step, StepContext
from voicefix.steps.flow_b.cleanup import Breaths, ClicksAndPlosives, SingingDenoise
from voicefix.steps.flow_b.pitch_space import PitchCorrection, Space
from voicefix.steps.flow_b.song import GapCleanup, SingingLoudnessExport, SongImport
from voicefix.steps.flow_b.tone import Compressor, DeEsser, ToneEQ
from voicefix.steps.timing import AlignPlan, BeatAlign, apply_pieces, phrase_pieces

MAX_LAG_SECONDS = 0.15
ALIGN_WORTH_MS = 10.0


def _variant(cls: type[Step], code: str, **attributes: Any) -> type[Step]:
    """The flow B step ``cls`` under a flow C id and code."""
    suffix = cls.id.split("_", 1)[1]
    return type(f"C{cls.__name__}", (cls,), {"id": f"{code}_{suffix}", "code": code, **attributes})


class TimeAlign(Step):
    id = "C7_align"
    code = "C7b"
    title = "對齊和聲與疊唱"
    description = "把和聲／疊唱的每一句微調到和主唱同時開始、同時結束。主唱是基準，不會被移動；音高不變。"
    term = "Vocal alignment (onset cross-correlation + WSOLA)"
    strength_label = "對齊程度"
    default_strength = 80
    scopes = ()
    uses_reference = True
    audition_with_backing = True
    pad_seconds = 0.0

    def map_strength(self, strength: int, option: Optional[str]) -> dict[str, float]:
        return {"amount": strength / 100.0}

    def measure_fingerprint(self, ctx: StepContext) -> Any:
        return ctx.reference_key

    def measure(self, audio: np.ndarray, sr: int, ctx: StepContext) -> Optional[dict[str, Any]]:
        lead = ctx.reference()
        if lead is None:
            return None  # this is the lead track
        curve = align.estimate_offsets(np.asarray(lead), np.asarray(audio), sr, MAX_LAG_SECONDS)
        segments, _, _ = an.speech_segments(audio, sr)
        return {"curve": curve, "segments": segments}

    def analyze(self, audio: np.ndarray, sr: int, ctx: StepContext) -> Analysis:
        measured = self.measure(audio, sr, ctx)
        if measured is None:
            return Analysis(("這是主唱軌：主唱是對齊的基準，不需要調整。",), False, self.default_strength)
        lags = [abs(measured["curve"].at_samples((s.start + s.end) // 2, 1)[0]) / sr * 1000 for s in measured["segments"]]
        if not lags:
            return Analysis(("這一軌沒有聲音，不需要對齊。",), False, self.default_strength)
        typical, worst = float(np.median(lags)), float(np.max(lags))
        lines = (f"和主唱的時間差：一般約 {typical:.0f} ms，最大約 {worst:.0f} ms（共 {len(lags)} 句）。",)
        if typical < ALIGN_WORTH_MS:
            return Analysis((*lines, "已經很整齊，可以略過。"), False, self.default_strength)
        return Analysis((*lines, "建議對齊，讓和聲聽起來和主唱是一起唱的。"), True, self.default_strength)

    def prepare(self, audio, sr, params, ctx, span, measured) -> Optional[AlignPlan]:
        if measured is None:
            return None
        return AlignPlan(audio, sr, measured["curve"], params["amount"], phrase_pieces(audio, sr, measured["segments"]))

    def process(self, chunk: np.ndarray, sr: int, params: Params, ctx: StepContext, plan: Any, offset: int) -> np.ndarray:
        return apply_pieces(chunk, sr, plan, offset)


class LinkedLoudnessExport(SingingLoudnessExport):
    """One gain for every track, measured on the lead, so the balance between
    the lead and the harmonies stays as recorded."""

    id = "C12_export"
    code = "C12"
    description = "以主唱決定整體音量，所有軌套用同一個增益並保護峰值，然後匯出各軌與混合試聽版。"
    uses_reference = True

    def measure_fingerprint(self, ctx: StepContext) -> Any:
        return ctx.reference_key

    def measure(self, audio: np.ndarray, sr: int, ctx: StepContext) -> dict[str, float]:
        lead = ctx.reference()
        return super().measure(lead if lead is not None else audio, sr, ctx)

    def analyze(self, audio: np.ndarray, sr: int, ctx: StepContext) -> Analysis:
        base = super().analyze(audio, sr, ctx)
        note = "（以主唱量測，所有軌用同一個增益）" if ctx.reference_loader else "（主唱，其他軌會跟著用同一個增益）"
        return Analysis((base.summary[0] + note,), base.recommend, base.suggested_strength, base.suggested_option, data=base.data)


FLOW_C_STEPS: tuple[type[Step], ...] = (
    _variant(SongImport, "C0", title="放入主唱與和聲", description="檢查每一軌的錄音，可以放入伴奏（只用來試聽），並確認歌曲調性。"),
    _variant(GapCleanup, "C2"),
    _variant(SingingDenoise, "C3"),
    _variant(ClicksAndPlosives, "C4"),
    _variant(Breaths, "C5"),
    _variant(PitchCorrection, "C6"),
    _variant(BeatAlign, "C7a", title="讓主唱跟上拍子", lead_only=True),
    TimeAlign,
    _variant(ToneEQ, "C8"),
    _variant(Compressor, "C9"),
    _variant(DeEsser, "C10"),
    _variant(Space, "C11"),
    LinkedLoudnessExport,
)
