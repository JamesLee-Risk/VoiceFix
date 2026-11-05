"""A0 import check and A1 editing (delete / mute / mark NG)."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from voicefix.core import analysis as an
from voicefix.core.editmap import EditMap
from voicefix.core.regions import Region, region_mask
from voicefix.core.timefmt import format_time as _fmt_time
from voicefix.steps.base import Analysis, Params, PassthroughStep, Step, StepContext

NOISE_SAMPLE_SECONDS = 3.0
MUTE_FADE_SECONDS = 0.005


class ImportCheck(PassthroughStep):
    id = "A0_import"
    code = "A0"
    title = "先看看你的錄音"
    description = "檢查錄音的基本狀況，並確認一段「只有背景噪音」的樣本給降噪使用。"
    term = "Import / Inspect"
    panel = "import"

    def analyze(self, audio: np.ndarray, sr: int, ctx: StepContext) -> Analysis:
        frame_db, frame = an.frame_power_db(audio, sr, 0.05)
        floor = an.noise_floor_db(frame_db)
        clips = an.clipped_runs(audio)
        candidate = an.quietest_window(frame_db, frame, int(NOISE_SAMPLE_SECONDS / 0.05))
        lines = [
            f"長度 {_fmt_time(len(audio), sr)}，取樣率 {sr} Hz，{'立體聲' if audio.shape[1] == 2 else '單聲道'}。",
            f"底噪約 {floor:.1f} dBFS" + ("（偏高，建議執行降噪）。" if floor > -60 else "。"),
        ]
        if clips:
            first = "、".join(_fmt_time(c.start, sr) for c in clips[:5])
            lines.append(f"⚠ 發現 {len(clips)} 處削波（破音），例如 {first}。削波無法完全修復，可考慮重錄。")
        else:
            lines.append("沒有發現削波。")
        if candidate:
            lines.append(f"建議的底噪樣本：{_fmt_time(candidate.start, sr)} 起 3 秒（最安靜的片段）。")
        else:
            lines.append("錄音太短，找不到 3 秒的底噪樣本；請手動選一段只有背景聲的區段。")
        return Analysis(
            summary=tuple(lines),
            recommend=True,
            marks=tuple(clips),
            data={"noise_sample": candidate, "noise_floor_db": floor, "clip_count": len(clips)},
        )


class EditStep(Step):
    """Settings: {"deletions": [[s, e]], "mutes": [[s, e]], "ng": [[s, e]]} (source timeline)."""

    id = "A1_edit"
    code = "A1"
    title = "剪掉不要的部分"
    description = "刪除或靜音不要的片段；可以先把 NG 片段標記起來。"
    term = "Edit"
    panel = "edit"
    has_strength = False
    scopes = ()
    respects_protection = False
    shared_keys = ()
    changes_length = True
    pad_seconds = 0.0

    def analyze(self, audio: np.ndarray, sr: int, ctx: StepContext) -> Analysis:
        segments, _, _ = an.speech_segments(audio, sr)
        return Analysis(
            summary=(
                f"依靜音切出 {len(segments)} 個台詞段落（波形上的淡色區塊）。",
                "在波形上拖曳選取後，按「刪除」「靜音」或「標記 NG」。沒有要剪的地方可以直接略過。",
            ),
            recommend=False,
            marks=tuple(segments),
            data={"segments": segments},
        )

    @staticmethod
    def regions(settings: Mapping[str, Any], name: str) -> list[Region]:
        return [Region.from_json(r) for r in settings.get(name, [])]

    def edit_map(self, settings: Mapping[str, Any]) -> EditMap:
        return EditMap.from_deletions(self.regions(settings, "deletions"))

    def process(self, chunk: np.ndarray, sr: int, params: Params, ctx: StepContext, plan: Any, offset: int) -> np.ndarray:
        # Deleted spans are silenced here so a preview stays time-aligned with the
        # input; the chain removes them for real after this pass.
        silenced = self.regions(ctx.settings, "mutes") + self.regions(ctx.settings, "deletions")
        if not silenced:
            return chunk
        fade = max(1, int(sr * MUTE_FADE_SECONDS))
        mask = region_mask(len(chunk), silenced, fade, fades_inside=True, offset=offset)
        return (chunk * (1.0 - mask)[:, None]).astype(np.float32)

