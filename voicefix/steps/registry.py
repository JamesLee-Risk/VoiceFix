"""Which steps make up each flow, in order."""

from __future__ import annotations

from voicefix.core.project import Flow
from voicefix.steps.base import Step
from voicefix.steps.flow_a.cleanup import DeHum, Denoise, LowCut, MouthClicks, Plosives
from voicefix.steps.flow_a.inspect_edit import EditStep, ImportCheck
from voicefix.steps.flow_a.level_export import Export, Leveler, LoudnessPeak
from voicefix.steps.flow_b.cleanup import Breaths, ClicksAndPlosives, SingingDenoise
from voicefix.steps.flow_b.song import GapCleanup, SingingLoudnessExport, SongImport
from voicefix.steps.flow_b.tone import Compressor, DeEsser, ToneEQ

# A7 (breaths) and A9 (room-tone fill) are scheduled for v1.
FLOW_A_STEPS: tuple[type[Step], ...] = (
    ImportCheck,
    EditStep,
    LowCut,
    DeHum,
    Denoise,
    MouthClicks,
    Plosives,
    Leveler,
    LoudnessPeak,
    Export,
)

# B1 (take comping) and B7 (timing alignment) are v2; B6 (pitch) and B11 (space) are v1.
FLOW_B_STEPS: tuple[type[Step], ...] = (
    SongImport,
    GapCleanup,
    SingingDenoise,
    ClicksAndPlosives,
    Breaths,
    ToneEQ,
    Compressor,
    DeEsser,
    SingingLoudnessExport,
)

FLOWS = {Flow.VOICE_WORK: FLOW_A_STEPS, Flow.SINGING: FLOW_B_STEPS}
IMPLEMENTED_FLOWS = tuple(FLOWS)


def build_steps(flow: Flow) -> list[Step]:
    return [cls() for cls in FLOWS[flow]]


def first_affected(steps: list[Step], key: str) -> int | None:
    """Index of the first step whose output depends on a shared project setting."""
    for i, step in enumerate(steps):
        if key in step.shared_keys:
            return i
    return None
