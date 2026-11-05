"""Human-readable time stamps for summaries and lists."""

from __future__ import annotations

from voicefix.core.regions import Region


def format_time(sample: int, sr: int) -> str:
    seconds = sample / sr
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"


def describe_region(region: Region, sr: int) -> str:
    return f"{format_time(region.start, sr)} – {format_time(region.end, sr)}"
