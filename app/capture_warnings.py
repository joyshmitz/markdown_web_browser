"""Capture warning heuristics for canvas/video-heavy pages and overlays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from playwright.async_api import Page

from app.settings import WarningSettings


@dataclass(slots=True)
class WarningStats:
    """Lightweight DOM metrics used to drive capture warnings."""

    canvas_count: int
    video_count: int


@dataclass(slots=True)
class CaptureWarningEntry:
    """Structured warning surfaced in manifests + telemetry."""

    code: str
    message: str
    count: int
    threshold: int


async def collect_warning_stats(page: Page) -> WarningStats:
    """Inspect the DOM and return counts that drive warning decisions."""

    result = await page.evaluate(
        """() => ({
            canvasCount: document.querySelectorAll('canvas').length,
            videoCount: document.querySelectorAll('video').length
        })"""
    )
    return WarningStats(
        canvas_count=int(result.get("canvasCount", 0)),
        video_count=int(result.get("videoCount", 0)),
    )


def build_warnings(
    stats: WarningStats,
    *,
    canvas_threshold: int,
    video_threshold: int,
) -> List[CaptureWarningEntry]:
    """Convert DOM stats into structured warnings."""

    warnings: List[CaptureWarningEntry] = []

    if canvas_threshold > 0 and stats.canvas_count >= canvas_threshold:
        warnings.append(
            CaptureWarningEntry(
                code="canvas-heavy",
                message="High canvas count may hide chart labels or overlays.",
                count=stats.canvas_count,
                threshold=canvas_threshold,
            )
        )

    if video_threshold > 0 and stats.video_count >= video_threshold:
        warnings.append(
            CaptureWarningEntry(
                code="video-heavy",
                message="Multiple video elements detected; screenshots may blur frames.",
                count=stats.video_count,
                threshold=video_threshold,
            )
        )

    return warnings


async def collect_capture_warnings(page: Page, settings: WarningSettings) -> List[CaptureWarningEntry]:
    """Gather DOM stats and produce warning entries using configured thresholds."""

    stats = await collect_warning_stats(page)
    return build_warnings(
        stats,
        canvas_threshold=settings.canvas_warning_threshold,
        video_threshold=settings.video_warning_threshold,
    )
