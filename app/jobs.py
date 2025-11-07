"""Job state machine and models for capture requests."""

from __future__ import annotations

from enum import Enum
from importlib import metadata
from typing import TypedDict

from app.schemas import ManifestMetadata
from app.settings import Settings, settings as global_settings

try:  # Playwright is an optional dependency in some CI environments
    PLAYWRIGHT_VERSION = metadata.version("playwright")
except metadata.PackageNotFoundError:  # pragma: no cover - development convenience
    PLAYWRIGHT_VERSION = None


class JobState(str, Enum):
    """Enumerated lifecycle states for a capture job."""

    BROWSER_STARTING = "BROWSER_STARTING"
    NAVIGATING = "NAVIGATING"
    SCROLLING = "SCROLLING"
    CAPTURING = "CAPTURING"
    TILING = "TILING"
    OCR_SUBMITTING = "OCR_SUBMITTING"
    OCR_WAITING = "OCR_WAITING"
    STITCHING = "STITCHING"
    DONE = "DONE"
    FAILED = "FAILED"


class JobSnapshot(TypedDict, total=False):
    """Serialized view of a job for API responses and SSE events."""

    id: str
    state: JobState
    url: str
    progress: dict[str, int]
    manifest_path: str
    manifest: dict[str, object]
    error: str | None


def build_initial_snapshot(
    url: str,
    *,
    job_id: str,
    settings: Settings | None = None,
) -> JobSnapshot:
    """Construct a basic snapshot stub used before persistence wiring exists."""

    manifest = None
    active_settings = settings or global_settings
    if active_settings:
        environment = active_settings.manifest_environment(playwright_version=PLAYWRIGHT_VERSION)
        manifest = ManifestMetadata(environment=environment)

    snapshot = JobSnapshot(
        id=job_id,
        url=url,
        state=JobState.BROWSER_STARTING,
        progress={"done": 0, "total": 0},
        manifest_path="",
        error=None,
    )
    if manifest:
        snapshot["manifest"] = manifest.model_dump()
    return snapshot
