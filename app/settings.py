"""Typed configuration objects backed by python-decouple settings."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

from decouple import Config as DecoupleConfig, RepositoryEnv

__all__ = [
    "BrowserSettings",
    "OCRSettings",
    "TelemetrySettings",
    "StorageSettings",
    "Settings",
    "load_config",
    "get_settings",
]


@dataclass(frozen=True, slots=True)
class BrowserSettings:
    """Capture/browser-related knobs that must be echoed into manifests."""

    cft_version: str
    playwright_channel: str
    browser_transport: str
    screenshot_style_hash: str


@dataclass(frozen=True, slots=True)
class OCRSettings:
    """Parameters for remote/local olmOCR orchestration."""

    server_url: str
    api_key: str | None
    model: str
    local_url: str | None
    use_fp8: bool
    min_concurrency: int
    max_concurrency: int


@dataclass(frozen=True, slots=True)
class TelemetrySettings:
    """Ports/intervals for Prometheus + HTMX SSE plumbing."""

    prometheus_port: int
    htmx_sse_heartbeat_ms: int


@dataclass(frozen=True, slots=True)
class StorageSettings:
    """Filesystem layout for cache + artifact persistence."""

    cache_root: Path


@dataclass(frozen=True, slots=True)
class Settings:
    """Top-level immutable configuration container."""

    env_path: str
    browser: BrowserSettings
    ocr: OCRSettings
    telemetry: TelemetrySettings
    storage: StorageSettings

    def manifest_environment(self, *, playwright_version: str | None = None) -> dict[str, object]:
        """Return a dict used by manifest builders to echo env metadata."""

        return {
            "cft_version": self.browser.cft_version,
            "playwright_channel": self.browser.playwright_channel,
            "playwright_version": playwright_version,
            "browser_transport": self.browser.browser_transport,
            "screenshot_style_hash": self.browser.screenshot_style_hash,
            "ocr_model": self.ocr.model,
            "ocr_use_fp8": self.ocr.use_fp8,
            "ocr_concurrency": {
                "min": self.ocr.min_concurrency,
                "max": self.ocr.max_concurrency,
            },
        }


def load_config(env_path: str = ".env") -> DecoupleConfig:
    """Return a python-decouple config anchored to the repository .env file."""

    return DecoupleConfig(RepositoryEnv(env_path))


def _int(cfg: DecoupleConfig, key: str, *, default: int) -> int:
    return cfg(key, cast=int, default=default)


def _bool(cfg: DecoupleConfig, key: str, *, default: bool) -> bool:
    return cfg(key, cast=bool, default=default)


@lru_cache(maxsize=1)
def get_settings(env_path: str = ".env") -> Settings:
    """Load and memoize structured settings for the current process."""

    cfg = load_config(env_path)
    browser = BrowserSettings(
        cft_version=cfg("CFT_VERSION", default="chrome-130.0.6723.69"),
        playwright_channel=cfg("PLAYWRIGHT_CHANNEL", default="cft"),
        browser_transport=cfg("PLAYWRIGHT_TRANSPORT", default="cdp"),
        screenshot_style_hash=cfg("SCREENSHOT_STYLE_HASH", default="dev-sweeps-v1"),
    )
    ocr = OCRSettings(
        server_url=cfg("OLMOCR_SERVER", default="https://ai2endpoints.cirrascale.ai/api"),
        api_key=cfg("OLMOCR_API_KEY", default=None),
        model=cfg("OLMOCR_MODEL", default="olmOCR-2-7B-1025-FP8"),
        local_url=cfg("OCR_LOCAL_URL", default=None),
        use_fp8=_bool(cfg, "OCR_USE_FP8", default=True),
        min_concurrency=_int(cfg, "OCR_MIN_CONCURRENCY", default=2),
        max_concurrency=_int(cfg, "OCR_MAX_CONCURRENCY", default=8),
    )
    if ocr.max_concurrency < ocr.min_concurrency:
        msg = "OCR_MAX_CONCURRENCY must be >= OCR_MIN_CONCURRENCY"
        raise ValueError(msg)
    telemetry = TelemetrySettings(
        prometheus_port=_int(cfg, "PROMETHEUS_PORT", default=9000),
        htmx_sse_heartbeat_ms=_int(cfg, "HTMX_SSE_HEARTBEAT_MS", default=4000),
    )
    storage = StorageSettings(cache_root=Path(cfg("CACHE_ROOT", default=".cache")))

    return Settings(
        env_path=env_path,
        browser=browser,
        ocr=ocr,
        telemetry=telemetry,
        storage=storage,
    )


# Statically importable settings singleton for convenience in modules that
# value module-level constants over injecting the loader everywhere.
settings: Final[Settings] = get_settings()
