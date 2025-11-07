"""Entry point for the FastAPI application."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Markdown Web Browser")
app.mount("/static", StaticFiles(directory="web"), name="static")

WEB_ROOT = Path("web")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Serve the current web UI shell."""

    return (WEB_ROOT / "index.html").read_text(encoding="utf-8")


@app.get("/health", tags=["health"])
async def healthcheck() -> dict[str, str]:
    """Return a simple status useful for smoke tests."""

    return {"status": "ok"}


@app.get("/jobs/demo/stream")
async def demo_job_stream(request: Request) -> StreamingResponse:
    """Emit a deterministic SSE stream so the frontend can exercise UI wiring."""

    async def event_generator() -> AsyncGenerator[str, None]:
        updates = [
            ("state", "<span class=\"badge badge--info\">CAPTURING</span>"),
            ("progress", "4 / 12 tiles"),
            ("runtime", "CfT Stable-1 · Playwright 1.55.0"),
            (
                "log",
                "<li>00:00:01 — Started viewport sweep (1280×2000, overlap 120px)</li>",
            ),
            (
                "log",
                "<li>00:00:02 — Tile t0 sha256 a1b2 captured · SSIM warm-up</li>",
            ),
            (
                "log",
                "<li>00:00:03 — OCR queued 6 tiles · remote policy olmocr-2-7b-1025-fp8</li>",
            ),
            ("state", "<span class=\"badge badge--warn\">OCR_WAITING</span>"),
            ("progress", "9 / 12 tiles"),
        ]

        for event_name, payload in updates:
            yield f"event: {event_name}\ndata: {payload}\n\n"
            await asyncio.sleep(0.75)
            if await request.is_disconnected():
                return

        heartbeat = 0
        while True:
            heartbeat += 1
            yield "event: log\n"
            yield f"data: <li>Heartbeat {heartbeat}: awaiting final stitch…</li>\n\n"
            await asyncio.sleep(4)
            if await request.is_disconnected():
                return

    return StreamingResponse(event_generator(), media_type="text/event-stream")
