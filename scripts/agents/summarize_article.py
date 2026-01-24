from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from scripts.agents import shared

cli = typer.Typer(help="Capture a URL (or reuse a job) and output a quick summary.")
console = Console()


@cli.command()
def summarize(
    url: str = typer.Option(
        "",
        "--url",
        help="URL to capture. Optional when --job-id is provided.",
    ),
    job_id: str = typer.Option(
        "",
        "--job-id",
        help="Reuse an existing job instead of starting a new capture.",
    ),
    api_base: str | None = typer.Option(None, help="Override API base URL."),
    profile: str | None = typer.Option(None, help="Browser profile id."),
    ocr_policy: str | None = typer.Option(None, help="OCR policy id."),
    sentences: int = typer.Option(
        5, min=1, max=12, help="Number of sentences to include in the summary."
    ),
    http2: bool = typer.Option(True, "--http2/--no-http2"),
    poll_interval: float = typer.Option(2.0, help="Seconds between polling /jobs/{id}."),
    timeout: float = typer.Option(300.0, help="Maximum seconds to wait for completion."),
    reuse_session: bool = typer.Option(
        True,
        "--reuse-session/--no-reuse-session",
        help="Reuse the same HTTP client across submit/poll/fetch.",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Write the summary (or raw Markdown when no summary is available) to this path.",
    ),
) -> None:
    """Capture a URL (if needed) and print a short summary."""

    settings = shared.resolve_settings(api_base)
    capture = shared.capture_markdown(
        url=url or None,
        job_id=job_id or None,
        settings=settings,
        http2=http2,
        profile=profile,
        ocr_policy=ocr_policy,
        poll_interval=poll_interval,
        timeout=timeout,
        reuse_session=reuse_session,
    )
    summary = shared.summarize_markdown(capture.markdown, sentences=sentences)
    if not summary:
        console.print("[yellow]No text content found; raw Markdown follows.[/]")
        console.print(capture.markdown)
        if out:
            shared.save_text(out, capture.markdown)
            console.print(f"[dim]Saved raw Markdown to {out}[/]")
        return
    console.rule(f"Summary for job {capture.job_id}")
    console.print(summary)
    if out:
        shared.save_text(out, summary)
        console.print(f"[dim]Saved summary to {out}[/]")


def main() -> None:  # pragma: no cover - Typer entry point
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
