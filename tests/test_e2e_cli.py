from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from typer.testing import CliRunner

from scripts import mdwb_cli
from scripts.agents import generate_todos, summarize_article
from scripts.agents import shared as agent_shared

runner = CliRunner()


def _setup_console(monkeypatch, targets: Iterable[object]) -> Console:
    console = Console(record=True)
    for target in targets:
        monkeypatch.setattr(target, "console", console)
    return console


def _resume_hash(identifier: str) -> str:
    return mdwb_cli._resume_hash(identifier)  # type: ignore[attr-defined]


def _capture_result(markdown: str) -> agent_shared.CaptureResult:
    return agent_shared.CaptureResult(job_id="demo-job", snapshot={"state": "DONE"}, markdown=markdown)


class FlowLogger:
    def __init__(self, console: Console, flow_name: str) -> None:
        self.console = console
        self.flow_name = flow_name

    def banner(self, text: str) -> None:
        self.console.rule(f"{self.flow_name}: {text}")

    def step(
        self,
        title: str,
        *,
        description: str,
        inputs: Mapping[str, Any] | None = None,
        functions: Sequence[str] | None = None,
        outputs: Mapping[str, Any] | None = None,
        command: str | None = None,
        syntax_blocks: Sequence[tuple[str, str]] | None = None,
    ) -> None:
        sections: list[Any] = [Text(description, style="bold cyan")]
        if inputs:
            sections.append(_mapping_table("Inputs", inputs))
        if functions:
            sections.append(_functions_table(functions))
        if outputs:
            sections.append(_mapping_table("Outputs", outputs))
        self.console.print(Panel(Group(*sections), title=f"Step ▸ {title}", border_style="blue"))
        if command:
            self.console.print(Panel(Syntax(command, "bash"), title=f"Command ▸ {title}", border_style="magenta"))
        if syntax_blocks:
            for language, snippet in syntax_blocks:
                self.console.print(
                    Panel(Syntax(snippet, language), title=f"Context ▸ {title}", border_style="cyan")
                )

    def finish(self, summary: Mapping[str, Any]) -> None:
        self.console.print(
            Panel(_mapping_table("Summary", summary), title=f"Summary ▸ {self.flow_name}", border_style="green")
        )


def _mapping_table(title: str, mapping: Mapping[str, Any]) -> Table:
    table = Table("Key", "Value", title=title, box=None, expand=True)
    for key, value in mapping.items():
        table.add_row(str(key), str(value))
    return table


def _functions_table(functions: Sequence[str]) -> Table:
    table = Table("Order", "Function", title="Functions", box=None)
    for idx, name in enumerate(functions, start=1):
        table.add_row(str(idx), name)
    return table


def test_e2e_fetch_resume_logs_with_rich(monkeypatch, tmp_path: Path) -> None:
    url = "https://example.com/article"
    resume_root = tmp_path / "resume"
    done_dir = resume_root / "done_flags"
    done_dir.mkdir(parents=True)
    done_flag = done_dir / f"done_{_resume_hash(url)}.flag"
    done_flag.write_text("done", encoding="utf-8")

    console = _setup_console(monkeypatch, [mdwb_cli])
    logger = FlowLogger(console, "mdwb fetch --resume")
    logger.banner("Resume guard")
    logger.step(
        "Workspace",
        description="Seeded resume workspace so the CLI must explain why the URL is skipped.",
        inputs={"url": url, "resume_root": resume_root},
        outputs={"done_flag": done_flag},
        command=f"touch {done_flag}",
    )

    function_calls: list[str] = []
    original_status = mdwb_cli.ResumeManager.status
    original_is_complete = mdwb_cli.ResumeManager.is_complete

    def tracking_status(self: mdwb_cli.ResumeManager) -> tuple[int, int | None]:
        function_calls.append("ResumeManager.status")
        return original_status(self)

    def tracking_is_complete(self: mdwb_cli.ResumeManager, entry: str) -> bool:
        function_calls.append("ResumeManager.is_complete")
        return original_is_complete(self, entry)

    monkeypatch.setattr(mdwb_cli.ResumeManager, "status", tracking_status)
    monkeypatch.setattr(mdwb_cli.ResumeManager, "is_complete", tracking_is_complete)
    monkeypatch.setattr(mdwb_cli, "_watch_events_with_fallback", lambda *args, **kwargs: function_calls.append("_watch_events_with_fallback"))

    command = f"mdwb fetch {url} --resume --resume-root {resume_root}"
    logger.step(
        "Invoke CLI",
        description="Trigger Typer command so ResumeManager short-circuits completed entries.",
        inputs={"resume": True, "watch": True},
        command=command,
    )

    result = runner.invoke(
        mdwb_cli.cli,
        [
            "fetch",
            url,
            "--resume",
            "--resume-root",
            str(resume_root),
        ],
    )

    assert result.exit_code == 0
    assert "skipping submission" in result.output
    logger.step(
        "Results",
        description="Summarize the skip path and the functions that executed.",
        functions=function_calls,
        outputs={"exit_code": result.exit_code, "message": result.output.strip()},
        syntax_blocks=[("text", result.output)],
    )
    logger.finish({"done_flags": done_dir, "functions_recorded": len(function_calls)})

    log_text = console.export_text()
    assert "ResumeManager.status" in log_text
    assert "ResumeManager.is_complete" in log_text
    assert "skip" in log_text.lower()


def test_e2e_agents_summarize_with_rich_logging(monkeypatch) -> None:
    console = _setup_console(monkeypatch, [agent_shared, summarize_article])
    logger = FlowLogger(console, "agents summarize")
    logger.banner("Markdown summarizer")

    called: dict[str, Any] = {}

    def fake_capture_markdown(**kwargs):  # noqa: ANN001
        called["capture_markdown"] = kwargs
        return _capture_result("# Title\n\n- [ ] Draft plan\n- [ ] Ship tests")

    def fake_summary(markdown: str, sentences: int) -> str:
        called["summarize_markdown"] = {"sentences": sentences, "markdown": markdown}
        return "Summary: Draft plan + Ship tests"

    monkeypatch.setattr(agent_shared, "capture_markdown", fake_capture_markdown)
    monkeypatch.setattr(agent_shared, "summarize_markdown", fake_summary)
    monkeypatch.setattr(
        agent_shared,
        "resolve_settings",
        lambda api_base: mdwb_cli.APISettings("http://api", None, Path("ops/warnings.jsonl")),
    )

    command = "agents summarize --job-id demo-job --sentences 3"
    logger.step(
        "Invoke CLI",
        description="Call summarize_article CLI to reuse a job and emit Rich summaries.",
        inputs={"job_id": "demo-job", "sentences": 3},
        command=command,
    )

    result = runner.invoke(summarize_article.cli, ["--job-id", "demo-job", "--sentences", "3"])
    assert result.exit_code == 0
    assert called["capture_markdown"]["job_id"] == "demo-job"
    assert called["summarize_markdown"]["sentences"] == 3

    logger.step(
        "Outputs",
        description="Captured Markdown + generated summary for downstream agents.",
        functions=["shared.capture_markdown", "shared.summarize_markdown"],
        outputs={"sentences": called["summarize_markdown"]["sentences"]},
        syntax_blocks=[
            ("markdown", called["summarize_markdown"]["markdown"]),
            ("text", result.output),
        ],
    )
    logger.finish({"job_id": "demo-job", "summary_len": len(result.output.splitlines())})

    log_text = console.export_text()
    assert "shared.capture_markdown" in log_text
    assert "Summary for job" in log_text


def test_e2e_agents_generate_todos_with_rich_logging(monkeypatch) -> None:
    console = _setup_console(monkeypatch, [agent_shared, generate_todos])
    logger = FlowLogger(console, "agents generate_todos")
    logger.banner("TODO extraction")

    called: dict[str, Any] = {}
    todos_payload = ["Fix warnings", "Ship report"]

    def fake_capture_markdown(**kwargs):  # noqa: ANN001
        called["capture_markdown"] = kwargs
        return _capture_result("# Tasks\n\n- [ ] Fix warnings\n- [ ] Ship report")

    def fake_extract_todos(markdown: str, max_tasks: int) -> list[str]:
        called["extract_todos"] = {"markdown": markdown, "max_tasks": max_tasks}
        return todos_payload

    monkeypatch.setattr(agent_shared, "capture_markdown", fake_capture_markdown)
    monkeypatch.setattr(agent_shared, "extract_todos", fake_extract_todos)
    monkeypatch.setattr(
        agent_shared,
        "resolve_settings",
        lambda api_base: mdwb_cli.APISettings("http://api", None, Path("ops/warnings.jsonl")),
    )

    command = "agents generate-todos --job-id demo-job --limit 2"
    logger.step(
        "Invoke CLI",
        description="Run the TODO generator so the CLI enumerates actionable tasks.",
        inputs={"job_id": "demo-job", "limit": 2},
        command=command,
    )

    result = runner.invoke(
        generate_todos.cli,
        ["--job-id", "demo-job", "--limit", "2"],
    )

    assert result.exit_code == 0
    assert called["capture_markdown"]["job_id"] == "demo-job"
    assert called["extract_todos"]["max_tasks"] == 2

    logger.step(
        "Outputs",
        description="Show the derived TODO items with numbered formatting.",
        functions=["shared.capture_markdown", "shared.extract_todos"],
        outputs={"todos": ", ".join(todos_payload)},
        syntax_blocks=[("text", result.output)],
    )
    logger.finish({"job_id": "demo-job", "todo_count": 2})

    log_text = console.export_text()
    assert "TODOs for job" in log_text
    assert "shared.extract_todos" in log_text


def test_e2e_warning_tail_rich_output(monkeypatch, tmp_path: Path) -> None:
    console = _setup_console(monkeypatch, [mdwb_cli])
    logger = FlowLogger(console, "mdwb warnings tail")
    logger.banner("Warning log triage")

    log_path = tmp_path / "warnings.jsonl"
    record = {
        "timestamp": "2025-11-08T08:00:00Z",
        "job_id": "run-1",
        "warnings": [
            {"code": "canvas-heavy", "message": "Many canvas elements", "count": 4, "threshold": 3},
        ],
        "sweep_stats": {"shrink_events": 1, "retry_attempts": 0, "overlap_pairs": 3, "overlap_match_ratio": 0.82},
        "validation_failures": ["Tile checksum mismatch"],
    }
    log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    logger.step(
        "Seed log",
        description="Write a structured warning entry to exercise the Rich tail output.",
        outputs={"log_path": log_path},
        syntax_blocks=[("json", json.dumps(record, indent=2))],
    )

    command = f"mdwb warnings tail --count 1 --json --log-path {log_path}"
    logger.step(
        "Invoke CLI",
        description="Call warnings tail in JSON mode for deterministic parsing.",
        inputs={"count": 1, "json": True},
        command=command,
    )

    result = runner.invoke(
        mdwb_cli.cli,
        [
            "warnings",
            "tail",
            "--count",
            "1",
            "--json",
            "--log-path",
            str(log_path),
        ],
    )

    assert result.exit_code == 0
    logger.step(
        "Outputs",
        description="Verify that canvas-heavy warnings and validation details are surfaced.",
        outputs={"exit_code": result.exit_code, "bytes": len(result.output)},
        syntax_blocks=[("json", result.output)],
    )
    logger.finish({"records": 1, "path": log_path})

    log_text = console.export_text()
    assert "canvas-heavy" in log_text
    assert "validation" in log_text.lower()
