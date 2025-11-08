from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer
from scripts import mdwb_cli
from scripts.agents import shared
from scripts.agents import summarize_article as summarize_article_module
from scripts.agents import generate_todos as generate_todos_module


SAMPLE_MD = """
# Example Article

Welcome to the **Markdown Web Browser** demo. We focus on deterministic captures.
Second sentence explains why reproducibility matters. Third sentence adds more context!

## Next Steps

- [ ] Wire nightly smoke results into dashboards.
- [x] Add SKIP_LIBVIPS_CHECK flag to run_checks.
- Prioritize TODO: add agent starter scripts.
- Note: Validate CfT label/build in manifests.

### Action Items

1. Update docs/ops.
2. Share sample scripts with other agents.
"""


def test_summarize_markdown_truncates_to_sentences():
    summary = shared.summarize_markdown(SAMPLE_MD, sentences=2)
    assert "Markdown Web Browser demo." in summary
    assert "We focus on deterministic captures." in summary
    assert "Second sentence explains why reproducibility matters." not in summary
    assert "Third sentence adds more context" not in summary


def test_extract_todos_prefers_checkboxes_and_heading_context():
    todos = shared.extract_todos(SAMPLE_MD, max_tasks=8)
    assert todos[0].startswith("Wire nightly smoke")
    assert any(task.startswith("Prioritize TODO") for task in todos)
    assert any("Validate CfT" in task for task in todos)
    assert any("Update docs/ops" in task for task in todos)


def _mock_capture(monkeypatch, markdown=SAMPLE_MD):
    capture = shared.CaptureResult(job_id="job-xyz", snapshot={}, markdown=markdown)

    def fake_capture(**kwargs):  # noqa: ANN001
        return capture

    for module in (shared, summarize_article_module.shared, generate_todos_module.shared):
        monkeypatch.setattr(module, "capture_markdown", fake_capture)
        monkeypatch.setattr(module, "resolve_settings", lambda api_base: object())


def test_summarize_article_cli_writes_out(monkeypatch, tmp_path: Path):
    _mock_capture(monkeypatch)
    out_path = tmp_path / "summary.txt"

    summarize_article_module.summarize(
        url="https://example.com",
        job_id="",
        api_base=None,
        profile=None,
        ocr_policy=None,
        sentences=2,
        http2=True,
        poll_interval=2.0,
        timeout=300.0,
        out=out_path,
    )
    expected = shared.summarize_markdown(SAMPLE_MD, sentences=2)
    assert out_path.read_text(encoding="utf-8") == expected


def test_generate_todos_cli_writes_text(monkeypatch, tmp_path: Path):
    _mock_capture(monkeypatch)
    out_path = tmp_path / "todos.txt"

    generate_todos_module.todos(
        url="https://example.com",
        job_id="",
        api_base=None,
        profile=None,
        ocr_policy=None,
        limit=5,
        json_output=False,
        http2=True,
        poll_interval=2.0,
        timeout=300.0,
        out=out_path,
    )
    expected = "\n".join(shared.extract_todos(SAMPLE_MD, max_tasks=5))
    assert out_path.read_text(encoding="utf-8") == expected


def test_generate_todos_cli_writes_json(monkeypatch, tmp_path: Path):
    _mock_capture(monkeypatch)
    out_path = tmp_path / "todos.json"

    generate_todos_module.todos(
        url="https://example.com",
        job_id="",
        api_base=None,
        profile=None,
        ocr_policy=None,
        limit=8,
        json_output=True,
        http2=True,
        poll_interval=2.0,
        timeout=300.0,
        out=out_path,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["job_id"] == "job-xyz"
    assert payload["todos"] == shared.extract_todos(SAMPLE_MD)


def test_capture_markdown_validates_missing_job_id(monkeypatch):
    settings = mdwb_cli.APISettings(base_url="http://localhost", api_key=None, warning_log_path=Path("ops/warnings.jsonl"))

    def fake_submit_job(**kwargs):  # noqa: ANN001
        return {"state": "BROWSER_STARTING"}

    monkeypatch.setattr(shared, "submit_job", fake_submit_job)
    with pytest.raises(RuntimeError, match="job id"):
        shared.capture_markdown(
            url="https://example.com",
            job_id=None,
            settings=settings,
            http2=True,
            profile=None,
            ocr_policy=None,
        )


def test_capture_markdown_requires_url_or_job_id():
    settings = mdwb_cli.APISettings(base_url="http://localhost", api_key=None, warning_log_path=Path("ops/warnings.jsonl"))
    with pytest.raises(typer.BadParameter):
        shared.capture_markdown(
            url=None,
            job_id=None,
            settings=settings,
            http2=True,
            profile=None,
            ocr_policy=None,
        )


def test_capture_markdown_raises_on_non_done_state(monkeypatch):
    settings = mdwb_cli.APISettings(base_url="http://localhost", api_key=None, warning_log_path=Path("ops/warnings.jsonl"))

    monkeypatch.setattr(shared, "submit_job", lambda **kwargs: {"id": "job-1"})
    monkeypatch.setattr(shared, "wait_for_completion", lambda *args, **kwargs: {"id": "job-1", "state": "FAILED", "manifest": {"error": "boom"}})
    monkeypatch.setattr(shared, "fetch_markdown", lambda *args, **kwargs: "")

    with pytest.raises(RuntimeError, match="FAILED"):
        shared.capture_markdown(
            url="https://example.com",
            job_id=None,
            settings=settings,
            http2=True,
            profile=None,
            ocr_policy=None,
        )


def test_generate_todos_cli_handles_empty_output(monkeypatch, tmp_path: Path):
    _mock_capture(monkeypatch, markdown="")
    out_path = tmp_path / "todos.txt"

    generate_todos_module.todos(
        url="https://example.com",
        job_id="",
        api_base=None,
        profile=None,
        ocr_policy=None,
        limit=3,
        json_output=False,
        http2=True,
        poll_interval=2.0,
        timeout=300.0,
        out=out_path,
    )

    assert out_path.read_text(encoding="utf-8") == ""
