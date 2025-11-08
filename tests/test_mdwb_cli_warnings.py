from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from scripts import mdwb_cli

runner = CliRunner()


def test_warnings_tail_json_includes_enriched_fields(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "warnings.jsonl"
    record = {
        "timestamp": "2025-11-08T08:00:00Z",
        "job_id": "run-1",
        "warnings": [],
        "blocklist_hits": {},
        "sweep_stats": {
            "shrink_events": 1,
            "retry_attempts": 1,
            "overlap_pairs": 4,
            "overlap_match_ratio": 0.9,
        },
        "validation_failures": ["tile checksum mismatch", "tile decode failed"],
    }
    log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

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
    assert "\"validation_failure_count\": 2" in result.output
    assert "sweep_summary" in result.output
    assert "overlap_match_ratio" in result.output
