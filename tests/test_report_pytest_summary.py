from __future__ import annotations

import json
from pathlib import Path

from scripts import report_pytest_summary


def _write_report(tmp_path: Path, content: str) -> Path:
    report = tmp_path / "report.xml"
    report.write_text(content, encoding="utf-8")
    return report


def test_summarize_junit_collects_failures(tmp_path: Path) -> None:
    report = _write_report(
        tmp_path,
        """<testsuite tests="2" failures="1" errors="0" skipped="0" time="0.2">
        <testcase classname="cli.test" name="test_ok" time="0.05"/>
        <testcase classname="cli.test" name="test_fail" time="0.15">
            <failure message="boom">Traceback</failure>
        </testcase>
        </testsuite>""",
    )

    summary = report_pytest_summary.summarize_junit(report, exit_code=1)

    assert summary["tests"] == 2
    assert summary["failures"] == 1
    assert summary["status"] == "failed"
    assert summary["failed_tests"] == [
        {"name": "test_fail", "classname": "cli.test", "type": "failure", "message": "boom"}
    ]


def test_summarize_handles_missing_report(tmp_path: Path) -> None:
    missing_report = tmp_path / "missing.xml"

    summary = report_pytest_summary.summarize_junit(missing_report, exit_code=2)

    assert summary["warning"].startswith("JUnit report not found")
    assert summary["status"] == "failed"
    assert summary["tests"] == 0


def test_cli_writes_summary_file(tmp_path: Path) -> None:
    report = _write_report(
        tmp_path,
        """<testsuites>
            <testsuite tests="1" failures="0" errors="0" skipped="0" time="0.01">
                <testcase classname="cli" name="test_ok" time="0.01"/>
            </testsuite>
        </testsuites>""",
    )
    summary_path = tmp_path / "summary.json"

    exit_code = report_pytest_summary.main(
        [
            "--junit",
            str(report),
            "--summary",
            str(summary_path),
            "--exit-code",
            "0",
        ]
    )

    assert exit_code == 0
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert data["status"] == "passed"
    assert data["tests"] == 1
