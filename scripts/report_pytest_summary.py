from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET


def _iter_testsuites(root: ET.Element) -> list[ET.Element]:
    if root.tag == "testsuite":
        return [root]
    if root.tag == "testsuites":
        suites = list(root.findall("testsuite"))
        if suites:
            return suites
    suites = list(root.findall(".//testsuite"))
    return suites or [root]


def summarize_junit(report_path: Path, *, exit_code: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "report_path": str(report_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "exit_code": exit_code,
        "status": "passed" if exit_code == 0 else "failed",
        "tests": 0,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "time": 0.0,
        "failed_tests": [],
    }
    if not report_path.exists():
        summary["warning"] = "JUnit report not found"
        return summary

    try:
        root = ET.parse(report_path).getroot()
    except ET.ParseError as exc:
        summary["warning"] = f"Failed to parse JUnit XML: {exc}"
        return summary

    failed_tests: list[dict[str, Any]] = []
    suites = _iter_testsuites(root)
    for suite in suites:
        summary["tests"] += int(suite.attrib.get("tests", 0) or 0)
        summary["failures"] += int(suite.attrib.get("failures", 0) or 0)
        summary["errors"] += int(suite.attrib.get("errors", 0) or 0)
        summary["skipped"] += int(suite.attrib.get("skipped", 0) or 0)
        try:
            summary["time"] += float(suite.attrib.get("time", 0) or 0)
        except (TypeError, ValueError):
            pass
        for case in suite.findall("testcase"):
            failure_node = case.find("failure")
            error_node = case.find("error")
            problem_node = failure_node if failure_node is not None else error_node
            if problem_node is None:
                continue
            message = problem_node.attrib.get("message") or (problem_node.text or "")
            failed_tests.append(
                {
                    "name": case.attrib.get("name"),
                    "classname": case.attrib.get("classname"),
                    "type": problem_node.tag,
                    "message": message.strip(),
                }
            )
    summary["failed_tests"] = failed_tests
    return summary


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit a JSON summary from a pytest JUnit XML report.")
    parser.add_argument("--junit", required=True, help="Path to the JUnit XML file produced by pytest.")
    parser.add_argument("--summary", required=True, help="Destination path for the JSON summary.")
    parser.add_argument("--exit-code", required=True, type=int, help="Exit code returned by pytest.")
    args = parser.parse_args(argv)

    junit_path = Path(args.junit)
    summary_path = Path(args.summary)
    summary = summarize_junit(junit_path, exit_code=args.exit_code)
    _write_summary(summary_path, summary)
    status = summary["status"]
    total = summary["tests"]
    failures = summary["failures"] + summary["errors"]
    print(f"Pytest summary ({status}): {failures} failures out of {total} tests — {summary_path}")
    if summary.get("failed_tests"):
        preview = ", ".join(filter(None, (case.get("name") for case in summary["failed_tests"][:5])))
        print(f"Failing tests: {preview}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
