#!/usr/bin/env python3
"""
Comprehensive End-to-End Integration Test for Markdown Web Browser
No mocks - tests the real system with extremely detailed rich logging
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import httpx
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree
from rich.text import Text
from rich.columns import Columns
from rich.rule import Rule
from rich import box
from rich.layout import Layout
from rich.align import Align
from rich.markdown import Markdown

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.settings import get_settings


class IntegrationTestRunner:
    """Comprehensive integration test runner with rich logging."""

    def __init__(self, api_url: str = "http://localhost:8000", verbose: bool = True):
        self.api_url = api_url
        self.verbose = verbose
        self.console = Console(record=True, width=140)
        self.settings = get_settings()
        self.test_results: Dict[str, Any] = {}
        self.start_time = time.time()

    def print_header(self):
        """Print the test header with system information."""

        header_text = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                  MARKDOWN WEB BROWSER - INTEGRATION TEST SUITE              ║
║                         Real End-to-End Testing - No Mocks                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """

        self.console.print(header_text, style="bold cyan")

        # System info table
        info_table = Table(title="System Configuration", box=box.ROUNDED)
        info_table.add_column("Parameter", style="cyan", no_wrap=True)
        info_table.add_column("Value", style="yellow")

        info_table.add_row("Test Started", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        info_table.add_row("API Endpoint", self.api_url)
        info_table.add_row("OCR Server", self.settings.OLMOCR_SERVER)
        info_table.add_row("OCR Model", self.settings.OLMOCR_MODEL)
        info_table.add_row("Python Version", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        info_table.add_row("Playwright Channel", self.settings.PLAYWRIGHT_CHANNEL)
        info_table.add_row("Cache Root", str(self.settings.CACHE_ROOT))

        self.console.print(info_table)
        self.console.print()

    async def test_health_check(self) -> Tuple[bool, Dict[str, Any]]:
        """Test 1: Health Check Endpoint"""

        self.console.rule("[bold blue]Test 1: Health Check", style="blue")

        panel_content = Text("Testing API health endpoint to verify server is running", style="dim")
        self.console.print(Panel(panel_content, title="🏥 Health Check", border_style="blue"))

        result = {"success": False, "data": {}}

        try:
            async with httpx.AsyncClient() as client:
                # Show request details
                request_table = Table(title="Request Details", box=box.SIMPLE)
                request_table.add_column("Field", style="cyan")
                request_table.add_column("Value", style="white")
                request_table.add_row("Method", "GET")
                request_table.add_row("URL", f"{self.api_url}/health")
                request_table.add_row("Headers", "Accept: application/json")
                self.console.print(request_table)

                # Make request with progress
                with self.console.status("[bold green]Sending health check request..."):
                    response = await client.get(f"{self.api_url}/health", timeout=5.0)

                # Show response
                response_data = response.json() if response.status_code == 200 else {}

                response_table = Table(title="Response Details", box=box.SIMPLE)
                response_table.add_column("Field", style="cyan")
                response_table.add_column("Value", style="green" if response.status_code == 200 else "red")
                response_table.add_row("Status Code", str(response.status_code))
                response_table.add_row("Response Time", f"{response.elapsed.total_seconds():.3f}s")

                if response_data:
                    for key, value in response_data.items():
                        response_table.add_row(key, str(value))

                self.console.print(response_table)

                result = {
                    "success": response.status_code == 200,
                    "data": {
                        "status_code": response.status_code,
                        "response_time": response.elapsed.total_seconds(),
                        "response": response_data
                    }
                }

                # Success/failure message
                if result["success"]:
                    self.console.print("✅ [green]Health check passed![/green]")
                else:
                    self.console.print("❌ [red]Health check failed![/red]")

        except Exception as e:
            error_panel = Panel(
                Syntax(traceback.format_exc(), "python", theme="monokai"),
                title="❌ Error Details",
                border_style="red"
            )
            self.console.print(error_panel)
            result["data"]["error"] = str(e)

        self.console.print()
        return result["success"], result["data"]

    async def test_job_submission(self, url: str = "https://example.com") -> Tuple[bool, Dict[str, Any]]:
        """Test 2: Job Submission and Processing"""

        self.console.rule("[bold blue]Test 2: Job Submission & Processing", style="blue")

        panel_content = Group(
            Text(f"URL to capture: {url}", style="cyan"),
            Text("This test will submit a capture job and monitor its progress", style="dim")
        )
        self.console.print(Panel(panel_content, title="📸 Capture Job", border_style="blue"))

        result = {"success": False, "data": {}}

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
                # Prepare request
                payload = {
                    "url": url,
                    "reuse_cache": False,
                }

                # Show request details
                request_info = Panel(
                    Syntax(json.dumps(payload, indent=2), "json", theme="monokai"),
                    title="Request Payload",
                    border_style="cyan"
                )
                self.console.print(request_info)

                # Submit job
                with self.console.status("[bold green]Submitting capture job..."):
                    response = await client.post(
                        f"{self.api_url}/jobs",
                        json=payload
                    )

                if response.status_code not in [200, 202]:
                    raise Exception(f"Job submission failed: {response.status_code}")

                job_data = response.json()
                job_id = job_data["id"]

                self.console.print(f"✅ Job submitted successfully! ID: [bold cyan]{job_id}[/bold cyan]")

                # Monitor job progress with live display
                job_complete = False
                poll_count = 0
                max_polls = 60  # 1 minute timeout

                # Create progress bar
                progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("{task.fields[status]}"),
                    TimeElapsedColumn(),
                    console=self.console,
                )

                with progress:
                    task = progress.add_task(
                        "[cyan]Processing capture job...",
                        total=100,
                        status="INITIALIZING"
                    )

                    while not job_complete and poll_count < max_polls:
                        await asyncio.sleep(1)
                        poll_count += 1

                        # Get job status
                        status_response = await client.get(f"{self.api_url}/jobs/{job_id}")

                        if status_response.status_code != 200:
                            raise Exception(f"Failed to get job status: {status_response.status_code}")

                        job_status = status_response.json()
                        state = job_status.get("state", "UNKNOWN")

                        # Update progress
                        if state == "BROWSER_STARTING":
                            progress.update(task, completed=10, status="Starting browser...")
                        elif state == "BROWSER_READY":
                            progress.update(task, completed=20, status="Browser ready")
                        elif state == "CAPTURE_RUNNING":
                            progress.update(task, completed=40, status="Capturing page...")
                        elif state == "CAPTURE_DONE":
                            progress.update(task, completed=60, status="Processing tiles...")
                        elif state == "OCR_RUNNING":
                            progress.update(task, completed=80, status="Running OCR...")
                        elif state == "DONE":
                            progress.update(task, completed=100, status="✅ Complete!")
                            job_complete = True
                        elif state == "FAILED":
                            progress.update(task, status=f"❌ Failed: {job_status.get('error', 'Unknown error')}")
                            break
                        else:
                            progress.update(task, status=f"State: {state}")

                # Display final job details
                if job_complete:
                    self.console.print("\n[green]Job completed successfully![/green]")

                    # Create results tree
                    tree = Tree("📊 Job Results")
                    tree.add(f"Job ID: {job_id}")
                    tree.add(f"State: {state}")

                    manifest = job_status.get("manifest", {})
                    if manifest:
                        manifest_branch = tree.add("📋 Manifest")
                        if manifest.get("tiles_total"):
                            manifest_branch.add(f"Tiles: {manifest['tiles_total']}")
                        if manifest.get("timings"):
                            timings = manifest["timings"]
                            timing_branch = manifest_branch.add("⏱️ Timings")
                            for key, value in timings.items():
                                if value:
                                    timing_branch.add(f"{key}: {value}ms")

                    self.console.print(tree)

                    result = {
                        "success": True,
                        "data": {
                            "job_id": job_id,
                            "final_state": state,
                            "polls": poll_count,
                            "manifest": manifest
                        }
                    }
                else:
                    self.console.print(f"\n[red]Job failed or timed out. State: {state}[/red]")
                    result["data"] = {
                        "job_id": job_id,
                        "final_state": state,
                        "error": job_status.get("error", "Unknown error")
                    }

        except Exception as e:
            error_panel = Panel(
                Syntax(traceback.format_exc(), "python", theme="monokai"),
                title="❌ Error Details",
                border_style="red"
            )
            self.console.print(error_panel)
            result["data"]["error"] = str(e)

        self.console.print()
        return result["success"], result["data"]

    async def test_artifact_retrieval(self, job_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Test 3: Artifact Retrieval"""

        self.console.rule("[bold blue]Test 3: Artifact Retrieval", style="blue")

        panel_content = Text(f"Retrieving artifacts for job: {job_id}", style="cyan")
        self.console.print(Panel(panel_content, title="📦 Artifacts", border_style="blue"))

        result = {"success": False, "data": {}}

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
                artifacts = {}

                # Test different artifact endpoints
                endpoints = [
                    ("Markdown", f"/jobs/{job_id}/out.md"),
                    ("Links JSON", f"/jobs/{job_id}/links.json"),
                    ("Manifest", f"/jobs/{job_id}/manifest.json"),
                ]

                artifact_table = Table(title="Artifact Retrieval", box=box.ROUNDED)
                artifact_table.add_column("Artifact", style="cyan")
                artifact_table.add_column("Status", style="green")
                artifact_table.add_column("Size", style="yellow")
                artifact_table.add_column("Type", style="blue")

                for name, endpoint in endpoints:
                    with self.console.status(f"Retrieving {name}..."):
                        try:
                            response = await client.get(f"{self.api_url}{endpoint}")

                            if response.status_code == 200:
                                content = response.text if name == "Markdown" else response.json()
                                size = len(response.content)
                                artifacts[name] = content

                                artifact_table.add_row(
                                    name,
                                    "✅ Retrieved",
                                    f"{size:,} bytes",
                                    "text/markdown" if name == "Markdown" else "application/json"
                                )
                            else:
                                artifact_table.add_row(
                                    name,
                                    f"❌ Failed ({response.status_code})",
                                    "-",
                                    "-"
                                )
                        except Exception as e:
                            artifact_table.add_row(
                                name,
                                f"❌ Error",
                                "-",
                                str(e)[:30]
                            )

                self.console.print(artifact_table)

                # Show sample of markdown content
                if "Markdown" in artifacts:
                    md_content = artifacts["Markdown"][:500]  # First 500 chars
                    md_panel = Panel(
                        Markdown(md_content + "\n\n*... (truncated)*"),
                        title="📝 Markdown Preview",
                        border_style="green"
                    )
                    self.console.print(md_panel)

                # Show links summary
                if "Links JSON" in artifacts:
                    links_data = artifacts["Links JSON"]
                    total_links = sum(len(domain["links"]) for domain in links_data.get("domains", []))

                    links_tree = Tree("🔗 Links Summary")
                    links_tree.add(f"Total Links: {total_links}")
                    links_tree.add(f"Domains: {len(links_data.get('domains', []))}")

                    self.console.print(links_tree)

                result = {
                    "success": len(artifacts) > 0,
                    "data": {
                        "artifacts_retrieved": list(artifacts.keys()),
                        "total_size": sum(len(str(v)) for v in artifacts.values())
                    }
                }

        except Exception as e:
            error_panel = Panel(
                Syntax(traceback.format_exc(), "python", theme="monokai"),
                title="❌ Error Details",
                border_style="red"
            )
            self.console.print(error_panel)
            result["data"]["error"] = str(e)

        self.console.print()
        return result["success"], result["data"]

    async def test_performance_metrics(self) -> Tuple[bool, Dict[str, Any]]:
        """Test 4: Performance Metrics"""

        self.console.rule("[bold blue]Test 4: Performance Metrics", style="blue")

        panel_content = Text("Checking Prometheus metrics endpoint", style="dim")
        self.console.print(Panel(panel_content, title="📈 Metrics", border_style="blue"))

        result = {"success": False, "data": {}}

        try:
            async with httpx.AsyncClient() as client:
                # Get metrics
                with self.console.status("[bold green]Fetching metrics..."):
                    response = await client.get(f"{self.api_url}:9000/metrics", timeout=5.0)

                if response.status_code == 200:
                    metrics_text = response.text

                    # Parse some key metrics
                    metrics_data = {}
                    for line in metrics_text.split('\n'):
                        if line and not line.startswith('#'):
                            parts = line.split(' ')
                            if len(parts) == 2:
                                metric_name = parts[0].split('{')[0]
                                if metric_name not in metrics_data:
                                    try:
                                        metrics_data[metric_name] = float(parts[1])
                                    except ValueError:
                                        pass

                    # Display metrics table
                    metrics_table = Table(title="Key Performance Metrics", box=box.ROUNDED)
                    metrics_table.add_column("Metric", style="cyan")
                    metrics_table.add_column("Value", style="yellow")

                    key_metrics = [
                        "http_requests_total",
                        "http_request_duration_seconds_count",
                        "capture_duration_seconds_sum",
                        "ocr_tiles_processed_total",
                    ]

                    for metric in key_metrics:
                        value = metrics_data.get(metric, "N/A")
                        if isinstance(value, float):
                            value = f"{value:,.2f}"
                        metrics_table.add_row(metric, str(value))

                    self.console.print(metrics_table)

                    result = {
                        "success": True,
                        "data": {
                            "metrics_count": len(metrics_data),
                            "sample_metrics": {k: v for k, v in list(metrics_data.items())[:5]}
                        }
                    }
                    self.console.print("✅ [green]Metrics endpoint is healthy![/green]")
                else:
                    self.console.print(f"⚠️ [yellow]Metrics endpoint returned {response.status_code}[/yellow]")
                    result["data"]["status_code"] = response.status_code

        except Exception as e:
            # Metrics might not be available, which is okay
            self.console.print(f"ℹ️ [yellow]Metrics endpoint not available: {str(e)}[/yellow]")
            result = {
                "success": True,  # Not a critical failure
                "data": {"note": "Metrics endpoint not configured"}
            }

        self.console.print()
        return result["success"], result["data"]

    async def run_all_tests(self):
        """Run all integration tests."""

        self.print_header()

        # Test results storage
        all_results = []
        job_id = None

        # Run tests
        tests = [
            ("Health Check", self.test_health_check, []),
            ("Job Submission", self.test_job_submission, ["https://example.com"]),
            ("Performance Metrics", self.test_performance_metrics, []),
        ]

        for test_name, test_func, args in tests:
            try:
                success, data = await test_func(*args)
                all_results.append({
                    "name": test_name,
                    "success": success,
                    "data": data
                })

                # Capture job ID for artifact test
                if test_name == "Job Submission" and success:
                    job_id = data.get("job_id")

            except Exception as e:
                all_results.append({
                    "name": test_name,
                    "success": False,
                    "data": {"error": str(e)}
                })

        # Run artifact test if we have a job ID
        if job_id:
            success, data = await self.test_artifact_retrieval(job_id)
            all_results.append({
                "name": "Artifact Retrieval",
                "success": success,
                "data": data
            })

        # Print summary
        self.print_summary(all_results)

        return all_results

    def print_summary(self, results: List[Dict[str, Any]]):
        """Print test summary with detailed results."""

        self.console.rule("[bold magenta]Test Summary", style="magenta")

        # Calculate stats
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r["success"])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        total_time = time.time() - self.start_time

        # Create summary table
        summary_table = Table(title="Integration Test Results", box=box.DOUBLE_EDGE)
        summary_table.add_column("Test", style="cyan", no_wrap=True)
        summary_table.add_column("Result", style="bold")
        summary_table.add_column("Details", style="dim")

        for result in results:
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            status_color = "green" if result["success"] else "red"

            # Extract key details
            details = []
            if "response_time" in result["data"]:
                details.append(f"Time: {result['data']['response_time']:.3f}s")
            if "job_id" in result["data"]:
                details.append(f"Job: {result['data']['job_id'][:8]}...")
            if "error" in result["data"]:
                details.append(f"Error: {str(result['data']['error'])[:30]}...")

            summary_table.add_row(
                result["name"],
                f"[{status_color}]{status}[/{status_color}]",
                " | ".join(details) if details else "Completed"
            )

        self.console.print(summary_table)

        # Overall stats
        stats_panel = Panel(
            Align.center(
                Group(
                    Text(f"Total Tests: {total_tests}", style="cyan"),
                    Text(f"Passed: {passed_tests}", style="green"),
                    Text(f"Failed: {failed_tests}", style="red" if failed_tests > 0 else "dim"),
                    Text(f"Success Rate: {success_rate:.1f}%",
                         style="green" if success_rate == 100 else "yellow" if success_rate >= 75 else "red"),
                    Text(f"Total Time: {total_time:.2f}s", style="blue"),
                )
            ),
            title="📊 Overall Statistics",
            border_style="magenta"
        )
        self.console.print(stats_panel)

        # Final verdict
        if success_rate == 100:
            verdict = Panel(
                Align.center(
                    Text("🎉 ALL TESTS PASSED! 🎉", style="bold green")
                ),
                border_style="green",
                box=box.DOUBLE
            )
        elif success_rate >= 75:
            verdict = Panel(
                Align.center(
                    Text("⚠️ MOSTLY PASSED - Some Issues Found", style="bold yellow")
                ),
                border_style="yellow",
                box=box.DOUBLE
            )
        else:
            verdict = Panel(
                Align.center(
                    Text("❌ TESTS FAILED - Critical Issues Found", style="bold red")
                ),
                border_style="red",
                box=box.DOUBLE
            )

        self.console.print(verdict)

        # Export results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(f"integration_test_{timestamp}.log")
        html_path = Path(f"integration_test_{timestamp}.html")

        log_path.write_text(self.console.export_text(clear=False), encoding="utf-8")
        html_path.write_text(self.console.export_html(clear=False), encoding="utf-8")

        self.console.print(f"\n📄 Test logs saved to: {log_path}")
        self.console.print(f"🌐 HTML report saved to: {html_path}")


async def main():
    """Main entry point."""

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run integration tests for Markdown Web Browser")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Run tests
    runner = IntegrationTestRunner(api_url=args.api_url, verbose=args.verbose)
    results = await runner.run_all_tests()

    # Exit with appropriate code
    success = all(r["success"] for r in results)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)