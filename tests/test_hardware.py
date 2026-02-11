from __future__ import annotations

import subprocess
from typing import Sequence

from app.hardware import (
    detect_host_capabilities,
    get_host_capabilities,
    reset_host_capabilities_cache,
)


def _runner(stdout: str, *, returncode: int = 0):
    def _invoke(command: Sequence[str], timeout_seconds: float) -> subprocess.CompletedProcess[str]:
        _ = timeout_seconds
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=returncode,
            stdout=stdout,
            stderr="",
        )

    return _invoke


def test_detect_host_capabilities_handles_missing_nvidia_smi() -> None:
    def _missing_runner(
        command: Sequence[str], timeout_seconds: float
    ) -> subprocess.CompletedProcess[str]:
        _ = (command, timeout_seconds)
        raise FileNotFoundError

    snapshot = detect_host_capabilities(command_runner=_missing_runner)
    assert snapshot.gpu_count == 0
    assert "nvidia-smi-missing" in snapshot.detection_warnings
    assert snapshot.preferred_hardware_path == "cpu"
    assert snapshot.cpu_logical_cores >= 1
    assert snapshot.cpu_physical_cores >= 1


def test_detect_host_capabilities_parses_nvidia_multi_gpu_inventory() -> None:
    snapshot = detect_host_capabilities(
        command_runner=_runner(
            "0, NVIDIA A100-SXM4-40GB, 40536, 550.54.15, 12.4\n"
            "1, NVIDIA A100-SXM4-40GB, 40536, 550.54.15, 12.4\n"
        )
    )
    assert snapshot.gpu_count == 2
    assert snapshot.has_gpu is True
    assert snapshot.preferred_hardware_path == "gpu"
    assert snapshot.detection_sources == ("nvidia-smi",)
    assert snapshot.gpu_devices[0].vendor == "nvidia"
    assert snapshot.gpu_devices[0].memory_total_mb == 40536
    assert snapshot.gpu_devices[0].runtime_version == "12.4"


def test_detect_host_capabilities_records_parse_warnings() -> None:
    snapshot = detect_host_capabilities(command_runner=_runner("abc,broken,row\n"))
    assert snapshot.gpu_count == 0
    assert "nvidia-smi-parse-error" in snapshot.detection_warnings


def test_hardware_snapshot_to_dict_shape() -> None:
    snapshot = detect_host_capabilities(command_runner=_runner("", returncode=0))
    payload = snapshot.to_dict()
    assert snapshot.cpu_logical_cores >= 1
    assert payload["gpu_count"] == snapshot.gpu_count
    assert payload["preferred_hardware_path"] == snapshot.preferred_hardware_path
    assert isinstance(payload["gpu_devices"], list)


def test_hardware_snapshot_cache_reset() -> None:
    reset_host_capabilities_cache()
    first = get_host_capabilities()
    second = get_host_capabilities()
    assert first == second
    reset_host_capabilities_cache()
