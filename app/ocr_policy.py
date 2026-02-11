"""Deterministic OCR autopilot policy engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

BACKEND_MODE_OPENAI_COMPATIBLE = "openai-compatible"
BACKEND_MODE_MAAS = "maas"

REASON_LOCAL_GPU_PREFERRED = "policy.local.gpu-preferred"
REASON_LOCAL_CPU_FALLBACK = "policy.local.cpu-fallback"
REASON_REMOTE_FALLBACK = "policy.remote.fallback"
REASON_SKIP_UNHEALTHY = "policy.skip.unhealthy"
REASON_REEVAL_TIMER = "policy.reeval.timer"
REASON_REEVAL_FAILURE = "policy.reeval.failure"
REASON_REEVAL_RECOVERED = "policy.reeval.recovered"
REASON_REEVAL_LATENCY = "policy.reeval.latency"
REASON_REEVAL_NOT_REQUIRED = "policy.reeval.not-required"


@dataclass(slots=True, frozen=True)
class OCRBackendCandidate:
    """Candidate backend considered by the autopilot selector."""

    backend_id: str
    backend_mode: str
    hardware_path: str
    healthy: bool | None = None


@dataclass(slots=True, frozen=True)
class OCRPolicyInputs:
    """Input payload used to select an OCR backend deterministically."""

    candidates: tuple[OCRBackendCandidate, ...]


@dataclass(slots=True, frozen=True)
class OCRPolicyDecision:
    """Final OCR backend decision with policy trace metadata."""

    backend_id: str
    backend_mode: str
    hardware_path: str
    fallback_chain: tuple[str, ...]
    reason_codes: tuple[str, ...]
    reevaluate_after_s: int


class OCRRuntimeSignal(str, Enum):
    """Runtime signals that can trigger policy re-evaluation."""

    REQUEST_FAILED = "request_failed"
    BACKEND_UNHEALTHY = "backend_unhealthy"
    BACKEND_RECOVERED = "backend_recovered"
    LATENCY_SPIKE = "latency_spike"
    PERIODIC_TIMER = "periodic_timer"
    NO_CHANGE = "no_change"


@dataclass(slots=True, frozen=True)
class OCRReevaluationDecision:
    should_reevaluate: bool
    reason_code: str


def select_ocr_backend(inputs: OCRPolicyInputs) -> OCRPolicyDecision:
    """Select the best backend using explicit GPU/CPU/remote priorities."""

    candidates = list(inputs.candidates)
    if not candidates:
        raise ValueError("OCR policy requires at least one backend candidate")

    reason_codes: list[str] = []
    selected: OCRBackendCandidate | None = None
    for candidate in candidates:
        if candidate.healthy is False:
            reason_codes.append(REASON_SKIP_UNHEALTHY)
            continue
        selected = candidate
        break

    if selected is None:
        selected = candidates[0]
        reason_codes.append(REASON_SKIP_UNHEALTHY)

    if selected.hardware_path == "gpu":
        reason_codes.append(REASON_LOCAL_GPU_PREFERRED)
    elif selected.hardware_path == "cpu":
        reason_codes.append(REASON_LOCAL_CPU_FALLBACK)
    else:
        reason_codes.append(REASON_REMOTE_FALLBACK)

    fallback_chain = tuple(
        candidate.backend_id
        for candidate in candidates
        if candidate.backend_id != selected.backend_id
    )
    # Re-evaluate faster when not on the top-tier local GPU path.
    reevaluate_after_s = 30 if selected.hardware_path != "gpu" else 120

    return OCRPolicyDecision(
        backend_id=selected.backend_id,
        backend_mode=selected.backend_mode,
        hardware_path=selected.hardware_path,
        fallback_chain=fallback_chain,
        reason_codes=tuple(reason_codes),
        reevaluate_after_s=reevaluate_after_s,
    )


def should_reevaluate_policy(
    *,
    signal: OCRRuntimeSignal,
    decision: OCRPolicyDecision,
) -> OCRReevaluationDecision:
    """Evaluate whether current runtime conditions should trigger failover logic."""

    if signal in {OCRRuntimeSignal.REQUEST_FAILED, OCRRuntimeSignal.BACKEND_UNHEALTHY}:
        return OCRReevaluationDecision(True, REASON_REEVAL_FAILURE)
    if signal == OCRRuntimeSignal.BACKEND_RECOVERED:
        return OCRReevaluationDecision(True, REASON_REEVAL_RECOVERED)
    if signal == OCRRuntimeSignal.LATENCY_SPIKE and decision.hardware_path != "gpu":
        return OCRReevaluationDecision(True, REASON_REEVAL_LATENCY)
    if signal == OCRRuntimeSignal.PERIODIC_TIMER:
        return OCRReevaluationDecision(True, REASON_REEVAL_TIMER)
    return OCRReevaluationDecision(False, REASON_REEVAL_NOT_REQUIRED)

