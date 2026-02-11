from __future__ import annotations

import pytest

from app.ocr_policy import (
    OCRBackendCandidate,
    OCRPolicyDecision,
    OCRPolicyInputs,
    OCRRuntimeSignal,
    REASON_LOCAL_CPU_FALLBACK,
    REASON_LOCAL_GPU_PREFERRED,
    REASON_REEVAL_FAILURE,
    REASON_REEVAL_LATENCY,
    REASON_REEVAL_NOT_REQUIRED,
    REASON_REEVAL_RECOVERED,
    REASON_REEVAL_TIMER,
    REASON_REMOTE_FALLBACK,
    REASON_SKIP_UNHEALTHY,
    select_ocr_backend,
    should_reevaluate_policy,
)


def test_select_ocr_backend_prefers_local_gpu() -> None:
    decision = select_ocr_backend(
        OCRPolicyInputs(
            candidates=(
                OCRBackendCandidate(
                    backend_id="glm-ocr-local-openai",
                    backend_mode="openai-compatible",
                    hardware_path="gpu",
                ),
                OCRBackendCandidate(
                    backend_id="glm-ocr-remote-openai",
                    backend_mode="openai-compatible",
                    hardware_path="remote",
                ),
            )
        )
    )
    assert decision.backend_id == "glm-ocr-local-openai"
    assert REASON_LOCAL_GPU_PREFERRED in decision.reason_codes
    assert decision.fallback_chain == ("glm-ocr-remote-openai",)
    assert decision.reevaluate_after_s == 120


def test_select_ocr_backend_uses_cpu_when_no_gpu_candidate() -> None:
    decision = select_ocr_backend(
        OCRPolicyInputs(
            candidates=(
                OCRBackendCandidate(
                    backend_id="glm-ocr-local-openai",
                    backend_mode="openai-compatible",
                    hardware_path="cpu",
                ),
                OCRBackendCandidate(
                    backend_id="glm-ocr-remote-openai",
                    backend_mode="openai-compatible",
                    hardware_path="remote",
                ),
            )
        )
    )
    assert decision.backend_id == "glm-ocr-local-openai"
    assert REASON_LOCAL_CPU_FALLBACK in decision.reason_codes
    assert decision.reevaluate_after_s == 30


def test_select_ocr_backend_skips_unhealthy_primary() -> None:
    decision = select_ocr_backend(
        OCRPolicyInputs(
            candidates=(
                OCRBackendCandidate(
                    backend_id="glm-ocr-local-openai",
                    backend_mode="openai-compatible",
                    hardware_path="gpu",
                    healthy=False,
                ),
                OCRBackendCandidate(
                    backend_id="glm-ocr-remote-openai",
                    backend_mode="openai-compatible",
                    hardware_path="remote",
                    healthy=True,
                ),
            )
        )
    )
    assert decision.backend_id == "glm-ocr-remote-openai"
    assert REASON_SKIP_UNHEALTHY in decision.reason_codes
    assert REASON_REMOTE_FALLBACK in decision.reason_codes


def test_select_ocr_backend_requires_candidates() -> None:
    with pytest.raises(ValueError):
        select_ocr_backend(OCRPolicyInputs(candidates=()))


def test_should_reevaluate_policy_signals() -> None:
    decision = OCRPolicyDecision(
        backend_id="glm-ocr-local-openai",
        backend_mode="openai-compatible",
        hardware_path="cpu",
        fallback_chain=("glm-ocr-remote-openai",),
        reason_codes=(REASON_LOCAL_CPU_FALLBACK,),
        reevaluate_after_s=30,
    )
    assert (
        should_reevaluate_policy(signal=OCRRuntimeSignal.REQUEST_FAILED, decision=decision).reason_code
        == REASON_REEVAL_FAILURE
    )
    assert (
        should_reevaluate_policy(signal=OCRRuntimeSignal.BACKEND_UNHEALTHY, decision=decision)
        .reason_code
        == REASON_REEVAL_FAILURE
    )
    assert (
        should_reevaluate_policy(signal=OCRRuntimeSignal.BACKEND_RECOVERED, decision=decision)
        .reason_code
        == REASON_REEVAL_RECOVERED
    )
    assert (
        should_reevaluate_policy(signal=OCRRuntimeSignal.LATENCY_SPIKE, decision=decision).reason_code
        == REASON_REEVAL_LATENCY
    )
    assert (
        should_reevaluate_policy(signal=OCRRuntimeSignal.PERIODIC_TIMER, decision=decision).reason_code
        == REASON_REEVAL_TIMER
    )
    assert (
        should_reevaluate_policy(signal=OCRRuntimeSignal.NO_CHANGE, decision=decision).reason_code
        == REASON_REEVAL_NOT_REQUIRED
    )
