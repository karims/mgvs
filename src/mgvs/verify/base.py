"""Base verification protocol used by all verifier components."""

from __future__ import annotations

from dataclasses import dataclass

from mgvs.state.models import ReasoningState


@dataclass
class VerificationResult:
    """Standard result object for verification stages."""

    ok: bool
    detail: str = ""


def verify(state: ReasoningState) -> VerificationResult:
    """Base verifier placeholder that always passes."""

    _ = state
    return VerificationResult(ok=True, detail="base verifier placeholder")
