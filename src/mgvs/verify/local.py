"""Local constraints verifier for single-state validation."""

from mgvs.state.models import ReasoningState
from mgvs.verify.base import VerificationResult


def verify_local(state: ReasoningState) -> VerificationResult:
    """Placeholder local check implementation."""

    _ = state
    return VerificationResult(ok=True, detail="local verifier placeholder")
