"""Global objective and invariant checks over aggregated search progress."""

from mgvs.state.models import ReasoningState
from mgvs.verify.base import VerificationResult


def verify_global(state: ReasoningState) -> VerificationResult:
    """Placeholder global check implementation."""

    _ = state
    return VerificationResult(ok=True, detail="global verifier placeholder")
