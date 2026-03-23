"""Base protocols and composite orchestration for MGVS verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from mgvs.actions.models import CandidateAction
    from mgvs.state.models import ReasoningState


@dataclass(frozen=True)
class VerificationResult:
    """Per-level verification outcome."""

    passed: bool
    level: str
    reason: str
    details: dict[str, Any]


@dataclass(frozen=True)
class CombinedVerificationResult:
    """Aggregate verification decision across multiple levels."""

    passed: bool
    results: list[VerificationResult]

    @property
    def first_failure(self) -> VerificationResult | None:
        """Return the first failed level if any."""

        for result in self.results:
            if not result.passed:
                return result
        return None


class LocalValidityVerifier(Protocol):
    """Interface for local structural validity checks."""

    def verify_action(self, state: "ReasoningState", action: "CandidateAction") -> VerificationResult:
        """Validate local structural correctness for an action."""

    def verify_state(self, state: "ReasoningState") -> VerificationResult:
        """Validate local structural correctness for a state."""


class StateConsistencyVerifier(Protocol):
    """Interface for consistency checks across status and transitions."""

    def verify_action(self, state: "ReasoningState", action: "CandidateAction") -> VerificationResult:
        """Validate action consistency against current state."""

    def verify_state(self, state: "ReasoningState") -> VerificationResult:
        """Validate internal consistency of a state."""


class GlobalCompatibilityVerifier(Protocol):
    """Interface for global/shared constraint compatibility checks."""

    def verify_action(self, state: "ReasoningState", action: "CandidateAction") -> VerificationResult:
        """Validate action compatibility with global constraints."""

    def verify_state(self, state: "ReasoningState") -> VerificationResult:
        """Validate global compatibility of a state."""


class CompositeVerifier:
    """Runs local, consistency, and global verifiers and combines decisions."""

    def __init__(
        self,
        *,
        local_verifier: LocalValidityVerifier,
        consistency_verifier: StateConsistencyVerifier,
        global_verifier: GlobalCompatibilityVerifier,
    ) -> None:
        self._local_verifier = local_verifier
        self._consistency_verifier = consistency_verifier
        self._global_verifier = global_verifier


    def verify_action(self, state: "ReasoningState", action: "CandidateAction") -> CombinedVerificationResult:
        """Run all action-level verifiers and return combined decision."""

        results = [
            self._local_verifier.verify_action(state, action),
            self._consistency_verifier.verify_action(state, action),
            self._global_verifier.verify_action(state, action),
        ]
        return CombinedVerificationResult(passed=all(result.passed for result in results), results=results)


    def verify_state(self, state: "ReasoningState") -> CombinedVerificationResult:
        """Run all state-level verifiers and return combined decision."""

        results = [
            self._local_verifier.verify_state(state),
            self._consistency_verifier.verify_state(state),
            self._global_verifier.verify_state(state),
        ]
        return CombinedVerificationResult(passed=all(result.passed for result in results), results=results)

    def is_action_valid(self, state: "ReasoningState", action: "CandidateAction") -> bool:
        """Compatibility adapter for controller boolean verifier contract."""

        return self.verify_action(state, action).passed

    def is_state_valid(self, state: "ReasoningState") -> bool:
        """Compatibility adapter for controller boolean verifier contract."""

        return self.verify_state(state).passed
