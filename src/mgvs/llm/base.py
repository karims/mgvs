"""Base interfaces for PT/PCT/LSS and endgame LLM-backed planning stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from mgvs.state.models import ReasoningState

if TYPE_CHECKING:
    from mgvs.actions.models import CandidateAction

LLMStage = Literal["pt", "pct", "lss", "endgame"]
PTContractVersion = "pt_v2"
PCTContractVersion = "pct_v2"
LSSContractVersion = "lss_v2"
ENDGAMEContractVersion = "endgame_v1"
DEFAULT_PCT_MAX_TACTICS = 4


class LLMClientError(RuntimeError):
    """Typed exception for recoverable/unrecoverable LLM client failures."""


@dataclass(frozen=True)
class LLMRequestOptions:
    """Runtime options for structured generation requests."""

    temperature: float
    max_tokens: int
    timeout: float


@dataclass(frozen=True)
class ParseIssue:
    """Non-fatal parser issue for structured output recovery."""

    stage: LLMStage
    reason: str


class PTClient(Protocol):
    """Problem translation stage client interface."""

    def generate_pt(self, prompt: str) -> str:
        """Return structured PT output text for the given prompt."""


class PCTClient(Protocol):
    """Concept-and-tactic proposal stage client interface."""

    def generate_pct(self, prompt: str) -> str:
        """Return structured PCT output text for the given prompt."""


class LSSClient(Protocol):
    """Local-step synthesis stage client interface."""

    def generate_lss(self, prompt: str) -> str:
        """Return structured LSS output text for the given prompt."""


class UnifiedLLMClient(PTClient, PCTClient, LSSClient, Protocol):
    """Unified client exposing all PT/PCT/LSS stage methods."""


class PTService(Protocol):
    """State-aware PT service boundary for higher-level orchestration."""

    def run_pt(self, state: ReasoningState) -> ReasoningState:
        """Apply PT stage updates to the provided state."""


class PCTService(Protocol):
    """State-aware PCT service boundary for higher-level orchestration."""

    def run_pct(self, state: ReasoningState) -> ReasoningState:
        """Apply PCT stage updates to the provided state."""


class LSSService(Protocol):
    """State-aware LSS service boundary for higher-level orchestration."""

    def propose(self, state: ReasoningState, depth: int) -> list["CandidateAction"]:
        """Return parsed LSS proposals for the given state/depth."""
