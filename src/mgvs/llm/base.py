"""Base interfaces for PT/PCT/LSS LLM-backed planning stages."""

from __future__ import annotations

from typing import Protocol

from mgvs.state.models import ReasoningState


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

    def propose(self, state: ReasoningState, depth: int) -> list[object]:
        """Return parsed LSS proposals for the given state/depth."""
