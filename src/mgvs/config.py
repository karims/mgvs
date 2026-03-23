"""Runtime configuration primitives for MGVS experiments and local runs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    """Minimal configuration container for bootstrap execution paths."""

    seed: int = 0
    max_steps: int = 100


@dataclass(frozen=True)
class RuntimeBudgetConfig:
    """Runtime budget controls for per-problem and session execution."""

    max_depth: int = 4
    per_problem_max_wall_time_s: float = 20.0
    session_max_wall_time_s: float = 0.0
    beam_width: int = 3
    candidate_action_cap_per_state: int = 3

    @classmethod
    def from_env(cls) -> "RuntimeBudgetConfig":
        """Load runtime budget controls from environment variables."""

        return cls(
            max_depth=int(os.getenv("MGVS_MAX_DEPTH", "4")),
            per_problem_max_wall_time_s=float(os.getenv("MGVS_MAX_PROBLEM_WALL_TIME_S", "20")),
            session_max_wall_time_s=float(os.getenv("MGVS_MAX_SESSION_WALL_TIME_S", "0")),
            beam_width=int(os.getenv("MGVS_BEAM_WIDTH", "3")),
            candidate_action_cap_per_state=int(os.getenv("MGVS_CANDIDATE_CAP_PER_STATE", "3")),
        )


@dataclass(frozen=True)
class SolveModeSettings:
    """Per-mode execution settings used by the policy layer."""

    beam_width: int
    max_depth: int
    max_candidates_per_state: int
    llm_retries: int
    allow_expensive_branching: bool
    use_pt: bool
    use_pct: bool
    use_lss: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize mode settings for logging/debug output."""

        return {
            "beam_width": self.beam_width,
            "max_depth": self.max_depth,
            "max_candidates_per_state": self.max_candidates_per_state,
            "llm_retries": self.llm_retries,
            "allow_expensive_branching": self.allow_expensive_branching,
            "use_pt": self.use_pt,
            "use_pct": self.use_pct,
            "use_lss": self.use_lss,
        }


@dataclass(frozen=True)
class SolvePolicyConfig:
    """Configurable policy defaults for solve-mode routing and fallback."""

    fast: SolveModeSettings
    balanced: SolveModeSettings
    deep: SolveModeSettings
    easy_problem_max_chars: int = 80
    hard_problem_min_chars: int = 220
    budget_pressure_fast_threshold: float = 0.2
    budget_pressure_fallback_threshold: float = 0.8
    malformed_retry_fallback_threshold: int = 2

    @classmethod
    def default(cls) -> "SolvePolicyConfig":
        """Create default deterministic solve mode profiles."""

        return cls(
            fast=SolveModeSettings(
                beam_width=1,
                max_depth=2,
                max_candidates_per_state=1,
                llm_retries=0,
                allow_expensive_branching=False,
                use_pt=True,
                use_pct=True,
                use_lss=True,
            ),
            balanced=SolveModeSettings(
                beam_width=3,
                max_depth=4,
                max_candidates_per_state=3,
                llm_retries=1,
                allow_expensive_branching=True,
                use_pt=True,
                use_pct=True,
                use_lss=True,
            ),
            deep=SolveModeSettings(
                beam_width=5,
                max_depth=7,
                max_candidates_per_state=5,
                llm_retries=2,
                allow_expensive_branching=True,
                use_pt=True,
                use_pct=True,
                use_lss=True,
            ),
        )

    @classmethod
    def from_env(cls) -> "SolvePolicyConfig":
        """Load solve-mode policy defaults from environment variables."""

        base = cls.default()
        return cls(
            fast=_mode_from_env("MGVS_FAST", base.fast),
            balanced=_mode_from_env("MGVS_BALANCED", base.balanced),
            deep=_mode_from_env("MGVS_DEEP", base.deep),
            easy_problem_max_chars=int(
                os.getenv("MGVS_POLICY_EASY_PROBLEM_MAX_CHARS", str(base.easy_problem_max_chars))
            ),
            hard_problem_min_chars=int(
                os.getenv("MGVS_POLICY_HARD_PROBLEM_MIN_CHARS", str(base.hard_problem_min_chars))
            ),
            budget_pressure_fast_threshold=float(
                os.getenv(
                    "MGVS_POLICY_BUDGET_PRESSURE_FAST_THRESHOLD",
                    str(base.budget_pressure_fast_threshold),
                )
            ),
            budget_pressure_fallback_threshold=float(
                os.getenv(
                    "MGVS_POLICY_BUDGET_PRESSURE_FALLBACK_THRESHOLD",
                    str(base.budget_pressure_fallback_threshold),
                )
            ),
            malformed_retry_fallback_threshold=int(
                os.getenv(
                    "MGVS_POLICY_MALFORMED_RETRY_FALLBACK_THRESHOLD",
                    str(base.malformed_retry_fallback_threshold),
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize policy config for reproducibility logs."""

        return {
            "fast": self.fast.to_dict(),
            "balanced": self.balanced.to_dict(),
            "deep": self.deep.to_dict(),
            "easy_problem_max_chars": self.easy_problem_max_chars,
            "hard_problem_min_chars": self.hard_problem_min_chars,
            "budget_pressure_fast_threshold": self.budget_pressure_fast_threshold,
            "budget_pressure_fallback_threshold": self.budget_pressure_fallback_threshold,
            "malformed_retry_fallback_threshold": self.malformed_retry_fallback_threshold,
        }


@dataclass(frozen=True)
class VLLMRuntimeConfig:
    """Runtime configuration for OpenAI-compatible vLLM endpoints."""

    base_url: str = "http://localhost:8000/v1"
    api_key: str = ""
    model_name: str = "default"
    temperature: float = 0.0
    max_tokens: int = 512
    timeout: float = 30.0
    retries: int = 1
    lss_retry_candidate_decay: float = 0.5

    @classmethod
    def from_env(cls) -> "VLLMRuntimeConfig":
        """Load vLLM configuration from environment variables."""

        return cls(
            base_url=os.getenv("MGVS_VLLM_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.getenv("MGVS_VLLM_API_KEY", ""),
            model_name=os.getenv("MGVS_VLLM_MODEL_NAME", "default"),
            temperature=float(os.getenv("MGVS_VLLM_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("MGVS_VLLM_MAX_TOKENS", "512")),
            timeout=float(os.getenv("MGVS_VLLM_TIMEOUT", "30.0")),
            retries=int(os.getenv("MGVS_VLLM_RETRIES", "1")),
            lss_retry_candidate_decay=float(os.getenv("MGVS_VLLM_LSS_RETRY_DECAY", "0.5")),
        )


def _mode_from_env(prefix: str, defaults: SolveModeSettings) -> SolveModeSettings:
    """Read solve-mode settings from env vars with fallback defaults."""

    return SolveModeSettings(
        beam_width=int(os.getenv(f"{prefix}_BEAM_WIDTH", str(defaults.beam_width))),
        max_depth=int(os.getenv(f"{prefix}_MAX_DEPTH", str(defaults.max_depth))),
        max_candidates_per_state=int(
            os.getenv(f"{prefix}_MAX_CANDIDATES", str(defaults.max_candidates_per_state))
        ),
        llm_retries=int(os.getenv(f"{prefix}_LLM_RETRIES", str(defaults.llm_retries))),
        allow_expensive_branching=_env_bool(
            f"{prefix}_ALLOW_EXPENSIVE_BRANCHING",
            defaults.allow_expensive_branching,
        ),
        use_pt=_env_bool(f"{prefix}_USE_PT", defaults.use_pt),
        use_pct=_env_bool(f"{prefix}_USE_PCT", defaults.use_pct),
        use_lss=_env_bool(f"{prefix}_USE_LSS", defaults.use_lss),
    )


def _env_bool(name: str, default: bool) -> bool:
    """Parse environment boolean with stable default."""

    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
