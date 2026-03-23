"""Runtime configuration primitives for MGVS experiments and local runs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RunConfig:
    """Minimal configuration container for bootstrap execution paths."""

    seed: int = 0
    max_steps: int = 100
