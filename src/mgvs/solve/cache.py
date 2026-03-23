"""Local in-memory stage caches for PT/PCT/LSS runtime stabilization."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StageCache:
    """Simple namespace-separated string cache."""

    _store: dict[str, dict[str, str]] = field(default_factory=dict)

    def get(self, namespace: str, key: str) -> str | None:
        """Get a cached value by namespace/key."""

        return self._store.get(namespace, {}).get(key)

    def set(self, namespace: str, key: str, value: str) -> None:
        """Set a cached value by namespace/key."""

        self._store.setdefault(namespace, {})[key] = value

    def clear(self) -> None:
        """Clear all cached data."""

        self._store.clear()
