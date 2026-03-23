"""Helpers to canonicalize state payloads for deterministic processing."""

from __future__ import annotations

from typing import Any


def canonicalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic ordering-friendly payload copy."""

    return dict(sorted(payload.items(), key=lambda item: item[0]))
