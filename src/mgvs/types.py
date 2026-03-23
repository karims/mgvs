"""Shared lightweight type aliases used across MGVS modules."""

from __future__ import annotations

from typing import NewType

StateId = NewType("StateId", str)
ActionId = NewType("ActionId", str)
