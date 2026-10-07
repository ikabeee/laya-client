"""Report whether the service can answer requests."""

from __future__ import annotations

from ...domain.entities import EngineStatus
from ...domain.ports import DecisionEngine


class GetHealth:
    def __init__(self, engine: DecisionEngine) -> None:
        self._engine = engine

    def execute(self) -> EngineStatus:
        return self._engine.status()
