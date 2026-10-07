"""List the checkpoints a caller may pin."""

from __future__ import annotations

from dataclasses import replace

from ...domain.entities import ModelInfo
from ...domain.ports import DecisionEngine, ModelCatalog


class ListModels:
    """Every checkpoint the catalog knows, flagged with whether the engine has it resident."""

    def __init__(self, catalog: ModelCatalog, engine: DecisionEngine) -> None:
        self._catalog = catalog
        self._engine = engine

    def execute(self) -> list[ModelInfo]:
        loaded = set(self._engine.status().loaded)
        return [replace(model, loaded=model.name in loaded) for model in self._catalog.list()]
