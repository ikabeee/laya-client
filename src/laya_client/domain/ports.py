"""Ports: the interfaces the application layer depends on and the infrastructure implements.

The use cases talk to a ``DecisionEngine``, a ``ModelCatalog`` and a ``PresetRepository`` without
knowing whether the engine is Laya on a GPU, Laya on a CPU or a deterministic mock in a test.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .entities import (
    BatchDecision,
    BatchDecisionRequest,
    Decision,
    DecisionRequest,
    EngineStatus,
    ModelInfo,
    Preset,
)


class DecisionEngine(ABC):
    """Answers decision requests. Implementations must be safe to call from several threads."""

    @abstractmethod
    def predict(self, request: DecisionRequest) -> Decision:
        """Answer every question about one state."""

    @abstractmethod
    def predict_batch(self, request: BatchDecisionRequest) -> BatchDecision:
        """Answer the same questions about several states, preserving their order."""

    @abstractmethod
    def status(self) -> EngineStatus:
        """Report readiness without loading anything."""

    def warmup(self) -> None:  # noqa: B027 -- optional hook, a no-op is a valid implementation
        """Load whatever the engine needs before the first request. Optional."""

    def shutdown(self) -> None:  # noqa: B027 -- optional hook, a no-op is a valid implementation
        """Release resources. Optional."""


class ModelCatalog(ABC):
    """The checkpoints a caller may name, and how their names resolve."""

    @abstractmethod
    def list(self) -> list[ModelInfo]:
        """Every known checkpoint."""

    @abstractmethod
    def resolve(self, name: str | None) -> str | None:
        """Canonical checkpoint name, or ``None`` to let the router choose.

        Raises ``ModelNotFoundError`` when ``name`` pins a checkpoint the deployment cannot load.
        """


class PresetRepository(ABC):
    """Ready-made question sets."""

    @abstractmethod
    def list(self) -> list[Preset]:
        """Every registered preset."""

    @abstractmethod
    def get(self, name: str) -> Preset:
        """The preset called ``name``. Raises ``PresetNotFoundError`` when there is none."""
