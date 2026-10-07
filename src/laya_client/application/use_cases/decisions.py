"""Answer decision requests: the ``/v1/systemone`` and ``/v1/systemone/batch`` use cases."""

from __future__ import annotations

from dataclasses import replace

from ...domain.entities import BatchDecision, BatchDecisionRequest, Decision, DecisionRequest
from ...domain.policies import RequestLimits, validate_batch_request, validate_decision_request
from ...domain.ports import DecisionEngine, ModelCatalog


class PredictDecision:
    """Validate a request, resolve the checkpoint it names, and let the engine answer it."""

    def __init__(self, engine: DecisionEngine, catalog: ModelCatalog, limits: RequestLimits) -> None:
        self._engine = engine
        self._catalog = catalog
        self._limits = limits

    def execute(self, request: DecisionRequest) -> Decision:
        validate_decision_request(request, self._limits)
        model = self._catalog.resolve(request.controls.model)
        return self._engine.predict(replace(request, controls=request.controls.with_model(model)))


class PredictBatchDecision:
    """Batch version of :class:`PredictDecision`: one set of questions, many states."""

    def __init__(self, engine: DecisionEngine, catalog: ModelCatalog, limits: RequestLimits) -> None:
        self._engine = engine
        self._catalog = catalog
        self._limits = limits

    def execute(self, request: BatchDecisionRequest) -> BatchDecision:
        validate_batch_request(request, self._limits)
        model = self._catalog.resolve(request.controls.model)
        return self._engine.predict_batch(replace(request, controls=request.controls.with_model(model)))
