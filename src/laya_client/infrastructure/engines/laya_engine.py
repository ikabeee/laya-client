"""The real engine: an in-process ``laya.Router``.

``laya`` (and torch) are imported lazily, the first time the engine is built, so the service starts,
serves its docs and answers ``/health`` even while the checkpoints are still downloading.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from ...domain.entities import (
    BatchDecision,
    BatchDecisionRequest,
    Decision,
    DecisionRequest,
    EngineStatus,
    Usage,
)
from ...domain.errors import (
    EngineUnavailableError,
    InferenceFailedError,
    InvalidRequestError,
    LayaClientError,
)
from ...domain.ports import DecisionEngine
from .mapping import controls_to_kwargs, decision_from_dict

_log = logging.getLogger("laya_client.engine")

# Row width assumed when a batch does not set max_len: the English checkpoint's own window.
_BATCH_ROW_TOKENS_ASSUMED = 512


@dataclass(frozen=True)
class LayaEngineConfig:
    device: str | None = None
    preload: bool = True
    models: tuple[str, ...] = field(default_factory=tuple)
    threads: int | None = None
    auto_task: bool = False
    default_model: str | None = None
    max_loaded: int | None = None
    max_batch_tokens: int = 131_072


class LayaRouterEngine(DecisionEngine):
    """Adapter from the ``DecisionEngine`` port to ``laya.Router``.

    One forward pass runs at a time (``_infer_lock``): that is what a single CPU or GPU wants, and
    it keeps memory bounded no matter how many HTTP requests are waiting.
    """

    def __init__(self, config: LayaEngineConfig, router: Any = None) -> None:
        self._config = config
        self._router = router
        self._build_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._last_error: str | None = None

    # -- lifecycle ----------------------------------------------------------------------------

    def _build_router(self) -> Any:
        try:
            from laya import Router
        except ImportError as error:
            raise EngineUnavailableError(
                "the 'laya' package is not installed: install 'laya-client[engine]' or set LAYA_ENGINE=mock"
            ) from error

        if self._config.threads:
            import torch

            torch.set_num_threads(self._config.threads)

        options: dict[str, Any] = {
            "device": self._config.device,
            "auto_task_detection": self._config.auto_task,
        }
        if self._config.max_loaded:
            options["max_loaded"] = self._config.max_loaded
        if self._config.default_model:
            options["default"] = self._config.default_model
        router = Router(**options)
        if self._config.preload:
            _log.info("preloading checkpoints: %s", ", ".join(self._config.models) or "all")
            router.preload(list(self._config.models) or None)
        return router

    def _get_router(self) -> Any:
        if self._router is not None:
            return self._router
        with self._build_lock:
            if self._router is None:
                started = time.perf_counter()
                try:
                    self._router = self._build_router()
                except LayaClientError as error:
                    self._last_error = error.message
                    raise
                except Exception as error:
                    _log.exception("could not start the Laya router")
                    self._last_error = "%s: %s" % (type(error).__name__, error)
                    raise EngineUnavailableError("the inference engine failed to start") from error
                self._last_error = None
                _log.info("Laya router ready in %.1fs", time.perf_counter() - started)
        return self._router

    def warmup(self) -> None:
        self._get_router()

    def shutdown(self) -> None:
        router = self._router
        if router is not None and hasattr(router, "unload"):
            try:
                router.unload()
            except Exception:  # noqa: BLE001 -- shutting down; nothing useful to do with it
                _log.exception("unload failed during shutdown")

    def status(self) -> EngineStatus:
        router = self._router
        if router is None:
            return EngineStatus(
                engine="laya",
                ready=False,
                device=self._config.device,
                details={"error": self._last_error} if self._last_error else {"state": "starting"},
            )
        return EngineStatus(
            engine="laya",
            ready=True,
            loaded=tuple(getattr(router, "loaded", None) or ()),
            device=self._config.device or "auto",
            details={"revisions": dict(getattr(router, "loaded_revisions", None) or {})},
        )

    # -- inference ----------------------------------------------------------------------------

    def _run(self, fn: Any) -> tuple[Any, float]:
        with self._infer_lock:
            started = time.perf_counter()
            try:
                result = fn()
            except LayaClientError:
                raise
            except (ValueError, TypeError) as error:
                # Laya raises ValueError for question/control problems; its messages name the
                # question and what to fix, so they are safe to return to the caller.
                raise InvalidRequestError(str(error)) from error
            except Exception as error:
                _log.exception("inference failed")
                raise InferenceFailedError("inference failed") from error
            return result, (time.perf_counter() - started) * 1000.0

    def predict(self, request: DecisionRequest) -> Decision:
        router = self._get_router()
        questions = request.question_definitions()
        kwargs = controls_to_kwargs(request.controls)
        result, elapsed = self._run(lambda: router.predict(request.state, questions, **kwargs))
        return decision_from_dict(result, inference_ms=elapsed)

    def _planned_batch_size(self, request: BatchDecisionRequest) -> int | None:
        """Split a batch so one forward pass stays inside ``max_batch_tokens``; ``None`` if it fits."""
        row_tokens = request.controls.max_len or _BATCH_ROW_TOKENS_ASSUMED
        per_state = max(1, len(request.questions)) * row_tokens
        if per_state * len(request.states) <= self._config.max_batch_tokens:
            return None
        return max(1, self._config.max_batch_tokens // per_state)

    def predict_batch(self, request: BatchDecisionRequest) -> BatchDecision:
        router = self._get_router()
        questions = request.question_definitions()
        item = controls_to_kwargs(request.controls)
        min_confidence = item.pop("min_confidence", None)
        call: dict[str, Any] = {}
        if min_confidence is not None:
            call["min_confidence"] = min_confidence
        batch_size = request.batch_size or self._planned_batch_size(request)
        if batch_size is not None:
            call["batch_size"] = batch_size
        if request.sort_by_length:
            call["sort_by_length"] = True
        requests = [dict(state=state, questions=questions, **item) for state in request.states]

        results, elapsed = self._run(lambda: router.predict_batch(requests, **call))
        decisions = tuple(decision_from_dict(result) for result in results)
        total = sum((d.usage for d in decisions), start=Usage())
        return BatchDecision(results=decisions, total_usage=total, inference_ms=elapsed)
