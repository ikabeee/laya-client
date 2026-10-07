"""The inference engine: an in-process ``laya.Router``.

``start()`` runs before the server accepts traffic and either leaves a fully working engine behind or
raises ``EngineUnavailableError`` with the reason. It checks the runtime, loads the checkpoints, verifies
they sit on the requested device, and runs one prediction per checkpoint. There is no lazy loading and
no fallback: a deployment that cannot answer with the real model does not start.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
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
from .runtime import RuntimeInfo, check_runtime

_log = logging.getLogger("laya_client.engine")

# Row width assumed when a batch does not set max_len: the English checkpoint's own window.
_BATCH_ROW_TOKENS_ASSUMED = 512

# The start-up prediction: proves tokenizer, weights and kernels all work on the chosen device.
_SMOKE_STATE = "Start-up check: I was charged twice for my order, please refund one payment."
_SMOKE_QUESTIONS = {"smoke_check": {"type": "noul", "instructions": "Does the text ask for a refund?"}}


@dataclass(frozen=True)
class LayaEngineConfig:
    device: str | None = None
    require_gpu: bool = False
    models: tuple[str, ...] = field(default_factory=tuple)
    threads: int | None = None
    auto_task: bool = False
    default_model: str | None = None
    max_loaded: int | None = None
    max_batch_tokens: int = 131_072


def _default_router_factory(**options: Any) -> Any:
    from laya import Router

    return Router(**options)


def _default_device_of(router: Any, name: str) -> str | None:
    """Where checkpoint ``name`` actually computes, read without side effects (``laya.mcp.device``)."""
    try:
        from laya.mcp.device import agent_device, router_agent
    except ImportError:
        return None
    return agent_device(router_agent(router, name))


class LayaRouterEngine(DecisionEngine):
    """Adapter from the ``DecisionEngine`` port to ``laya.Router``.

    One forward pass runs at a time (``_infer_lock``): that is what a single CPU or GPU wants, and
    it keeps memory bounded no matter how many HTTP requests are waiting.
    """

    def __init__(
        self,
        config: LayaEngineConfig,
        router: Any = None,
        *,
        runtime_check: Callable[..., RuntimeInfo] = check_runtime,
        router_factory: Callable[..., Any] = _default_router_factory,
        device_of: Callable[[Any, str], str | None] = _default_device_of,
    ) -> None:
        self._config = config
        self._router = router
        self._runtime: RuntimeInfo | None = None
        self._runtime_check = runtime_check
        self._router_factory = router_factory
        self._device_of = device_of
        self._infer_lock = threading.Lock()

    # -- lifecycle ----------------------------------------------------------------------------

    def start(self) -> None:
        if self._router is not None:
            return
        started = time.perf_counter()
        runtime = self._runtime_check(self._config.device, require_gpu=self._config.require_gpu)
        _log.info(
            "runtime: device=%s torch=%s cuda=%s accelerator=%s laya=%s",
            runtime.device,
            runtime.torch_version,
            runtime.torch_cuda or "none",
            runtime.accelerator or "none",
            runtime.laya_version,
        )
        for note in runtime.notes:
            _log.warning(note)

        if self._config.threads:
            import torch

            torch.set_num_threads(self._config.threads)

        names = list(self._config.models)
        router = self._load(runtime, names)
        self._verify_devices(router, runtime)
        self._smoke_test(router)

        self._runtime = runtime
        self._router = router
        _log.info(
            "Laya engine ready in %.1fs: %s on %s",
            time.perf_counter() - started,
            ", ".join(router.loaded or ()) or "no checkpoint",
            runtime.accelerator or runtime.device,
        )

    def _load(self, runtime: RuntimeInfo, names: list[str]) -> Any:
        options: dict[str, Any] = {"device": runtime.device, "auto_task_detection": self._config.auto_task}
        if self._config.max_loaded:
            options["max_loaded"] = self._config.max_loaded
        if self._config.default_model:
            options["default"] = self._config.default_model
        label = ", ".join(names) or "all checkpoints"
        _log.info("loading %s (the first start downloads the weights from Hugging Face)", label)
        try:
            router = self._router_factory(**options)
            router.preload(names or None)
        except LayaClientError:
            raise
        except Exception as error:
            raise EngineUnavailableError(
                "could not load %s: %s: %s. Check the model names in LAYA_MODELS, network access to "
                "huggingface.co, free disk space for the weights and free memory."
                % (label, type(error).__name__, error)
            ) from error
        if not router.loaded:
            raise EngineUnavailableError("no checkpoint is loaded after preloading %s" % label)
        return router

    def _verify_devices(self, router: Any, runtime: RuntimeInfo) -> None:
        if runtime.accelerator is None:
            return
        wanted = runtime.device.split(":", 1)[0]
        for name in router.loaded:
            actual = self._device_of(router, name)
            if actual is None:
                _log.warning("cannot read the device of checkpoint %s; skipping the placement check", name)
                continue
            if not str(actual).startswith(wanted):
                raise EngineUnavailableError(
                    "checkpoint %s was loaded on %s instead of %s (%s): Laya fell back from the GPU. "
                    "Check free GPU memory and the server log above for the reason."
                    % (name, actual, runtime.device, runtime.accelerator)
                )

    def _smoke_test(self, router: Any) -> None:
        for name in router.loaded:
            try:
                result = router.predict(_SMOKE_STATE, _SMOKE_QUESTIONS, model=name)
                p_true = float(result["answers"]["smoke_check"]["noul"])
            except Exception as error:
                raise EngineUnavailableError(
                    "checkpoint %s loaded but a test prediction failed: %s: %s" % (name, type(error).__name__, error)
                ) from error
            if not 0.0 <= p_true <= 1.0:
                raise EngineUnavailableError(
                    "checkpoint %s returned an invalid probability (%r) in the start-up check" % (name, p_true)
                )
            _log.info("checkpoint %s passed the start-up prediction (p=%.3f)", name, p_true)

    def shutdown(self) -> None:
        router = self._router
        if router is not None and hasattr(router, "unload"):
            try:
                router.unload()
            except Exception:  # noqa: BLE001 -- shutting down; nothing useful to do with it
                _log.exception("unload failed during shutdown")

    def status(self) -> EngineStatus:
        router = self._router
        runtime = self._runtime
        if router is None:
            return EngineStatus(engine="laya", ready=False, device=self._config.device, details={"state": "stopped"})
        details: dict[str, Any] = {"revisions": dict(getattr(router, "loaded_revisions", None) or {})}
        if runtime is not None:
            details["runtime"] = runtime.as_dict()
        return EngineStatus(
            engine="laya",
            ready=True,
            loaded=tuple(getattr(router, "loaded", None) or ()),
            device=runtime.device if runtime else self._config.device,
            details=details,
        )

    # -- inference ----------------------------------------------------------------------------

    def _get_router(self) -> Any:
        if self._router is None:
            raise EngineUnavailableError("the inference engine is not running")
        return self._router

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
