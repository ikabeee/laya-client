"""Map domain errors onto HTTP responses, in one place.

The body keeps FastAPI's ``{"detail": ...}`` shape, which is what Jev clients and ``laya-serve``
clients already parse, and adds a stable ``error`` code.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ...domain.errors import (
    EngineBusyError,
    EngineUnavailableError,
    InferenceFailedError,
    InvalidRequestError,
    LayaClientError,
    MissingStateError,
    ModelNotFoundError,
    PayloadTooLargeError,
    PresetNotFoundError,
)

_log = logging.getLogger("laya_client.http")

# Most specific first: the first matching class wins.
_STATUS: tuple[tuple[type[LayaClientError], int, str], ...] = (
    (MissingStateError, 400, "missing_state"),
    (ModelNotFoundError, 422, "model_not_found"),
    (InvalidRequestError, 422, "invalid_request"),
    (PayloadTooLargeError, 413, "payload_too_large"),
    (PresetNotFoundError, 404, "preset_not_found"),
    (EngineBusyError, 503, "server_busy"),
    (EngineUnavailableError, 503, "engine_unavailable"),
    (InferenceFailedError, 500, "inference_failed"),
)


def error_response(status: int, code: str, detail: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": detail, "error": code}, headers=headers)


async def _handle_domain_error(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, LayaClientError)
    for cls, status, code in _STATUS:
        if isinstance(exc, cls):
            headers = {"Retry-After": "1"} if isinstance(exc, EngineBusyError) else None
            if isinstance(exc, EngineUnavailableError):
                headers = {"Retry-After": "10"}
            return error_response(status, code, exc.message, headers)
    _log.error("unmapped domain error %s: %s", type(exc).__name__, exc.message)
    return error_response(500, "internal_error", "internal error")


async def _handle_validation_error(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    # FastAPI's list-shaped detail, minus the echoed input: a rejected body can be megabytes long.
    errors = [{k: v for k, v in error.items() if k not in ("input", "ctx", "url")} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors), "error": "validation_error"})


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(LayaClientError, _handle_domain_error)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
