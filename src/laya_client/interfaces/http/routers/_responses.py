"""Error responses shared by the OpenAPI document of several routes."""

from __future__ import annotations

from typing import Any

from ..schemas.responses import ErrorResponse

AUTH = {401: {"model": ErrorResponse, "description": "Missing or invalid bearer token."}}
INFERENCE: dict[int | str, dict[str, Any]] = {
    **AUTH,
    400: {"model": ErrorResponse, "description": "No `state` was sent."},
    413: {"model": ErrorResponse, "description": "A size limit was exceeded (body, state, questions or options)."},
    422: {"model": ErrorResponse, "description": "The request is invalid (question definition, model, controls)."},
    500: {"model": ErrorResponse, "description": "Inference failed; details are in the server log only."},
    503: {"model": ErrorResponse, "description": "Server busy or engine still starting. Honour `Retry-After`."},
}
