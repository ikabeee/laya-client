"""Liveness and readiness probes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials

from .... import __version__
from ....container import Container
from ..dependencies import get_container
from ..presenters import readiness_payload
from ..schemas.responses import HealthResponse, ReadinessResponse
from ..security import is_authorized, optional_credentials

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description=(
        "Always open, so container and load-balancer probes need no credential. Callers that "
        "authenticate (or every caller, when no API key is set) also get engine details."
    ),
)
async def health(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(optional_credentials),
    container: Container = Depends(get_container),
) -> dict:
    if not is_authorized(request, credentials):
        return {"status": "ok"}
    status = container.get_health.execute()
    return {"status": "ok", "version": __version__, **readiness_payload(status)}


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse, "description": "The engine is still loading or failed."}},
    summary="Readiness probe",
    description="200 once the engine can answer requests; 503 while checkpoints are still loading.",
)
async def ready(container: Container = Depends(get_container)) -> JSONResponse:
    status = container.get_health.execute()
    payload = readiness_payload(status)
    if not status.ready:
        # Keep the reason out of an unauthenticated probe: it can name host paths or hardware.
        payload["details"] = {}
    return JSONResponse(payload, status_code=200 if status.ready else 503)
