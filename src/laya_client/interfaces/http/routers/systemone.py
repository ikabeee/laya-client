"""``/v1/systemone``: the Jev-compatible decision endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from ....container import Container
from ..dependencies import admit, get_container
from ..mappers import to_batch_request, to_decision_request
from ..presenters import batch_payload, decision_payload
from ..schemas.requests import SystemOneBatchRequest, SystemOneRequest
from ..schemas.responses import BatchDecisionResponse, DecisionResponse
from ..security import require_api_key
from ._responses import INFERENCE

router = APIRouter(prefix="/v1/systemone", tags=["SystemOne"], dependencies=[Depends(require_api_key)])


def _timing_headers(inference_ms: float | None) -> dict[str, str]:
    if inference_ms is None:
        return {}
    return {"Server-Timing": "inference;dur=%.2f" % inference_ms, "X-Inference-Time-Ms": "%.2f" % inference_ms}


@router.post(
    "",
    response_model=DecisionResponse,
    responses=INFERENCE,
    summary="Answer typed questions about one state",
    description=(
        "Every question (`choice`, `score`, `noul`) is answered in a single forward pass. Wire-compatible "
        "with Jev's `POST /v1/systemone`: point an existing Jev client's base URL here."
    ),
    dependencies=[Depends(admit)],
)
async def systemone(body: SystemOneRequest, container: Container = Depends(get_container)) -> JSONResponse:
    decision = await run_in_threadpool(container.predict_decision.execute, to_decision_request(body))
    return JSONResponse(
        decision_payload(decision, strict=container.settings.jev_strict),
        headers=_timing_headers(decision.inference_ms),
    )


@router.post(
    "/batch",
    response_model=BatchDecisionResponse,
    responses=INFERENCE,
    summary="Answer the same questions about several states",
    description="States share forward passes; `results` come back in the order the states were sent.",
    dependencies=[Depends(admit)],
)
async def systemone_batch(body: SystemOneBatchRequest, container: Container = Depends(get_container)) -> JSONResponse:
    batch = await run_in_threadpool(container.predict_batch.execute, to_batch_request(body))
    return JSONResponse(
        batch_payload(batch, strict=container.settings.jev_strict),
        headers=_timing_headers(batch.inference_ms),
    )
