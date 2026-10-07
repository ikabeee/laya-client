"""``/v1/presets``: ready-made question sets for common workflows."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Path
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from ....container import Container
from ..dependencies import admit, get_container
from ..mappers import to_controls
from ..presenters import decision_payload, preset_detail_payload, preset_summary_payload
from ..schemas.requests import PresetPredictRequest
from ..schemas.responses import DecisionResponse, ErrorResponse, PresetDetail, PresetListResponse
from ..security import require_api_key
from ._responses import AUTH, INFERENCE

router = APIRouter(prefix="/v1/presets", tags=["Presets"], dependencies=[Depends(require_api_key)])

PresetName = Path(description="Preset name: triage, email, guard, moderation or router.", examples=["triage"])
NOT_FOUND = {404: {"model": ErrorResponse, "description": "No preset with that name."}}


@router.get("", response_model=PresetListResponse, responses=AUTH, summary="List presets")
async def list_presets(container: Container = Depends(get_container)) -> dict:
    return {"object": "list", "data": [preset_summary_payload(p) for p in container.list_presets.execute()]}


@router.get(
    "/{name}",
    response_model=PresetDetail,
    responses={**AUTH, **NOT_FOUND},
    summary="Get a preset's questions",
    description="The `questions` object can be sent as-is to `POST /v1/systemone`.",
)
async def get_preset(name: str = PresetName, container: Container = Depends(get_container)) -> dict:
    return preset_detail_payload(container.get_preset.execute(name))


@router.post(
    "/{name}",
    response_model=DecisionResponse,
    responses={**INFERENCE, **NOT_FOUND},
    summary="Answer a preset's questions about a state",
    dependencies=[Depends(admit)],
)
async def predict_with_preset(
    body: PresetPredictRequest,
    name: str = PresetName,
    container: Container = Depends(get_container),
) -> JSONResponse:
    decision = await run_in_threadpool(container.predict_with_preset.execute, name, body.state, to_controls(body))
    return JSONResponse(decision_payload(decision, strict=container.settings.jev_strict))
