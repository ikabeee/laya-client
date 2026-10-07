"""``/v1/models``: the checkpoints a request may pin."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ....container import Container
from ..dependencies import get_container
from ..presenters import model_payload
from ..schemas.responses import ModelListResponse
from ..security import require_api_key
from ._responses import AUTH

router = APIRouter(prefix="/v1/models", tags=["Models"], dependencies=[Depends(require_api_key)])


@router.get("", response_model=ModelListResponse, responses=AUTH, summary="List checkpoints and aliases")
async def list_models(container: Container = Depends(get_container)) -> dict:
    return {"object": "list", "data": [model_payload(m) for m in container.list_models.execute()]}
