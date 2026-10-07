"""API reference: the OpenAPI document rendered by Scalar at ``/docs``."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from scalar_fastapi import get_scalar_api_reference

DESCRIPTION = """
Self-hosted [Laya](https://github.com/NandhaKishorM/laya) System-1 decision engine behind a REST API
that is **wire-compatible with Jev's `/v1/systemone`**. Point an existing Jev client at this server
and it keeps working; your data never leaves your infrastructure.

### How a decision works
Send a `state` (text or any JSON) and a set of typed `questions`. Each question is answered in a
single forward pass — no tokens are generated:

| type | answers with |
|---|---|
| `choice` | `choice` (the most probable label) and `probabilities` per label |
| `score` | `score` (expected rubric index, may fall between levels), `probabilities`, `legend` |
| `noul` | `noul`, the probability the statement is true |

### Authentication
When the server sets `LAYA_API_KEY`, send `Authorization: Bearer <key>` (the same header Jev uses).
`/health` and `/ready` stay open for probes.
"""

TAGS = [
    {"name": "SystemOne", "description": "Jev-compatible decision endpoints."},
    {"name": "Presets", "description": "Ready-made question sets: triage, email, guard, moderation, router."},
    {"name": "Models", "description": "Checkpoints a request may pin with `model`."},
    {"name": "Health", "description": "Liveness and readiness probes."},
]


def mount_scalar(app: FastAPI, path: str = "/docs") -> None:
    @app.get(path, include_in_schema=False)
    async def scalar_reference() -> HTMLResponse:
        return get_scalar_api_reference(
            openapi_url=app.root_path + app.openapi_url if app.openapi_url else None,
            title=app.title,
            # Self-hosted means self-contained: no usage pings from the docs page.
            telemetry=False,
            hide_client_button=False,
        )
