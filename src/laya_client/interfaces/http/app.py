"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from ... import __version__
from ...container import Container, build_container
from ...infrastructure.config import Settings, get_settings
from .dependencies import Admission
from .docs import DESCRIPTION, TAGS, mount_scalar
from .errors import register_error_handlers
from .middleware import BodySizeLimitMiddleware, RequestIdMiddleware
from .routers import health, models, presets, systemone

_log = logging.getLogger("laya_client")


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    """Build the app. Pass a ``container`` to inject test doubles; otherwise one is built from settings."""
    settings = settings or (container.settings if container else get_settings())
    container = container or build_container(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # Load checkpoints in the background: the server answers /health (and serves its docs)
        # while weights download, and /ready flips to 200 once the engine can answer.
        warmup = None
        if settings.preload:

            async def _warm() -> None:
                try:
                    await asyncio.to_thread(container.engine.warmup)
                except Exception:  # noqa: BLE001 -- reported through /ready and the next request
                    _log.exception("engine warm-up failed; it will be retried on the next request")

            warmup = asyncio.create_task(_warm())
        try:
            yield
        finally:
            if warmup is not None and not warmup.done():
                warmup.cancel()
            await asyncio.to_thread(container.engine.shutdown)

    app = FastAPI(
        title="Laya Client API",
        version=__version__,
        summary="Self-hosted System-1 decisions, wire-compatible with Jev's /v1/systemone.",
        description=DESCRIPTION,
        openapi_tags=TAGS,
        root_path=settings.root_path,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )
    app.state.container = container
    app.state.admission = Admission(settings.max_concurrent)
    app.state.api_keys = [key.encode("utf-8", "surrogateescape") for key in settings.api_keys]

    register_error_handlers(app)
    for module in (systemone, presets, models, health):
        app.include_router(module.router)

    if settings.docs_enabled:
        mount_scalar(app)

        @app.get("/", include_in_schema=False)
        async def root() -> RedirectResponse:
            return RedirectResponse(url=app.root_path + "/docs")

    if settings.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origin_list,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
            expose_headers=["X-Request-ID", "X-Inference-Time-Ms", "Server-Timing", "Retry-After"],
        )
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_body_bytes)
    app.add_middleware(RequestIdMiddleware)
    return app
