"""Entry point: ``laya-client`` (or ``python -m laya_client``) starts the server with uvicorn."""

from __future__ import annotations

import logging

import uvicorn

from .infrastructure.config import get_settings


def run() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper() if settings.log_level != "trace" else "DEBUG")
    # One worker on purpose: each worker process would hold its own copy of the model weights.
    uvicorn.run(
        "laya_client.interfaces.http.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        proxy_headers=True,
        workers=1,
    )


if __name__ == "__main__":
    run()
