"""ASGI middleware: request ids and a cap on request body size."""

from __future__ import annotations

import json
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class _BodyTooLarge(Exception):
    pass


async def _send_json(send: Send, status: int, body: dict, extra_headers: list | None = None) -> None:
    payload = json.dumps(body).encode("utf-8")
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())]
    await send({"type": "http.response.start", "status": status, "headers": headers + (extra_headers or [])})
    await send({"type": "http.response.body", "body": payload})


class BodySizeLimitMiddleware:
    """Refuse bodies over ``max_bytes`` with 413, checking both Content-Length and the streamed bytes."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        too_large = {"detail": "request body too large", "error": "payload_too_large"}
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = 0
                if declared > self.max_bytes:
                    await _send_json(send, 413, too_large)
                    return

        received = 0
        started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLarge:
            if not started:
                await _send_json(send, 413, too_large)


class RequestIdMiddleware:
    """Echo the caller's ``X-Request-ID`` (or mint one) on every response."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = None
        for name, value in scope.get("headers", []):
            if name == b"x-request-id":
                request_id = value.decode("latin-1")[:128]
                break
        request_id = request_id or uuid.uuid4().hex

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                message.setdefault("headers", [])
                message["headers"] = list(message["headers"]) + [(b"x-request-id", request_id.encode("latin-1"))]
            await send(message)

        await self.app(scope, receive, send_with_id)
