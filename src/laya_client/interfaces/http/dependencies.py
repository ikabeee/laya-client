"""FastAPI dependencies: access to the container and the admission limit."""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Request

from ...container import Container
from ...domain.errors import EngineBusyError


class Admission:
    """Non-blocking bound on in-flight inference requests.

    Excess load is refused (503 + ``Retry-After``) rather than queued, so the request bodies held in
    memory stay bounded however many clients connect. Only touched from the event loop thread.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.in_flight = 0

    def try_acquire(self) -> bool:
        if self.in_flight >= self.limit:
            return False
        self.in_flight += 1
        return True

    def release(self) -> None:
        self.in_flight = max(0, self.in_flight - 1)


def get_container(request: Request) -> Container:
    return request.app.state.container


async def admit(request: Request) -> AsyncIterator[None]:
    admission: Admission = request.app.state.admission
    if not admission.try_acquire():
        raise EngineBusyError("server busy, try again later")
    try:
        yield
    finally:
        admission.release()
