"""Domain errors.

Each error names a business failure, not a transport one. The HTTP interface maps them onto status
codes in one place (``interfaces/http/errors.py``), so the inner layers never know HTTP exists.
"""

from __future__ import annotations


class LayaClientError(Exception):
    """Base class for every error this service raises on purpose."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidRequestError(LayaClientError):
    """The request is well-formed JSON but asks for something the engine cannot answer."""


class MissingStateError(InvalidRequestError):
    """No state was sent, so there is nothing to decide about."""


class PayloadTooLargeError(LayaClientError):
    """The request exceeds one of the deployment's size limits."""


class ModelNotFoundError(InvalidRequestError):
    """The caller pinned a checkpoint this deployment cannot load."""


class PresetNotFoundError(LayaClientError):
    """No preset is registered under the requested name."""


class EngineBusyError(LayaClientError):
    """Too many requests are in flight; the caller should retry shortly."""


class EngineUnavailableError(LayaClientError):
    """The engine could not be started (missing dependency, failed model download ...)."""


class InferenceFailedError(LayaClientError):
    """The engine failed for a reason the caller cannot fix. Details stay in the server log."""
