"""REST interface built on FastAPI, documented with OpenAPI and rendered by Scalar."""

from .app import create_app

__all__ = ["create_app"]
