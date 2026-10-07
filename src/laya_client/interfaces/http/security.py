"""Optional bearer-token authentication, compatible with Jev's ``Authorization: Bearer <key>``."""

from __future__ import annotations

import hmac

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

bearer_scheme = HTTPBearer(
    auto_error=False,
    scheme_name="BearerAuth",
    description="Send `Authorization: Bearer <LAYA_API_KEY>`. Only enforced when the server sets `LAYA_API_KEY`.",
)


def is_authorized(request: Request, credentials: HTTPAuthorizationCredentials | None) -> bool:
    keys: list[bytes] = request.app.state.api_keys
    if not keys:
        return True
    if credentials is None or credentials.scheme.lower() != "bearer":
        return False
    supplied = credentials.credentials.encode("utf-8", "surrogateescape")
    # Check every key so the time taken does not reveal which one (if any) matched.
    matched = False
    for key in keys:
        matched |= hmac.compare_digest(supplied, key)
    return matched


async def optional_credentials(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> HTTPAuthorizationCredentials | None:
    return credentials


async def require_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> None:
    if not is_authorized(request, credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
