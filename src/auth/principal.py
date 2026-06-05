"""``Principal`` — the resolved identity for a request — and its dependency.

A ``Principal`` is what every authz decision reads: which firm (slug), which
user (if known), what role, and whether the caller is anonymous.

``get_current_principal`` is the single resolution path:
  * ``AUTH_ENABLED=false`` (P0/dev) → mirror today's ``get_current_firm_id``
    exactly: the firm comes from ``X-Dev-Firm`` or ``DEV_FIRM_ID``; the caller
    is treated as that known firm (not anonymous). **No behaviour change.**
  * ``AUTH_ENABLED=true`` → verify the bearer token via the configured
    ``TokenVerifier``. No/invalid token → an **anonymous** principal (the access
    policy then decides what's allowed). Valid token → an authenticated
    principal. (User-row provisioning + the real ``user_id`` land in P1.)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from fastapi import Header, HTTPException, status

from src.auth.verifier import get_verifier
from src.config import settings

# Sentinel firm slug for unauthenticated callers (never matches a real firm).
ANONYMOUS_FIRM = "__anonymous__"


@dataclass(frozen=True)
class Principal:
    """The identity behind a request."""

    firm_id: str                      # firm slug (canonical wire id everywhere)
    user_id: uuid.UUID | None = None  # None in dev / for anonymous callers
    role: str | None = None           # FirmMembership.role when known
    is_anonymous: bool = False
    email: str | None = None
    full_name: str | None = None

    @property
    def is_authenticated(self) -> bool:
        return not self.is_anonymous


def _bearer(authorization: str | None) -> str | None:
    """Extract the token from an ``Authorization: Bearer <token>`` header."""
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip() or None
    return None


async def get_current_principal(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_dev_firm: str | None = Header(default=None, alias="X-Dev-Firm"),
    authorization: str | None = Header(default=None),
) -> Principal:
    """Resolve the calling principal. See module docstring for the two paths."""
    if not settings.AUTH_ENABLED:
        # Dev stub — identical firm resolution to get_current_firm_id today.
        return Principal(firm_id=x_dev_firm or settings.DEV_FIRM_ID, is_anonymous=False)

    verifier = get_verifier()
    if verifier is None:
        # AUTH_ENABLED but no provider wired yet (pre-P1) — fail closed, not open.
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Real auth not yet wired — set AUTH_ENABLED=false for now.",
        )

    token = _bearer(authorization)
    claims = await verifier.verify(token) if token else None
    if claims is None:
        # No / invalid credential → anonymous; the access policy decides access.
        return Principal(firm_id=ANONYMOUS_FIRM, is_anonymous=True)

    # Valid token → JIT-provision (or look up) the user + personal firm and
    # return a fully-populated principal. Uses its own session (independent of
    # the request's get_session) so it commits the first-login rows cleanly.
    from src.auth.provisioning import ensure_principal
    from src.core.database import session_scope

    async with session_scope() as session:
        return await ensure_principal(session, claims)
