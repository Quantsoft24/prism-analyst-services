"""Auth dependencies — Phase 1 W3 will wire real Clerk/Supabase JWT verification.

For Slice 1, ``get_current_firm_id`` is a dev-mode stub that resolves the
calling firm in this priority order:

1. ``X-API-Key`` header — reserved for programmatic third-party consumers
   (real validation lands when we issue API keys in Phase 4).
2. ``X-Dev-Firm`` header — dev/test override.
3. ``settings.DEV_FIRM_ID`` — the bootstrapped ``QUANTSOFT`` firm.

The interface returned (``str`` firm slug) is the same one the production
auth will return, so router code does not change when we wire Clerk.
"""

from __future__ import annotations

from fastapi import Header, HTTPException, status

from src.config import settings


async def get_current_firm_id(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_dev_firm: str | None = Header(default=None, alias="X-Dev-Firm"),
) -> str:
    """Resolve the firm slug for the current request.

    Returns the firm slug (e.g. ``"QUANTSOFT"``); routers use this for
    tenant-scoping every query. In Slice 1 there is no real auth, so we
    accept a header override and fall back to the dev firm.
    """
    if settings.AUTH_ENABLED:
        # Real Clerk/Supabase JWT verification lands in Phase 1 W3.
        # For now, refuse the request if AUTH_ENABLED is set but no provider
        # is wired up — fail closed rather than open.
        if not x_api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required.",
            )
        # TODO(Phase 1 W3): validate JWT / API key against the issuer.
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Real auth not yet wired — set AUTH_ENABLED=false for now.",
        )

    return x_dev_firm or settings.DEV_FIRM_ID
