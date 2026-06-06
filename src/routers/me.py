"""``/api/v1/me`` — the signed-in user's account surface (profile + preferences).

Powers the Settings → Profile / Preferences screens (which are mock today). Uses
the provider-agnostic ``Principal`` so it works unchanged when real auth lands:

  * ``AUTH_ENABLED=false`` (dev): the principal is the dev firm with no user, so
    GET returns the firm + empty preferences, and PATCH is rejected (you can't
    persist per-user prefs without a user identity — that arrives with P1 auth).
  * ``AUTH_ENABLED=true``: GET returns the real user + their stored preferences;
    PATCH merges and persists them.

Access is gated via the configurable policy (``require("profile.*")``) — open by
default in dev, lockable before launch without touching this code.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.policy import require
from src.auth.principal import Principal, get_current_principal
from src.core.database import get_session
from src.models.user import User
from src.repositories.preferences_repo import PreferencesRepository
from src.repositories.usage_repo import UsageRepository
from src.schemas.me import MeRead, PreferencesRead, PreferencesUpdate, UsageSummary, UserRead

router = APIRouter(prefix="/me", tags=["Account"])


@router.get("", response_model=MeRead, summary="Current user: firm, role, profile, preferences")
async def read_me(
    principal: Annotated[Principal, Depends(require("profile.read"))],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> MeRead:
    user: UserRead | None = None
    preferences: dict = {}
    if principal.user_id is not None:
        db_user = await session.get(User, principal.user_id)
        if db_user is not None:
            user = UserRead(id=db_user.id, email=db_user.email, full_name=db_user.full_name)
        preferences = await PreferencesRepository(session).get(principal.user_id)
    return MeRead(
        firm_id=principal.firm_id,
        role=principal.role,
        is_anonymous=principal.is_anonymous,
        user=user,
        preferences=preferences,
    )


@router.patch(
    "/preferences",
    response_model=PreferencesRead,
    summary="Merge-update the current user's preferences",
)
async def update_preferences(
    body: PreferencesUpdate,
    principal: Annotated[Principal, Depends(require("profile.write"))],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> PreferencesRead:
    if principal.user_id is None:
        # No user identity (dev stub / anonymous) → nothing to persist against.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sign in to save preferences.",
        )
    repo = PreferencesRepository(session)
    current = await repo.get(principal.user_id)
    merged = {**current, **body.preferences}
    saved = await repo.upsert(principal.user_id, merged)
    return PreferencesRead(preferences=saved)


@router.get("/usage", response_model=UsageSummary, summary="The user's aggregate usage")
async def read_usage(
    request: Request,
    principal: Annotated[Principal, Depends(get_current_principal)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> UsageSummary:
    # Guests share the anonymous firm → isolate per browser by X-Guest-Id.
    client_key = (
        request.headers.get("X-Guest-Id") or (request.client.host if request.client else None)
        if principal.is_anonymous
        else None
    )
    data = await UsageRepository(session).summary(
        firm_id=principal.firm_id, user_id=principal.user_id, client_key=client_key
    )
    return UsageSummary(**data)
