"""JIT provisioning — turn verified token claims into PRISM rows.

On a user's first authenticated request we ensure their ``User`` exists and that
they have a workspace. For the ≤100-user pilot every user gets a **personal,
single-member firm** (Supabase has no org concept yet); team/org invites are a
later phase but the schema (``firms`` / ``firm_memberships``) already supports
them. Steady-state cost is one SELECT (user found by ``external_id``); writes
happen only on first-ever login.

Org-based provisioning (map an IdP org → existing ``Firm``) slots in here when
enterprise SSO lands — same function, extra branch on ``claims.org_slug``.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.principal import Principal
from src.auth.verifier import TokenClaims
from src.models.firm import Firm
from src.models.user import FirmMembership, User


def _personal_slug(user: User) -> str:
    """Stable, unique slug for a user's personal firm (≤64 chars)."""
    return f"u-{user.id.hex[:16]}"


async def ensure_principal(session: AsyncSession, claims: TokenClaims) -> Principal:
    """Resolve (creating if needed) the User + personal Firm + membership for a
    verified token, and return the request ``Principal``."""
    # 1) User — by provider id first, then link an existing email, else create.
    user = await session.scalar(select(User).where(User.external_id == claims.sub))
    if user is None and claims.email:
        user = await session.scalar(select(User).where(User.email == claims.email))
        if user is not None and not user.external_id:
            user.external_id = claims.sub  # link the provider id to a known email
    if user is None:
        user = User(
            email=claims.email or f"{claims.sub}@users.noreply.prism",
            full_name=claims.full_name or "",
            external_id=claims.sub,
        )
        session.add(user)
        await session.flush()  # populate user.id
    else:
        # Keep the PRISM mirror in sync with the IdP on each login — so a name
        # change made via Supabase shows up in /me on the next request.
        if claims.full_name and user.full_name != claims.full_name:
            user.full_name = claims.full_name
        if claims.email and user.email != claims.email:
            user.email = claims.email

    # 2) Personal firm — if the token carried an org we'd resolve that instead.
    slug = claims.org_slug or _personal_slug(user)
    firm = await session.scalar(select(Firm).where(Firm.slug == slug))
    if firm is None:
        firm = Firm(slug=slug, name=(user.full_name or user.email or slug))
        session.add(firm)
        await session.flush()

    # 3) Membership — owner of their personal firm (or the token's org role).
    role = claims.org_role or "owner"
    membership = await session.scalar(
        select(FirmMembership).where(
            FirmMembership.firm_id == firm.id, FirmMembership.user_id == user.id
        )
    )
    if membership is None:
        session.add(FirmMembership(firm_id=firm.id, user_id=user.id, role=role))

    await session.commit()
    return Principal(
        firm_id=slug,
        user_id=user.id,
        role=role,
        is_anonymous=False,
        email=user.email,
        full_name=user.full_name,
    )
