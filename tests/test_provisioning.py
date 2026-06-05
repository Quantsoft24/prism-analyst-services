"""Integration tests for JIT provisioning (DB-backed — needs the test Postgres).

ensure_principal must create the User + personal Firm + membership on first
login and be idempotent on subsequent logins (no duplicate rows)."""

from __future__ import annotations

from sqlalchemy import func, select

from src.auth.provisioning import ensure_principal
from src.auth.verifier import TokenClaims
from src.models.firm import Firm
from src.models.user import FirmMembership, User


async def test_first_login_provisions_user_firm_membership(db_session):
    claims = TokenClaims(sub="prov-1", email="new@example.com", full_name="New User")
    principal = await ensure_principal(db_session, claims)

    assert principal.user_id is not None
    assert principal.email == "new@example.com"
    assert principal.role == "owner"
    assert principal.is_anonymous is False

    user = await db_session.scalar(select(User).where(User.external_id == "prov-1"))
    assert user is not None and user.email == "new@example.com"
    firm = await db_session.scalar(select(Firm).where(Firm.slug == principal.firm_id))
    assert firm is not None
    membership = await db_session.scalar(
        select(FirmMembership).where(
            FirmMembership.user_id == user.id, FirmMembership.firm_id == firm.id
        )
    )
    assert membership is not None and membership.role == "owner"


async def test_second_login_is_idempotent(db_session):
    claims = TokenClaims(sub="prov-2", email="repeat@example.com", full_name="Repeat")
    p1 = await ensure_principal(db_session, claims)
    p2 = await ensure_principal(db_session, claims)

    assert p1.user_id == p2.user_id
    assert p1.firm_id == p2.firm_id
    users = await db_session.scalar(
        select(func.count()).select_from(User).where(User.external_id == "prov-2")
    )
    assert users == 1
    memberships = await db_session.scalar(
        select(func.count()).select_from(FirmMembership).where(FirmMembership.user_id == p1.user_id)
    )
    assert memberships == 1


async def test_existing_email_gets_linked(db_session):
    """A pre-existing user (no external_id) is linked, not duplicated."""
    existing = User(email="known@example.com", full_name="Known")
    db_session.add(existing)
    await db_session.flush()

    claims = TokenClaims(sub="prov-3", email="known@example.com", full_name="Known")
    principal = await ensure_principal(db_session, claims)

    assert principal.user_id == existing.id
    await db_session.refresh(existing)
    assert existing.external_id == "prov-3"
