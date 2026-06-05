"""Integration tests for ``/api/v1/me`` (DB-backed — needs the test Postgres).

Covers the dev path (no user) and the authenticated round-trip (preferences
merge) by overriding ``get_current_principal`` with a seeded user.
"""

from __future__ import annotations

from src.auth.principal import Principal, get_current_principal
from src.main import app
from src.models.user import User


async def test_me_dev_path_no_user(client):
    """Dev stub (AUTH_ENABLED=false): firm resolves, no user, empty prefs."""
    resp = await client.get("/api/v1/me", headers={"X-Dev-Firm": "QUANTSOFT"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["firm_id"] == "QUANTSOFT"
    assert body["is_anonymous"] is False
    assert body["user"] is None
    assert body["preferences"] == {}


async def test_patch_preferences_requires_user(client):
    """Without a user identity (dev/anonymous), preferences can't be persisted."""
    resp = await client.patch(
        "/api/v1/me/preferences",
        json={"preferences": {"theme": "dark"}},
        headers={"X-Dev-Firm": "QUANTSOFT"},
    )
    assert resp.status_code == 401


async def test_me_roundtrip_with_user(client, db_session):
    """With a real user, GET returns identity + PATCH merges preferences."""
    user = User(email="analyst@example.com", full_name="Test Analyst")
    db_session.add(user)
    await db_session.flush()
    uid = user.id

    app.dependency_overrides[get_current_principal] = lambda: Principal(
        firm_id="acme", user_id=uid, role="member"
    )
    try:
        me = await client.get("/api/v1/me")
        assert me.status_code == 200
        assert me.json()["user"]["email"] == "analyst@example.com"

        first = await client.patch(
            "/api/v1/me/preferences", json={"preferences": {"theme": "dark"}}
        )
        assert first.status_code == 200
        assert first.json()["preferences"] == {"theme": "dark"}

        # PATCH merges (doesn't replace).
        second = await client.patch(
            "/api/v1/me/preferences", json={"preferences": {"model": "pro"}}
        )
        assert second.json()["preferences"] == {"theme": "dark", "model": "pro"}

        again = await client.get("/api/v1/me")
        assert again.json()["preferences"] == {"theme": "dark", "model": "pro"}
    finally:
        app.dependency_overrides.pop(get_current_principal, None)
