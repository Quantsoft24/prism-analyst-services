"""Unit tests for the access policy (no DB needed).

Verifies the default-open behaviour and that ``require()`` enforces a level by
calling the produced dependency directly with a ``Principal``.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi import HTTPException

from src.auth import policy
from src.auth.principal import ANONYMOUS_FIRM, Principal


def test_unlisted_feature_defaults_to_anonymous():
    # The shipped config has `default: anonymous` and no feature overrides.
    assert policy.required_level("anything.not.listed") == policy.ANONYMOUS


async def test_require_anonymous_feature_is_open():
    dep = policy.require("open.feature")  # anonymous by default
    principal = Principal(firm_id="QUANTSOFT")
    assert await dep(principal=principal) is principal


async def test_require_authenticated_blocks_anonymous(monkeypatch):
    monkeypatch.setattr(policy, "required_level", lambda _f: policy.AUTHENTICATED)
    dep = policy.require("locked.feature")
    anon = Principal(firm_id=ANONYMOUS_FIRM, is_anonymous=True)
    with pytest.raises(HTTPException) as exc:
        await dep(principal=anon)
    assert exc.value.status_code == 401


async def test_require_authenticated_allows_signed_in(monkeypatch):
    monkeypatch.setattr(policy, "required_level", lambda _f: policy.AUTHENTICATED)
    dep = policy.require("locked.feature")
    user = Principal(firm_id="acme", user_id=uuid.uuid4(), role="member")
    assert await dep(principal=user) is user
