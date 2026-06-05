"""Access policy — one configurable, server-enforced source of truth.

`config/access_policy.yml` maps a *feature key* to the level required to use it:

    default: anonymous          # level for any feature not listed
    features:
      profile.write: authenticated

Levels: ``anonymous`` | ``authenticated`` | ``entitlement:<key>`` (the last is
billing-gated; enforced in P4 — for now it just requires authentication).

`require("feature.key")` is a FastAPI dependency that enforces this and returns
the ``Principal``. Routers depend on it instead of hard-coding ``if`` checks, so
locking a feature down before launch is a one-line config change (NOT a refactor).

⚠️ DEV DEFAULT is ``anonymous`` for everything (per the team's call). The gating
matrix MUST be set before ``AUTH_ENABLED=true`` ships to prod — see
``final_docs/12_AUTH_USER_PROFILES_BILLING.md`` §8.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import yaml
from fastapi import Depends, HTTPException, status

from src.auth.principal import Principal, get_current_principal
from src.config import settings

logger = logging.getLogger(__name__)

ANONYMOUS = "anonymous"
AUTHENTICATED = "authenticated"
_ENTITLEMENT_PREFIX = "entitlement:"


@lru_cache(maxsize=1)
def _policy() -> dict:
    """Load + cache the access policy. Missing file → safe default (all open in
    dev). The cache is process-lifetime; the file isn't hot-reloaded."""
    path = Path(getattr(settings, "ACCESS_POLICY_PATH", "config/access_policy.yml"))
    if not path.exists():
        logger.info("No access_policy.yml at %s — defaulting all features to '%s'.", path, ANONYMOUS)
        return {"default": ANONYMOUS, "features": {}}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {
        "default": data.get("default", ANONYMOUS),
        "features": data.get("features") or {},
    }


def required_level(feature: str) -> str:
    """The access level a feature requires (its override, else the default)."""
    pol = _policy()
    return pol["features"].get(feature, pol["default"])


def require(feature: str):
    """FastAPI dependency factory: enforce ``feature``'s required level and
    return the ``Principal``. ``anonymous`` features are open to everyone."""

    async def _dep(principal: Principal = Depends(get_current_principal)) -> Principal:
        level = required_level(feature)
        if level == ANONYMOUS:
            return principal
        if principal.is_anonymous:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Sign in to use this feature.",
            )
        # entitlement:<key> → real quota/plan checks land with billing (P4).
        # Until then, an authenticated principal satisfies it.
        return principal

    return _dep
