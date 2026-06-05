"""Daily message rate limiting — configurable per tier.

Caps live in ``config/rate_limits.yml`` (tweak the numbers, restart). A "message"
is one ``POST /api/v1/chat/run``. We count today's ``agent_runs`` for the caller:

  * signed-in   → by ``user_id``           cap = tiers[firm.subscription_tier] or default_signed_in
  * anonymous   → by ``client_key`` (guest id / IP)   cap = guest_daily

Counting the audit log we already write means no extra table and no double-count
on a rejected message (a blocked attempt never creates a run row).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import yaml
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.models.agent_run import AgentRun

_DEFAULTS = {"enabled": True, "guest_daily": 10, "default_signed_in": 50, "tiers": {}}


@lru_cache(maxsize=1)
def _config() -> dict:
    path = Path(getattr(settings, "RATE_LIMITS_PATH", "config/rate_limits.yml"))
    if not path.exists():
        return dict(_DEFAULTS)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {**_DEFAULTS, **data}


def is_enabled() -> bool:
    return bool(_config().get("enabled", True))


def cap_for(*, is_anonymous: bool, tier: str | None) -> int:
    """The daily message cap for this caller."""
    cfg = _config()
    if is_anonymous:
        return int(cfg.get("guest_daily", 10))
    tiers = cfg.get("tiers") or {}
    if tier and tier in tiers:
        return int(tiers[tier])
    return int(cfg.get("default_signed_in", 50))


def _day_start() -> datetime:
    return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


async def used_today(
    session: AsyncSession, *, user_id: uuid.UUID | None, client_key: str | None
) -> int:
    """How many messages the caller has sent since 00:00 UTC today."""
    if user_id is not None:
        scope = AgentRun.user_id == user_id
    elif client_key:
        scope = AgentRun.client_key == client_key
    else:
        return 0  # unidentifiable caller → don't block (shouldn't happen)
    count = await session.scalar(
        select(func.count()).where(scope, AgentRun.created_at >= _day_start())
    )
    return int(count or 0)
