"""Usage aggregates for the current principal (from ``agent_runs``).

Powers the Dashboard stats + Settings → Billing usage. Scoped to the user when
authenticated, else the firm. Includes hidden conversations (the cost was still
incurred). All from the audit log we already write — no new tables.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import false, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.principal import ANONYMOUS_FIRM
from src.models.agent_run import AgentRun


class UsageRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def summary(
        self, *, firm_id: str, user_id: uuid.UUID | None, client_key: str | None = None
    ) -> dict[str, Any]:
        if user_id is not None:
            scope = AgentRun.user_id == user_id
        elif firm_id == ANONYMOUS_FIRM:
            # Guests share the anonymous firm → isolate per browser via client_key.
            scope = AgentRun.client_key == client_key if client_key else false()
        else:
            scope = AgentRun.firm_id == firm_id

        row = (
            await self._session.execute(
                select(
                    func.count(func.distinct(AgentRun.session_id)).label("conversations"),
                    func.count().label("runs"),
                    func.coalesce(
                        func.sum(func.coalesce(func.jsonb_array_length(AgentRun.tool_trace), 0)), 0
                    ).label("tool_calls"),
                    func.coalesce(func.sum(AgentRun.input_tokens), 0).label("input_tokens"),
                    func.coalesce(func.sum(AgentRun.output_tokens), 0).label("output_tokens"),
                    func.coalesce(func.sum(AgentRun.cost_usd), 0).label("cost_usd"),
                    func.count()
                    .filter(AgentRun.created_at > func.now() - func.make_interval(0, 0, 0, 7))
                    .label("runs_7d"),
                ).where(scope)
            )
        ).one()

        return {
            "conversations": int(row.conversations or 0),
            "runs": int(row.runs or 0),
            "tool_calls": int(row.tool_calls or 0),
            "input_tokens": int(row.input_tokens or 0),
            "output_tokens": int(row.output_tokens or 0),
            "cost_usd": float(row.cost_usd or 0),
            "runs_7d": int(row.runs_7d or 0),
        }
