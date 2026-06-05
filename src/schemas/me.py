"""Schemas for the ``/api/v1/me`` account surface (profile + preferences)."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field


class UserRead(BaseModel):
    """The signed-in user's identity. ``null`` on the dev/anonymous path."""

    id: uuid.UUID
    email: str | None = None
    full_name: str | None = None


class MeRead(BaseModel):
    """Current principal: firm, role, identity, and preferences."""

    firm_id: str
    role: str | None = None
    is_anonymous: bool
    user: UserRead | None = None
    preferences: dict = Field(default_factory=dict)


class PreferencesUpdate(BaseModel):
    """PATCH body — keys here are merged over the existing preference blob."""

    preferences: dict


class PreferencesRead(BaseModel):
    preferences: dict


class UsageSummary(BaseModel):
    """Aggregate usage for the current user (from agent_runs)."""

    conversations: int
    runs: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    runs_7d: int
