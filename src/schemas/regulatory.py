"""Pydantic schemas for the ``/api/v1/regulatory`` API (Regulatory Lens).

Backed by the read-only SEBI DB (``content`` + ``weekly_summaries`` +
``insight_feed``). Mirrors the ``ai_tags`` JSON shape
(``intent / topics / stakeholders / severity / action_required / deadlines``)
into a typed ``AiTags`` object so the frontend never has to guess.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# The 13 active content types in the corpus (3 more exist in the enum but are
# dormant: CONSULTATION_PAPER, FAQ, SPEECH).
RegType = str  # kept loose — the enum is owned by the external DB
Severity = str  # "High" | "Medium" | "Low" | None


class AiTags(BaseModel):
    """Parsed ``ai_tags`` JSON. All fields optional — older rows may lack tags."""

    intent: str | None = None
    topics: list[str] = Field(default_factory=list)
    stakeholders: list[str] = Field(default_factory=list)
    severity: Severity | None = None
    action_required: bool = False
    deadlines: list[str] = Field(default_factory=list)


class RegDocSummary(BaseModel):
    """Compact document row for feeds / lists / search results."""

    id: int
    type: str
    sub_type: str | None = None
    title: str
    date: datetime | None = None
    summary: str | None = None
    sebi_id: str | None = None
    sebi_department: str | None = None
    ai_tags: AiTags = Field(default_factory=AiTags)


class RegDocDetail(RegDocSummary):
    """Full document — adds the body text, source link, and all metadata."""

    sebi_url: str | None = None
    sebi_section: str | None = None
    sebi_sub_section: str | None = None
    sebi_info_for: str | None = None
    meeting_date: datetime | None = None
    extracted_text: str | None = None
    language: str | None = None
    related_content_id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class FeedResponse(BaseModel):
    """Paginated feed / search response."""

    items: list[RegDocSummary]
    total: int
    page: int
    limit: int
    total_pages: int


class TypeCount(BaseModel):
    type: str
    count: int


class SeverityCount(BaseModel):
    severity: str
    count: int


class IntentCount(BaseModel):
    intent: str
    count: int


class TopicCount(BaseModel):
    topic: str
    count: int


class RegStats(BaseModel):
    """Dashboard headline stats."""

    total_documents: int
    this_week: int
    today: int
    action_required: int
    high_severity_week: int
    open_deadlines: int
    type_counts: list[TypeCount]
    severity_counts: list[SeverityCount]
    intent_counts: list[IntentCount]


class DeadlineItem(BaseModel):
    """One upcoming deadline derived from a document's ``ai_tags.deadlines``."""

    id: int
    type: str
    title: str
    date: datetime | None = None
    deadline: str  # YYYY-MM-DD
    severity: Severity | None = None
    intent: str | None = None


class DeadlinesResponse(BaseModel):
    items: list[DeadlineItem]
    total: int


class WeeklySummary(BaseModel):
    id: int
    week_start_date: datetime | None = None
    week_end_date: datetime | None = None
    generated_at: datetime | None = None
    summary_text: str | None = None


class WeeklySummariesResponse(BaseModel):
    items: list[WeeklySummary]


class CalendarEvent(BaseModel):
    """One dated event for the regulatory calendar — a compliance deadline
    (from ai_tags.deadlines) or a board meeting (from meeting_date)."""

    id: int
    type: str
    title: str
    date: str  # YYYY-MM-DD
    kind: Literal["deadline", "board"]
    severity: Severity | None = None


class CalendarResponse(BaseModel):
    events: list[CalendarEvent]


class InsightFeedItem(BaseModel):
    id: int
    generated_at: datetime | None = None
    insights: list[str] = Field(default_factory=list)


# ── Per-user personalization (stored in user_preferences.prefs['regulatory']) ──


class TrackedTerm(BaseModel):
    """A topic/theme or named entity the user follows for alerts."""

    term: str
    kind: Literal["topic", "entity"] = "topic"


class AlertRules(BaseModel):
    """Toggle set for what should surface as an alert."""

    orders_naming_entity: bool = True
    circular_matching_topic: bool = True
    deadline_soon: bool = True


class RegPersonalization(BaseModel):
    """The signed-in user's Regulatory Lens state."""

    bookmarks: list[int] = Field(default_factory=list)
    tracked: list[TrackedTerm] = Field(default_factory=list)
    alert_rules: AlertRules = Field(default_factory=AlertRules)


class RegAlert(RegDocSummary):
    """A document surfaced as an alert because it matched a tracked term."""

    matched_term: str | None = None


class AlertsResponse(BaseModel):
    items: list[RegAlert]
    total: int
