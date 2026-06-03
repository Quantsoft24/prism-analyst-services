"""Pydantic schemas for the Systematic Portfolio Builder API (`/api/v1/portfolio`)."""

from __future__ import annotations

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, Field

from src.portfolio.calendar import Frequency
from src.portfolio.screening import Operator
from src.portfolio.weighting import Scheme


# ── Catalog / universe ───────────────────────────────────────────────────────
class UniverseRead(BaseModel):
    index_id: int
    index_name: str | None
    exchange: str | None


class FactorMetaRead(BaseModel):
    id: str
    name: str
    category: str
    unit: str
    direction: str
    default_operator: str
    data_kind: str
    source_tables: list[str]
    description: str
    exclude_sectors: list[str]
    decimals: int


class IndexSeriesResponse(BaseModel):
    """A benchmark index's cumulative NAV (growth of ₹1) over a window — used to
    overlay/switch the benchmark on the backtest NAV chart without a re-run."""

    index_id: int
    index_name: str | None = None
    dates: list[str]
    nav: list[float]


# ── Custom factors ───────────────────────────────────────────────────────────
class CustomFactorSpec(BaseModel):
    """A user-composed factor, referenced by ``id`` in filters/weighting/display.
    Sent inline so unsaved factors ('backtest this factor') work too."""

    id: str
    name: str
    expression: str                      # e.g. "(roe + earnings_yield) / pb"
    direction: str = "higher_better"     # higher_better | lower_better
    normalization: str = "none"          # none | zscore | rank


# ── Screen request ───────────────────────────────────────────────────────────
class FilterSpec(BaseModel):
    factor_id: str
    op: Operator
    value: float | None = None
    value2: float | None = None          # upper bound for `between`
    k: int | None = Field(default=None, ge=1)  # for top_k / bottom_k


class WeightingSpec(BaseModel):
    scheme: Scheme = "equal"
    score_factor_id: str | None = None   # required for `factor_score`
    max_weight: float | None = Field(default=None, gt=0, le=1)
    max_sector_weight: float | None = Field(default=None, gt=0, le=1)


class ScreenRequest(BaseModel):
    index_id: int
    filters: list[FilterSpec] = Field(default_factory=list)
    weighting: WeightingSpec = Field(default_factory=WeightingSpec)
    basis: str | None = None             # 'consolidated' | 'standalone'
    as_of: date | None = None            # defaults to the latest trading day
    display_factors: list[str] = Field(default_factory=list)
    custom_factors: list[CustomFactorSpec] = Field(default_factory=list)


# ── Screen response ──────────────────────────────────────────────────────────
class HoldingRead(BaseModel):
    security_id: int
    symbol: str | None
    name: str | None
    sector: str | None
    weight: float
    factors: dict[str, float | None]


class CoverageRead(BaseModel):
    factor_id: str
    computable: int
    total: int


class FunnelStepRead(BaseModel):
    label: str
    remaining: int


class ScreenResponse(BaseModel):
    as_of: date
    universe: UniverseRead
    membership_count: int
    basis: str
    weighting_scheme: str
    holdings: list[HoldingRead]
    funnel: list[FunnelStepRead]
    coverage: list[CoverageRead]
    dropped_no_weight: int
    notes: list[str] = Field(default_factory=list)


# ── Backtest (async job) ─────────────────────────────────────────────────────
class BacktestRequest(BaseModel):
    index_id: int
    start: date
    end: date
    frequency: Frequency = "quarterly"
    filters: list[FilterSpec] = Field(default_factory=list)
    weighting: WeightingSpec = Field(default_factory=WeightingSpec)
    basis: str | None = None
    benchmark_index_id: int | None = None
    custom_factors: list[CustomFactorSpec] = Field(default_factory=list)
    name: str | None = None


class BacktestJobRead(BaseModel):
    id: UUID
    name: str | None
    status: str                       # queued | running | succeeded | failed | cancelled
    progress: float
    stage: str | None
    error: str | None
    spec: dict
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    result: dict | None = None        # populated only when succeeded (omitted in lists)


# ── Persistence: custom factors + saved strategies ───────────────────────────
class CustomFactorCreate(BaseModel):
    name: str
    expression: str
    direction: str = "higher_better"
    normalization: str = "none"


class CustomFactorRead(BaseModel):
    id: UUID
    name: str
    expression: str
    direction: str
    normalization: str
    created_at: datetime


class ExpressionValidateRequest(BaseModel):
    expression: str


class ExpressionValidateResponse(BaseModel):
    ok: bool
    refs: list[str] = Field(default_factory=list)
    error: str | None = None


class StrategyCreate(BaseModel):
    name: str
    config: dict                      # the full builder config (universe/filters/rules/weighting/customs)


class StrategyRead(BaseModel):
    id: UUID
    name: str
    config: dict
    created_at: datetime
    updated_at: datetime


# ── Factor preview (live ranking for the Factor Builder) ─────────────────────
class FactorPreviewRequest(BaseModel):
    index_id: int
    factor_id: str | None = None      # rank by an existing factor…
    custom: CustomFactorSpec | None = None   # …or an inline custom expression
    as_of: date | None = None
    basis: str | None = None
    limit: int = Field(default=25, ge=1, le=100)


class FactorPreviewRow(BaseModel):
    security_id: int
    symbol: str | None
    name: str | None
    sector: str | None
    value: float | None


class FactorPreviewResponse(BaseModel):
    as_of: date
    factor_id: str
    computable: int
    total: int
    top: list[FactorPreviewRow]
    bottom: list[FactorPreviewRow]
