"""Declarative ingestion source registry.

The list of "what filings to ingest" lives in a versioned YAML file
(``config/ingestion_sources.yml``) rather than being hardcoded or scattered.
This makes the coverage universe a reviewable artifact: adding NSE-100 is a
PR that appends entries, not a code change.

Migration path (Slice 5C): a DB-backed ``ingestion_sources`` table replaces
the YAML. ``FilingsRegistry`` is the seam — swap ``load_registry`` to read
from the DB and every consumer keeps working. The ``IngestionSource`` shape
stays identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from src.models.filing import FILING_TYPES


@dataclass(slots=True, frozen=True)
class IngestionSource:
    """One filing to ingest. Maps 1:1 to a future ``filings`` row."""

    ticker: str            # NSE ticker — resolved to company_id at ingest time
    filing_type: str       # must be in FILING_TYPES
    source_url: str
    fiscal_period: str | None = None
    title: str | None = None
    filed_at: date | None = None
    exchange: str = "NSE"

    def __post_init__(self) -> None:
        if self.filing_type not in FILING_TYPES:
            raise ValueError(
                f"Unknown filing_type {self.filing_type!r} for {self.ticker}. "
                f"Allowed: {sorted(FILING_TYPES)}"
            )
        if not self.source_url.startswith(("http://", "https://", "file://", "s3://")):
            raise ValueError(
                f"source_url for {self.ticker} must be an absolute URL, got {self.source_url!r}"
            )


class FilingsRegistry:
    """In-memory registry of ingestion sources, loaded from YAML."""

    def __init__(self, sources: list[IngestionSource]) -> None:
        self._sources = sources

    def __len__(self) -> int:
        return len(self._sources)

    def all(self) -> list[IngestionSource]:
        return list(self._sources)

    def for_ticker(self, ticker: str) -> list[IngestionSource]:
        t = ticker.strip().upper()
        return [s for s in self._sources if s.ticker.upper() == t]

    def tickers(self) -> list[str]:
        # Preserve first-seen order, de-duplicated.
        seen: dict[str, None] = {}
        for s in self._sources:
            seen.setdefault(s.ticker.upper(), None)
        return list(seen.keys())


def load_registry(path: str | Path) -> FilingsRegistry:
    """Parse the YAML registry file into a ``FilingsRegistry``.

    Expected YAML shape — a top-level list of entries::

        - ticker: TCS
          filing_type: quarterly_result
          fiscal_period: Q4-FY26
          source_url: https://nsearchives.nseindia.com/...
          filed_at: 2026-05-02
          title: TCS Q4 FY26 Results

    Raises ``FileNotFoundError`` if the file is missing and ``ValueError`` on
    malformed entries (validation happens in ``IngestionSource.__post_init__``).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Ingestion registry not found at {p}")

    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or []
    if not isinstance(raw, list):
        raise ValueError(f"Registry {p} must be a YAML list of entries, got {type(raw).__name__}")

    sources: list[IngestionSource] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"Registry entry #{i} must be a mapping, got {type(entry).__name__}")
        filed_at = entry.get("filed_at")
        # PyYAML parses ISO dates to ``date`` automatically; tolerate strings too.
        if isinstance(filed_at, str):
            filed_at = date.fromisoformat(filed_at)
        sources.append(
            IngestionSource(
                ticker=entry["ticker"],
                filing_type=entry["filing_type"],
                source_url=entry["source_url"],
                fiscal_period=entry.get("fiscal_period"),
                title=entry.get("title"),
                filed_at=filed_at,
                exchange=entry.get("exchange", "NSE"),
            )
        )
    return FilingsRegistry(sources)
