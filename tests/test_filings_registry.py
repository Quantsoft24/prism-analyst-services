"""Tests for the YAML-backed ingestion source registry."""

from __future__ import annotations

from datetime import date

import pytest

from src.services.ingestion.registry import (
    FilingsRegistry,
    IngestionSource,
    load_registry,
)

VALID_YAML = """\
- ticker: TCS
  filing_type: quarterly_result
  fiscal_period: Q4-FY26
  source_url: https://example.com/tcs.pdf
  filed_at: 2026-04-15
  title: TCS Q4 FY26
- ticker: RELIANCE
  filing_type: annual_report
  fiscal_period: FY26
  source_url: https://example.com/ril.pdf
"""


def _write(tmp_path, content: str):
    p = tmp_path / "sources.yml"
    p.write_text(content, encoding="utf-8")
    return p


def test_load_valid_registry(tmp_path):
    reg = load_registry(_write(tmp_path, VALID_YAML))
    assert len(reg) == 2
    assert reg.tickers() == ["TCS", "RELIANCE"]
    tcs = reg.for_ticker("tcs")  # case-insensitive
    assert len(tcs) == 1
    assert tcs[0].filing_type == "quarterly_result"
    assert tcs[0].filed_at == date(2026, 4, 15)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_registry(tmp_path / "nope.yml")


def test_invalid_filing_type_rejected(tmp_path):
    bad = """\
- ticker: TCS
  filing_type: not_a_real_type
  source_url: https://example.com/x.pdf
"""
    with pytest.raises(ValueError, match="Unknown filing_type"):
        load_registry(_write(tmp_path, bad))


def test_relative_url_rejected(tmp_path):
    bad = """\
- ticker: TCS
  filing_type: quarterly_result
  source_url: /local/path/not/absolute.pdf
"""
    with pytest.raises(ValueError, match="absolute URL"):
        load_registry(_write(tmp_path, bad))


def test_non_list_yaml_rejected(tmp_path):
    with pytest.raises(ValueError, match="must be a YAML list"):
        load_registry(_write(tmp_path, "ticker: TCS\nfiling_type: quarterly_result"))


def test_empty_yaml_is_empty_registry(tmp_path):
    reg = load_registry(_write(tmp_path, ""))
    assert len(reg) == 0


def test_ingestion_source_validates_on_construct():
    with pytest.raises(ValueError):
        IngestionSource(ticker="X", filing_type="bogus", source_url="https://x.com/a.pdf")


def test_shipped_registry_is_valid():
    """The real config/ingestion_sources.yml must always parse — guards against
    a malformed edit landing in a PR."""
    reg = load_registry("config/ingestion_sources.yml")
    assert len(reg) >= 1
    for src in reg.all():
        assert isinstance(src, IngestionSource)
        assert src.source_url.startswith(("http://", "https://", "file://", "s3://"))
