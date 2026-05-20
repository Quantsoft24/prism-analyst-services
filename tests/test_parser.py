"""Tests for the parser factory + parser error handling.

Happy-path PDF parsing (real filings) is validated end-to-end in Slice 5B-2's
ingestion test — we don't add a PDF-generation dependency just to unit-test it.
Here we cover the seams: factory dispatch + graceful failure on bad input.
"""

from __future__ import annotations

import pytest

from src.services.ingestion.parser import get_parser
from src.services.ingestion.parser.base import ParseError, PdfParser
from src.services.ingestion.parser.docling_parser import DoclingParser
from src.services.ingestion.parser.pdfplumber_parser import PdfPlumberParser


def test_factory_returns_pdfplumber_by_default():
    parser = get_parser("pdfplumber")
    assert isinstance(parser, PdfPlumberParser)
    assert parser.backend_name == "pdfplumber"


def test_factory_returns_docling():
    parser = get_parser("docling")
    assert isinstance(parser, DoclingParser)
    assert parser.backend_name == "docling"


def test_factory_is_case_insensitive():
    assert isinstance(get_parser("PDFPlumber"), PdfPlumberParser)


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown PARSER_BACKEND"):
        get_parser("not_a_real_parser")


def test_all_parsers_implement_interface():
    for name in ("pdfplumber", "docling"):
        assert isinstance(get_parser(name), PdfParser)


@pytest.mark.asyncio
async def test_pdfplumber_raises_parse_error_on_garbage():
    parser = PdfPlumberParser()
    with pytest.raises(ParseError):
        await parser.parse(b"this is definitely not a pdf", filename="junk.pdf")


@pytest.mark.asyncio
async def test_pdfplumber_raises_parse_error_on_empty():
    parser = PdfPlumberParser()
    with pytest.raises(ParseError):
        await parser.parse(b"", filename="empty.pdf")


@pytest.mark.asyncio
async def test_docling_parser_unreachable_sidecar_raises_helpful_error():
    # Point at a port nothing is listening on → connection error → ParseError
    # with a message that tells the operator the sidecar isn't running.
    parser = DoclingParser(service_url="http://localhost:59999", timeout=2)
    with pytest.raises(ParseError, match="docling sidecar"):
        await parser.parse(b"%PDF-1.7 fake", filename="x.pdf")


@pytest.mark.asyncio
async def test_docling_health_false_when_down():
    parser = DoclingParser(service_url="http://localhost:59999", timeout=2)
    assert await parser.health() is False
