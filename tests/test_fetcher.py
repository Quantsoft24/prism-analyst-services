"""Tests for the document fetcher — no network (uses file:// + temp files)."""

from __future__ import annotations

import hashlib

import pytest

from src.services.ingestion.fetcher import fetch_document


@pytest.mark.asyncio
async def test_fetch_file_scheme(tmp_path):
    pdf = tmp_path / "doc.pdf"
    payload = b"%PDF-1.7 some bytes here"
    pdf.write_bytes(payload)

    result = await fetch_document(f"file://{pdf.as_posix()}")
    assert result.content == payload
    assert result.size_bytes == len(payload)
    # Fingerprint is the SHA-256 of the bytes — deterministic + idempotency anchor.
    assert result.fingerprint == hashlib.sha256(payload).hexdigest()


@pytest.mark.asyncio
async def test_fingerprint_is_deterministic(tmp_path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"identical content")
    r1 = await fetch_document(f"file://{pdf.as_posix()}")
    r2 = await fetch_document(f"file://{pdf.as_posix()}")
    assert r1.fingerprint == r2.fingerprint


@pytest.mark.asyncio
async def test_different_content_different_fingerprint(tmp_path):
    a = tmp_path / "a.pdf"
    b = tmp_path / "b.pdf"
    a.write_bytes(b"content A")
    b.write_bytes(b"content B")
    ra = await fetch_document(f"file://{a.as_posix()}")
    rb = await fetch_document(f"file://{b.as_posix()}")
    assert ra.fingerprint != rb.fingerprint


@pytest.mark.asyncio
async def test_unsupported_scheme_raises():
    with pytest.raises(ValueError, match="Unsupported source scheme"):
        await fetch_document("ftp://example.com/file.pdf")
