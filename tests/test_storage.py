"""Tests for the fsspec-backed ObjectStore against a real local temp dir."""

from __future__ import annotations

import pytest

from src.services.storage import open_storage


@pytest.fixture
def store(tmp_path):
    # file:// URL pointed at a pytest temp dir — exercises the real fsspec
    # local backend, no mocks.
    return open_storage(f"file://{tmp_path.as_posix()}")


@pytest.mark.asyncio
async def test_put_then_get_roundtrip(store):
    data = b"%PDF-1.7 fake filing bytes"
    result = await store.put("filings/tcs/q4fy26.pdf", data, content_type="application/pdf")
    assert result.size_bytes == len(data)
    assert result.key == "filings/tcs/q4fy26.pdf"
    fetched = await store.get("filings/tcs/q4fy26.pdf")
    assert fetched == data


@pytest.mark.asyncio
async def test_exists(store):
    assert await store.exists("missing.pdf") is False
    await store.put("present.pdf", b"x")
    assert await store.exists("present.pdf") is True


@pytest.mark.asyncio
async def test_delete_is_idempotent(store):
    await store.put("temp.pdf", b"x")
    await store.delete("temp.pdf")
    assert await store.exists("temp.pdf") is False
    # Deleting again must not raise.
    await store.delete("temp.pdf")


@pytest.mark.asyncio
async def test_get_missing_raises(store):
    with pytest.raises(FileNotFoundError):
        await store.get("nope.pdf")


@pytest.mark.asyncio
async def test_uri_for_includes_scheme(store):
    uri = store.uri_for("a/b.pdf")
    assert uri.startswith("file://")
    assert uri.endswith("a/b.pdf")


@pytest.mark.asyncio
async def test_nested_paths_create_parents(store):
    # Deeply nested key must work — parent dirs auto-created.
    await store.put("a/b/c/d/deep.pdf", b"deep")
    assert await store.exists("a/b/c/d/deep.pdf") is True
