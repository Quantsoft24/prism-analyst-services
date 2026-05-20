"""fsspec-backed ObjectStore — works for local disk, S3, GCS, Azure, etc.

The base URL (e.g. ``file://./.data/filings`` or ``s3://bucket/filings``)
fixes the protocol + root. ``key`` is always a relative path joined under it.

fsspec is predominantly synchronous, so we offload its blocking calls to a
thread via ``asyncio.to_thread`` to avoid stalling the event loop. For S3
at high throughput we'd switch to the async ``s3fs`` API, but that's a
Slice 5C+ optimization — correctness first.
"""

from __future__ import annotations

import asyncio
from urllib.parse import urlsplit

import fsspec

from src.services.storage.base import ObjectStore, StoredObject


class FsspecObjectStore(ObjectStore):
    """Generic store. Construct via ``open_storage(url)`` rather than directly."""

    def __init__(self, base_url: str) -> None:
        # Split scheme from path. For "file://./x" fsspec wants the protocol
        # "file" and a root path "./x". For "s3://bucket/x" it's protocol "s3"
        # and root "bucket/x".
        parts = urlsplit(base_url)
        self._protocol = parts.scheme or "file"
        # Reassemble the root path (netloc + path) without the scheme.
        root = (parts.netloc + parts.path) if parts.netloc else parts.path
        self._root = root.rstrip("/")
        self._fs = fsspec.filesystem(self._protocol)

        # For local file systems, ensure the root directory exists upfront.
        if self._protocol == "file":
            self._fs.makedirs(self._root, exist_ok=True)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _full_path(self, key: str) -> str:
        """Join key under root for fsspec calls (no scheme)."""
        return f"{self._root}/{key.lstrip('/')}"

    def uri_for(self, key: str) -> str:
        return f"{self._protocol}://{self._full_path(key)}"

    # ── Operations (sync work offloaded to threads) ──────────────────────

    async def put(self, key: str, data: bytes, *, content_type: str | None = None) -> StoredObject:
        path = self._full_path(key)

        def _write() -> int:
            # Ensure the parent directory exists (local fs). For object stores
            # this is a no-op since they have no real directories.
            parent = path.rsplit("/", 1)[0]
            try:
                self._fs.makedirs(parent, exist_ok=True)
            except (NotImplementedError, FileExistsError):
                pass
            with self._fs.open(path, "wb") as f:
                f.write(data)
            return len(data)

        size = await asyncio.to_thread(_write)
        return StoredObject(key=key, uri=self.uri_for(key), size_bytes=size)

    async def get(self, key: str) -> bytes:
        path = self._full_path(key)

        def _read() -> bytes:
            with self._fs.open(path, "rb") as f:
                return f.read()

        return await asyncio.to_thread(_read)

    async def exists(self, key: str) -> bool:
        path = self._full_path(key)
        return await asyncio.to_thread(self._fs.exists, path)

    async def delete(self, key: str) -> None:
        path = self._full_path(key)

        def _rm() -> None:
            try:
                self._fs.rm(path)
            except FileNotFoundError:
                pass

        await asyncio.to_thread(_rm)


def open_storage(base_url: str) -> ObjectStore:
    """Factory — returns the appropriate ObjectStore for ``base_url``'s scheme.

    Currently always an ``FsspecObjectStore`` (fsspec covers file/s3/gcs/az/
    http). The factory exists so callers depend on the interface, and so we
    can swap in a specialized backend later without touching call sites.
    """
    return FsspecObjectStore(base_url)
