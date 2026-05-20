"""Storage interface — the contract every backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class StoredObject:
    """Result of a successful ``put`` — what we persist on the Filing row."""

    key: str          # relative key within the store (goes into filings.storage_path)
    uri: str          # fully-qualified URI (e.g. s3://bucket/filings/abc.pdf)
    size_bytes: int


class ObjectStore(ABC):
    """Async object storage. Keys are relative paths within a configured base.

    Implementations must be safe to call from async request handlers — any
    blocking I/O should be offloaded (e.g. ``asyncio.to_thread``).
    """

    @abstractmethod
    async def put(self, key: str, data: bytes, *, content_type: str | None = None) -> StoredObject:
        """Write ``data`` at ``key``. Overwrites if it already exists."""

    @abstractmethod
    async def get(self, key: str) -> bytes:
        """Read the bytes at ``key``. Raises ``FileNotFoundError`` if absent."""

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """True if an object exists at ``key``."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Remove the object at ``key``. No-op if it doesn't exist."""

    @abstractmethod
    def uri_for(self, key: str) -> str:
        """Return the fully-qualified URI for ``key`` (no I/O)."""
