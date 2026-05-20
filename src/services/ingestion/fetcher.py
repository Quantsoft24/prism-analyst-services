"""Document fetcher — downloads a filing source URL into bytes.

Keeps fetching concerns (HTTP, retries, content-type, fingerprinting) out of
the pipeline orchestrator. Supports http/https now; ``file://`` is handled too
so tests + local fixtures work without a network.

Fingerprint = SHA-256 of the raw bytes. This is the idempotency anchor: the
pipeline skips re-ingesting content it has already stored (see
``FilingRepository.get_by_fingerprint``).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import httpx


@dataclass(slots=True)
class FetchedDocument:
    """Raw bytes + provenance for a fetched source."""

    content: bytes
    content_type: str | None
    fingerprint: str
    size_bytes: int


# A realistic UA — NSE/BSE archive endpoints reject obvious bot agents.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 PRISM-Ingest/0.1"
)


async def fetch_document(source_url: str, *, timeout: int = 60) -> FetchedDocument:
    """Fetch ``source_url`` → bytes + fingerprint.

    Raises ``ValueError`` for unsupported schemes and ``httpx.HTTPError`` /
    ``FileNotFoundError`` on fetch failure (the pipeline catches + records
    these as a failed filing).
    """
    scheme = urlsplit(source_url).scheme.lower()

    if scheme in ("http", "https"):
        content, content_type = await _fetch_http(source_url, timeout)
    elif scheme == "file":
        content, content_type = _fetch_file(source_url)
    else:
        raise ValueError(f"Unsupported source scheme {scheme!r} for {source_url}")

    return FetchedDocument(
        content=content,
        content_type=content_type,
        fingerprint=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
    )


async def _fetch_http(url: str, timeout: int) -> tuple[bytes, str | None]:
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": _USER_AGENT},
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content, resp.headers.get("content-type")


def _fetch_file(url: str) -> tuple[bytes, str | None]:
    # file://./relative/path or file:///abs/path
    parts = urlsplit(url)
    path = parts.netloc + parts.path if parts.netloc else parts.path
    data = Path(path).read_bytes()
    return data, "application/pdf"
