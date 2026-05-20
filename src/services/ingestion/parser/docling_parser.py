"""Docling parser — best-in-class financial-PDF parsing via a Docker sidecar.

The heavy docling stack (PyTorch + layout/table models, multi-GB) lives in a
separate container (``docling-service/``), NOT in the backend image. This
class is a thin HTTP client that POSTs PDF bytes to the sidecar and maps the
response back into our ``ParsedDocument`` shape.

Why a sidecar instead of in-process docling:
  * Keeps the backend image lean + fast to build/deploy.
  * Keeps the dev venv (Windows) free of torch + native model deps.
  * Parsing is batch/offline (ingestion), so an HTTP hop is fine.
  * The sidecar can scale / GPU-accelerate independently later.

Contract with the sidecar (see docling-service/app.py):
  POST {DOCLING_SERVICE_URL}/parse   (multipart file=<pdf>)
  → 200 {"pages": [{"page_number": int, "text": str, "markdown": str}],
         "has_structure": true, "metadata": {...}}
"""

from __future__ import annotations

import httpx

from src.config import settings
from src.services.ingestion.parser.base import (
    ParsedDocument,
    ParsedPage,
    ParseError,
    PdfParser,
)


class DoclingParser(PdfParser):
    backend_name = "docling"

    def __init__(self, service_url: str | None = None, timeout: int | None = None) -> None:
        self._url = (service_url or settings.DOCLING_SERVICE_URL).rstrip("/")
        self._timeout = timeout or settings.DOCLING_TIMEOUT_SECONDS

    async def parse(self, pdf_bytes: bytes, *, filename: str | None = None) -> ParsedDocument:
        files = {"file": (filename or "document.pdf", pdf_bytes, "application/pdf")}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(f"{self._url}/parse", files=files)
        except httpx.HTTPError as exc:
            raise ParseError(
                f"Could not reach docling sidecar at {self._url} — is the "
                f"container running? ({exc})"
            ) from exc

        if resp.status_code != 200:
            raise ParseError(
                f"docling sidecar returned {resp.status_code}: {resp.text[:300]}"
            )

        data = resp.json()
        pages = [
            ParsedPage(
                page_number=p["page_number"],
                text=p.get("text", ""),
                markdown=p.get("markdown"),
            )
            for p in data.get("pages", [])
        ]
        if not pages:
            raise ParseError("docling returned no pages.")

        return ParsedDocument(
            pages=pages,
            parser_backend=self.backend_name,
            has_structure=bool(data.get("has_structure", True)),
            metadata=data.get("metadata", {}),
        )

    async def health(self) -> bool:
        """Probe the sidecar — used by the parser factory to fail fast with a
        helpful message if PARSER_BACKEND=docling but the container isn't up."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self._url}/health")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False
