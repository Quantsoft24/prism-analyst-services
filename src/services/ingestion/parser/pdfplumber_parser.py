"""Lightweight in-process PDF parser using pdfplumber.

Pure Python, no ML, installs in seconds. Good text extraction; tables come
through as flattened text (we don't attempt structured table reconstruction
here — that's docling's job). This is the default backend so the ingestion
pipeline works on any machine without Docker.

Runs the (synchronous) pdfplumber work in a thread to keep the async event
loop free.
"""

from __future__ import annotations

import asyncio
import io

from src.services.ingestion.parser.base import (
    ParsedDocument,
    ParsedPage,
    ParseError,
    PdfParser,
)


class PdfPlumberParser(PdfParser):
    backend_name = "pdfplumber"

    async def parse(self, pdf_bytes: bytes, *, filename: str | None = None) -> ParsedDocument:
        try:
            return await asyncio.to_thread(self._parse_sync, pdf_bytes)
        except ParseError:
            raise
        except Exception as exc:  # pdfplumber raises a variety of errors
            raise ParseError(f"pdfplumber failed to parse {filename or 'document'}: {exc}") from exc

    def _parse_sync(self, pdf_bytes: bytes) -> ParsedDocument:
        import pdfplumber

        pages: list[ParsedPage] = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                # extract_text returns None for image-only pages.
                text = page.extract_text() or ""
                # Append flattened tables so numbers aren't lost even though
                # we don't preserve their structure. docling does this properly.
                for table in page.extract_tables() or []:
                    rows = [
                        " | ".join(cell or "" for cell in row)
                        for row in table
                        if any(cell for cell in row)
                    ]
                    if rows:
                        text += "\n\n[TABLE]\n" + "\n".join(rows)
                pages.append(ParsedPage(page_number=i, text=text.strip()))

        if not any(p.text for p in pages):
            raise ParseError("No extractable text — document may be scanned/image-only.")

        return ParsedDocument(
            pages=pages,
            parser_backend=self.backend_name,
            has_structure=False,
            metadata={"page_count": len(pages)},
        )
