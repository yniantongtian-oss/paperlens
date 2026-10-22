"""Read paper text from a PDF or a plain-text file."""

from __future__ import annotations

import os
import re

_PAGE_PART = re.compile(r"^(\d+)\s*(?:-\s*(\d+))?$")


def extract_text(path: str | os.PathLike[str], *, pages: str | None = None) -> str:
    """Return the text of a PDF or UTF-8 text file.

    PDFs are read with pypdf. *pages* is a 1-based spec such as ``1-5``,
    ``3``, or ``1-3,8`` and applies to PDFs only. Every other file is decoded
    as UTF-8, and invalid bytes are dropped. An empty string means the file
    had no text layer (a scanned PDF, for example).
    """
    file_path = os.fspath(path)
    if not os.path.isfile(file_path):
        raise SystemExit(f"[!] Input not found: {file_path}")
    if file_path.lower().endswith(".pdf"):
        return _read_pdf(file_path, pages)
    if pages:
        raise SystemExit("[!] --pages applies to PDF files only.")
    with open(file_path, encoding="utf-8", errors="ignore") as handle:
        return handle.read()


def _read_pdf(path: str, pages: str | None) -> str:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(path)
    except PdfReadError as exc:
        raise SystemExit(f"[!] Could not read PDF: {exc}") from None
    selected = reader.pages
    if pages is not None:
        indexes = _page_indexes(pages, len(reader.pages))
        selected = [reader.pages[index] for index in indexes]
    parts = [(page.extract_text() or "") for page in selected]
    return "\n".join(parts)


def _page_indexes(spec: str, page_count: int) -> list[int]:
    """Return 0-based indexes for a 1-based spec. Ranges are inclusive."""
    if page_count < 1:
        raise SystemExit("[!] PDF has no pages.")
    indexes: list[int] = []
    seen: set[int] = set()
    for raw in spec.split(","):
        part = raw.strip()
        match = _PAGE_PART.fullmatch(part)
        if not match:
            raise SystemExit(f"[!] Invalid --pages range: {spec!r}")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start < 1 or end < start:
            raise SystemExit(f"[!] Invalid --pages range: {part!r}")
        if end > page_count:
            raise SystemExit(f"[!] --pages {spec!r} is outside this PDF (1-{page_count}).")
        for number in range(start, end + 1):
            if number not in seen:
                seen.add(number)
                indexes.append(number - 1)
    if not indexes:
        raise SystemExit(f"[!] Invalid --pages range: {spec!r}")
    return indexes
