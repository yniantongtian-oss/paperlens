"""Read paper text from a PDF or a plain-text file."""

from __future__ import annotations

import os


def extract_text(path: str | os.PathLike[str]) -> str:
    """Return the text of a PDF or UTF-8 text file.

    PDFs are read with pypdf. Every other file is decoded as UTF-8, and
    invalid bytes are dropped. An empty string means the file had no text
    layer (a scanned PDF, for example).
    """
    file_path = os.fspath(path)
    if not os.path.isfile(file_path):
        raise SystemExit(f"[!] Input not found: {file_path}")
    if file_path.lower().endswith(".pdf"):
        return _read_pdf(file_path)
    with open(file_path, encoding="utf-8", errors="ignore") as handle:
        return handle.read()


def _read_pdf(path: str) -> str:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(path)
    except PdfReadError as exc:
        raise SystemExit(f"[!] Could not read PDF: {exc}") from None
    parts = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(parts)
