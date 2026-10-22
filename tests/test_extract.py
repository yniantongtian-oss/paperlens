"""Text extraction from text files and the tiny PDF fixture."""

from pathlib import Path

import pytest

from paperlens import extract_text

FIXTURE = Path(__file__).parent / "fixtures" / "tiny.pdf"
FIXTURE_TEXT = "PaperLens tiny fixture. Widgets improve by 12 percent."


def test_tiny_pdf_fixture():
    assert FIXTURE_TEXT in extract_text(FIXTURE)


def test_text_file(tmp_path: Path):
    path = tmp_path / "notes.txt"
    path.write_text("Widgets improve by 12 percent.\n", encoding="utf-8")
    assert extract_text(path) == "Widgets improve by 12 percent.\n"


def test_missing_file(tmp_path: Path):
    with pytest.raises(SystemExit, match="Input not found"):
        extract_text(tmp_path / "missing.pdf")


def test_corrupt_pdf(tmp_path: Path):
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"%PDF-1.4\nnot a real pdf")
    with pytest.raises(SystemExit, match="Could not read PDF"):
        extract_text(path)
