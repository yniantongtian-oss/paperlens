"""Text extraction from text files and the tiny PDF fixture."""

from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter

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


def test_tiny_pdf_page_selection():
    assert FIXTURE_TEXT in extract_text(FIXTURE, pages="1")
    assert FIXTURE_TEXT in extract_text(FIXTURE, pages="1-1")
    with pytest.raises(SystemExit, match="outside"):
        extract_text(FIXTURE, pages="1-5")


def test_page_range_on_copies_of_tiny_pdf(tmp_path: Path):
    reader = PdfReader(str(FIXTURE))
    writer = PdfWriter()
    for _ in range(5):
        writer.add_page(reader.pages[0])
    dest = tmp_path / "five.pdf"
    with dest.open("wb") as handle:
        writer.write(handle)

    assert extract_text(dest, pages="1-5").count(FIXTURE_TEXT) == 5
    assert extract_text(dest, pages="2-4").count(FIXTURE_TEXT) == 3
    assert extract_text(dest, pages="1,3").count(FIXTURE_TEXT) == 2
    assert extract_text(dest, pages="1").count(FIXTURE_TEXT) == 1


@pytest.mark.parametrize("spec", ["foo", "5-1", "0", ""])
def test_invalid_page_spec(spec: str):
    with pytest.raises(SystemExit, match="--pages"):
        extract_text(FIXTURE, pages=spec)


def test_pages_rejected_for_text(tmp_path: Path):
    path = tmp_path / "notes.txt"
    path.write_text("hello\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="PDF"):
        extract_text(path, pages="1-5")
