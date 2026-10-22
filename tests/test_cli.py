"""CLI behavior: echo, output files, JSON, and argument errors."""

import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from paperlens import main
from paperlens.version import __version__
from tests.helpers import long_paper

ROOT = Path(__file__).parents[1]
CHAT = {"choices": [{"message": {"content": "GROQ REPORT"}}]}


def test_echo_backend(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    paper = tmp_path / "paper.txt"
    paper.write_text("Widgets improve by 12 percent.\n", encoding="utf-8")
    assert main([str(paper), "--backend", "echo"]) == 0
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert "## TL;DR" in out
    assert "Widgets improve by 12 percent." in out


def test_echo_output_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    paper = tmp_path / "paper.txt"
    paper.write_text("A tiny note about fusion.\n", encoding="utf-8")
    dest = tmp_path / "report.md"
    assert main([str(paper), "--backend", "echo", "-o", str(dest)]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"Report written to {dest}" in captured.err
    written = dest.read_text(encoding="utf-8")
    assert "fusion" in written
    assert written.startswith("<!-- dry-run:")


def test_output_file_with_mocked_groq(tmp_path, monkeypatch, capture_http, capsys):
    monkeypatch.setenv("GROQ_API_KEY", "secret-key")
    calls = capture_http(CHAT)
    paper = tmp_path / "paper.txt"
    paper.write_text("Groq me.\n", encoding="utf-8")
    dest = tmp_path / "out.md"
    assert main([str(paper), "--backend", "groq", "-o", str(dest)]) == 0
    assert dest.read_text(encoding="utf-8") == "GROQ REPORT\n"
    assert capsys.readouterr().out == ""
    assert calls[0].get_header("Authorization") == "Bearer secret-key"


def test_missing_groq_key_from_cli(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    paper = tmp_path / "paper.txt"
    paper.write_text("Hello.\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="GROQ_API_KEY"):
        main([str(paper), "--backend", "groq"])


def test_json_format_on_stdout(tmp_path, capture_http, capsys):
    capture_http(
        {
            "response": json.dumps(
                {
                    "tldr": "Fusion works.",
                    "key_contributions": ["a cell"],
                    "method": "stack layers",
                    "key_results": ["3x"],
                    "reproduction_notes": "needs the dataset",
                }
            )
        }
    )
    paper = tmp_path / "paper.txt"
    paper.write_text("Fusion paper.\n", encoding="utf-8")
    assert main([str(paper), "--backend", "ollama", "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["tldr"] == "Fusion works."
    assert data["meta"]["backend"] == "ollama"
    assert data["meta"]["map_reduce"] is False


def test_max_chars_boundary(tmp_path, capsys):
    paper = tmp_path / "paper.txt"
    paper.write_text("A" * 20, encoding="utf-8")
    assert main([str(paper), "--backend", "echo", "--max-chars", "20"]) == 0
    assert "map-reduce" not in capsys.readouterr().err

    assert main([str(paper), "--backend", "echo", "--max-chars", "19"]) == 0
    captured = capsys.readouterr()
    assert "map-reduce" in captured.err
    assert "truncated" in captured.err
    assert "A" * 19 in captured.out
    assert "### Document (part 2/2)\nA\n" in captured.out


def test_echo_pdf_fixture():
    fixture = ROOT / "tests" / "fixtures" / "tiny.pdf"
    proc = subprocess.run(
        [sys.executable, "-m", "paperlens", str(fixture), "--backend", "echo"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Widgets improve by 12 percent." in proc.stdout
    assert "## Method" in proc.stdout


def test_long_paper_echo_cli(tmp_path, capsys):
    paper = tmp_path / "long.txt"
    paper.write_text(long_paper(), encoding="utf-8")
    assert main([str(paper), "--backend", "echo", "--max-chars", "200", "--format", "json"]) == 0
    captured = capsys.readouterr()
    assert "RESULT_TOKEN" in captured.out
    assert "tldr" in captured.out
    assert "map-reduce" in captured.err


def test_empty_input(tmp_path):
    paper = tmp_path / "empty.txt"
    paper.write_text("   \n", encoding="utf-8")
    with pytest.raises(SystemExit, match="No text extracted"):
        main([str(paper), "--backend", "echo"])


def test_max_chars_must_be_positive(tmp_path):
    paper = tmp_path / "paper.txt"
    paper.write_text("Hello.\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="max-chars"):
        main([str(paper), "--max-chars", "0"])


def test_base_url_requires_openai_backend(tmp_path):
    paper = tmp_path / "paper.txt"
    paper.write_text("Hello.\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="base-url"):
        main([str(paper), "--backend", "ollama", "--base-url", "http://localhost:11434/v1"])


def test_version_matches_pyproject_and_flag():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["version"] == __version__
    assert project["project"]["scripts"]["paperlens"] == "paperlens.cli:main"
    proc = subprocess.run(
        [sys.executable, "-m", "paperlens", "--version"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert __version__ in proc.stdout


def test_pages_on_tiny_pdf(capsys):
    fixture = ROOT / "tests" / "fixtures" / "tiny.pdf"
    assert main([str(fixture), "--backend", "echo", "--pages", "1"]) == 0
    assert "Widgets improve by 12 percent." in capsys.readouterr().out
    with pytest.raises(SystemExit, match="outside"):
        main([str(fixture), "--backend", "echo", "--pages", "1-5"])


def test_batch_directory_writes_one_report_per_file(tmp_path: Path, capsys):
    papers = tmp_path / "papers"
    nested = papers / "nested"
    nested.mkdir(parents=True)
    (papers / "alpha.txt").write_text("Alpha finding.\n", encoding="utf-8")
    (papers / "beta.txt").write_text("Beta finding.\n", encoding="utf-8")
    (papers / "notes.csv").write_text("comma,separated\n", encoding="utf-8")
    (papers / ".hidden.txt").write_text("secret\n", encoding="utf-8")
    (nested / "skipped.txt").write_text("nested finding.\n", encoding="utf-8")
    dest = tmp_path / "reports"
    assert main([str(papers), "--backend", "echo", "-o", str(dest)]) == 0
    alpha = (dest / "alpha.md").read_text(encoding="utf-8")
    beta = (dest / "beta.md").read_text(encoding="utf-8")
    assert "Alpha finding." in alpha
    assert "Beta finding." in beta
    assert not (dest / "notes.md").exists()
    assert not (dest / ".hidden.md").exists()
    assert not (dest / "skipped.md").exists()
    err = capsys.readouterr().err
    assert str(dest / "alpha.md") in err
    assert str(dest / "beta.md") in err


def test_batch_files_stdout_and_json_directory(tmp_path: Path, capsys):
    alpha = tmp_path / "alpha.txt"
    beta = tmp_path / "beta.txt"
    alpha.write_text("File A.\n", encoding="utf-8")
    beta.write_text("File B.\n", encoding="utf-8")
    assert main([str(alpha), str(beta), "--backend", "echo"]) == 0
    out = capsys.readouterr().out
    assert "<!-- alpha.txt -->" in out
    assert "<!-- beta.txt -->" in out
    assert out.index("File A.") < out.index("File B.")

    dest = tmp_path / "reports"
    assert (
        main([str(alpha), str(beta), "--backend", "echo", "--format", "json", "-o", str(dest)]) == 0
    )
    written = (dest / "alpha.json").read_text(encoding="utf-8")
    assert "File A." in written
    assert "File B." in (dest / "beta.json").read_text(encoding="utf-8")
    assert not (dest / "alpha.md").exists()


def test_batch_output_must_be_a_directory(tmp_path: Path):
    alpha = tmp_path / "alpha.txt"
    beta = tmp_path / "beta.txt"
    alpha.write_text("File A.\n", encoding="utf-8")
    beta.write_text("File B.\n", encoding="utf-8")
    dest = tmp_path / "report.md"
    dest.write_text("existing\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="directory"):
        main([str(alpha), str(beta), "--backend", "echo", "-o", str(dest)])


def test_timeout_retries_and_forwarded_timeout(tmp_path, capture_http):
    calls = capture_http({"response": "ok"})
    paper = tmp_path / "paper.txt"
    paper.write_text("Hello.\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="timeout"):
        main([str(paper), "--timeout", "0"])
    with pytest.raises(SystemExit, match="retries"):
        main([str(paper), "--retries", "-1"])
    assert (
        main(
            [
                str(paper),
                "--backend",
                "ollama",
                "--timeout",
                "15",
                "--retries",
                "0",
                "--no-cache",
            ]
        )
        == 0
    )
    assert calls[0].timeout_seconds == 15


def test_console_script_help():
    exe = Path(sys.executable).with_name("paperlens")
    if not exe.is_file():
        found = shutil.which("paperlens")
        if found is None:
            pytest.skip("paperlens console script is not installed")
        exe = Path(found)
    proc = subprocess.run([str(exe), "--help"], check=False, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "--max-chars" in proc.stdout
    assert "12000" in proc.stdout
    assert "openai" in proc.stdout
    assert "--format" in proc.stdout
    assert "--timeout" in proc.stdout
    assert "--retries" in proc.stdout
    assert "--pages" in proc.stdout
    assert "--no-cache" in proc.stdout
