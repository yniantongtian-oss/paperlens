"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from paperlens.backends import DEFAULT_MODELS, DEFAULT_RETRIES
from paperlens.extract import extract_text
from paperlens.summarize import DEFAULT_MAX_CHARS, summarize
from paperlens.version import __version__

_DIR_SUFFIXES = {".pdf", ".txt", ".md"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paperlens",
        description="Summarize a research paper with a local or OpenAI-compatible model.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="PDF or text files, or a directory of them",
    )
    parser.add_argument(
        "--backend",
        choices=tuple(DEFAULT_MODELS),
        default="ollama",
        help="model backend (default: ollama)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "model name "
            "(defaults: ollama qwen3:8b, groq llama-3.3-70b-versatile, openai gpt-4o-mini)"
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible API root, for example https://api.openai.com/v1",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help=(
            "max characters of paper or notes per model call "
            f"(default: {DEFAULT_MAX_CHARS}). Longer papers use section-aware map-reduce."
        ),
    )
    parser.add_argument(
        "--pages",
        default=None,
        help="1-based PDF pages to read, for example 1-5 or 1,3-4",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="HTTP timeout in seconds (default: 600 for ollama, 300 otherwise)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=(
            "extra attempts after HTTP 429 or 5xx, with exponential backoff "
            f"(default: {DEFAULT_RETRIES})"
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="do not read or write the on-disk response cache",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        dest="output_format",
        help="report format (default: markdown)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="write the report here, or a directory when there are several inputs",
    )
    parser.add_argument("--version", action="version", version=f"paperlens {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run PaperLens. *argv* is the argument list, without the program name."""
    args = build_parser().parse_args(argv)
    _validate_args(args)
    files = collect_inputs(args.inputs)
    if len(files) == 1:
        report = _summarize_file(args, files[0])
        if args.output:
            _write_report(report, Path(args.output))
        else:
            sys.stdout.write(report)
        return 0

    if args.output:
        planned = _batch_destinations(files, Path(args.output), args.output_format)
        for source, dest in planned:
            _write_report(_summarize_file(args, source), dest)
        return 0

    for source in files:
        report = _summarize_file(args, source)
        sys.stdout.write(f"<!-- {source.name} -->\n")
        sys.stdout.write(report)
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_chars < 1:
        raise SystemExit("[!] --max-chars must be a positive integer.")
    if args.base_url and args.backend != "openai":
        raise SystemExit("[!] --base-url applies to the openai backend only.")
    if args.timeout is not None and not args.timeout > 0:
        raise SystemExit("[!] --timeout must be a positive number.")
    if args.retries < 0:
        raise SystemExit("[!] --retries must be zero or a positive integer.")


def collect_inputs(raw_paths: list[str]) -> list[Path]:
    """Expand files and one directory level into the papers to summarize."""
    files: list[Path] = []
    for raw in raw_paths:
        path = Path(raw)
        if path.is_dir():
            found = [
                child
                for child in sorted(path.iterdir())
                if child.is_file()
                and not child.name.startswith(".")
                and child.suffix.lower() in _DIR_SUFFIXES
            ]
            if not found:
                raise SystemExit(f"[!] No PDF or text files in {path}")
            files.extend(found)
        elif path.is_file():
            files.append(path)
        else:
            raise SystemExit(f"[!] Input not found: {path}")
    return files


def _summarize_file(args: argparse.Namespace, path: Path) -> str:
    text = extract_text(path, pages=args.pages).strip()
    if not text:
        raise SystemExit("[!] No text extracted — is this a scanned PDF? (OCR is out of scope.)")
    return summarize(
        text,
        backend=args.backend,
        model=args.model,
        max_chars=args.max_chars,
        output_format=args.output_format,
        base_url=args.base_url,
        timeout=args.timeout,
        retries=args.retries,
        no_cache=args.no_cache,
    )


def _report_name(source: Path, output_format: str) -> str:
    suffix = ".json" if output_format == "json" else ".md"
    return f"{source.stem}{suffix}"


def _batch_destinations(
    files: list[Path],
    dest_dir: Path,
    output_format: str,
) -> list[tuple[Path, Path]]:
    if dest_dir.exists() and not dest_dir.is_dir():
        raise SystemExit("[!] --output must be a directory when summarizing more than one file.")
    planned: list[tuple[Path, Path]] = []
    seen: dict[str, Path] = {}
    for source in files:
        name = _report_name(source, output_format)
        if name in seen:
            raise SystemExit(
                f"[!] {source} and {seen[name]} would both write {name}. Rename one of them."
            )
        dest = dest_dir / name
        if dest.resolve() == source.resolve():
            raise SystemExit(f"[!] Refusing to overwrite input {source}")
        seen[name] = source
        planned.append((source, dest))
    dest_dir.mkdir(parents=True, exist_ok=True)
    return planned


def _write_report(report: str, dest: Path) -> None:
    try:
        with open(dest, "w", encoding="utf-8") as handle:
            handle.write(report)
    except OSError as exc:
        raise SystemExit(f"[!] Could not write {dest}: {exc}") from None
    print(f"[*] Report written to {dest}", file=sys.stderr)
