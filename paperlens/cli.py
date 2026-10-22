"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys

from paperlens.backends import DEFAULT_MODELS
from paperlens.extract import extract_text
from paperlens.summarize import DEFAULT_MAX_CHARS, summarize
from paperlens.version import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paperlens",
        description="Summarize a research paper with a local or OpenAI-compatible model.",
    )
    parser.add_argument("input", help="PDF or text file of the paper")
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
        help="write the report to this path (default: stdout)",
    )
    parser.add_argument("--version", action="version", version=f"paperlens {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run PaperLens. *argv* is the argument list, without the program name."""
    args = build_parser().parse_args(argv)
    if args.max_chars < 1:
        raise SystemExit("[!] --max-chars must be a positive integer.")
    if args.base_url and args.backend != "openai":
        raise SystemExit("[!] --base-url applies to the openai backend only.")

    text = extract_text(args.input).strip()
    if not text:
        raise SystemExit("[!] No text extracted — is this a scanned PDF? (OCR is out of scope.)")

    report = summarize(
        text,
        backend=args.backend,
        model=args.model,
        max_chars=args.max_chars,
        output_format=args.output_format,
        base_url=args.base_url,
    )
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(report)
        except OSError as exc:
            raise SystemExit(f"[!] Could not write {args.output}: {exc}") from None
        print(f"[*] Report written to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(report)
    return 0
