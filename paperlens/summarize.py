"""Single-pass and section-aware map-reduce summarization."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable

from paperlens.backends import DEFAULT_RETRIES, call_backend, resolve_model
from paperlens.prompt import build_prompt
from paperlens.report import parse_model_json
from paperlens.sections import plan_chunks

DEFAULT_MAX_CHARS = 12000


def summarize(
    text: str,
    *,
    backend: str,
    model: str | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    output_format: str = "markdown",
    base_url: str | None = None,
    timeout: float | None = None,
    retries: int = DEFAULT_RETRIES,
    no_cache: bool = False,
) -> str:
    """Summarize *text* with *backend*.

    A paper that fits in ``max_chars`` is one model call. A longer paper is
    split on section headings, each chunk is summarized, and those notes are
    reduced to one briefing. Every real model call carries at most
    ``max_chars`` characters of paper or notes.
    """
    if max_chars < 1:
        raise SystemExit("[!] --max-chars must be a positive integer.")
    cleaned = text.strip()
    if not cleaned:
        raise SystemExit("[!] No text extracted — is this a scanned PDF? (OCR is out of scope.)")

    def complete(prompt: str) -> str:
        return call_backend(
            prompt,
            backend,
            model,
            base_url=base_url,
            timeout=timeout,
            retries=retries,
            no_cache=no_cache,
        )

    chunks = plan_chunks(cleaned, max_chars)
    if not chunks:
        raise SystemExit("[!] No text extracted — is this a scanned PDF? (OCR is out of scope.)")

    map_reduce = len(cleaned) > max_chars
    if map_reduce:
        print(
            f"[*] Paper is {len(cleaned)} chars, above --max-chars {max_chars}; "
            f"truncated into {len(chunks)} chunk(s) for section-aware map-reduce.",
            file=sys.stderr,
        )
        partials = [
            _map_chunk(chunk.title, chunk.text, backend, max_chars, complete) for chunk in chunks
        ]
        raw, prompt = _reduce(
            partials,
            backend=backend,
            max_chars=max_chars,
            output_format=output_format,
            complete=complete,
        )
        section_titles = [chunk.title for chunk in chunks]
        chunk_count = len(chunks)
    else:
        prompt = build_prompt(cleaned, output_format=output_format, mode="full")
        raw = complete(prompt)
        section_titles = []
        chunk_count = 1

    meta = {
        "backend": backend,
        "model": model or resolve_model(backend, None),
        "map_reduce": map_reduce,
        "chunks": chunk_count,
        "max_chars": max_chars,
        "sections": section_titles,
    }
    return _present(raw, prompt, backend, output_format, meta)


def _map_chunk(
    title: str,
    text: str,
    backend: str,
    max_chars: int,
    complete: Callable[[str], str],
) -> str:
    if backend == "echo":
        # Keep the whole chunk. Echo never sends it to a model.
        return _labeled(title, text, limit=len(title) + len(text) + 16)
    prompt = build_prompt(text, mode="section", section_title=title)
    summary = complete(prompt)
    return _labeled(title, summary, limit=_note_limit(max_chars))


def _note_limit(max_chars: int) -> int:
    """Largest note that still lets two notes fit in one reduce call."""
    return max(1, (max_chars - 2) // 2)


def _labeled(title: str, body: str, limit: int) -> str:
    header = f"### {title}\n"
    if len(header) >= limit:
        return header[:limit]
    return header + body.strip()[: limit - len(header)]


def _reduce(
    partials: list[str],
    *,
    backend: str,
    max_chars: int,
    output_format: str,
    complete: Callable[[str], str],
    depth: int = 0,
) -> tuple[str, str]:
    if depth > 12:
        raise SystemExit("[!] Map-reduce did not converge; raise --max-chars and retry.")
    if backend == "echo":
        combined = "\n\n".join(part.strip() for part in partials if part.strip())
        prompt = build_prompt(combined, output_format=output_format, mode="reduce")
        raw = complete(prompt)
        return raw, prompt

    limit = _note_limit(max_chars)
    clipped = [part.strip()[:limit] for part in partials if part.strip()]
    if not clipped:
        clipped = ["(no section notes)"]
    groups = _pack(clipped, max_chars)
    if len(groups) >= len(clipped) and len(clipped) > 1:
        groups = [
            "\n\n".join(clipped[index : index + 2])[:max_chars]
            for index in range(0, len(clipped), 2)
        ]
    if len(groups) == 1:
        prompt = build_prompt(groups[0], output_format=output_format, mode="reduce")
        raw = complete(prompt)
        return raw, prompt

    intermediates = [complete(build_prompt(group, mode="intermediate")) for group in groups]
    return _reduce(
        intermediates,
        backend=backend,
        max_chars=max_chars,
        output_format=output_format,
        complete=complete,
        depth=depth + 1,
    )


def _pack(blocks: list[str], max_chars: int) -> list[str]:
    groups: list[str] = []
    buf = ""
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if len(block) > max_chars:
            block = block[:max_chars]
        if not buf:
            buf = block
            continue
        candidate = f"{buf}\n\n{block}"
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            groups.append(buf)
            buf = block
    if buf:
        groups.append(buf)
    return groups


def _present(raw: str, prompt: str, backend: str, output_format: str, meta: dict) -> str:
    if backend == "echo":
        return _format_echo(prompt)
    if output_format == "json":
        data = parse_model_json(raw)
        data["meta"] = meta
        return json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if raw.endswith("\n"):
        return raw
    return raw + "\n"


def _format_echo(prompt: str) -> str:
    fence = "```"
    while fence in prompt:
        fence += "`"
    return (
        "<!-- dry-run: the prompt below would be sent to the model -->\n\n"
        f"{fence}\n{prompt}\n{fence}\n"
    )
