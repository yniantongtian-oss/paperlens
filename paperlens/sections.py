"""Split paper text into sections and context-sized chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Academic headings, matched on their own line (optionally numbered or hashed).
_KNOWN = re.compile(
    r"^(?:"
    r"abstract|introduction|related work|background|preliminaries|"
    r"methods?|methodology|approach|experiments?|experimental setup|"
    r"evaluation|results?|discussion|conclusions?|references|bibliography|"
    r"appendix|acknowledgements?|acknowledgments?|limitations|future work|"
    r"broader impact"
    r")\b",
    re.IGNORECASE,
)
_NUMBERED = re.compile(r"^(?:(?:\d+|[IVXLC]+)(?:\.\d+)*)[.)]?\s+(?P<title>.+)$")
_HASH = re.compile(r"^#{1,6}\s+")
_AND = re.compile(r"^(?:and|&)\b", re.IGNORECASE)
_SENTENCE_START = re.compile(
    r"^(?:we|this|our|the|it|in|to|for|a|an|these|those|on|by|as|at|from|"
    r"with|using|when|if|that|there)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Chunk:
    """One model-sized piece of a paper, labeled with its section title."""

    title: str
    text: str


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split *text* into ``(title, body)`` pairs.

    A heading is a line of its own: a known section name (``Abstract``,
    ``1. Introduction``) or a short numbered title (``3.1 Architecture``).
    Prose before the first heading is titled ``Preamble``. With no headings,
    the document is a single ``Document`` section.
    """
    sections: list[tuple[str, list[str]]] = [("Preamble", [])]
    saw_heading = False
    for line in text.splitlines():
        title = heading_title(line)
        if title:
            saw_heading = True
            sections.append((title, []))
            continue
        sections[-1][1].append(line)

    result = [(title, "\n".join(lines).strip()) for title, lines in sections]
    result = [(title, body) for title, body in result if body]
    if not result:
        stripped = text.strip()
        return [("Document", stripped)] if stripped else []
    if len(result) == 1 and not saw_heading:
        return [("Document", result[0][1])]
    return result


def heading_title(line: str) -> str | None:
    """Return the heading text, or ``None`` when *line* is body text."""
    raw = line.strip()
    if not raw or len(raw) > 90:
        return None
    raw = _HASH.sub("", raw).strip().rstrip(":").strip()
    if not raw or len(raw) > 90:
        return None

    numbered = _NUMBERED.match(raw)
    title = numbered.group("title").strip().rstrip(":").strip() if numbered else raw
    if _is_known(title):
        return raw
    if numbered and _is_short_title(title):
        return raw
    return None


def _is_known(title: str) -> bool:
    match = _KNOWN.match(title)
    if not match:
        return False
    rest = title[match.end() :].strip()
    if not rest:
        return True
    if len(title) > 70 or rest.endswith("."):
        return False
    if _AND.match(rest):
        return True
    return bool(rest[0].isupper() and len(rest.split()) <= 6)


def _is_short_title(title: str) -> bool:
    if not title or len(title) > 50 or title.endswith((".", ",", ";")):
        return False
    words = title.split()
    if not 1 <= len(words) <= 6:
        return False
    if _SENTENCE_START.match(title):
        return False
    return title[0].isupper()


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split *text* into pieces of at most *max_chars* characters.

    Paragraphs stay intact when they fit. Longer paragraphs break on a line
    or a word, and only then on a hard character cut.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    buf = ""

    def flush() -> None:
        nonlocal buf
        if buf:
            chunks.append(buf)
            buf = ""

    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) > max_chars:
            flush()
            chunks.extend(_hard_split(para, max_chars))
            continue
        candidate = f"{buf}\n\n{para}" if buf else para
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            flush()
            buf = para
    flush()
    return chunks


def _hard_split(text: str, max_chars: int) -> list[str]:
    parts: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            window = text[start:end]
            cut = window.rfind("\n")
            if cut < max_chars // 2:
                cut = window.rfind(" ")
            if cut >= max_chars // 2:
                end = start + cut
        if end <= start:
            end = min(start + max(max_chars, 1), length)
        piece = text[start:end].strip()
        if piece:
            parts.append(piece)
        if end >= length:
            break
        nxt = end
        while nxt < length and text[nxt].isspace():
            nxt += 1
        start = nxt if nxt > start else end
    return parts


def plan_chunks(text: str, max_chars: int) -> list[Chunk]:
    """Return section-aware chunks, each at most *max_chars* characters."""
    chunks: list[Chunk] = []
    for title, body in split_sections(text):
        parts = chunk_text(body, max_chars)
        if len(parts) <= 1:
            if parts:
                chunks.append(Chunk(title, parts[0]))
            continue
        total = len(parts)
        for index, part in enumerate(parts, start=1):
            chunks.append(Chunk(f"{title} (part {index}/{total})", part))
    return chunks
