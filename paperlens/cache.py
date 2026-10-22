"""Disk cache for model responses.

The file name is the SHA-256 of the length-prefixed triple
``(prompt, backend, model)``. Nothing else is part of the key.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path


def cache_key(prompt: str, backend: str, model: str) -> str:
    """Return the hex SHA-256 of *(prompt, backend, model)*."""
    digest = hashlib.sha256()
    for part in (prompt, backend, model):
        encoded = part.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def cache_dir() -> Path:
    """Directory that holds cached responses.

    ``PAPERLENS_CACHE`` wins, then ``$XDG_CACHE_HOME/paperlens``, then
    ``~/.cache/paperlens``.
    """
    override = os.environ.get("PAPERLENS_CACHE", "").strip()
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    root = Path(xdg) if xdg else Path.home() / ".cache"
    return root / "paperlens"


def load_cached(prompt: str, backend: str, model: str) -> str | None:
    """Return a cached response, or ``None`` when there is no entry."""
    path = cache_dir() / cache_key(prompt, backend, model)
    try:
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"[!] Could not read cache: {exc}", file=sys.stderr)
        return None


def store_cached(prompt: str, backend: str, model: str, text: str) -> None:
    """Write *text* under the key for *(prompt, backend, model)*."""
    directory = cache_dir()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / cache_key(prompt, backend, model)
        temporary = directory / f".{path.name}.tmp"
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        print(f"[!] Could not write cache: {exc}", file=sys.stderr)
