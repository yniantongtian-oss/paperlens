"""HTTP stand-ins for the Ollama and chat-completions backends."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._raw = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


@pytest.fixture
def capture_http(monkeypatch: pytest.MonkeyPatch):
    """Patch urlopen and record each request. *reply* is a payload or a function.

    Each request grows a ``timeout_seconds`` attribute with the timeout that
    ``urlopen`` received.
    """

    def install(reply: dict[str, Any] | Callable[..., dict[str, Any]]):
        calls = []

        def fake(req, timeout=None):
            req.timeout_seconds = timeout
            calls.append(req)
            payload = reply(req) if callable(reply) else reply
            return FakeResponse(payload)

        monkeypatch.setattr("paperlens.backends.urllib.request.urlopen", fake)
        return calls

    return install


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Keep response caches out of the home directory and out of other tests."""
    monkeypatch.setenv("PAPERLENS_CACHE", str(tmp_path / "paperlens-cache"))
