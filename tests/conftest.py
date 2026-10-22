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
    """Patch urlopen and record each request. *reply* is a payload or a function."""

    def install(reply: dict[str, Any] | Callable[..., dict[str, Any]]):
        calls = []

        def fake(req, timeout=None):
            calls.append(req)
            payload = reply(req) if callable(reply) else reply
            return FakeResponse(payload)

        monkeypatch.setattr("paperlens.backends.urllib.request.urlopen", fake)
        return calls

    return install
