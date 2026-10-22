"""Small helpers shared by the tests."""

from __future__ import annotations

import json
import re

_BODY = re.compile(
    r"--- (?:PAPER|SECTION) TEXT START ---\n(.*?)\n--- (?:PAPER|SECTION) TEXT END ---",
    re.DOTALL,
)


def prompt_body(prompt: str) -> str:
    match = _BODY.search(prompt)
    assert match, prompt
    return match.group(1)


def request_prompt(req) -> str:
    payload = json.loads(req.data.decode())
    if "prompt" in payload:
        return payload["prompt"]
    return payload["messages"][0]["content"]


def long_paper() -> str:
    """A three-section paper whose bodies are longer than a small chunk budget."""

    def section(title: str, token: str) -> str:
        return f"{title}\n" + ("lorem " * 80) + token + "\n"

    return (
        section("1. Introduction", "INTRO_TOKEN")
        + "\n"
        + section("2. Methods", "METHOD_TOKEN")
        + "\n"
        + section("3. Results", "RESULT_TOKEN")
    )
