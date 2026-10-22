"""Model backends: echo, Ollama, Groq, and any OpenAI-compatible server."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable

DEFAULT_MODELS = {
    "ollama": "qwen3:8b",
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "echo": "echo",
}

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OLLAMA_URL = "http://localhost:11434/api/generate"
OPENAI_DEFAULT_ROOT = "https://api.openai.com/v1"


def resolve_model(backend: str, model: str | None) -> str:
    """Return *model*, or the default for *backend*."""
    if model:
        return model
    try:
        return DEFAULT_MODELS[backend]
    except KeyError:
        known = ", ".join(DEFAULT_MODELS)
        raise SystemExit(f"[!] Unknown backend: {backend}. Choose from {known}.") from None


def call_backend(
    prompt: str,
    backend: str,
    model: str | None = None,
    *,
    base_url: str | None = None,
) -> str:
    """Send *prompt* to *backend* and return the model text.

    ``echo`` returns the prompt unchanged (the CLI wraps it as a dry-run).
    ``ollama`` talks to a local Ollama server. ``groq`` uses Groq's
    OpenAI-compatible chat API. ``openai`` uses Chat Completions at
    ``base_url`` or ``OPENAI_BASE_URL`` (default ``https://api.openai.com/v1``).
    """
    name = backend.lower().strip()
    if base_url and name != "openai":
        raise SystemExit("[!] --base-url applies to the openai backend only.")
    chosen = resolve_model(name, model)
    if name == "echo":
        return prompt
    if name == "ollama":
        return _call_ollama(prompt, chosen)
    if name == "groq":
        return _call_groq(prompt, chosen)
    if name == "openai":
        return _call_openai(prompt, chosen, base_url=base_url)
    known = ", ".join(DEFAULT_MODELS)
    raise SystemExit(f"[!] Unknown backend: {name}. Choose from {known}.")


def _call_ollama(prompt: str, model: str) -> str:
    data = _request_json(
        OLLAMA_URL,
        {"model": model, "prompt": prompt, "stream": False},
        headers={"Content-Type": "application/json"},
        timeout=600,
        connection_error=lambda exc: (
            f"[!] Could not reach Ollama at localhost:11434: {exc}\n"
            "    Start it with `ollama serve` and pull a model, e.g. `ollama pull qwen3:8b`."
        ),
    )
    try:
        return data["response"]
    except (KeyError, TypeError) as exc:
        raise SystemExit(f"[!] Unexpected Ollama response: {exc}") from None


def _call_groq(prompt: str, model: str) -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise SystemExit(
            "[!] GROQ_API_KEY is not set. Create a free key at https://console.groq.com/keys"
        )
    return _chat_completions(url=GROQ_URL, api_key=key, model=model, prompt=prompt, timeout=300)


def _call_openai(prompt: str, model: str, *, base_url: str | None) -> str:
    url, official = _openai_url(base_url)
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key and official:
        raise SystemExit(
            "[!] OPENAI_API_KEY is not set. Create a key at "
            "https://platform.openai.com/api-keys, or pass --base-url for another "
            "OpenAI-compatible server."
        )
    return _chat_completions(
        url=url,
        api_key=key or "local",
        model=model,
        prompt=prompt,
        timeout=300,
    )


def _openai_url(base_url: str | None) -> tuple[str, bool]:
    env_root = os.environ.get("OPENAI_BASE_URL", "")
    root = (base_url or env_root or OPENAI_DEFAULT_ROOT).strip().rstrip("/")
    official = "api.openai.com" in root
    if root.endswith("/chat/completions"):
        return root, official
    return f"{root}/chat/completions", official


def _chat_completions(*, url: str, api_key: str, model: str, prompt: str, timeout: int) -> str:
    data = _request_json(
        url,
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        },
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        timeout=timeout,
        connection_error=lambda exc: f"[!] Could not reach {url}: {exc}",
    )
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise SystemExit(f"[!] Unexpected chat response from {url}: {exc}") from None


def _request_json(
    url: str,
    payload: dict,
    *,
    headers: dict[str, str],
    timeout: int,
    connection_error: Callable[[OSError], str],
) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise SystemExit(f"[!] HTTP {exc.code} from {url}: {detail}") from None
    except OSError as exc:
        raise SystemExit(connection_error(exc)) from None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[!] Response from {url} was not JSON: {exc}") from None
    if not isinstance(parsed, dict):
        raise SystemExit(f"[!] Response from {url} was not a JSON object.")
    return parsed
