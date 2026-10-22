"""Model backends: echo, Ollama, Groq, and any OpenAI-compatible server."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable

from paperlens.cache import load_cached, store_cached

DEFAULT_MODELS = {
    "ollama": "qwen3:8b",
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "echo": "echo",
}

DEFAULT_RETRIES = 2
_DEFAULT_TIMEOUT = {
    "ollama": 600.0,
    "groq": 300.0,
    "openai": 300.0,
}
_BACKOFF_BASE = 0.5
_BACKOFF_CAP = 8.0

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
    timeout: float | None = None,
    retries: int = DEFAULT_RETRIES,
    no_cache: bool = False,
) -> str:
    """Send *prompt* to *backend* and return the model text.

    ``echo`` returns the prompt unchanged (the CLI wraps it as a dry-run).
    ``ollama`` talks to a local Ollama server. ``groq`` uses Groq's
    OpenAI-compatible chat API. ``openai`` uses Chat Completions at
    ``base_url`` or ``OPENAI_BASE_URL`` (default ``https://api.openai.com/v1``).

    HTTP 429 and 5xx are retried with exponential backoff. A successful
    response is reused from the disk cache unless *no_cache* is set.
    """
    name = backend.lower().strip()
    if base_url and name != "openai":
        raise SystemExit("[!] --base-url applies to the openai backend only.")
    _check_transport(timeout, retries)
    chosen = resolve_model(name, model)
    if name == "echo":
        return prompt
    if name not in _DEFAULT_TIMEOUT:
        known = ", ".join(DEFAULT_MODELS)
        raise SystemExit(f"[!] Unknown backend: {name}. Choose from {known}.")

    limit = float(timeout if timeout is not None else _DEFAULT_TIMEOUT[name])
    if not no_cache:
        cached = load_cached(prompt, name, chosen)
        if cached is not None:
            print(f"[*] Cache hit ({name}, {chosen}).", file=sys.stderr)
            return cached

    if name == "ollama":
        text = _call_ollama(prompt, chosen, timeout=limit, retries=retries)
    elif name == "groq":
        text = _call_groq(prompt, chosen, timeout=limit, retries=retries)
    else:
        text = _call_openai(prompt, chosen, base_url=base_url, timeout=limit, retries=retries)

    if not no_cache:
        store_cached(prompt, name, chosen, text)
    return text


def _check_transport(timeout: float | None, retries: int) -> None:
    if timeout is not None and (
        isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not timeout > 0
    ):
        raise SystemExit("[!] --timeout must be a positive number.")
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise SystemExit("[!] --retries must be zero or a positive integer.")


def _call_ollama(prompt: str, model: str, *, timeout: float, retries: int) -> str:
    data = _request_json(
        OLLAMA_URL,
        {"model": model, "prompt": prompt, "stream": False},
        headers={"Content-Type": "application/json"},
        timeout=timeout,
        retries=retries,
        connection_error=lambda exc: (
            f"[!] Could not reach Ollama at localhost:11434: {exc}\n"
            "    Start it with `ollama serve` and pull a model, e.g. `ollama pull qwen3:8b`."
        ),
    )
    try:
        return data["response"]
    except (KeyError, TypeError) as exc:
        raise SystemExit(f"[!] Unexpected Ollama response: {exc}") from None


def _call_groq(prompt: str, model: str, *, timeout: float, retries: int) -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise SystemExit(
            "[!] GROQ_API_KEY is not set. Create a free key at https://console.groq.com/keys"
        )
    return _chat_completions(
        url=GROQ_URL,
        api_key=key,
        model=model,
        prompt=prompt,
        timeout=timeout,
        retries=retries,
    )


def _call_openai(
    prompt: str,
    model: str,
    *,
    base_url: str | None,
    timeout: float,
    retries: int,
) -> str:
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
        timeout=timeout,
        retries=retries,
    )


def _openai_url(base_url: str | None) -> tuple[str, bool]:
    env_root = os.environ.get("OPENAI_BASE_URL", "")
    root = (base_url or env_root or OPENAI_DEFAULT_ROOT).strip().rstrip("/")
    official = "api.openai.com" in root
    if root.endswith("/chat/completions"):
        return root, official
    return f"{root}/chat/completions", official


def _chat_completions(
    *,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout: float,
    retries: int,
) -> str:
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
        retries=retries,
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
    timeout: float,
    retries: int,
    connection_error: Callable[[OSError], str],
) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers=headers,
    )
    raw = _read_with_retries(
        request,
        timeout=timeout,
        retries=retries,
        connection_error=connection_error,
    )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[!] Response from {url} was not JSON: {exc}") from None
    if not isinstance(parsed, dict):
        raise SystemExit(f"[!] Response from {url} was not a JSON object.")
    return parsed


def _read_with_retries(
    request: urllib.request.Request,
    *,
    timeout: float,
    retries: int,
    connection_error: Callable[[OSError], str],
) -> bytes:
    """Read *request*, retrying HTTP 429 and 5xx with exponential backoff."""
    attempts = retries + 1
    url = request.full_url
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            if _retryable(exc.code) and attempt + 1 < attempts:
                delay = min(_BACKOFF_BASE * (2**attempt), _BACKOFF_CAP)
                print(
                    f"[*] HTTP {exc.code} from {url}; retrying in {delay:.1f}s "
                    f"(attempt {attempt + 2}/{attempts})",
                    file=sys.stderr,
                )
                time.sleep(delay)
                continue
            raise SystemExit(
                _http_failure(url, exc.code, detail, attempts, retried=_retryable(exc.code))
            ) from None
        except TimeoutError as exc:
            raise SystemExit(f"[!] Timed out after {timeout:g}s calling {url}: {exc}") from None
        except OSError as exc:
            raise SystemExit(connection_error(exc)) from None
    raise SystemExit(f"[!] No response from {url}")


def _retryable(status: int) -> bool:
    return status == 429 or 500 <= status <= 599


def _http_failure(url: str, status: int, detail: str, attempts: int, *, retried: bool) -> str:
    if retried:
        message = f"[!] HTTP {status} from {url} after {attempts} attempts"
    else:
        message = f"[!] HTTP {status} from {url}"
    if detail:
        return f"{message}: {detail}"
    return message
