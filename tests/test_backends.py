"""Backend calls, including mocked HTTP and the missing-key error."""

import io
import json
import urllib.error

import pytest

from paperlens import call_backend
from paperlens.cache import cache_dir, cache_key

CHAT = {"choices": [{"message": {"content": "MODEL REPORT"}}]}


def test_echo_returns_the_prompt():
    assert call_backend("hello prompt", "echo") == "hello prompt"


def test_ollama_request(capture_http):
    calls = capture_http({"response": "local summary"})
    assert call_backend("paper", "ollama") == "local summary"
    assert len(calls) == 1
    req = calls[0]
    assert req.full_url == "http://localhost:11434/api/generate"
    body = json.loads(req.data.decode())
    assert body == {"model": "qwen3:8b", "prompt": "paper", "stream": False}


def test_groq_request(monkeypatch, capture_http):
    monkeypatch.setenv("GROQ_API_KEY", "secret-key")
    calls = capture_http(CHAT)
    assert call_backend("paper", "groq", "llama-test") == "MODEL REPORT"
    req = calls[0]
    assert req.full_url == "https://api.groq.com/openai/v1/chat/completions"
    assert req.get_header("Authorization") == "Bearer secret-key"
    body = json.loads(req.data.decode())
    assert body["model"] == "llama-test"
    assert body["temperature"] == 0.2
    assert body["messages"] == [{"role": "user", "content": "paper"}]


def test_groq_default_model(monkeypatch, capture_http):
    monkeypatch.setenv("GROQ_API_KEY", "secret-key")
    calls = capture_http(CHAT)
    call_backend("paper", "groq")
    body = json.loads(calls[0].data.decode())
    assert body["model"] == "llama-3.3-70b-versatile"


def test_missing_groq_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    def fail(*args, **kwargs):
        raise AssertionError("urlopen should not be called")

    monkeypatch.setattr("paperlens.backends.urllib.request.urlopen", fail)
    with pytest.raises(SystemExit, match="GROQ_API_KEY"):
        call_backend("paper", "groq")


def test_openai_request_and_base_url(monkeypatch, capture_http):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    calls = capture_http(CHAT)
    text = call_backend(
        "paper",
        "openai",
        "gpt-test",
        base_url="https://example.test/v1",
    )
    assert text == "MODEL REPORT"
    req = calls[0]
    assert req.full_url == "https://example.test/v1/chat/completions"
    assert req.get_header("Authorization") == "Bearer sk-test"
    assert json.loads(req.data.decode())["model"] == "gpt-test"


def test_openai_base_url_already_includes_chat_path(monkeypatch, capture_http):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    calls = capture_http(CHAT)
    call_backend(
        "paper",
        "openai",
        base_url="http://localhost:11434/v1/chat/completions",
    )
    assert calls[0].full_url == "http://localhost:11434/v1/chat/completions"
    assert calls[0].get_header("Authorization") == "Bearer local"


def test_missing_openai_key_on_official_host(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        call_backend("paper", "openai")


def test_openai_base_url_from_env(monkeypatch, capture_http):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    calls = capture_http(CHAT)
    call_backend("paper", "openai", "local-model")
    assert calls[0].full_url == "http://127.0.0.1:8000/v1/chat/completions"


def test_base_url_rejected_for_other_backends():
    with pytest.raises(SystemExit, match="base-url"):
        call_backend("paper", "echo", base_url="http://localhost:1")


def test_ollama_unreachable(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr("paperlens.backends.urllib.request.urlopen", boom)
    with pytest.raises(SystemExit, match="Ollama"):
        call_backend("paper", "ollama")


def test_http_error_is_reported(monkeypatch):
    calls = []

    def boom(req, timeout=None):
        calls.append(req)
        raise urllib.error.HTTPError(
            req.full_url,
            401,
            "Unauthorized",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"nope"}'),
        )

    monkeypatch.setattr("paperlens.backends.urllib.request.urlopen", boom)
    with pytest.raises(SystemExit, match="HTTP 401"):
        call_backend("paper", "ollama")
    assert len(calls) == 1


def test_retries_after_two_429s_then_succeeds(monkeypatch, capture_http, capsys):
    delays = []
    monkeypatch.setattr("paperlens.backends.time.sleep", delays.append)
    state = {"n": 0}

    def reply(req):
        state["n"] += 1
        if state["n"] < 3:
            raise urllib.error.HTTPError(
                req.full_url,
                429,
                "Too Many Requests",
                hdrs=None,
                fp=io.BytesIO(b"slow down"),
            )
        return {"response": "recovered"}

    calls = capture_http(reply)
    assert call_backend("paper", "ollama", timeout=7) == "recovered"
    assert len(calls) == 3
    assert [call.timeout_seconds for call in calls] == [7, 7, 7]
    assert delays == [0.5, 1.0]
    err = capsys.readouterr().err
    assert "HTTP 429" in err
    assert "retrying in 0.5s" in err
    assert "retrying in 1.0s" in err


def test_retries_exhausted_on_5xx(monkeypatch, capture_http):
    monkeypatch.setattr("paperlens.backends.time.sleep", lambda delay: None)

    def reply(req):
        raise urllib.error.HTTPError(
            req.full_url,
            503,
            "unavailable",
            hdrs=None,
            fp=io.BytesIO(b"down"),
        )

    capture_http(reply)
    with pytest.raises(SystemExit, match=r"HTTP 503 from .+ after 3 attempts: down"):
        call_backend("paper", "ollama")


def test_second_call_skips_backend(monkeypatch, capture_http):
    monkeypatch.setenv("GROQ_API_KEY", "secret-key")

    def reply(req):
        if req.full_url.startswith("http://localhost:11434"):
            return {"response": "from-ollama"}
        return {"choices": [{"message": {"content": "from-groq"}}]}

    calls = capture_http(reply)
    assert call_backend("same prompt", "ollama", "qwen3:8b") == "from-ollama"
    assert call_backend("same prompt", "ollama", "qwen3:8b") == "from-ollama"
    assert len(calls) == 1
    stored = cache_dir() / cache_key("same prompt", "ollama", "qwen3:8b")
    assert stored.is_file()
    assert stored.read_text(encoding="utf-8") == "from-ollama"

    assert call_backend("same prompt", "ollama", "other-model") == "from-ollama"
    assert call_backend("other prompt", "ollama", "qwen3:8b") == "from-ollama"
    assert call_backend("same prompt", "groq", "qwen3:8b") == "from-groq"
    assert len(calls) == 4
    assert call_backend("same prompt", "ollama", "qwen3:8b") == "from-ollama"
    assert len(calls) == 4


def test_no_cache_calls_backend_again(capture_http):
    calls = capture_http({"response": "from-model"})
    assert call_backend("paper", "ollama", no_cache=True) == "from-model"
    assert call_backend("paper", "ollama", no_cache=True) == "from-model"
    assert len(calls) == 2
    assert list(cache_dir().glob("*")) == []


def test_cache_key_changes_with_each_field():
    base = cache_key("prompt", "ollama", "qwen3:8b")
    assert len(base) == 64
    assert int(base, 16) >= 0
    assert base == cache_key("prompt", "ollama", "qwen3:8b")
    assert base != cache_key("other", "ollama", "qwen3:8b")
    assert base != cache_key("prompt", "groq", "qwen3:8b")
    assert base != cache_key("prompt", "ollama", "other")


def test_unknown_backend():
    with pytest.raises(SystemExit, match="Unknown backend"):
        call_backend("paper", "nope")
