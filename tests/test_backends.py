"""Backend calls, including mocked HTTP and the missing-key error."""

import io
import json
import urllib.error

import pytest

from paperlens import call_backend

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
    def boom(req, timeout=None):
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


def test_unknown_backend():
    with pytest.raises(SystemExit, match="Unknown backend"):
        call_backend("paper", "nope")
