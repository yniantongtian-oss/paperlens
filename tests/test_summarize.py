"""Map-reduce coverage: truncation, section labels, and hierarchical reduce."""

import json

from paperlens.summarize import summarize
from tests.helpers import long_paper, prompt_body, request_prompt

CHAT = {"choices": [{"message": {"content": "brief note"}}]}


def test_short_paper_is_one_call(capture_http, capsys):
    calls = capture_http({"response": "done"})
    report = summarize("Widgets improve by 12 percent.", backend="ollama", max_chars=1000)
    assert report.strip() == "done"
    assert len(calls) == 1
    assert "map-reduce" not in capsys.readouterr().err
    assert "12 percent" in request_prompt(calls[0])


def test_truncation_keeps_each_call_inside_max_chars_and_covers_the_tail(capture_http, capsys):
    calls = capture_http({"response": "section note"})
    paper = long_paper()
    max_chars = 200
    summarize(paper, backend="ollama", max_chars=max_chars)

    bodies = [prompt_body(request_prompt(req)) for req in calls]
    assert len(bodies) > 1
    assert all(len(body) <= max_chars for body in bodies)
    assert any("INTRO_TOKEN" in body for body in bodies)
    assert any("RESULT_TOKEN" in body for body in bodies)
    intro_calls = [body for body in bodies if "INTRO_TOKEN" in body]
    assert all("RESULT_TOKEN" not in body for body in intro_calls)

    prompts = [request_prompt(req) for req in calls]
    assert any("Section title: 1. Introduction" in prompt for prompt in prompts)
    assert any("Section title: 2. Methods" in prompt for prompt in prompts)
    assert any(prompt.startswith("You are a research assistant") for prompt in prompts)

    err = capsys.readouterr().err
    assert "truncated" in err
    assert "map-reduce" in err


def test_groq_map_reduce_uses_chat_completions(monkeypatch, capture_http):
    monkeypatch.setenv("GROQ_API_KEY", "secret-key")
    calls = capture_http(CHAT)
    summarize(long_paper(), backend="groq", max_chars=200)
    assert len(calls) > 1
    assert all("api.groq.com" in req.full_url for req in calls)
    bodies = [prompt_body(request_prompt(req)) for req in calls]
    assert all(len(body) <= 200 for body in bodies)
    assert any("RESULT_TOKEN" in body for body in bodies)


def test_hierarchical_reduce_makes_progress(capture_http):
    """A model that echoes the prompt must still finish in a bounded number of calls."""

    def echo_prompt(req):
        return {"response": request_prompt(req)}

    calls = capture_http(echo_prompt)
    summarize(long_paper(), backend="ollama", max_chars=200)
    assert 1 < len(calls) < 40
    bodies = [prompt_body(request_prompt(req)) for req in calls]
    assert all(len(body) <= 200 for body in bodies)


def test_echo_long_paper_keeps_later_sections(capsys):
    report = summarize(long_paper(), backend="echo", max_chars=200)
    assert "dry-run" in report
    assert "INTRO_TOKEN" in report
    assert "METHOD_TOKEN" in report
    assert "RESULT_TOKEN" in report
    assert "## TL;DR" in report
    err = capsys.readouterr().err
    assert "map-reduce" in err


def test_json_report_adds_meta_and_normalizes_lists(capture_http):
    payload = {
        "response": (
            "```json\n"
            '{"tldr": "Widgets win.", "key_contributions": "a new loss", '
            '"method": "train widgets", "key_results": ["12 percent"], '
            '"reproduction_notes": "code is unclear", "meta": {"chunks": 99}}\n'
            "```"
        )
    }
    capture_http(payload)
    report = summarize(
        "Widgets improve by 12 percent.",
        backend="ollama",
        model="qwen3:8b",
        output_format="json",
    )
    data = json.loads(report)
    assert data["tldr"] == "Widgets win."
    assert data["key_contributions"] == ["a new loss"]
    assert data["key_results"] == ["12 percent"]
    assert data["meta"]["map_reduce"] is False
    assert data["meta"]["chunks"] == 1
    assert data["meta"]["backend"] == "ollama"
    assert data["meta"]["model"] == "qwen3:8b"


def test_invalid_json_report(capture_http):
    import pytest

    capture_http({"response": "this is not json"})
    with pytest.raises(SystemExit, match="JSON"):
        summarize("A short paper.", backend="ollama", output_format="json")
