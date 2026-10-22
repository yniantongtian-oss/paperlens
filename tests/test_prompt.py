"""Prompt construction for markdown, JSON, and map-reduce modes."""

import pytest

from paperlens import build_prompt


def test_full_markdown_prompt_has_required_sections():
    prompt = build_prompt("Widgets improve by 12 percent.")
    for heading in (
        "## TL;DR",
        "## Key contributions",
        "## Method",
        "## Key results",
        "## Reproduction notes",
    ):
        assert heading in prompt
    assert "Widgets improve by 12 percent." in prompt
    assert "inventing details" in prompt
    assert prompt.strip().endswith("--- PAPER TEXT END ---")


def test_json_prompt_requests_object_keys():
    prompt = build_prompt("A short paper.", output_format="json")
    for key in ("tldr", "key_contributions", "method", "key_results", "reproduction_notes"):
        assert key in prompt
    assert "## TL;DR" not in prompt
    assert "A short paper." in prompt


def test_section_prompt_names_the_section():
    prompt = build_prompt("accuracy is 91", mode="section", section_title="3. Results")
    assert "Section title: 3. Results" in prompt
    assert "--- SECTION TEXT START ---" in prompt
    assert "accuracy is 91" in prompt


def test_reduce_prompt_treats_notes_as_the_paper():
    prompt = build_prompt("notes", mode="reduce")
    assert "read in chunks" in prompt
    assert "notes" in prompt


def test_unknown_mode_and_format():
    with pytest.raises(ValueError, match="output_format"):
        build_prompt("x", output_format="html")
    with pytest.raises(ValueError, match="mode"):
        build_prompt("x", mode="sideways")
