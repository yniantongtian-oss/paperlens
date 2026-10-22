"""Prompt templates for a single pass, a section note, and the final reduce."""

from __future__ import annotations

_FULL_MARKDOWN = """\
You are a research assistant helping an undergraduate student read a scientific paper.
Read the paper text below and produce a structured summary in Markdown with EXACTLY these sections:

## TL;DR
One or two sentences: what problem does this paper solve and what is the headline result?

## Key contributions
Bullet list of the paper's main contributions.

## Method
How the proposed method works, in plain language a sophomore can follow.

## Key results
The most important experimental/quantitative results mentioned.

## Reproduction notes
What would be needed to reproduce this work (data, code, compute)? Flag anything unclear.

Be honest: if the text is truncated or a section is missing, say so instead of inventing details.
"""

_REDUCE_MARKDOWN = """\
You are a research assistant helping an undergraduate student read a scientific paper.
The notes below are faithful summaries of each section. The paper was read in chunks;
treat these notes as the full text and produce a structured summary in Markdown
with EXACTLY these sections:

## TL;DR
One or two sentences: what problem does this paper solve and what is the headline result?

## Key contributions
Bullet list of the paper's main contributions.

## Method
How the proposed method works, in plain language a sophomore can follow.

## Key results
The most important experimental/quantitative results mentioned.

## Reproduction notes
What would be needed to reproduce this work (data, code, compute)? Flag anything unclear.

Be honest: if a section is missing from the notes, say so instead of inventing details.
"""

_JSON_INSTRUCTIONS = """\
Return ONLY a JSON object with these keys and no surrounding prose:
{"tldr": "one or two sentences", "key_contributions": ["contribution"], \
"method": "plain-language method", "key_results": ["result"], \
"reproduction_notes": "what is needed to reproduce this, and what is unclear"}
Be honest: if the source is thin or a section is missing, say so in the relevant
field instead of inventing details.
"""

_FULL_JSON_LEAD = """\
You are a research assistant helping an undergraduate student read a scientific paper.
Read the paper text below and answer from that text only.
"""

_REDUCE_JSON_LEAD = """\
You are a research assistant helping an undergraduate student read a scientific paper.
The notes below are faithful summaries of each section of the paper, which was read
in chunks. Treat these notes as the full text and answer from them only.
"""


def build_prompt(
    paper_text: str,
    *,
    output_format: str = "markdown",
    mode: str = "full",
    section_title: str | None = None,
) -> str:
    """Build the prompt for one model call.

    ``mode="full"`` is a single-pass briefing. ``mode="section"`` asks for notes
    on one chunk. ``mode="intermediate"`` merges notes that still do not fit in
    one reduce call. ``mode="reduce"`` asks for the final briefing.

    ``output_format`` is ``markdown`` or ``json``. Section and intermediate
    calls always ask for prose, so their notes can be merged.
    """
    if output_format not in {"markdown", "json"}:
        raise ValueError(f"unknown output_format: {output_format}")
    if mode == "section":
        return _section_prompt(paper_text, section_title or "Section")
    if mode == "intermediate":
        return _intermediate_prompt(paper_text)
    if mode not in {"full", "reduce"}:
        raise ValueError(f"unknown mode: {mode}")

    if output_format == "json":
        lead = _REDUCE_JSON_LEAD if mode == "reduce" else _FULL_JSON_LEAD
        instructions = f"{lead}{_JSON_INSTRUCTIONS}"
    else:
        instructions = _REDUCE_MARKDOWN if mode == "reduce" else _FULL_MARKDOWN
    return (
        f"{instructions}\n"
        "--- PAPER TEXT START ---\n"
        f"{paper_text}\n"
        "--- PAPER TEXT END ---\n"
    )


def _section_prompt(text: str, title: str) -> str:
    return (
        "You are taking notes on one section of a scientific paper for a later synthesis.\n"
        f"Section title: {title}\n"
        "\n"
        "Summarize only this section for an undergraduate reader. Preserve numbers,\n"
        "dataset names, and method names. If the section is only references, list\n"
        "up to eight relevant citations. Do not invent details.\n"
        "\n"
        "--- SECTION TEXT START ---\n"
        f"{text}\n"
        "--- SECTION TEXT END ---\n"
    )


def _intermediate_prompt(text: str) -> str:
    return (
        "You are consolidating notes from several sections of a scientific paper.\n"
        "Merge them into a shorter briefing. Preserve concrete claims, numbers,\n"
        "dataset names, and method names. Do not add facts that are not in the notes.\n"
        "\n"
        "--- PAPER TEXT START ---\n"
        f"{text}\n"
        "--- PAPER TEXT END ---\n"
    )
