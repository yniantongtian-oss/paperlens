"""Parse a model JSON report into a stable shape."""

from __future__ import annotations

import json
import re

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_KEYS = (
    "tldr",
    "key_contributions",
    "method",
    "key_results",
    "reproduction_notes",
)


def parse_model_json(raw: str) -> dict:
    """Parse a model response into the PaperLens report object.

    Markdown fences and surrounding prose are ignored. ``key_contributions``
    and ``key_results`` accept either a string or a list of strings.
    """
    text = raw.strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise SystemExit("[!] Model did not return a JSON object.")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[!] Could not parse JSON report: {exc}") from None
    if not isinstance(data, dict):
        raise SystemExit("[!] JSON report was not an object.")
    missing = [key for key in _KEYS if key not in data]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(f"[!] JSON report is missing keys: {joined}")

    ordered: dict = {
        "tldr": _as_str(data["tldr"]),
        "key_contributions": _as_list(data["key_contributions"]),
        "method": _as_str(data["method"]),
        "key_results": _as_list(data["key_results"]),
        "reproduction_notes": _as_str(data["reproduction_notes"]),
    }
    for key, value in data.items():
        if key not in ordered and key != "meta":
            ordered[key] = value
    return ordered


def _as_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_as_str(item) for item in value)
    return str(value)


def _as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        return [item for item in (_as_str(entry) for entry in value) if item]
    return [_as_str(value)]
