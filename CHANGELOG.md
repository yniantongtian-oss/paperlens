# Changelog

## 0.3.0 — 2026-10-22

### Added

- MIT `LICENSE`. Project metadata already declared MIT in `pyproject.toml` (license text and the OSI classifier).
- `--timeout` and `--retries`. HTTP backends retry 429 and 5xx with exponential backoff, then report the status, the attempt count, and a snippet of the body.
- `--pages`, a 1-based page range for PDFs (`1-5`, `3`, or `1-3,8`).
- An on-disk response cache keyed by SHA-256 of the prompt, backend, and model. `--no-cache` skips it.
- Batch input: several files, or a directory of PDF and text files, produce one report each.
- JSON reports are checked against the report schema. Missing fields are filled with defaults and named on stderr.

## 0.2.0

- Importable package, section-aware map-reduce summaries, an OpenAI-compatible backend, JSON reports, and CI (ruff and pytest on Python 3.11 and 3.12).
