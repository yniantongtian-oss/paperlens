# PaperLens

PaperLens turns a research paper into a structured briefing: a TL;DR, the contributions, the method in plain language, the key numbers, and what you would need to reproduce the work.

A PDF or a text file is extracted with [pypdf](https://pypdf.readthedocs.io/), packed into a prompt, and sent to a model you choose. Ollama keeps the manuscript on your machine. Groq and any OpenAI-compatible server are there when you want a hosted model.

```
PDF or text → extract_text → section chunks → prompt → model → Markdown or JSON
```

## Install

```bash
pip install .          # runtime
pip install -e ".[dev]"  # checkout, with pytest and ruff
```

That installs a `paperlens` command. `python -m paperlens` is the same entry point.

## Examples

Dry-run the prompt. No model and no API key:

```bash
paperlens paper.pdf --backend echo
```

Summarize locally with Ollama (the default backend, model `qwen3:8b`):

```bash
ollama pull qwen3:8b
paperlens paper.pdf -o report.md
```

Groq's OpenAI-compatible free tier:

```bash
export GROQ_API_KEY=your_key_here   # https://console.groq.com/keys
paperlens paper.pdf --backend groq --model llama-3.3-70b-versatile -o report.md
```

OpenAI, or any Chat Completions server. `--base-url` defaults to `https://api.openai.com/v1`. `OPENAI_BASE_URL` sets the same root. A custom host accepts a run with no API key.

```bash
export OPENAI_API_KEY=your_key_here
paperlens paper.pdf --backend openai --model gpt-4o-mini --format json -o report.json

# Ollama's OpenAI-compatible port, vLLM, or LM Studio
paperlens paper.pdf --backend openai \
  --base-url http://localhost:11434/v1 \
  --model qwen3:8b
```

Text notes take the same flags as a PDF:

```bash
printf 'Abstract\nWe show that widgets improve by 12%%.\n' > notes.txt
paperlens notes.txt --backend echo --format json
```

`--max-chars` is the character budget for one model call (default `12000`). A longer paper is split on section headings, each chunk is summarized, and those notes are reduced into one briefing:

```bash
paperlens paper.pdf --max-chars 8000 -o report.md
```

`--pages` reads a 1-based inclusive range from a PDF. `1-5` is pages 1 through 5, `3` is one page, and `1-3,8` is a list of ranges. It applies to PDFs only.

```bash
paperlens paper.pdf --pages 1-5 -o report.md
```

`--timeout` is the HTTP timeout in seconds (600 for Ollama, 300 for Groq and OpenAI-compatible hosts). `--retries` is how many extra attempts follow an HTTP 429 or 5xx (default `2`). The wait doubles each time, starting at half a second and stopping at 8 seconds. Any other HTTP status fails immediately. The final error includes the status, the number of attempts, and a short snippet of the body.

```bash
paperlens paper.pdf --timeout 180 --retries 4
```

A successful model response is stored on disk. The file name is the SHA-256 of the prompt, the backend, and the model, so the same request is not sent twice. `--no-cache` skips the lookup and the write. `echo` is never cached. A hit does not contact the provider. Set `PAPERLENS_CACHE` to choose the directory; otherwise PaperLens uses `$XDG_CACHE_HOME/paperlens` or `~/.cache/paperlens`. The key does not include `--base-url`, so two OpenAI-compatible hosts that share a model name share a cache entry.

```bash
paperlens paper.pdf --no-cache
```

Several files, or a directory, produce one report per paper. A directory is scanned one level deep for `.pdf`, `.txt`, and `.md` (hidden files are skipped). With more than one paper, `-o` must be a directory and each report is named from the source stem (`.md` or `.json`). Without `-o`, the reports go to stdout, each marked with the file name.

```bash
paperlens a.pdf b.pdf -o reports/
paperlens papers/ -o reports/
```

## Report shape

Markdown reports use these sections, in order:

- TL;DR
- Key contributions
- Method
- Key results
- Reproduction notes

The prompt tells the model to say when the text is missing or unclear.

`--format json` asks the model for one object. PaperLens parses it (fences are fine), checks the report schema, and adds a `meta` object about the run. A response that is not a JSON object is an error. Missing fields are filled with defaults (`""` for text, `[]` for lists) and named on stderr:

```json
{
  "tldr": "One or two sentences.",
  "key_contributions": ["..."],
  "method": "...",
  "key_results": ["..."],
  "reproduction_notes": "...",
  "meta": {
    "backend": "groq",
    "model": "llama-3.3-70b-versatile",
    "map_reduce": true,
    "chunks": 4,
    "max_chars": 12000,
    "sections": ["Abstract", "1. Introduction", "3. Results (part 1/2)"]
  }
}
```

`key_contributions` and `key_results` accept a string or a list of strings, and are stored as lists.

## How a long paper is read

1. A heading on its own line starts a section: `Abstract`, `1. Introduction`, `3.1 Architecture`, or a markdown `#` heading.
2. Each section is packed into chunks of at most `--max-chars` characters, breaking on paragraphs first.
3. Each chunk is summarized on its own (map).
4. Those notes are synthesized into the final briefing (reduce). Notes that still overflow the budget are merged in pairs until they fit.

A paper that fits in `--max-chars` stays a single call. With `--backend echo`, a long paper prints the final reduce prompt, including every section, so you can see that the tail was kept. Real backends keep every call inside the budget. Echo can run past it, because nothing is sent to a model.

## Backends

| Backend | Default model | Auth | Request goes to |
| --- | --- | --- | --- |
| `ollama` (default) | `qwen3:8b` | none | `localhost:11434` |
| `groq` | `llama-3.3-70b-versatile` | `GROQ_API_KEY` | `api.groq.com` |
| `openai` | `gpt-4o-mini` | `OPENAI_API_KEY` on the official host | `--base-url` or `OPENAI_BASE_URL` |
| `echo` | — | none | nowhere; the prompt is printed |

## Library

```python
from paperlens import extract_text, build_prompt, call_backend, main

text = extract_text("paper.pdf")
prompt = build_prompt(text)  # or build_prompt(text, output_format="json")
print(call_backend(prompt, "echo"))
raise SystemExit(main(["paper.pdf", "--backend", "echo", "-o", "report.md"]))
```

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

CI runs ruff and pytest on Python 3.11 and 3.12.

## Limitations

- A scanned PDF has no text layer. PaperLens stops and says so. `--pages` cannot OCR a page.
- Section detection needs the heading on its own line. When extraction glues a heading onto the paragraph, that stretch is chunked by size and titled `Document`.
- Cached responses are reused until you pass `--no-cache`. Changing `--base-url` but not the model name still hits the old entry.
- The briefing is only as careful as the model. Check claims against the paper.

## License

[MIT](LICENSE)
