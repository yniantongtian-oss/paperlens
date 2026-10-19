# PaperLens

Read a research paper with an **open-weight model** — no API bills, no data leaving your device (Ollama backend).

Upload a PDF, get a structured Markdown report: TL;DR, key contributions, method in plain language, key results, and reproduction notes. Built for students who read papers faster than they can fully digest them.

## How it works

```
PDF → text extraction (pypdf) → structured prompt → open-weight model → Markdown report
```

The model's job is narrow and honest: the prompt forces fixed sections and tells the model to say "unclear" instead of inventing details when the text is truncated.

## Quick start

```bash
pip install -r requirements.txt

# Option A: fully local with Ollama (data never leaves your machine)
ollama pull qwen3:8b
python paperlens.py paper.pdf -o report.md

# Option B: Groq Cloud free tier (open-weight Llama, OpenAI-compatible, no credit card)
export GROQ_API_KEY=your_key_here   # free at https://console.groq.com/keys
python paperlens.py paper.pdf --backend groq -o report.md

# Option C: dry-run — verify the pipeline without any model
python paperlens.py paper.pdf --backend echo
```

## Why open?

- **Cost:** SOTA proprietary APIs charge per token; a local 8B model or Groq's free tier summarizes papers for $0.
- **Privacy:** unpublished manuscripts and thesis drafts stay on your own device with the Ollama backend.
- **Reproducibility:** the whole pipeline is ~130 lines of stdlib Python plus `pypdf` — anyone can read, run, and fork it.

## Limitations (v0.1)

- Long papers are truncated to 12,000 chars to fit the context window.
- Scanned PDFs need OCR first — out of scope for now.
- The summary is only as good as the model; always verify against the paper itself.

## License

MIT
