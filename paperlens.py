#!/usr/bin/env python3
"""
PaperLens — read a research paper with an open-weight model.

Pipeline: PDF -> text -> structured prompt -> open model -> Markdown report.
Backends:
  ollama  local Ollama server (default, model e.g. qwen3:8b) — no API key, data stays on device
  groq    Groq Cloud free tier (open-weight Llama etc., OpenAI-compatible) — needs GROQ_API_KEY
  echo    dry-run: prints the prompt instead of calling a model (pipeline self-test)

Usage:
  python paperlens.py paper.pdf
  python paperlens.py paper.pdf --backend groq --model llama-3.3-70b-versatile -o report.md
  python paperlens.py notes.txt --backend echo   # self-test without a model
"""

import argparse
import json
import os
import sys
import urllib.request

MAX_CHARS = 12000  # keep the paper excerpt inside the model's context window

PROMPT_TEMPLATE = """You are a research assistant helping an undergraduate student read a scientific paper.
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

--- PAPER TEXT START ---
{paper_text}
--- PAPER TEXT END ---
"""


def extract_text(path: str) -> str:
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(path)
        parts = [(page.extract_text() or "") for page in reader.pages]
        return "\n".join(parts)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def call_ollama(prompt: str, model: str) -> str:
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps({"model": model, "prompt": prompt, "stream": False}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read())["response"]
    except OSError as e:
        sys.exit(f"[!] Could not reach Ollama at localhost:11434: {e}\n"
                 "    Start it with `ollama serve` and pull a model, e.g. `ollama pull qwen3:8b`.")


def call_groq(prompt: str, model: str) -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        sys.exit("[!] GROQ_API_KEY is not set. Create a free key at https://console.groq.com/keys")
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize a paper with an open-weight model.")
    ap.add_argument("input", help="PDF or .txt file of the paper")
    ap.add_argument("--backend", choices=["ollama", "groq", "echo"], default="ollama")
    ap.add_argument("--model", default=None, help="model name (defaults: qwen3:8b / llama-3.3-70b-versatile)")
    ap.add_argument("-o", "--output", default=None, help="write Markdown report here (default: stdout)")
    args = ap.parse_args()

    text = extract_text(args.input).strip()
    if not text:
        sys.exit("[!] No text extracted — is this a scanned PDF? (OCR is out of scope for v0.1)")
    excerpt = text[:MAX_CHARS]
    if len(text) > MAX_CHARS:
        print(f"[*] Paper truncated to {MAX_CHARS} chars for context window.", file=sys.stderr)
    prompt = PROMPT_TEMPLATE.format(paper_text=excerpt)

    if args.backend == "echo":
        report = "<!-- dry-run: the prompt below would be sent to the model -->\n\n```\n" + prompt + "\n```"
    elif args.backend == "ollama":
        report = call_ollama(prompt, args.model or "qwen3:8b")
    else:
        report = call_groq(prompt, args.model or "llama-3.3-70b-versatile")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[*] Report written to {args.output}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
