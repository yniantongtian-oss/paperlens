"""Summarize research papers with local or OpenAI-compatible models."""

from paperlens.backends import call_backend
from paperlens.cli import main
from paperlens.extract import extract_text
from paperlens.prompt import build_prompt
from paperlens.version import __version__

__all__ = [
    "__version__",
    "build_prompt",
    "call_backend",
    "extract_text",
    "main",
]
