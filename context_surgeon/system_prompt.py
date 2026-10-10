"""Preserve system prompts without rewriting their instructions."""

from __future__ import annotations


def compress_system_prompt(prompt: str) -> tuple[str, bool]:
    """Return the system prompt unchanged; system instructions are immutable."""
    return prompt, False
