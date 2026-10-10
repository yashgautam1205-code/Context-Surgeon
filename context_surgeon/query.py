"""Preserve the active user query without rewriting its instructions."""

from __future__ import annotations

from collections.abc import Callable


def compress_query(query: str, count_tokens: Callable[[str], int]) -> tuple[str, bool]:
    """Return the active query unchanged; query instructions are immutable."""
    return query, False
