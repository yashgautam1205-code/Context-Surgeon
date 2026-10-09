"""Query cleanup limited to unambiguous leading politeness filler."""

from __future__ import annotations

import re

_PREFIXES = (
    re.compile(r"^\s*Could you please\s+", re.IGNORECASE),
    re.compile(r"^\s*Would you please\s+", re.IGNORECASE),
    re.compile(r"^\s*I would like you to\s+", re.IGNORECASE),
    re.compile(r"^\s*Please\s+", re.IGNORECASE),
)
_PROTECTED = re.compile(r"\b(?:not|never|no|unless|except|only|if|when|before|after)\b|\d+(?:[.,]\d+)*%?", re.IGNORECASE)


def compress_query(query: str, count_tokens) -> tuple[str, bool]:
    """Trim safe filler only when protected constraints remain and tokens decrease."""
    if not query:
        return query, False
    original = query.strip()
    candidate = original
    for pattern in _PREFIXES:
        candidate = pattern.sub("", candidate, count=1)
        if candidate != original:
            break
    candidate = re.sub(r"\s+", " ", candidate).strip()
    if not candidate or candidate == original:
        return query, False
    if _PROTECTED.findall(original) != _PROTECTED.findall(candidate):
        return query, False
    if count_tokens(candidate) >= count_tokens(query):
        return query, False
    return candidate, True