"""Deterministic checks for whether compression retained explicit key markers."""

from __future__ import annotations

import re

_NUMBER = re.compile(
    r"(?<![\w])(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}:\d{2}(?::\d{2})?|"
    r"[-+]?\d+(?:[.,]\d+)*(?:%|[A-Za-z]+)?)(?!\w)"
)
_IDENTIFIER = re.compile(
    r"\b(?:[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]+|[A-Z][A-Z0-9_]{1,}|"
    r"[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]+)+|[A-Z]{2,}[A-Z][a-z][A-Za-z0-9]*|"
    r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)\b"
)
_CONSTRAINT = re.compile(
    r"\b(?:not|no|never|cannot|can't|don't|doesn't|isn't|aren't|won't|shouldn't|mustn't|"
    r"without|unless|except|only|if|when|before|after|must|required|mandatory)\b",
    re.IGNORECASE,
)
_FORMAT = re.compile(
    r"\b(?:json|csv|xml|yaml|markdown|bullet(?:\s+list)?|table|one paragraph|"
    r"no markdown|valid json)\b",
    re.IGNORECASE,
)
_USER_LINE = re.compile(r"^\s*(?:user|human)\s*:\s*.+$", re.IGNORECASE)


def _markers(text: str) -> dict[str, set[str]]:
    numbers = {match.strip() for match in _NUMBER.findall(text)}
    identifiers = {match.strip() for match in _IDENTIFIER.findall(text)}
    constraints = {match.casefold() for match in _CONSTRAINT.findall(text)}
    formats = {match.casefold() for match in _FORMAT.findall(text)}
    table_rows = {
        re.sub(r"\s+", " ", line).strip()
        for line in text.splitlines()
        if "|" in line and re.search(r"[A-Za-z0-9]", line)
    }
    recent_user_turns = [
        re.sub(r"\s+", " ", line).strip()
        for line in text.splitlines()
        if _USER_LINE.match(line)
    ][-2:]
    return {
        "Numbers and dates": numbers,
        "Code identifiers": identifiers,
        "Negations and conditions": constraints,
        "Output formats": formats,
        "Table rows": table_rows,
        "Latest user turns": set(recent_user_turns),
    }


def audit_preservation(originals: dict[str, str], compressed: dict[str, str]) -> dict:
    """Report explicit-marker retention; this is not a semantic answer-quality score."""
    rows: list[dict] = []
    required_total = 0
    retained_total = 0
    for label in ("system_prompt", "context", "user_query"):
        original_markers = _markers(originals.get(label, ""))
        compressed_markers = _markers(compressed.get(label, ""))
        for category, required in original_markers.items():
            if not required:
                continue
            retained = required & compressed_markers[category]
            missing = sorted(required - retained)
            required_total += len(required)
            retained_total += len(retained)
            rows.append(
                {
                    "input": label.replace("_", " ").title(),
                    "category": category,
                    "required": len(required),
                    "retained": len(retained),
                    "missing": missing,
                    "coverage_percentage": (len(retained) / len(required) * 100)
                    if required
                    else 100.0,
                }
            )

    coverage = (retained_total / required_total * 100) if required_total else None
    return {
        "status": "not_applicable"
        if required_total == 0
        else "pass"
        if retained_total == required_total
        else "warning",
        "retained_markers": retained_total,
        "required_markers": required_total,
        "coverage_percentage": coverage,
        "rows": rows,
        "scope": "Explicit markers only; this does not prove semantic answer equivalence.",
    }
