"""Deterministic, source-order-preserving context selection."""

from __future__ import annotations

import re
from dataclasses import dataclass

_WORD = re.compile(r"[\w'-]+", re.UNICODE)
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_CODE_LINE = re.compile(r"^\s{4,}(?:\S|$)|^\s*(?:def |class |import |from |if |for |while |return |const |let |function )")
_USER_MESSAGE = re.compile(r"^\s*(?:user|human)\s*:", re.IGNORECASE)
_NUMBER_MARKER = re.compile(r"(?<![\w])[-+]?\d+(?:[,_]\d{3})*(?:\.\d+)?(?:%|[A-Za-z]+)?(?![\w])")
_IDENTIFIER_MARKER = re.compile(
    r"\b(?:[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+|[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]+)+|"
    r"[A-Z]{2,}[A-Z][a-z][A-Za-z0-9]*|"
    r"[A-Z]{2,}[A-Z0-9_]*|[A-Za-z_]\w*\.[A-Za-z_]\w*)\b"
)
_CONSTRAINT_MARKER = re.compile(
    r"\b(?:not|no|never|cannot|can't|don't|doesn't|isn't|aren't|won't|shouldn't|mustn't|without|"
    r"unless|except|only|if|when|before|after|must|required|mandatory)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ContextCompression:
    text: str
    warnings: tuple[str, ...]
    method: str


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _deduplicate_prose_sentences(text: str) -> tuple[str, bool]:
    if _is_structured_context(text) or any(_USER_MESSAGE.match(line) for line in text.splitlines()):
        return text, False

    paragraphs = _PARAGRAPH_SPLIT.split(text)
    seen: set[str] = set()
    cleaned_paragraphs: list[str] = []
    removed = False
    for paragraph in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())
        kept: list[str] = []
        for sentence in sentences:
            key = _normalized(sentence)
            if key in seen:
                removed = True
                continue
            seen.add(key)
            kept.append(sentence)
        cleaned_paragraphs.append(" ".join(kept))

    if not removed:
        return text, False
    return "\n\n".join(paragraph for paragraph in cleaned_paragraphs if paragraph), True


def _split_chunks(text: str, count_tokens, max_chunk_tokens: int) -> list[str]:
    paragraphs = [part.strip() for part in _PARAGRAPH_SPLIT.split(text) if part.strip()]
    chunks: list[str] = []
    for paragraph in paragraphs:
        if count_tokens(paragraph) <= max_chunk_tokens:
            chunks.append(paragraph)
            continue
        # Sentence boundaries preserve exact text and avoid slicing token byte sequences.
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        current = ""
        for sentence in sentences:
            proposed = f"{current} {sentence}".strip()
            if current and count_tokens(proposed) > max_chunk_tokens:
                chunks.append(current)
                current = sentence
            else:
                current = proposed
        if current:
            if count_tokens(current) <= max_chunk_tokens:
                chunks.append(current)
            else:
                # An indivisible long sentence remains whole rather than being corrupted.
                chunks.append(current)
    return chunks or [text.strip()]


def _is_structured_context(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    code_lines = sum(bool(_CODE_LINE.match(line)) for line in lines)
    digit_chars = sum(char.isdigit() for char in text)
    table_lines = sum(line.count("|") >= 2 or "\t" in line for line in lines)
    return code_lines / len(lines) >= 0.2 or table_lines >= 2 or digit_chars / max(len(text), 1) >= 0.12


def _has_critical_markers(text: str) -> bool:
    return any(pattern.search(text) for pattern in (_NUMBER_MARKER, _IDENTIFIER_MARKER, _CONSTRAINT_MARKER))


def _joined_chunks(chunks: list[tuple[int, str]], selected: set[int]) -> str:
    return "\n\n".join(chunk for index, chunk in chunks if index in selected)


def _relevance(chunk: str, query_terms: set[str]) -> int:
    terms = {term.casefold() for term in _WORD.findall(chunk)}
    return sum(2 if term in terms else 0 for term in query_terms) + len(terms & query_terms)


def compress_context(
    context: str,
    query: str,
    token_budget: int | None,
    count_tokens,
    max_chunk_tokens: int = 256,
) -> ContextCompression:
    if not context:
        return ContextCompression(context, (), "unchanged-empty-context")
    if token_budget is not None and token_budget < 0:
        raise ValueError("token_budget must be non-negative or None")

    context, duplicate_sentences_removed = _deduplicate_prose_sentences(context)
    chunks = _split_chunks(context, count_tokens, max_chunk_tokens)
    recent_user_indices = {
        index for index, _ in [
            (index, chunk) for index, chunk in enumerate(chunks) if _USER_MESSAGE.match(chunk)
        ][-2:]
    }
    unique_chunks: list[tuple[int, str]] = []
    seen: set[str] = set()
    duplicates_removed = duplicate_sentences_removed
    for index, chunk in enumerate(chunks):
        key = _normalized(chunk)
        if key in seen and index not in recent_user_indices:
            duplicates_removed = True
            continue
        seen.add(key)
        unique_chunks.append((index, chunk))

    base_text = "\n\n".join(chunk for _, chunk in unique_chunks)
    warnings: list[str] = []
    if duplicates_removed:
        warnings.append("Exact duplicate context passages were removed.")
    if token_budget is None or count_tokens(base_text) <= token_budget:
        return ContextCompression(base_text, tuple(warnings), "exact-deduplication")

    if _is_structured_context(context):
        warnings.append("Structured/code/numerical context exceeded the budget; retained it to avoid damaging dependencies or data.")
        return ContextCompression(base_text, tuple(warnings), "structured-context-retained")

    protected = {
        index for index, chunk in unique_chunks
        if _has_critical_markers(chunk) or index in recent_user_indices
    }
    selected = set(protected)
    if count_tokens(_joined_chunks(unique_chunks, selected)) > token_budget:
        warnings.append(
            "Numeric, code-identifier, constraint, or recent-user content exceeded the budget; retained all context instead of dropping it."
        )
        return ContextCompression(base_text, tuple(warnings), "critical-content-budget-fallback")

    query_terms = {term.casefold() for term in _WORD.findall(query) if len(term) > 1}
    ranked = sorted(
        unique_chunks,
        key=lambda item: (-_relevance(item[1], query_terms), item[0]),
    )
    for index, chunk in ranked:
        if index in selected:
            continue
        proposed = selected | {index}
        if count_tokens(_joined_chunks(unique_chunks, proposed)) <= token_budget:
            selected = proposed

    if not selected:
        warnings.append("No complete context chunk fit the token budget; retained context rather than silently dropping it.")
        return ContextCompression(base_text, tuple(warnings), "budget-fallback-retain")

    compressed = _joined_chunks(unique_chunks, selected)
    if len(selected) < len(unique_chunks):
        warnings.append(
            f"{len(unique_chunks) - len(selected)} lower-ranked context chunk(s) were omitted to meet the budget; review them for task-specific importance."
        )
    if count_tokens(compressed) > token_budget:
        warnings.append("The context remains over budget because complete chunks are retained.")
    return ContextCompression(compressed, tuple(warnings), "query-ranked-complete-chunks")