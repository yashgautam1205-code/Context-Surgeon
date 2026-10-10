"""Deterministic, source-order-preserving context selection."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

_WORD = re.compile(r"[\w'-]+", re.UNICODE)
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_FENCE_START = re.compile(r"^\s*(`{3,}|~{3,})")
_FENCE_CLOSE = re.compile(r"^\s*(`{3,}|~{3,})\s*$")
_TAG = re.compile(r"<(/?)([A-Za-z][\w:.-]*)(?:\s[^<>]*)?(/?)>")
_VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
_CODE_LINE = re.compile(
    r"^\s{4,}(?:\S|$)|^\s*(?:def |class |import |from |if |for |while |return |const |let |function )"
)
_USER_MESSAGE = re.compile(r"^\s*(?:user|human)\s*:", re.IGNORECASE)
_PROTECTED_BLOCK = re.compile(
    r"^\s*(?:system|developer|tool|few[- ]shot|example|demonstration)\s*:",
    re.IGNORECASE,
)
_BOILERPLATE_LINE = re.compile(
    r"^\s*(?:as an ai language model[,!.]?|sure,\s+here(?:'s| is) your answer:)\s*$",
    re.IGNORECASE,
)
_NUMBER_MARKER = re.compile(
    r"(?<![\w])[-+]?\d+(?:[,_]\d{3})*(?:\.\d+)?(?:%|[A-Za-z]+)?(?![\w])"
)
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
_FORMAT_MARKER = re.compile(
    r"\b(?:json|csv|xml|yaml|markdown|bullet(?:\s+list)?|table|one paragraph|"
    r"no markdown|valid json)\b",
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
    if _is_structured_context(text) or any(
        _USER_MESSAGE.match(line) or _PROTECTED_BLOCK.match(line)
        for line in text.splitlines()
    ):
        return text, False
    paragraphs = _PARAGRAPH_SPLIT.split(text)
    if any(
        _FENCE_START.match(line)
        or _TAG.search(line)
        or "|" in line
        or "\t" in line
        or _CODE_LINE.match(line)
        for line in text.splitlines()
    ) or any(
        _is_json_value(paragraph.strip()) or not _is_balanced_delimiters(paragraph)
        for paragraph in paragraphs
    ):
        return text, False

    normalized = re.sub(r"\n{3,}", "\n\n", text)
    normalized = "\n".join(
        re.sub(r"(?<=\S) {3,}(?=\S)", " ", line)
        for line in normalized.splitlines()
        if not _BOILERPLATE_LINE.match(line)
    ).strip()
    changed = normalized != text
    text = normalized
    paragraphs = _PARAGRAPH_SPLIT.split(text)
    seen: set[str] = set()
    cleaned_paragraphs: list[str] = []
    removed = changed
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


def _split_chunks(
    text,
    count_tokens,
    max_chunk_tokens: int,
    allow_sentence_chunks: bool = False,
) -> list[str]:
    paragraphs: list[str] = []
    prose_lines: list[str] = []
    fenced_lines: list[str] = []
    fence_marker: tuple[str, int] | None = None

    def flush_prose() -> None:
        if prose_lines:
            paragraphs.extend(_PARAGRAPH_SPLIT.split("".join(prose_lines)))
            prose_lines.clear()

    for line in text.splitlines(keepends=True):
        fence_match = _FENCE_START.match(line)
        if fence_marker is not None:
            fenced_lines.append(line)
            close_match = _FENCE_CLOSE.match(line)
            if (
                close_match
                and close_match.group(1)[0] == fence_marker[0]
                and len(close_match.group(1)) >= fence_marker[1]
            ):
                paragraphs.append("".join(fenced_lines).strip())
                fenced_lines.clear()
                fence_marker = None
        elif fence_match:
            flush_prose()
            fence_marker = (fence_match.group(1)[0], len(fence_match.group(1)))
            fenced_lines.append(line)
        else:
            prose_lines.append(line)

    if fenced_lines:
        paragraphs.append("".join(fenced_lines).strip())
    flush_prose()
    paragraphs = [part.strip() for part in paragraphs if part.strip()]
    chunks: list[str] = []
    for paragraph in paragraphs:
        if _is_atomic_chunk(paragraph):
            chunks.append(paragraph)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        if allow_sentence_chunks and len(sentences) > 1:
            chunks.extend(sentence for sentence in sentences if sentence)
            continue
        if count_tokens(paragraph) <= max_chunk_tokens:
            chunks.append(paragraph)
            continue
        # Sentence boundaries preserve exact text and avoid slicing token byte sequences.
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
    return (
        code_lines / len(lines) >= 0.2
        or table_lines >= 2
        or digit_chars / max(len(text), 1) >= 0.12
    )


def _is_balanced_delimiters(text: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for char in text:
        if char in "([{":
            stack.append(char)
        elif char in pairs and (not stack or stack.pop() != pairs[char]):
            return False
    return not stack


def _is_json_value(text: str) -> bool:
    if not text.lstrip().startswith(("{", "[")):
        return False
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(value, (dict, list))


def _has_balanced_tags(text: str) -> bool:
    stack: list[str] = []
    for match in _TAG.finditer(text):
        closing, name, self_closing = match.groups()
        name = name.casefold()
        if closing:
            if not stack or stack.pop() != name:
                return False
        elif not self_closing and name not in _VOID_TAGS:
            stack.append(name)
    return not stack


def _is_atomic_chunk(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    is_table = any("|" in line or "\t" in line for line in lines)
    is_code = any(_CODE_LINE.match(line) for line in lines)
    return (
        _FENCE_START.match(text)
        or _PROTECTED_BLOCK.match(text)
        or _is_json_value(text)
        or is_table
        or is_code
        or not _is_balanced_delimiters(text)
        or not _has_balanced_tags(text)
    )


def _has_critical_markers(text: str) -> bool:
    return any(
        pattern.search(text)
        for pattern in (
            _NUMBER_MARKER,
            _IDENTIFIER_MARKER,
            _CONSTRAINT_MARKER,
            _FORMAT_MARKER,
        )
    )


def _joined_chunks(chunks: list[tuple[int, str]], selected: set[int]) -> str:
    return "\n\n".join(chunk for index, chunk in chunks if index in selected)


def _relevance(
    chunk: str,
    query_terms: set[str],
    documents: list[str],
    query_phrase: str,
) -> float:
    """Score a chunk with BM25 term relevance plus exact-phrase preference."""
    if not query_terms:
        return 0.0
    tokens = [term.casefold() for term in _WORD.findall(chunk)]
    frequencies = {term: tokens.count(term) for term in query_terms}
    average_length = (
        sum(len(_WORD.findall(document)) for document in documents) / len(documents)
        if documents
        else 0.0
    )
    document_frequency = {
        term: sum(
            term in {word.casefold() for word in _WORD.findall(document)}
            for document in documents
        )
        for term in query_terms
    }
    score = 0.0
    for term, frequency in frequencies.items():
        if not frequency:
            continue
        idf = math.log(
            1
            + (len(documents) - document_frequency[term] + 0.5)
            / (document_frequency[term] + 0.5)
        )
        length_ratio = len(tokens) / average_length if average_length else 0.0
        score += (
            idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length_ratio))
        )
    phrase = " ".join(term.casefold() for term in _WORD.findall(chunk))
    if query_phrase and query_phrase in phrase:
        score += 1.0
    return score


def compress_context(
    context: str,
    query: str,
    token_budget: int | None,
    count_tokens,
    max_chunk_tokens: int = 256,
    allow_sentence_chunks: bool = False,
) -> ContextCompression:
    if not context:
        return ContextCompression(context, (), "unchanged-empty-context")
    if token_budget is not None and token_budget < 0:
        raise ValueError("token_budget must be non-negative or None")

    original_tokens = count_tokens(context)
    context, duplicate_sentences_removed = _deduplicate_prose_sentences(context)
    chunks = _split_chunks(
        context,
        count_tokens,
        max_chunk_tokens,
        allow_sentence_chunks=allow_sentence_chunks,
    )
    recent_user_indices = {
        index
        for index, _ in [
            (index, chunk)
            for index, chunk in enumerate(chunks)
            if _USER_MESSAGE.match(chunk)
        ][-2:]
    }
    protected_block_indices = {
        index for index, chunk in enumerate(chunks) if _PROTECTED_BLOCK.match(chunk)
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
        warnings.append(
            "Duplicate passages, redundant whitespace, or boilerplate were removed."
        )
    effective_budget = token_budget
    if effective_budget is None or count_tokens(base_text) <= effective_budget:
        return ContextCompression(base_text, tuple(warnings), "exact-deduplication")

    protected = {
        index
        for index, chunk in unique_chunks
        if _is_atomic_chunk(chunk)
        or index in recent_user_indices
        or index in protected_block_indices
        or (
            token_budget is not None
            and token_budget < original_tokens
            and _has_critical_markers(chunk)
        )
    }
    selected = set(protected)

    query_terms = {term.casefold() for term in _WORD.findall(query) if len(term) > 1}
    query_phrase = " ".join(term.casefold() for term in _WORD.findall(query))
    documents = [chunk for _, chunk in unique_chunks]
    ranked = sorted(
        unique_chunks,
        key=lambda item: (
            -_relevance(item[1], query_terms, documents, query_phrase),
            item[0],
        ),
    )
    for index, chunk in ranked:
        if index in selected:
            continue
        proposed = selected | {index}
        if count_tokens(_joined_chunks(unique_chunks, proposed)) <= effective_budget:
            selected = proposed

    if not selected:
        return ContextCompression(base_text, tuple(warnings), "budget-fallback-retain")

    compressed = _joined_chunks(unique_chunks, selected)
    return ContextCompression(
        compressed, tuple(warnings), "query-ranked-complete-chunks"
    )
