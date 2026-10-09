"""Public compression API and structured metrics."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from .context import compress_context
from .query import compress_query
from .system_prompt import compress_system_prompt
from .tokenizer import TokenizerAdapter


@dataclass(frozen=True)
class InputMetrics:
    original: str
    compressed: str
    original_tokens: int
    compressed_tokens: int


@dataclass(frozen=True)
class CompressionResult:
    system_prompt: InputMetrics
    context: InputMetrics
    user_query: InputMetrics
    original_tokens: int
    compressed_tokens: int
    tokens_saved: int
    reduction_percentage: float
    compression_latency_ms: float
    compression_method: str
    tokenizer_model: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict:
        result = asdict(self)
        result["warnings"] = list(self.warnings)
        return result


def compress_inputs(
    system_prompt: str,
    context: str,
    user_query: str,
    token_budget: int | None = None,
    *,
    model: str = "gpt-4o-mini",
    tokenizer=None,
) -> CompressionResult:
    """Compress prompt components conservatively using the target model tokenizer.

    ``token_budget`` applies to context only. Token counts are local tokenizer
    estimates, not provider-reported API usage or chat-template overhead.
    ``tokenizer`` is injectable for deterministic tests and must expose ``count``.
    """
    started = time.perf_counter()
    counter = tokenizer or TokenizerAdapter(model)
    count_tokens = counter.count

    compressed_system, _ = compress_system_prompt(system_prompt)
    compressed_query, _ = compress_query(user_query, count_tokens)
    context_result = compress_context(context, compressed_query, token_budget, count_tokens)

    values = {
        "system_prompt": (system_prompt, compressed_system),
        "context": (context, context_result.text),
        "user_query": (user_query, compressed_query),
    }
    inputs = {
        name: InputMetrics(original, compressed, count_tokens(original), count_tokens(compressed))
        for name, (original, compressed) in values.items()
    }
    original_total = sum(item.original_tokens for item in inputs.values())
    compressed_total = sum(item.compressed_tokens for item in inputs.values())
    saved = original_total - compressed_total
    reduction = (saved / original_total * 100) if original_total else 0.0
    warnings = list(context_result.warnings)
    if saved < 0:
        warnings.append("Compression increased tokenizer-reported token usage; review the output before using it.")
    latency = (time.perf_counter() - started) * 1000
    return CompressionResult(
        **inputs,
        original_tokens=original_total,
        compressed_tokens=compressed_total,
        tokens_saved=saved,
        reduction_percentage=reduction,
        compression_latency_ms=latency,
        compression_method=f"system-deduplication+query-trim+{context_result.method}",
        tokenizer_model=getattr(counter, "model", model),
        warnings=tuple(dict.fromkeys(warnings)),
    )