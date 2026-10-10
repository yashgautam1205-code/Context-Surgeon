"""Public compression API and structured metrics."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from .context import compress_context
from .preservation import audit_preservation
from .query import compress_query
from .system_prompt import compress_system_prompt
from .tokenizer import TokenizerAdapter

MINIMUM_REDUCTION_RATIO = 0.20


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
    minimum_reduction_percentage: float
    reduction_target_met: bool
    preservation_coverage_percentage: float
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
    """Aim for 20% total token reduction while preserving all audited markers.

    The minimum reduction goal and ``token_budget`` both apply to context;
    whichever is stricter determines the context target. System prompts and
    user queries are immutable. If reaching the token target would remove any
    audited marker, the original context is restored. Token counts are local
    tokenizer estimates, not provider-reported API usage or chat-template
    overhead.
    ``tokenizer`` is injectable for deterministic tests and must expose ``count``.
    """
    if token_budget is not None and token_budget < 0:
        raise ValueError("token_budget must be non-negative or None")
    started = time.perf_counter()
    counter = tokenizer or TokenizerAdapter(model)
    count_tokens = counter.count

    compressed_system, _ = compress_system_prompt(system_prompt)
    compressed_query, _ = compress_query(user_query, count_tokens)
    system_tokens = count_tokens(system_prompt)
    context_tokens = count_tokens(context)
    query_tokens = count_tokens(user_query)
    original_total = system_tokens + context_tokens + query_tokens
    target_total = int(original_total * (1 - MINIMUM_REDUCTION_RATIO))
    context_target = max(0, target_total - system_tokens - query_tokens)
    effective_context_budget = (
        min(token_budget, context_target)
        if token_budget is not None
        else context_target
    )
    context_result = compress_context(
        context,
        compressed_query,
        effective_context_budget,
        count_tokens,
        allow_sentence_chunks=True,
    )
    warnings = list(context_result.warnings)
    original_values = {
        "system_prompt": system_prompt,
        "context": context,
        "user_query": user_query,
    }
    candidate_values = {
        "system_prompt": compressed_system,
        "context": context_result.text,
        "user_query": compressed_query,
    }
    preservation = audit_preservation(original_values, candidate_values)
    if count_tokens(candidate_values["context"]) > context_tokens:
        candidate_values["context"] = context
        warnings.append(
            "The candidate increased context token usage; original context restored."
        )
        preservation = audit_preservation(original_values, candidate_values)
    if preservation["status"] == "warning":
        candidate_values["context"] = context
        warnings.append(
            "The candidate missed explicit preservation markers; original context restored."
        )
        preservation = audit_preservation(original_values, candidate_values)

    values = {
        "system_prompt": (system_prompt, candidate_values["system_prompt"]),
        "context": (context, candidate_values["context"]),
        "user_query": (user_query, candidate_values["user_query"]),
    }
    inputs = {
        name: InputMetrics(
            original, compressed, count_tokens(original), count_tokens(compressed)
        )
        for name, (original, compressed) in values.items()
    }
    original_total = sum(item.original_tokens for item in inputs.values())
    compressed_total = sum(item.compressed_tokens for item in inputs.values())
    saved = original_total - compressed_total
    reduction = (saved / original_total * 100) if original_total else 0.0
    if saved < 0:
        warnings.append(
            "Compression increased tokenizer-reported token usage; review the output before using it."
        )
    reduction_target_met = reduction >= MINIMUM_REDUCTION_RATIO * 100
    latency = (time.perf_counter() - started) * 1000
    return CompressionResult(
        **inputs,
        original_tokens=original_total,
        compressed_tokens=compressed_total,
        tokens_saved=saved,
        reduction_percentage=reduction,
        compression_latency_ms=latency,
        compression_method=f"preserve-system+preserve-query+{context_result.method}",
        minimum_reduction_percentage=MINIMUM_REDUCTION_RATIO * 100,
        reduction_target_met=reduction_target_met,
        preservation_coverage_percentage=(
            preservation["coverage_percentage"]
            if preservation["coverage_percentage"] is not None
            else 100.0
        ),
        tokenizer_model=getattr(counter, "model", model),
        warnings=tuple(dict.fromkeys(warnings)),
    )
