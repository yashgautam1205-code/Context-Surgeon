"""Optional OpenAI-compatible baseline/compressed answer evaluation."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from urllib.parse import quote

RUBRIC = {
    "semantic_equivalence": "Does the compressed answer preserve the same material conclusions, facts, and requested outcome as the baseline? Ignore harmless wording differences.",
    "correctness": "Does the answer state facts supported by the supplied inputs?",
    "completeness": "Does it address all material parts of the request?",
    "constraint_preservation": "Does it preserve negations, conditions, numbers, entities, and requested format?",
    "clarity": "Is it understandable, direct, and internally consistent?",
}


def _chat(
    messages: list[dict],
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, float]:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=90) as response:
        data = json.loads(response.read().decode("utf-8"))
    elapsed_ms = (time.perf_counter() - started) * 1000
    return data["choices"][0]["message"]["content"], elapsed_ms


def _provider_error_message(status: int) -> str:
    if status == 401:
        return "The provider rejected this credential (401 Unauthorized). Confirm it is a full, active OpenAI Platform API key with no surrounding whitespace, belongs to the intended project, and has not been revoked. Do not reuse a key exposed in chat."
    if status == 403:
        return "The provider denied this request (403 Forbidden). Check project permissions, model access, and account billing."
    if status == 429:
        return "The provider rate limit or quota was reached (429). Check usage/billing or retry after the limit resets."
    return f"The provider returned HTTP {status}. Check the configured endpoint, model access, and provider status."


def validate_api_key(
    api_key: str | None = None, model: str | None = None, base_url: str | None = None
) -> dict:
    """Check credential access to a model metadata endpoint without generating tokens."""
    key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        return {"status": "not run", "reason": "Enter an API key before checking it."}
    provider_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    endpoint = (
        base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    ).rstrip("/")
    request = urllib.request.Request(
        f"{endpoint}/models/{quote(provider_model, safe='')}",
        headers={"Authorization": f"Bearer {key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=30):
            return {"status": "valid", "model": provider_model}
    except urllib.error.HTTPError as exc:
        return {
            "status": "error",
            "http_status": exc.code,
            "reason": _provider_error_message(exc.code),
        }
    except (urllib.error.URLError, TimeoutError) as exc:
        return {"status": "error", "reason": f"Could not reach the provider: {exc}"}


def evaluate_answers(
    result,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> dict:
    key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        return {"status": "not run", "reason": "OPENAI_API_KEY is not configured."}
    provider_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    endpoint = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    settings = {"temperature": 0, "max_tokens": 512}
    baseline_messages = [
        {"role": "system", "content": result.system_prompt.original},
        {
            "role": "user",
            "content": f"Context:\n{result.context.original}\n\nQuery:\n{result.user_query.original}",
        },
    ]
    compressed_messages = [
        {"role": "system", "content": result.system_prompt.compressed},
        {
            "role": "user",
            "content": f"Context:\n{result.context.compressed}\n\nQuery:\n{result.user_query.compressed}",
        },
    ]
    try:
        baseline, baseline_latency = _chat(
            baseline_messages, provider_model, key, endpoint, **settings
        )
        compressed, compressed_latency = _chat(
            compressed_messages, provider_model, key, endpoint, **settings
        )
        judge_prompt = (
            "Evaluate two answers to the same task. Score each dimension from 1 (poor) to 5 (excellent), "
            "using the supplied source inputs as ground truth. Also decide whether the answers are materially "
            "equivalent; wording differences alone do not make them different. A changed or omitted key number, "
            "identifier, negation, condition, or requested constraint means they are not equivalent. Return only JSON: "
            '{"baseline":{"semantic_equivalence":5,"correctness":5,"completeness":5,"constraint_preservation":5,"clarity":5},'
            '"compressed":{"semantic_equivalence":5,"correctness":5,"completeness":5,"constraint_preservation":5,"clarity":5},'
            '"materially_equivalent":true,"equivalence_notes":"...","notes":"..."}.\n'
            f"Rubric: {json.dumps(RUBRIC)}\n\nOriginal inputs: {result.system_prompt.original}\n{result.context.original}\n{result.user_query.original}"
        )
        judged, judge_latency = _chat(
            [
                {
                    "role": "system",
                    "content": "You are a careful answer-quality evaluator. Follow the rubric and return valid JSON.",
                },
                {
                    "role": "user",
                    "content": f"Baseline answer:\n{baseline}\n\nCompressed answer:\n{compressed}\n\n{judge_prompt}",
                },
            ],
            provider_model,
            key,
            endpoint,
            **settings,
        )
        start = judged.find("{")
        end = judged.rfind("}")
        scores = (
            json.loads(judged[start : end + 1])
            if start >= 0 and end > start
            else {"raw": judged}
        )
        return {
            "status": "completed",
            "model": provider_model,
            "generation_settings": settings,
            "rubric": RUBRIC,
            "baseline_answer": baseline,
            "compressed_answer": compressed,
            "baseline_latency_ms": baseline_latency,
            "compressed_latency_ms": compressed_latency,
            "judge_latency_ms": judge_latency,
            "scores": scores,
        }
    except urllib.error.HTTPError as exc:
        return {
            "status": "error",
            "model": provider_model,
            "reason": _provider_error_message(exc.code),
        }
    except (
        urllib.error.URLError,
        TimeoutError,
        KeyError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        return {"status": "error", "model": provider_model, "reason": str(exc)}
