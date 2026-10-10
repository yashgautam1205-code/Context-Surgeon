from __future__ import annotations

import csv
import io
import json
import urllib.error

import pytest

from context_surgeon.api import compress_inputs
from context_surgeon.benchmarks import (
    BENCHMARK_EXAMPLES,
    export_benchmarks,
    run_benchmarks,
)
from context_surgeon.context import ContextCompression, compress_context
from context_surgeon.evaluation import RUBRIC, evaluate_answers, validate_api_key
from context_surgeon.preservation import audit_preservation
from context_surgeon.query import compress_query
from context_surgeon.system_prompt import compress_system_prompt


class FakeTokenizer:
    model = "test-whitespace"

    def count(self, text: str) -> int:
        return len(text.split())


TOKENIZER = FakeTokenizer()


def test_token_count_reduction_and_result_structure():
    result = compress_inputs(
        "Rule.\nRule.",
        "Paris is the capital.\n\nParis is the capital.",
        "Please state the capital of France.",
        tokenizer=TOKENIZER,
    )
    assert result.original_tokens >= result.compressed_tokens
    assert result.tokens_saved == result.original_tokens - result.compressed_tokens
    assert result.reduction_percentage == pytest.approx(
        result.tokens_saved / result.original_tokens * 100
    )
    assert result.minimum_reduction_percentage == 20
    assert result.preservation_coverage_percentage == 100
    payload = result.to_dict()
    assert payload["system_prompt"]["original"] == "Rule.\nRule."
    assert {
        "context",
        "user_query",
        "compression_latency_ms",
        "compression_method",
        "warnings",
    } <= payload.keys()


def test_api_targets_twenty_percent_total_reduction_and_preserves_marker_coverage():
    result = compress_inputs(
        "",
        (
            "The garden has green flowers. "
            "The store sells fresh bread. "
            "The river crosses the quiet town."
        ),
        "What color are the flowers?",
        tokenizer=TOKENIZER,
    )
    assert result.reduction_percentage >= 20
    assert result.reduction_target_met
    assert result.preservation_coverage_percentage == 100
    assert result.context.compressed != result.context.original


def test_api_restores_context_if_candidate_drops_audit_marker(monkeypatch):
    context = (
        "Return a CSV report about tree coverage. "
        "The bay contains seven small islands. "
        "The rural station has five platforms. "
        "The orchard grows several pear varieties."
    )
    monkeypatch.setattr(
        "context_surgeon.api.compress_context",
        lambda *args, **kwargs: ContextCompression(
            "The bay contains seven small islands.", (), "query-ranked-complete-chunks"
        ),
    )
    result = compress_inputs(
        "",
        context,
        "Tell me about the islands and bay.",
        tokenizer=TOKENIZER,
    )
    audit = audit_preservation(
        {
            "system_prompt": "",
            "context": context,
            "user_query": result.user_query.original,
        },
        {
            "system_prompt": result.system_prompt.compressed,
            "context": result.context.compressed,
            "user_query": result.user_query.compressed,
        },
    )
    assert result.preservation_coverage_percentage == 100
    assert audit["coverage_percentage"] == 100
    assert "CSV" in result.context.compressed
    assert any("original context restored" in warning for warning in result.warnings)
    assert not result.reduction_target_met
    assert not any(
        "20% total reduction target" in warning for warning in result.warnings
    )


def test_empty_inputs_are_safe():
    result = compress_inputs("", "", "", tokenizer=TOKENIZER)
    assert result.original_tokens == result.compressed_tokens == 0
    assert result.reduction_percentage == 0


def test_system_prompt_is_immutable():
    prompt = "Rule A.\n\nRule A.\nRule B."
    assert compress_system_prompt(prompt) == (prompt, False)


def test_context_removes_duplicate_passages():
    result = compress_context(
        "alpha fact\n\nbeta fact\n\nalpha fact", "alpha", None, TOKENIZER.count
    )
    assert result.text == "alpha fact\n\nbeta fact"
    assert result.warnings


def test_context_removes_repeated_inline_sentence_from_realistic_incident_text():
    context = (
        "Checkout errors began at 09:42 UTC. 12% of checkout attempts failed. "
        "Login and payments were unaffected. Rollback completed at 10:05 UTC. "
        "Rollback completed at 10:05 UTC."
    )
    result = compress_context(
        context, "Summarize the incident timeline", None, TOKENIZER.count
    )
    assert result.text.count("Rollback completed at 10:05 UTC.") == 1
    assert "09:42 UTC" in result.text
    assert "12%" in result.text
    assert result.warnings


def test_relevant_chunk_is_selected_within_budget():
    context = "Garden benches are wooden.\n\nIncident resolved at 10:05 UTC.\n\nLogo color is green."
    result = compress_context(
        context, "When was incident resolved?", 5, TOKENIZER.count
    )
    assert "10:05" in result.text
    assert "benches" not in result.text
    assert not any(
        "lower-ranked context chunk" in warning for warning in result.warnings
    )


def test_reduction_target_selects_complete_high_relevance_chunks():
    context = (
        "Cats sleep on mats.\n\n"
        "Blue mugs sit on shelves.\n\n"
        "Sailboats cross the bay.\n\n"
        "Pine trees line roads."
    )
    result = compress_context(
        context,
        "Where do sailboats cross?",
        12,
        TOKENIZER.count,
        allow_sentence_chunks=True,
    )
    assert "Sailboats cross the bay." in result.text
    assert result.text.count("\n\n") == 2
    assert TOKENIZER.count(result.text) <= TOKENIZER.count(context) * 0.75


def test_fenced_json_is_atomic_while_surrounding_prose_can_be_pruned():
    json_block = '```json\n{"status": "ready"}\n```'
    context = (
        "The old office uses blue chairs.\n\n"
        f"{json_block}\n\n"
        "The garden has wooden benches.\n\n"
        "The kitchen serves hot tea."
    )
    result = compress_context(
        context,
        "What is the status?",
        int(TOKENIZER.count(context) * 0.4),
        TOKENIZER.count,
        allow_sentence_chunks=True,
    )
    assert json_block in result.text
    assert "```json\n{" in result.text
    assert result.text.count("```") == 2
    assert "The old office" not in result.text


def test_unbalanced_delimiter_chunk_is_kept_whole():
    malformed = "Retain this incomplete [reference"
    context = f"{malformed}\n\nDiscard this unrelated paragraph.\n\nAnother spare note."
    result = compress_context(
        context,
        "What must be retained?",
        5,
        TOKENIZER.count,
        allow_sentence_chunks=True,
    )
    assert malformed in result.text
    assert "[reference" in result.text


def test_unreachable_reduction_target_preserves_code_without_warning():
    code = "```python\ndef calculate(value):\n    return value + 1\n```"
    result = compress_context(
        code,
        "calculate",
        int(TOKENIZER.count(code) * 0.5),
        TOKENIZER.count,
        allow_sentence_chunks=True,
    )
    assert result.text == code
    assert not any("exceed the target" in warning for warning in result.warnings)


def test_system_tool_and_few_shot_context_blocks_are_protected():
    system_block = "System: Never disclose the example's secret."
    tool_block = 'Tool: {"name": "lookup", "parameters": {"id": "string"}}'
    example_block = "Few-shot: User asks for JSON. Assistant returns valid JSON."
    context = (
        f"{system_block}\n\n{tool_block}\n\n{example_block}\n\n"
        "The unused lobby was painted blue."
    )
    result = compress_context(
        context,
        "What must the response format be?",
        int(TOKENIZER.count(context) * 0.75),
        TOKENIZER.count,
        allow_sentence_chunks=True,
    )
    assert system_block in result.text
    assert tool_block in result.text
    assert example_block in result.text


def test_context_normalizer_removes_whitespace_and_standalone_boilerplate():
    context = (
        "Sure, here is your answer:\n"
        "The   cache   stores responses.\n\n\n"
        "The cache stores responses."
    )
    result = compress_context(context, "cache", None, TOKENIZER.count)
    assert result.text == "The cache stores responses."
    assert result.warnings


def test_query_is_immutable():
    query = "Please do not exceed 12 retries."
    assert compress_query(query, TOKENIZER.count) == (query, False)


def test_code_and_numeric_context_are_retained_over_budget_without_warning():
    code = "LIMIT = 64\n\ndef take(items):\n    return items[:LIMIT]"
    result = compress_context(code, "limit", 1, TOKENIZER.count)
    assert result.text == code
    assert not result.warnings


def test_numeric_identifiers_and_constraints_are_protected_when_selecting_chunks():
    context = (
        "The garden has wooden benches.\n\n"
        "Use parse_event_v2 with HTTPClient for the request.\n\n"
        "The policy is not approved before 2025-03-01 and allows 12,000 requests per window."
    )
    result = compress_context(context, "What is in the garden?", 5, TOKENIZER.count)
    assert "parse_event_v2" in result.text
    assert "HTTPClient" in result.text
    assert "not approved" in result.text
    assert "2025-03-01" in result.text
    assert "12,000" in result.text
    assert result.text.count("\n\n") == 1


def test_acronym_pascal_case_identifier_is_protected():
    context = "Call HTTPClient for requests.\n\nThe garden has green benches."
    result = compress_context(context, "garden benches", 5, TOKENIZER.count)
    assert "HTTPClient" in result.text


@pytest.mark.parametrize(
    "critical_chunk",
    [
        "Quota is 12,000 requests per hour.",
        "Call parse_event_v2 before retry.",
        "Do not deploy unless tests pass.",
        "User: Use concise answers.",
    ],
)
def test_critical_and_recent_user_chunks_survive_tight_budget(critical_chunk):
    context = f"{critical_chunk}\n\nThe garden has six wooden benches."
    result = compress_context(
        context,
        "How many garden benches are there?",
        len(critical_chunk.split()),
        TOKENIZER.count,
    )
    assert critical_chunk in result.text


def test_latest_two_user_messages_are_prioritized_over_irrelevant_history():
    context = (
        "User: I like green themes.\n\n"
        "Assistant: Noted.\n\n"
        "User: Prefer concise summaries.\n\n"
        "Assistant: Understood.\n\n"
        "User: Use bullet lists."
    )
    result = compress_context(
        context, "Recommend a database engine.", 9, TOKENIZER.count
    )
    assert "Prefer concise summaries" in result.text
    assert "Use bullet lists" in result.text
    assert "I like green themes" not in result.text


def test_query_without_safe_shortening_is_unchanged():
    query = "Is it safe to delete the only backup?"
    assert compress_query(query, TOKENIZER.count) == (query, False)


def test_zero_budget_falls_back_instead_of_silently_dropping():
    context = "The service uses a cache."
    result = compress_context(context, "cache", 0, TOKENIZER.count)
    assert result.text == context
    assert not result.warnings


def test_api_restores_original_when_compression_increases_context_token_count():
    class ExpandingTokenizer:
        model = "expanding-test"

        def count(self, text):
            return 20 if text == "Rule." else len(text.split())

    result = compress_inputs("", "Rule.\nRule.", "", tokenizer=ExpandingTokenizer())
    assert result.original_tokens == 2
    assert result.compressed_tokens == 2
    assert result.tokens_saved == 0
    assert any(
        "increased context token usage" in warning for warning in result.warnings
    )


def test_negative_budget_rejected():
    with pytest.raises(ValueError):
        compress_context("some text", "text", -1, TOKENIZER.count)


def test_negative_api_token_budget_rejected():
    with pytest.raises(ValueError):
        compress_inputs("", "context", "", token_budget=-1, tokenizer=TOKENIZER)


def test_quality_evaluation_is_explicitly_not_run_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert evaluate_answers(None) == {
        "status": "not run",
        "reason": "OPENAI_API_KEY is not configured.",
    }
    assert "semantic_equivalence" in RUBRIC


def test_keyless_preservation_audit_covers_numeric_code_format_table_and_recent_turns():
    originals = {
        "system_prompt": "Return valid JSON. Do not omit version 3.12.",
        "context": (
            "At 09:42 UTC use parse_event_v2. Never deploy before 2026-10-09.\n"
            "Region | Eligible\nWest | no\n"
            "User: Older request.\nUser: Keep 12 rows.\nUser: Never remove IDs."
        ),
        "user_query": "Return JSON and do not drop 12 rows.",
    }
    compressed = {
        "system_prompt": originals["system_prompt"],
        "context": (
            "At 09:42 UTC use parse_event_v2. Never deploy before 2026-10-09.\n"
            "Region | Eligible\nWest | no\n"
            "User: Keep 12 rows.\nUser: Never remove IDs."
        ),
        "user_query": originals["user_query"],
    }

    audit = audit_preservation(originals, compressed)

    assert audit["status"] == "pass"
    assert audit["coverage_percentage"] == 100
    assert audit["required_markers"] > 0
    assert not any(row["missing"] for row in audit["rows"])
    assert {row["category"] for row in audit["rows"]} >= {
        "Numbers and dates",
        "Code identifiers",
        "Negations and conditions",
        "Output formats",
        "Table rows",
        "Latest user turns",
    }


def test_keyless_preservation_audit_warns_on_missing_markers():
    audit = audit_preservation(
        {
            "system_prompt": "Do not deploy before 2026-10-09 with APIClient.",
            "context": "",
            "user_query": "",
        },
        {"system_prompt": "Deploy now.", "context": "", "user_query": ""},
    )

    assert audit["status"] == "warning"
    assert audit["coverage_percentage"] < 100
    assert any(row["missing"] for row in audit["rows"])


def test_keyless_preservation_audit_handles_empty_inputs():
    audit = audit_preservation(
        {"system_prompt": "", "context": "", "user_query": ""}, {}
    )

    assert audit["status"] == "not_applicable"
    assert audit["coverage_percentage"] is None
    assert audit["required_markers"] == audit["retained_markers"] == 0


def test_quality_evaluation_success_path_uses_three_provider_calls(monkeypatch):
    calls = []
    judge_response = {
        "baseline": {key: 5 for key in RUBRIC},
        "compressed": {key: 5 for key in RUBRIC},
        "materially_equivalent": True,
        "equivalence_notes": "Same facts and constraints.",
        "notes": "No material differences.",
    }
    responses = [
        ("Original answer", 12.0),
        ("Compressed answer", 10.0),
        (json.dumps(judge_response), 3.0),
    ]

    def fake_chat(messages, model, api_key, base_url, temperature, max_tokens):
        calls.append(
            {
                "model": model,
                "api_key": api_key,
                "base_url": base_url,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return responses[len(calls) - 1]

    monkeypatch.setattr("context_surgeon.evaluation._chat", fake_chat)
    result = compress_inputs(
        "Answer precisely.",
        "The release is 14 June.",
        "When is release?",
        tokenizer=TOKENIZER,
    )
    evaluation = evaluate_answers(
        result,
        api_key=" \t test-key\r\n",
        model="test-model",
        base_url="https://provider.test/v1",
    )

    assert evaluation["status"] == "completed"
    assert evaluation["scores"]["materially_equivalent"] is True
    assert evaluation["baseline_answer"] == "Original answer"
    assert evaluation["compressed_answer"] == "Compressed answer"
    assert len(calls) == 3
    assert all(
        call
        == {
            "model": "test-model",
            "api_key": "test-key",
            "base_url": "https://provider.test/v1",
            "temperature": 0,
            "max_tokens": 512,
        }
        for call in calls
    )


def test_unauthorized_provider_error_is_actionable_and_does_not_echo_key(monkeypatch):
    exposed_key = "test-secret-that-must-not-appear"

    def unauthorized(*args, **kwargs):
        raise urllib.error.HTTPError(
            "https://provider.test/v1/chat/completions",
            401,
            "Unauthorized",
            {},
            io.BytesIO(),
        )

    monkeypatch.setattr("context_surgeon.evaluation._chat", unauthorized)
    result = compress_inputs(
        "Answer precisely.", "A fact.", "What fact?", tokenizer=TOKENIZER
    )
    evaluation = evaluate_answers(result, api_key=exposed_key, model="test-model")

    assert evaluation["status"] == "error"
    assert "401 Unauthorized" in evaluation["reason"]
    assert "active OpenAI Platform API key" in evaluation["reason"]
    assert exposed_key not in evaluation["reason"]


def test_api_key_check_uses_model_metadata_endpoint_without_generation(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    requests = []

    def fake_open(request, timeout):
        requests.append(request)
        assert timeout == 30
        return FakeResponse()

    monkeypatch.setattr("context_surgeon.evaluation.urllib.request.urlopen", fake_open)
    result = validate_api_key(
        "  test-key\n", model="gpt-4o-mini", base_url="https://provider.test/v1"
    )

    assert result == {"status": "valid", "model": "gpt-4o-mini"}
    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert requests[0].full_url == "https://provider.test/v1/models/gpt-4o-mini"
    assert requests[0].get_header("Authorization") == "Bearer test-key"


def test_benchmark_runs_all_examples_and_exports_csv_json(tmp_path):
    progress_updates = []
    rows, aggregate = run_benchmarks(
        tokenizer=TOKENIZER,
        repeats=3,
        progress_callback=lambda completed, total: progress_updates.append(
            (completed, total)
        ),
    )
    assert len(BENCHMARK_EXAMPLES) >= 30
    assert len(rows) == len(BENCHMARK_EXAMPLES)
    assert aggregate["examples"] == len(rows)
    assert aggregate["runs_per_example"] == 3
    assert aggregate["median_compression_latency_ms"] >= 0
    assert (
        aggregate["p95_compression_latency_ms"]
        >= aggregate["median_compression_latency_ms"]
    )
    assert aggregate["answer_quality_evaluation"] == "not run"
    assert progress_updates == [
        (completed, len(BENCHMARK_EXAMPLES))
        for completed in range(1, len(BENCHMARK_EXAMPLES) + 1)
    ]
    csv_path, json_path = export_benchmarks(rows, aggregate, tmp_path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == len(rows)
    exported = json.loads(json_path.read_text(encoding="utf-8"))
    assert exported["aggregate"]["examples"] == len(rows)
    assert {
        "original_tokens",
        "compressed_tokens",
        "token_reduction",
        "minimum_reduction_percentage",
        "reduction_target_met",
        "preservation_coverage_percentage",
        "compression_latency_ms",
        "warnings",
    } <= exported["results"][0].keys()
    assert exported["aggregate"]["minimum_reduction_percentage"] == 20
    assert exported["aggregate"]["all_cases_preservation_100"]
