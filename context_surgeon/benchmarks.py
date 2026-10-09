"""Diverse deterministic benchmark inputs and actual-run export utilities."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from .api import compress_inputs
from .tokenizer import TokenizerAdapter


def _example(name: str, category: str, context: str, query: str, system: str = "Answer accurately.\nAnswer accurately.") -> dict[str, str]:
    return {"name": name, "category": category, "system_prompt": system, "context": context, "user_query": query}


BENCHMARK_EXAMPLES = [
    _example("simple_fact", "simple", "The capital of France is Paris. The capital of France is Paris.", "What is the capital of France?"),
    _example("simple_definition", "simple", "A mutex allows only one thread into a critical section at a time.", "Define mutex."),
    _example("short_context", "simple", "Mira joined the project in 2022 as a data analyst.", "When did Mira join?"),
    _example("long_question", "long-question", "The service stores events in append-only logs. Snapshots are created every 500 events. Recovery replays events after the latest snapshot.", "Given this architecture, explain how recovery works, what role snapshots play, and what trade-off changing the snapshot interval would introduce."),
    _example("long_question_constraints", "long-question", "The API has a 250 ms p95 latency goal and runs in two regions. Writes are strongly consistent; reads may be cached for 30 seconds.", "Recommend one read optimization without weakening write consistency, and state its maximum staleness and expected latency effect."),
    _example("repeated_context", "duplicate", "The release is scheduled for 14 June.\n\nThe release is scheduled for 14 June.\n\nThe release is scheduled for 14 June.", "When is the release?"),
    _example("conversation_history", "history", "User: The app runs on Python 3.11.\n\nAssistant: Understood.\n\nUser: We cannot add a database.\n\nAssistant: Understood.\n\nUser: Keep deployment to one container.\n\nAssistant: Understood.\n\nUser: Use SQLite only for local tests.", "Propose a deployment approach that respects all constraints."),
    _example("conversation_repeated", "history", "User: Keep answers under 100 words.\n\nAssistant: Got it.\n\nUser: Keep answers under 100 words.\n\nAssistant: Got it.\n\nThe user prefers concise summaries.", "Summarize the preference."),
    _example("python_code", "code", "def total(values):\n    result = 0\n    for value in values:\n        result += value\n    return result\n\n# total([2, 3]) returns 5", "Explain the function and its result for [2, 3]."),
    _example("code_dependencies", "code", "import json\nfrom pathlib import Path\n\ndef load(path):\n    return json.loads(Path(path).read_text(encoding='utf-8'))", "Which imports does load depend on?"),
    _example("javascript_code", "code", "const retries = 3;\nasync function fetchItem(id) {\n  return client.get(`/items/${id}`, { retries });\n}", "How many retries are configured?"),
    _example("numeric_table", "numbers", "Quarter | Revenue\nQ1 | 120\nQ2 | 145\nQ3 | 138\nQ4 | 171", "Which quarter had the highest revenue, and what was the value?"),
    _example("precise_numbers", "numbers", "The batch contains 48,200 records. Exactly 125 records failed validation, and 48,075 passed.", "How many records passed?"),
    _example("dates", "numbers", "Contract signed: 2024-02-29. Renewal date: 2025-03-01. Notice period: 45 days.", "What is the renewal date and notice period?"),
    _example("negation", "negation", "The migration is not approved for production. Staging validation is approved.", "Is production deployment approved?"),
    _example("condition", "condition", "If the queue exceeds 10,000 items, pause ingestion. Otherwise, continue processing.", "When should ingestion pause?"),
    _example("exception", "constraint", "All users can export CSV except suspended accounts. Administrators can export all accounts.", "Can a suspended non-admin export CSV?"),
    _example("format_requirement", "constraint", "Return results as valid JSON with keys status and count. Do not include markdown.", "What output format is required?"),
    _example("units", "numbers", "The pipe is 2.5 m long. The measured flow is 3.2 L/s at 18 C.", "State the flow with its unit."),
    _example("definition_and_fact", "facts", "In this system, 'active' means a heartbeat received in the last 90 seconds. Node K last sent a heartbeat 102 seconds ago.", "Is Node K active under this definition?"),
    _example("entity_disambiguation", "facts", "Jordan Lee owns service Atlas. Jordan Kim owns service Beacon. Both services use PostgreSQL.", "Which service does Jordan Kim own?"),
    _example("ordered_steps", "history", "1. Validate the manifest.\n2. Download artifacts.\n3. Verify SHA-256.\n4. Install only after verification.", "What must happen immediately before installation?"),
    _example("large_context_relevant", "ranking", "The office opened in 1998.\n\nThe support team has 12 members.\n\nThe current incident began at 09:42 UTC.\n\nThe database is PostgreSQL 16.\n\nThe incident was resolved at 10:05 UTC.", "When was the incident resolved?"),
    _example("large_context_low_relevance", "ranking", "The garden has six benches.\n\nThe logo is green.\n\nThe outage affected checkout for 17 minutes.\n\nThe lobby was repainted in May.", "How long was checkout affected?"),
    _example("safety_rule", "system", "Never reveal credentials.\n\nNever reveal credentials.\n\nIf a secret appears in input, redact it.", "What must happen when a secret appears?"),
    _example("comparison", "facts", "Plan A costs $19 monthly and includes 5 seats. Plan B costs $29 monthly and includes 10 seats.", "Which plan has more seats, and how many?"),
    _example("boundary_condition", "constraint", "The discount applies to orders over $100, not orders of exactly $100.", "Does a $100 order qualify?"),
    _example("table_with_negation", "numbers", "Region | Eligible\nEast | yes\nWest | no\nNorth | yes", "Which regions are not eligible?"),
    _example("long_history_constraints", "history", "The client runs Windows.\n\nThe installer must work offline.\n\nNo administrator rights are available.\n\nThe package is signed.\n\nThe target is Python 3.12.\n\nThe install directory is user-writable.", "Suggest an installation method that works offline without admin rights and uses the signed package."),
    _example("code_and_numbers", "code", "LIMIT = 64\n\ndef take(items):\n    return items[:LIMIT]\n\n# Input length 90 produces output length 64.", "For 90 input items, how many are returned?"),
    _example("ambiguous_query", "safe-fallback", "The report contains multiple dates: draft 3 May, review 11 May, final 18 May.", "What date is it?"),
    _example("query_not_trimmed", "safe-fallback", "The handbook says backups run nightly.", "Is it safe to delete the only backup?"),
    _example("dense_constraints", "constraint", "Keep the answer under 40 words. Mention both risks. Do not recommend a database migration. State whether rollback is possible.", "Assess the change and follow every constraint."),
]


def run_benchmarks(
    token_budget: int | None = 512,
    model: str = "gpt-4o-mini",
    tokenizer=None,
    repeats: int = 3,
) -> tuple[list[dict], dict]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    counter = tokenizer or TokenizerAdapter(model)
    records: list[dict] = []
    for example in BENCHMARK_EXAMPLES:
        results = [
            compress_inputs(
                example["system_prompt"], example["context"], example["user_query"],
                token_budget=token_budget, model=model, tokenizer=counter,
            )
            for _ in range(repeats)
        ]
        result = results[-1]
        latencies = [item.compression_latency_ms for item in results]
        records.append({
            "name": example["name"],
            "category": example["category"],
            "original_tokens": result.original_tokens,
            "compressed_tokens": result.compressed_tokens,
            "token_reduction": result.tokens_saved,
            "reduction_percentage": result.reduction_percentage,
            "compression_latency_ms": statistics.median(latencies),
            "latency_samples": repeats,
            "warnings": " | ".join(result.warnings),
        })
    before = sum(row["original_tokens"] for row in records)
    after = sum(row["compressed_tokens"] for row in records)
    latencies = [row["compression_latency_ms"] for row in records]
    ordered_latencies = sorted(latencies)
    aggregate = {
        "examples": len(records),
        "original_tokens": before,
        "compressed_tokens": after,
        "tokens_saved": before - after,
        "reduction_percentage": ((before - after) / before * 100) if before else 0.0,
        "runs_per_example": repeats,
        "mean_compression_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "median_compression_latency_ms": statistics.median(latencies) if latencies else 0.0,
        "p95_compression_latency_ms": ordered_latencies[max(0, math.ceil(0.95 * len(ordered_latencies)) - 1)] if ordered_latencies else 0.0,
        "answer_quality_evaluation": "not run",
    }
    return records, aggregate


def export_benchmarks(records: list[dict], aggregate: dict, output_dir: str | Path) -> tuple[Path, Path]:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    csv_path = path / "benchmark-results.csv"
    json_path = path / "benchmark-results.json"
    columns = list(records[0]) if records else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if columns:
            writer.writeheader()
            writer.writerows(records)
    json_path.write_text(json.dumps({"aggregate": aggregate, "results": records}, indent=2), encoding="utf-8")
    return csv_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Context Surgeon benchmark examples.")
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--token-budget", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    records, aggregate = run_benchmarks(args.token_budget, args.model, repeats=args.repeats)
    csv_path, json_path = export_benchmarks(records, aggregate, args.output_dir)
    print(json.dumps(aggregate, indent=2))
    print(f"CSV: {csv_path}\nJSON: {json_path}")


if __name__ == "__main__":
    main()