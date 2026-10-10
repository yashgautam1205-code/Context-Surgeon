"""Streamlit dashboard for Context Surgeon."""

from __future__ import annotations

import csv
import difflib
import html
import io
import json
import re
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from context_surgeon.api import compress_inputs
from context_surgeon.benchmarks import run_benchmarks
from context_surgeon.preservation import audit_preservation
from context_surgeon.tokenizer import TokenizerAdapter, TokenizerUnavailableError

load_dotenv()

st.set_page_config(page_title="Context Surgeon", page_icon="✂", layout="wide")
st.markdown(
    f"""
<style>{Path(__file__).with_name("style.css").read_text(encoding="utf-8")}</style>
<div class="brandbar">
  <div><p class="brand-kicker">Prompt optimization / Local analysis</p><h1>Context Surgeon</h1><p class="brand-subtitle">Reduce input tokens. Keep the details that matter.</p></div>
  <div class="brand-stamp">Target-aware<br>tiktoken counts</div>
</div>
""",
    unsafe_allow_html=True,
)

SAMPLE = {
    "system_prompt": "You are an incident analyst. Use only supplied facts. Preserve exact timestamps, percentages, affected services, and negations. Clearly mark unknown causes. Respond in exactly three bullets.",
    "context": "Checkout errors began at 09:42 UTC. 12% of checkout attempts failed. Login and payments were unaffected. Rollback completed at 10:05 UTC. Rollback completed at 10:05 UTC.",
    "user_query": "Summarize the impact and timeline in three bullets. Do not claim a cause; it has not been confirmed.",
}


@st.cache_resource
def _get_tokenizer(model: str) -> TokenizerAdapter:
    return TokenizerAdapter(model)


def _timestamped_filename(stem: str, suffix: str, timestamp: str | None) -> str:
    timestamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stem}-{timestamp}.{suffix}"


def _render_metrics(
    metrics: list[tuple[str, str | int | float] | tuple[str, str | int | float, str]],
) -> None:
    columns = st.columns(len(metrics))
    for column, metric in zip(columns, metrics):
        label, value = metric[:2]
        if len(metric) == 3:
            column.metric(label, value, delta=metric[2], delta_color="inverse")
        else:
            column.metric(label, value)


def _render_download_buttons(downloads: list[tuple[str, str, str, str]]) -> None:
    columns = st.columns(len(downloads))
    for column, (label, data, filename, mime) in zip(columns, downloads):
        column.download_button(label, data, filename, mime, width="stretch")


def _render_audit_table(rows: list[dict]) -> None:
    st.dataframe(
        [
            {
                "Input": row["input"],
                "Marker type": row["category"],
                "Original": row["required"],
                "Retained": row["retained"],
                "Coverage": f"{row['coverage_percentage']:.1f}%",
                "Missing": ", ".join(row["missing"]) or "None",
            }
            for row in rows
        ],
        hide_index=True,
        width="stretch",
    )


def _render_inline_diff(original: str, compressed: str) -> None:
    original_tokens = re.findall(r"\s+|\S+", original)
    compressed_tokens = re.findall(r"\s+|\S+", compressed)
    matcher = difflib.SequenceMatcher(
        None, original_tokens, compressed_tokens, autojunk=False
    )
    original_markup: list[str] = []
    compressed_markup: list[str] = []
    for (
        tag,
        start_original,
        end_original,
        start_compressed,
        end_compressed,
    ) in matcher.get_opcodes():
        if tag == "equal":
            original_markup.extend(
                f'<span class="diff-kept">{html.escape(token)}</span>'
                for token in original_tokens[start_original:end_original]
            )
            compressed_markup.extend(
                f'<span class="diff-kept">{html.escape(token)}</span>'
                for token in compressed_tokens[start_compressed:end_compressed]
            )
        elif tag == "delete":
            original_markup.extend(
                f'<del class="diff-removed">{html.escape(token)}</del>'
                for token in original_tokens[start_original:end_original]
            )
        elif tag == "insert":
            compressed_markup.extend(
                f'<ins class="diff-added">{html.escape(token)}</ins>'
                for token in compressed_tokens[start_compressed:end_compressed]
            )
        else:
            original_markup.extend(
                f'<del class="diff-removed">{html.escape(token)}</del>'
                for token in original_tokens[start_original:end_original]
            )
            compressed_markup.extend(
                f'<ins class="diff-added">{html.escape(token)}</ins>'
                for token in compressed_tokens[start_compressed:end_compressed]
            )
    original_col, compressed_col = st.columns(2)
    with original_col:
        st.markdown("**Original · removed text highlighted**")
        st.markdown(
            f'<div class="diff-output">{"".join(original_markup)}</div>',
            unsafe_allow_html=True,
        )
    with compressed_col:
        st.markdown("**Compressed · retained and added text highlighted**")
        st.markdown(
            f'<div class="diff-output">{"".join(compressed_markup)}</div>',
            unsafe_allow_html=True,
        )


def _clear_compression_outputs() -> None:
    st.session_state.result = None
    st.session_state.compression_timestamp = None
    st.session_state.benchmark_rows = None
    st.session_state.benchmark_aggregate = None
    st.session_state.benchmark_timestamp = None


def _clear_all_outputs() -> None:
    _clear_compression_outputs()


if "_page_initialized" not in st.session_state:
    for name in SAMPLE:
        st.session_state[name] = ""
    _clear_all_outputs()
    st.session_state._page_initialized = True
else:
    for name in SAMPLE:
        st.session_state.setdefault(name, "")
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("compression_timestamp", None)
    st.session_state.setdefault("benchmark_rows", None)
    st.session_state.setdefault("benchmark_aggregate", None)
    st.session_state.setdefault("benchmark_timestamp", None)
st.session_state.pop("provider_api_key", None)
st.session_state.pop("api_key_check", None)
st.session_state.pop("evaluation", None)

sample_col, reset_col, spacer = st.columns([1, 1, 6])
if sample_col.button("Load sample", icon="📄"):
    for name, value in SAMPLE.items():
        st.session_state[name] = value
    _clear_all_outputs()
    st.rerun()
if reset_col.button("Reset", icon="🗑"):
    for name in SAMPLE:
        st.session_state[name] = ""
    _clear_all_outputs()
    st.rerun()

left, right = st.columns(2)
with left:
    st.markdown(
        '<div class="section-kicker">01 / Instructions and evidence</div>',
        unsafe_allow_html=True,
    )
    st.text_area(
        "System Prompt",
        key="system_prompt",
        height=150,
        placeholder="Instructions, priorities, and constraints",
        on_change=_clear_compression_outputs,
    )
    st.text_area(
        "Context",
        key="context",
        height=260,
        placeholder="Conversation history, facts, code, or data",
        on_change=_clear_compression_outputs,
    )
with right:
    st.markdown(
        '<div class="section-kicker">02 / Request and token settings</div>',
        unsafe_allow_html=True,
    )
    st.text_area(
        "User Query",
        key="user_query",
        height=150,
        placeholder="The current request",
        on_change=_clear_compression_outputs,
    )
    model = st.selectbox(
        "Target tokenizer model",
        ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
        index=0,
        on_change=_clear_compression_outputs,
    )
    token_budget = st.number_input(
        "Context token budget",
        min_value=0,
        max_value=100000,
        value=512,
        step=32,
        on_change=_clear_compression_outputs,
    )
    st.caption(
        "Compression targets at least 20% total token reduction. The local preservation audit must remain at 100%; if both goals cannot be met, the app keeps the markers and reports the missed reduction. System prompts and active queries stay unchanged."
    )
    st.caption(
        "Token reduction, benchmark, and local quality proxy run without API credentials."
    )

if st.button("Compress", type="primary", width="stretch"):
    try:
        with st.spinner("Compressing inputs..."):
            st.session_state.result = compress_inputs(
                st.session_state.system_prompt,
                st.session_state.context,
                st.session_state.user_query,
                token_budget=int(token_budget),
                model=model,
                tokenizer=_get_tokenizer(model),
            )
            st.session_state.compression_timestamp = datetime.now().strftime(
                "%Y%m%d-%H%M%S"
            )
    except TokenizerUnavailableError as exc:
        st.error(str(exc))

if st.button("Run 33-case benchmark", key="run_benchmark"):
    try:
        progress_bar = st.progress(0, text="Starting benchmark...")
        rows, aggregate = run_benchmarks(
            int(token_budget),
            model,
            _get_tokenizer(model),
            progress_callback=lambda completed, total: progress_bar.progress(
                completed / total,
                text=f"Benchmarking case {completed} of {total}...",
            ),
        )
        st.session_state.benchmark_rows = rows
        st.session_state.benchmark_aggregate = aggregate
        st.session_state.benchmark_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        progress_bar.empty()
    except TokenizerUnavailableError as exc:
        progress_bar.empty()
        st.error(str(exc))

result = st.session_state.result
if result:
    preservation_audit = audit_preservation(
        {
            "system_prompt": result.system_prompt.original,
            "context": result.context.original,
            "user_query": result.user_query.original,
        },
        {
            "system_prompt": result.system_prompt.compressed,
            "context": result.context.compressed,
            "user_query": result.user_query.compressed,
        },
    )
    st.divider()
    st.markdown(
        '<div class="section-kicker">03 / Compression report</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Token reduction")
    _render_metrics(
        [
            ("Original tokens", result.original_tokens),
            (
                "Compressed tokens",
                result.compressed_tokens,
                str(result.compressed_tokens - result.original_tokens),
            ),
            ("Tokens saved", result.tokens_saved),
            ("Reduction", f"{result.reduction_percentage:.1f}%"),
            ("Latency", f"{result.compression_latency_ms:.2f} ms"),
        ]
    )
    st.caption(
        "The compressor aims for 20% total token reduction when it can maintain 100% local marker coverage. The actual reduction is shown above."
    )
    if result.tokens_saved < 0:
        st.warning(
            "Compression increased token usage. Review the outputs before using them."
        )
    for warning in result.warnings:
        st.warning(warning)

    comparison = []
    for label, key in [
        ("System Prompt", "system_prompt"),
        ("Context", "context"),
        ("User Query", "user_query"),
    ]:
        item = getattr(result, key)
        saved = item.original_tokens - item.compressed_tokens
        reduction = (
            (saved / item.original_tokens * 100) if item.original_tokens else 0.0
        )
        comparison.append(
            {
                "Input": label,
                "Original": item.original_tokens,
                "Compressed": item.compressed_tokens,
                "Saved": saved,
                "Reduction": f"{reduction:.1f}%",
            }
        )
    st.markdown("**Per-input token accounting**")
    st.dataframe(comparison, hide_index=True, width="stretch")
    st.markdown("**Token usage by input**")
    st.bar_chart(
        {
            "Input": ["Original", "Compressed"],
            "System Prompt": [
                result.system_prompt.original_tokens,
                result.system_prompt.compressed_tokens,
            ],
            "Context": [
                result.context.original_tokens,
                result.context.compressed_tokens,
            ],
            "User Query": [
                result.user_query.original_tokens,
                result.user_query.compressed_tokens,
            ],
        },
        x="Input",
        y=["System Prompt", "Context", "User Query"],
        x_label="Input version",
        y_label="Tokens",
        horizontal=True,
        stack=True,
    )

    st.markdown("**Inline text diff**")
    for label, key in [
        ("System Prompt", "system_prompt"),
        ("Context", "context"),
        ("User Query", "user_query"),
    ]:
        item = getattr(result, key)
        with st.expander(f"{label} diff"):
            _render_inline_diff(item.original, item.compressed)

    result_payload = result.to_dict()
    result_payload["preservation_audit"] = preservation_audit
    result_payload["quality_assessment"] = {
        "method": "keyless explicit-marker preservation audit",
        "status": preservation_audit["status"],
        "coverage_percentage": preservation_audit["coverage_percentage"],
        "scope": preservation_audit["scope"],
    }
    result_json = json.dumps(result_payload, indent=2)
    csv_stream = io.StringIO()
    writer = csv.writer(csv_stream)
    writer.writerow(
        ["input", "original_tokens", "compressed_tokens", "original", "compressed"]
    )
    for label, key in [
        ("System Prompt", "system_prompt"),
        ("Context", "context"),
        ("User Query", "user_query"),
    ]:
        item = getattr(result, key)
        writer.writerow(
            [
                label,
                item.original_tokens,
                item.compressed_tokens,
                item.original,
                item.compressed,
            ]
        )
    compression_timestamp = st.session_state.compression_timestamp
    _render_download_buttons(
        [
            (
                "Download compressed result (JSON)",
                result_json,
                _timestamped_filename(
                    "context-surgeon-result", "json", compression_timestamp
                ),
                "application/json",
            ),
            (
                "Download comparison (CSV)",
                csv_stream.getvalue(),
                _timestamped_filename(
                    "context-surgeon-result", "csv", compression_timestamp
                ),
                "text/csv",
            ),
        ]
    )

    st.divider()
    st.markdown(
        '<div class="section-kicker">04 / Keyless quality proxy</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Quality comparison · input preservation")
    audit_coverage = preservation_audit["coverage_percentage"]
    audit_status = (
        "no markers"
        if preservation_audit["status"] == "not_applicable"
        else preservation_audit["status"].upper()
    )
    _render_metrics(
        [
            (
                "Markers retained",
                f"{preservation_audit['retained_markers']}/{preservation_audit['required_markers']}",
            ),
            (
                "Marker coverage",
                f"{audit_coverage:.1f}%"
                if audit_coverage is not None
                else f"{result.preservation_coverage_percentage:.1f}% (no markers)",
            ),
            ("Audit status", audit_status),
        ]
    )
    if preservation_audit["rows"]:
        _render_audit_table(preservation_audit["rows"])
    else:
        st.info(
            "No explicit numeric, identifier, constraint, format, table, or recent-user markers were found to audit."
        )
    st.caption(preservation_audit["scope"])


benchmark_rows = st.session_state.benchmark_rows
benchmark_aggregate = st.session_state.benchmark_aggregate
if benchmark_rows and benchmark_aggregate:
    st.divider()
    st.markdown(
        '<div class="section-kicker">05 / Benchmark</div>', unsafe_allow_html=True
    )
    st.subheader("33-case compression benchmark")
    _render_metrics(
        [
            ("Cases", benchmark_aggregate["examples"]),
            ("Original tokens", benchmark_aggregate["original_tokens"]),
            ("Compressed tokens", benchmark_aggregate["compressed_tokens"]),
            ("Tokens saved", benchmark_aggregate["tokens_saved"]),
            ("Reduction", f"{benchmark_aggregate['reduction_percentage']:.1f}%"),
            (
                "Cases reaching 20%",
                f"{benchmark_aggregate['cases_meeting_minimum_reduction']}/{benchmark_aggregate['examples']}",
            ),
        ]
    )
    _render_metrics(
        [
            (
                "Mean latency",
                f"{benchmark_aggregate['mean_compression_latency_ms']:.3f} ms",
            ),
            (
                "Median latency",
                f"{benchmark_aggregate['median_compression_latency_ms']:.3f} ms",
            ),
            (
                "P95 latency",
                f"{benchmark_aggregate['p95_compression_latency_ms']:.3f} ms",
            ),
        ]
    )
    st.caption(
        f"{benchmark_aggregate['runs_per_example']} runs per case. "
        f"100% marker preservation across cases: {benchmark_aggregate['all_cases_preservation_100']}. "
        f"Answer-quality evaluation: {benchmark_aggregate['answer_quality_evaluation']}."
    )
    st.dataframe(benchmark_rows, hide_index=True, width="stretch")

    benchmark_csv = io.StringIO()
    benchmark_writer = csv.DictWriter(benchmark_csv, fieldnames=list(benchmark_rows[0]))
    benchmark_writer.writeheader()
    benchmark_writer.writerows(benchmark_rows)
    benchmark_json = json.dumps(
        {"aggregate": benchmark_aggregate, "results": benchmark_rows},
        indent=2,
    )
    benchmark_timestamp = st.session_state.benchmark_timestamp
    _render_download_buttons(
        [
            (
                "Download benchmark (CSV)",
                benchmark_csv.getvalue(),
                _timestamped_filename("benchmark-results", "csv", benchmark_timestamp),
                "text/csv",
            ),
            (
                "Download benchmark (JSON)",
                benchmark_json,
                _timestamped_filename("benchmark-results", "json", benchmark_timestamp),
                "application/json",
            ),
        ]
    )
