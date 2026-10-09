"""Streamlit dashboard for Context Surgeon."""

from __future__ import annotations

import csv
import io
import json

import streamlit as st

from context_surgeon.api import compress_inputs
from context_surgeon.benchmarks import run_benchmarks
from context_surgeon.preservation import audit_preservation
from context_surgeon.tokenization_experiment import run_tokenization_experiment
from context_surgeon.tokenizer import TokenizerAdapter, TokenizerUnavailableError


st.set_page_config(page_title="Context Surgeon", page_icon="✂", layout="wide")
st.markdown("""
<style>
    :root {
        --ink: #163a35;
        --muted: #5e716c;
        --line: #d4dfda;
        --paper: #ffffff;
        --canvas: #eef3f0;
        --green: #176b55;
        --mint: #d8eee4;
        --copper: #b84c2d;
    }
    .stApp {
        background-color: var(--canvas);
        background-image: linear-gradient(rgba(22,58,53,.025) 1px, transparent 1px), linear-gradient(90deg, rgba(22,58,53,.025) 1px, transparent 1px);
        background-size: 28px 28px;
        color: var(--ink);
    }
    .block-container { max-width: 1440px; padding-top: 1.4rem; padding-bottom: 3rem; }
    .brandbar {
        display: flex; align-items: flex-end; justify-content: space-between; gap: 1rem;
        padding: 1.45rem 1.7rem; margin-bottom: 1.4rem;
        color: #f4faf6; background: var(--ink); border-bottom: 4px solid #db8052;
    }
    .brand-kicker { margin: 0 0 .4rem; color: #a9d8c2; font: 700 .72rem/1.2 'Aptos', 'Segoe UI', sans-serif; text-transform: uppercase; letter-spacing: .12em; }
    .brandbar h1 { margin: 0; color: #fff; font: 500 clamp(2rem, 4vw, 3.1rem)/1.04 Georgia, 'Times New Roman', serif; letter-spacing: 0; }
    .brand-subtitle { margin: .45rem 0 0; color: #d0e0d9; font-size: .96rem; }
    .brand-stamp { padding: .45rem .7rem; border: 1px solid #78978d; color: #d6e7df; font-size: .73rem; font-weight: 700; text-align: center; text-transform: uppercase; }
    h2, h3 { color: var(--ink); letter-spacing: 0; }
    h2 { font-family: Georgia, 'Times New Roman', serif; }
    .section-kicker { margin: .25rem 0 .65rem; color: var(--green); font-size: .73rem; font-weight: 800; text-transform: uppercase; letter-spacing: .1em; }
    [data-testid="stMetric"] { min-height: 102px; padding: 14px 16px; background: var(--paper); border: 1px solid var(--line); border-top: 3px solid var(--green); border-radius: 5px; }
    [data-testid="stMetricLabel"] { color: var(--muted); font-weight: 650; }
    [data-testid="stMetricValue"] { color: var(--ink); font-variant-numeric: tabular-nums; }
    div[data-testid="stTextArea"] label, div[data-testid="stTextInput"] label,
    div[data-testid="stSelectbox"] label, div[data-testid="stNumberInput"] label { color: var(--ink) !important; font-weight: 700; }
    div[data-testid="stTextInput"] label p, div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input { color: var(--ink) !important; }
    div[data-testid="stTextInput"] [data-baseweb="input"],
    div[data-testid="stNumberInput"] [data-baseweb="input"],
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
        color: var(--ink) !important; background: #fff !important; border-color: var(--line) !important;
    }
    div[data-testid="stTextInputRootElement"] {
        min-height: 42px;
        background: #fff !important;
        border: 1.5px solid #78978d !important;
        border-radius: 5px !important;
        box-shadow: 0 0 0 1px rgba(22, 58, 53, .08) !important;
    }
    div[data-testid="stTextInputRootElement"]:focus-within {
        border-color: var(--green) !important;
        box-shadow: 0 0 0 2px rgba(23, 107, 85, .18) !important;
    }
    div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input { background: transparent !important; }
    div[data-testid="stTextInput"] button svg { color: var(--ink) !important; fill: var(--ink) !important; }
    div[data-testid="stTextArea"] textarea { min-height: 130px; font-family: ui-monospace, 'Cascadia Code', Consolas, monospace; font-size: .88rem; line-height: 1.5; background: #fff; border-color: var(--line); border-radius: 5px; }
    div.stButton > button[kind="primary"] { color: #fff; background: var(--green); border: 1px solid var(--green); font-weight: 750; }
    div.stButton > button[kind="primary"]:hover { color: #fff; background: #10533f; border-color: #10533f; }
    div.stButton > button:not([kind="primary"]) { color: var(--ink) !important; background: #fff !important; border: 1px solid var(--line) !important; }
    div[data-testid="stDownloadButton"] button { color: var(--ink) !important; background: #fff !important; border: 1px solid var(--line) !important; }
    div[data-testid="stDownloadButton"] button:hover { color: #fff !important; background: var(--green) !important; border-color: var(--green) !important; }
    div[data-testid="stExpander"] details > summary { color: var(--ink) !important; background: #e5efeb !important; border: 1px solid var(--line) !important; border-radius: 5px !important; }
    div[data-testid="stExpander"] details > summary p,
    div[data-testid="stExpander"] details > summary span,
    div[data-testid="stExpander"] details > summary svg { color: var(--ink) !important; fill: var(--ink) !important; }
    div[data-testid="stExpander"] details > summary:hover { background: #d9e8e1 !important; }
    div[data-testid="stExpander"] [data-testid="stExpanderDetails"] { color: var(--ink) !important; background: #fff !important; border: 1px solid var(--line) !important; border-top: 0 !important; }
    div[data-testid="stCodeBlock"], div[data-testid="stCode"] { color: var(--ink) !important; background: #f7faf8 !important; border: 1px solid var(--line) !important; }
    div[data-testid="stCodeBlock"] pre, div[data-testid="stCodeBlock"] code,
    div[data-testid="stCode"] pre, div[data-testid="stCode"] code { color: var(--ink) !important; background: transparent !important; white-space: pre-wrap !important; overflow-wrap: anywhere !important; }
    [data-testid="stDataFrame"] {
        --gdg-bg-cell: #ffffff;
        --gdg-bg-cell-medium: #f1f6f3;
        --gdg-bg-header: #e5efeb;
        --gdg-bg-header-hovered: #d9e8e1;
        --gdg-text-dark: #163a35;
        --gdg-text-medium: #38564e;
        --gdg-text-light: #5e716c;
        --gdg-text-header: #163a35;
        --gdg-text-group-header: #163a35;
        --gdg-border-color: #d4dfda;
        --gdg-horizontal-border-color: #d4dfda;
        color: var(--ink) !important;
        background: #fff !important;
        border: 1px solid var(--line);
        border-radius: 5px;
        overflow: hidden;
    }
    .quality-status { display: inline-block; padding: .34rem .6rem; color: var(--ink); background: var(--mint); border-left: 3px solid var(--green); font-size: .75rem; font-weight: 800; text-transform: uppercase; }
    @media (max-width: 700px) {
        .brandbar { align-items: flex-start; flex-direction: column; padding: 1.2rem; }
        .brand-stamp { align-self: flex-start; }
        .block-container { padding-left: 1rem; padding-right: 1rem; }
    }
</style>
<div class="brandbar">
  <div><p class="brand-kicker">Prompt optimization / Local analysis</p><h1>Context Surgeon</h1><p class="brand-subtitle">Reduce input tokens. Keep the details that matter.</p></div>
  <div class="brand-stamp">Target-aware<br>tiktoken counts</div>
</div>
""", unsafe_allow_html=True)

SAMPLE = {
    "system_prompt": "You are an incident analyst. Use only supplied facts. Preserve exact timestamps, percentages, affected services, and negations. Clearly mark unknown causes. Respond in exactly three bullets.",
    "context": "Checkout errors began at 09:42 UTC. 12% of checkout attempts failed. Login and payments were unaffected. Rollback completed at 10:05 UTC. Rollback completed at 10:05 UTC.",
    "user_query": "Summarize the impact and timeline in three bullets. Do not claim a cause; it has not been confirmed.",
}

def _clear_compression_outputs() -> None:
    st.session_state.result = None
    st.session_state.benchmark_rows = None
    st.session_state.benchmark_aggregate = None


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
    st.session_state.setdefault("benchmark_rows", None)
    st.session_state.setdefault("benchmark_aggregate", None)
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
    st.markdown('<div class="section-kicker">01 / Instructions and evidence</div>', unsafe_allow_html=True)
    st.text_area("System Prompt", key="system_prompt", height=150, placeholder="Instructions, priorities, and constraints", on_change=_clear_compression_outputs)
    st.text_area("Context", key="context", height=260, placeholder="Conversation history, facts, code, or data", on_change=_clear_compression_outputs)
with right:
    st.markdown('<div class="section-kicker">02 / Request and token settings</div>', unsafe_allow_html=True)
    st.text_area("User Query", key="user_query", height=150, placeholder="The current request", on_change=_clear_compression_outputs)
    model = st.selectbox("Target tokenizer model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"], index=0, on_change=_clear_compression_outputs)
    token_budget = st.number_input("Context token budget", min_value=0, max_value=100000, value=512, step=32, on_change=_clear_compression_outputs)
    st.caption("The budget applies to context. Counts use tiktoken and exclude provider chat-template overhead.")
    st.caption("Token reduction, benchmark, and local quality proxy run without API credentials.")

if st.button("Compress", type="primary", width="stretch"):
    try:
        with st.spinner("Compressing inputs and running the 33-case benchmark..."):
            st.session_state.result = compress_inputs(
                st.session_state.system_prompt,
                st.session_state.context,
                st.session_state.user_query,
                token_budget=int(token_budget),
                model=model,
            )
            rows, aggregate = run_benchmarks(int(token_budget), model, TokenizerAdapter(model))
            st.session_state.benchmark_rows = rows
            st.session_state.benchmark_aggregate = aggregate
    except TokenizerUnavailableError as exc:
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
    st.markdown('<div class="section-kicker">03 / Compression report</div>', unsafe_allow_html=True)
    st.subheader("Token reduction")
    metric_cols = st.columns(5)
    metric_cols[0].metric("Original tokens", result.original_tokens)
    metric_cols[1].metric("Compressed tokens", result.compressed_tokens, delta=result.compressed_tokens - result.original_tokens, delta_color="inverse")
    metric_cols[2].metric("Tokens saved", result.tokens_saved)
    metric_cols[3].metric("Reduction", f"{result.reduction_percentage:.1f}%")
    metric_cols[4].metric("Latency", f"{result.compression_latency_ms:.2f} ms")
    st.caption("Savings and reduction are calculated from actual before/after tokenizer counts; no target percentage is applied.")
    if result.tokens_saved < 0:
        st.warning("Compression increased token usage. Review the outputs before using them.")
    for warning in result.warnings:
        st.warning(warning)

    comparison = []
    for label, key in [("System Prompt", "system_prompt"), ("Context", "context"), ("User Query", "user_query")]:
        item = getattr(result, key)
        saved = item.original_tokens - item.compressed_tokens
        reduction = (saved / item.original_tokens * 100) if item.original_tokens else 0.0
        comparison.append({
            "Input": label,
            "Original": item.original_tokens,
            "Compressed": item.compressed_tokens,
            "Saved": saved,
            "Reduction": f"{reduction:.1f}%",
        })
    st.markdown("**Per-input token accounting**")
    st.dataframe(comparison, hide_index=True, width="stretch")
    st.markdown("**Total token usage**")
    st.bar_chart({"Original": [result.original_tokens], "Compressed": [result.compressed_tokens]}, horizontal=True)

    original_col, compressed_col = st.columns(2)
    for container, title, field in [(original_col, "Original inputs", "original"), (compressed_col, "Compressed inputs", "compressed")]:
        with container:
            st.markdown(f"**{title}**")
            for label, key in [("System Prompt", "system_prompt"), ("Context", "context"), ("User Query", "user_query")]:
                st.markdown(f"_{label}_ · {getattr(result, key).original_tokens if field == 'original' else getattr(result, key).compressed_tokens} tokens")
                st.code(getattr(getattr(result, key), field) or "", language="text")

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
    writer.writerow(["input", "original_tokens", "compressed_tokens", "original", "compressed"])
    for label, key in [("System Prompt", "system_prompt"), ("Context", "context"), ("User Query", "user_query")]:
        item = getattr(result, key)
        writer.writerow([label, item.original_tokens, item.compressed_tokens, item.original, item.compressed])
    download_a, download_b = st.columns(2)
    download_a.download_button("Download compressed result (JSON)", result_json, "context-surgeon-result.json", "application/json", width="stretch")
    download_b.download_button("Download comparison (CSV)", csv_stream.getvalue(), "context-surgeon-result.csv", "text/csv", width="stretch")

    st.divider()
    st.markdown('<div class="section-kicker">04 / Keyless quality proxy</div>', unsafe_allow_html=True)
    st.subheader("Quality comparison · input preservation")
    audit_columns = st.columns(3)
    audit_columns[0].metric("Markers retained", f"{preservation_audit['retained_markers']}/{preservation_audit['required_markers']}")
    audit_coverage = preservation_audit["coverage_percentage"]
    audit_columns[1].metric("Marker coverage", f"{audit_coverage:.1f}%" if audit_coverage is not None else "N/A")
    audit_status = "no markers" if preservation_audit["status"] == "not_applicable" else preservation_audit["status"].upper()
    audit_columns[2].metric("Audit status", audit_status)
    if preservation_audit["rows"]:
        st.dataframe([
            {
                "Input": row["input"],
                "Marker type": row["category"],
                "Original": row["required"],
                "Retained": row["retained"],
                "Coverage": f"{row['coverage_percentage']:.1f}%",
                "Missing": ", ".join(row["missing"]) or "None",
            }
            for row in preservation_audit["rows"]
        ], hide_index=True, width="stretch")
    else:
        st.info("No explicit numeric, identifier, constraint, format, table, or recent-user markers were found to audit.")
    st.caption(preservation_audit["scope"])

