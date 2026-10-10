# Context Surgeon

Context Surgeon is a local Python library and Streamlit dashboard for conservative, target-model-aware prompt compression. It aims for 20% reduction in total input tokens when that can be achieved while retaining 100% coverage of the local explicit-marker preservation audit. System prompts and active user queries remain unchanged; context is normalized and ranked in complete units.

## Install and launch

Requires Python 3.10 or later. From the project directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

On systems without the `py` launcher, create the environment with `python -m venv .venv`. The UI opens in the browser. The default tokenizer is `gpt-4o-mini`; select another supported model in the dashboard if needed. No API key is required for compression or benchmarks.

## Python API

```python
from context_surgeon import compress_inputs

result = compress_inputs(
    system_prompt="Follow the task constraints.",
    context="Relevant facts and conversation history.",
    user_query="Answer the user's question.",
    token_budget=512,
)
print(result.to_dict())
```

The app applies a fixed 20% reduction goal to total input tokens when possible; the optional `token_budget` is an additional hard context limit. `CompressionResult` reports whether 20% was reached and the measured explicit-marker coverage. If a candidate would miss any audited marker, the original context is restored even if this means missing the reduction goal. A zero-token original total yields a 0% reduction. Token counts come from the selected model's `tiktoken` encoding; they are not provider-reported usage and exclude chat-template, tool, and request framing overhead.

## Benchmarks

Run the actual deterministic compression pipeline over 33 varied cases, three times per case by default, and export CSV and JSON:

```powershell
python -m context_surgeon.benchmarks --output-dir benchmark_results
```

Use `--repeats 5` to change the run count. Each case reports median compression latency across its repeats; aggregates include mean, median, and p95 across the same 33 cases. The tokenizer is initialized once outside the timed compression calls. In the dashboard, **Compress** and **Run 33-case benchmark** are separate actions; the latter shows progress, summary metrics, all per-case results, and timestamped CSV/JSON downloads. The compression report includes a per-input stacked token chart and visual inline text diffs. Answer-quality scores are explicitly `not run` unless optional evaluation is invoked with a configured provider.

After compression, the dashboard also runs a local preservation audit without a key. The compressor restores original context whenever a candidate fails to retain every explicit number/date, common code identifier, negation/condition term, output-format marker, table row, and latest labeled user turn. This deterministic marker check is not proof of semantic equivalence or generated-answer quality.

## Quality assessment

The dashboard uses a keyless local preservation proxy. It compares explicit markers in original and compressed input and reports coverage/missing values. It does not generate answers, claim semantic equivalence, or require an API key. `python-dotenv` loads an optional `.env` file for integrations using `OPENAI_API_KEY`, `OPENAI_MODEL`, or `OPENAI_BASE_URL`. The optional `context_surgeon.evaluation` module can perform paired model-answer judging for integrations that configure an OpenAI-compatible endpoint; it is not part of the dashboard flow and can incur provider charges.

## Interpreting results and limitations

- Reduction is `(original tokenizer tokens - compressed tokenizer tokens) / original tokenizer tokens * 100`; negative savings mean the compressed form costs more tokens. Empty input totals report 0%.
- Context selection protects chunks containing numeric values, common code-identifier forms, negation/condition and output-format markers, malformed/unbalanced markup, valid JSON, Markdown tables, code blocks, and the two most recent explicitly labeled `User:` or `Human:` messages. Code fences and structured units are atomic, not line-pruned. If a candidate would miss any audited marker, the original context is restored. Other omitted chunks may be required to achieve the token-reduction goal; deterministic BM25-style lexical relevance is not proof that they are irrelevant.
- System prompts and active user queries are immutable. The 20% goal is measured over all three inputs and may not be reachable when context is short or protected. Marker coverage is enforced at 100%; it checks only explicit markers and is not semantic answer-quality comparison.
- The tokenizer experiment reports hypothetical two-character segmentation separately. Such pieces are not compatible with an existing model vocabulary or token IDs and are not used in compression.
- A provider's actual usage can differ from local tokenizer counts. Benchmark latency measures local compression only, not answer generation. The keyless preservation proxy is not a semantic answer-quality score or a guarantee of answer equivalence.
- `tiktoken` must recognize the selected OpenAI model. Unsupported models fail clearly rather than substituting character counts. The library currently focuses on supported OpenAI encodings and deterministic lexical relevance, not semantic embeddings.

## Tests and troubleshooting

```powershell
python -m pytest -q
ruff check .
ruff format --check .
```

GitHub Actions runs lint, formatting, and tests on pushes and pull requests.

If the dashboard reports a missing tokenizer, install dependencies with `python -m pip install -r requirements.txt`. If PowerShell blocks virtual-environment activation, use `.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.venv\Scripts\streamlit.exe run app.py`. An unrecognized model needs a supported `tiktoken` model name. Provider errors affect only optional evaluation; local compression and benchmarks work without credentials.