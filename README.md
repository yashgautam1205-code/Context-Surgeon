# Context Surgeon

Context Surgeon is a local Python library and Streamlit dashboard for conservative, target-model-aware compression of a system prompt, context, and user query. Its default behavior removes exact duplicate instructions/passages, trims a small set of unambiguous query prefixes, and ranks complete context chunks against the query. It preserves the source order of selected chunks and warns when lower-ranked material is omitted.

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

`token_budget` applies only to context. The result contains original and compressed text/counts for each input, totals, signed savings, percentage reduction, latency, method, tokenizer model, and warnings. A zero-token original total yields a 0% reduction. Token counts come from the selected model's `tiktoken` encoding; they are not provider-reported usage and exclude chat-template, tool, and request framing overhead.

## Benchmarks

Run the actual deterministic compression pipeline over 33 varied cases, three times per case by default, and export CSV and JSON:

```powershell
python -m context_surgeon.benchmarks --output-dir benchmark_results
```

Use `--repeats 5` to change the run count. Each case reports median compression latency across its repeats; aggregates include mean, median, and p95 across the same 33 cases. The tokenizer is initialized once outside the timed compression calls. The dashboard has a matching benchmark action and downloadable exports. Answer-quality scores are explicitly `not run` unless optional evaluation is invoked with a configured provider.

After compression, the dashboard also runs a local preservation audit without a key. It checks explicit numbers/dates, common code identifiers, negation/condition terms, output-format markers, table rows, and the latest two labeled user turns. Missing markers are listed and coverage is reported. This is a deterministic input-retention check, not proof of semantic equivalence or generated-answer quality.

## Quality assessment

The dashboard uses a keyless local preservation proxy. It compares explicit markers in original and compressed input and reports coverage/missing values. It does not generate answers, claim semantic equivalence, or require an API key. The optional `context_surgeon.evaluation` module can perform paired model-answer judging for integrations that configure an OpenAI-compatible endpoint, but that is not part of the dashboard flow and can incur provider charges.

## Interpreting results and limitations

- Reduction is `(original tokenizer tokens - compressed tokenizer tokens) / original tokenizer tokens * 100`; negative savings mean the compressed form costs more tokens. Empty input totals report 0%.
- Context selection protects chunks containing numeric values, common code-identifier forms, negation/condition markers, and the two most recent explicitly labeled `User:` or `Human:` messages. If those chunks exceed budget, the full deduplicated context is retained with a warning, so output may exceed budget. Structured, code-heavy, or numerical context is also retained rather than sliced. Other omitted chunks are reported; lexical relevance is not proof that they are irrelevant.
- System prompt editing is limited to duplicate lines and redundant blank lines. Query rewriting removes only a small set of leading politeness phrases, and falls back unless tokenizer count decreases and protected terms are unchanged.
- The tokenizer experiment reports hypothetical two-character segmentation separately. Such pieces are not compatible with an existing model vocabulary or token IDs and are not used in compression.
- A provider's actual usage can differ from local tokenizer counts. Benchmark latency measures local compression only, not answer generation. The keyless preservation proxy is not a semantic answer-quality score or a guarantee of answer equivalence.
- `tiktoken` must recognize the selected OpenAI model. Unsupported models fail clearly rather than substituting character counts. The library currently focuses on supported OpenAI encodings and deterministic lexical relevance, not semantic embeddings.

## Tests and troubleshooting

```powershell
python -m pytest -q
```

If the dashboard reports a missing tokenizer, install dependencies with `python -m pip install -r requirements.txt`. If PowerShell blocks virtual-environment activation, use `.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.venv\Scripts\streamlit.exe run app.py`. An unrecognized model needs a supported `tiktoken` model name. Provider errors affect only optional evaluation; local compression and benchmarks work without credentials.