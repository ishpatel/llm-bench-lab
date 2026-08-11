# Architecture

How llm-bench-lab is put together, and why each decision was made. Roughly
5,300 lines across 17 files, standard library only.

## The thesis

**The model is one component of an AI product, not the product.** Model quality,
quantization, memory residency, retrieval quality, harness design, guardrails
and evaluation all shape what the user actually experiences. So this tool
measures the whole system, and every layer stays transparent enough to explain.

Three constraints fall out of that thesis, and they explain most of the design.

### 1. Zero third-party dependencies

Standard library only. This is not minimalism for its own sake: it means the
*identical* code runs on Apple Silicon and on an NVIDIA machine with
`git clone && python bench.py serve`, no pip, no version drift, no
"works on my machine". For a cross-system comparison, that is what makes the
comparison valid rather than a comparison of two software environments.

The cost is real and deliberate: a hand-written PDF parser, a hand-built HTTP
server, inline SVG charts, and PNG cropping in pure Python, instead of pypdf,
Flask, matplotlib and Pillow.

### 2. Honest failure over quiet success

Every measurement records how it was obtained. Every extraction reports its
method and any quality warnings. Hardware detection reports what is actually
true rather than what would be convenient.

This is the trait that keeps paying off. The built-in PDF parser flagged
"no extractable text" instead of returning an empty string, which is how a real
bug got caught: ReportLab writes text through `/Filter [/ASCII85Decode
/FlateDecode]`, a filter *chain*, and the decoder only tried raw Flate. A silent
failure would have quietly corrupted three eval cases.

### 3. Explain every layer

No framework hiding the interesting parts. Chunking, cosine similarity, the
grounded prompt, the guardrail decisions: all visible in readable code, so every
choice can be defended rather than attributed to a library.

## Layer map

```
                        ┌───────────────────────────────────┐
  Surfaces              │  web/index.html    ·    bench.py   │
                        └────────────────┬──────────────────┘
                                         │
  Web layer             ┌────────────────▼──────────────────┐
                        │ server.py   (stdlib HTTP + JSON API)│  composition root
                        │ jobs.py (1-worker queue)   store.py │
                        └──┬─────────┬─────────┬─────────┬───┘
                           │         │         │         │
  Capabilities      ┌──────▼───┐ ┌───▼───┐ ┌───▼────┐ ┌──▼──────┐
                    │ runner.py│ │ rag.py│ │agent.py│ │ evals.py│
                    │methodology│ │       │ │        │ │         │
                    │+ matrix   │ │       │ │        │ │         │
                    └──┬───┬────┘ └──┬────┘ └───┬────┘ └─────────┘
                       │   │         │          │
  Shared         ┌─────▼┐ ┌▼─────────▼┐ ┌───────▼─────┐ ┌──────────┐
  services       │config│ │attachments│ │guardrails.py│ │telemetry │
                 │  .py │ │+ extract  │ │             │ │   .py    │
                 └──────┘ └───────────┘ └─────────────┘ └──────────┘

  Engine              ┌──────────────────────────────────┐
  contract            │ ollama.py     |    backends.py    │
                      │  GenerationResult is the interface │
                      └────────┬──────────────┬───────────┘
                               │              │
                     ┌─────────▼────┐  ┌──────▼────────────────┐
                     │ Ollama /api  │  │ OpenAI-compatible /v1  │
                     │ (llama.cpp)  │  │ (TRT-LLM / NIM / vLLM) │
                     └──────────────┘  └───────────────────────┘
```

`server.py` is the composition root: the one module that imports nearly
everything and wires it together. Everything below is a leaf or near-leaf.
`extract.py`, `telemetry.py` and `guardrails.py` import no internal modules at
all. That shallow, acyclic graph is why each capability can be exercised
standalone.

## The two contracts

Everything composes because of two small agreements.

### Contract 1: `GenerationResult` (the engine interface)

One dataclass in `ollama.py` carries every per-generation metric: `ttft_ms`,
`gen_tps`, `prompt_tps`, `load_ms`, token counts, `response_text`,
`thinking_chars`.

A "client" is anything exposing `generate() -> GenerationResult` plus
`list_models()`, `is_up()` and `unload()`. Two classes satisfy it,
`OllamaClient` and `OpenAICompatClient`, and nothing downstream knows which one
it is talking to.

That is why adding the TensorRT-LLM engine cost about 150 lines and touched no
metric code. `backends.py` speaks a different wire protocol and maps the result
onto the same object.

### Contract 2: the config dict (the run description)

`runner.Runner` consumes a plain dict: models, prompts, options, repeats,
warmup, an optional `backend`. The CLI *loads* that dict from JSON; the web
server *builds* it in memory from a form POST. Same runner, same methodology,
two front doors, no possibility of drift between them.

## Layer by layer

### Engine clients: `ollama.py`, `backends.py`

`ollama.py` streams `/api/generate` as newline-delimited JSON. Decisions worth
knowing:

- **TTFT is wall-clock time to the first streamed token**, not derived from
  Ollama's internal timings. TTFT is a user-experience metric, so it is measured
  the way the user experiences it.
- It watches both the `response` and `thinking` fields for that first token.
  Reasoning models stream hidden thinking separately; without this, TTFT is null
  on every thinking model.
- `gen_tps = eval_count / eval_duration` isolates decode speed from prefill,
  because those are different bottlenecks and a system can be strong at one and
  weak at the other.

`backends.py` does the same job over the OpenAI-compatible
`/v1/chat/completions` SSE protocol, taking token counts from the `usage` chunk.
It exists because Ollama has no TensorRT mode, so the only honest way to
benchmark TensorRT-LLM is to talk to a real TensorRT-LLM server. NIM and vLLM
work for free as a result.

### `telemetry.py`

Platform detection plus a background `nvidia-smi` sampler polling every 250 ms
during a generation, capturing peak and average utilization, VRAM, power and
temperature. Cross-platform and best-effort: on Apple Silicon it is a no-op,
because there is no per-process VRAM counter without elevated `powermetrics`,
and inventing one would be dishonest. `describe_system_deep()` produces the
CPU / GPU / NPU / memory / environment breakdown, cached because
`system_profiler` is slow. `tensorrt_status()` reports plain facts: NVIDIA GPU
present, python `tensorrt` importable, endpoint reachable.

### `extract.py` and `attachments.py`

`extract.py` is a tiered document-to-text layer. Where a high-quality system
converter exists it is used (`pdftotext`, macOS `textutil`); otherwise a
stdlib-only fallback keeps the harness working anywhere. OOXML and ODT formats
are zipped XML and handled with `zipfile`. The PDF path is a hand-written
parser: decode the content streams, scan for text-showing operators.

`_pdf_decode_stream` walks the plausible filter chains (Flate, ASCII85+Flate,
ASCII85, ASCIIHex+Flate) rather than parsing each stream's filter dictionary,
which is both simpler and more tolerant of real-world writers.

`attachments.py` turns files into an injected context block, measuring the token
cost, and images into base64 for vision models.

### `config.py` and `runner.py`

The measurement engine, and where the defensible methodology lives:

- **Warm-up runs, discarded.** Separates model-load and cache effects from
  steady-state inference.
- **N measured repeats, median reported with min-max spread.** Median for
  outlier resistance on a small sample; the spread so a reader can see when a
  number should not be trusted.
- **Cold start measured separately** by evicting the model (`keep_alive: 0`) and
  timing the reload. The one-time load cost is a different user-experience event
  from warm inference, and averaging them together misrepresents both.
- **Generation options held constant** across a comparison so only the intended
  variable moves.

The runner expands a matrix of models x prompts x context lengths. It is
engine-aware: with an external backend it skips cold start (the engine manages
its own loading), labels residency by engine, and treats model listing as
advisory rather than gating.

### `report.py`

Self-contained HTML with hand-built inline SVG bar charts. No chart library and
no external requests, so a report opens correctly offline years later. Every
number carries a plain-English verdict and the page includes a "how to read
these numbers" glossary, because a report a non-expert cannot read is a report
that does not travel.

### `store.py` and `jobs.py`

`store.py` writes each run as a self-contained folder (`meta.json`,
`result.json`, `attachments/`). Portable by design: copying a folder between
machines merges history, which is the mechanism behind cross-system comparison.

`jobs.py` is a single-worker FIFO queue. The single worker is the point. Two
benchmarks running concurrently would share the GPU and corrupt each other's
timings, so this is a measurement-integrity decision wearing the clothes of a
concurrency detail.

### `server.py` and `web/index.html`

A stdlib `ThreadingHTTPServer` exposing a small JSON API and serving one HTML
file. Threaded so that progress polling is served while the worker benchmarks.
The UI is a single dependency-free vanilla-JS page: seven tabs, no framework, no
build step. It stays thin on purpose, building config dicts and rendering JSON
while all real work happens in the same backend the CLI uses.

### `rag.py`

The full retrieval pipeline with every layer exposed: extract, chunk (900
characters with 150 overlap), embed through Ollama, retrieve by cosine
similarity in pure Python, build a grounded prompt with numbered sources, and
answer with `[n]` citations or an honest abstention.

Embeddings are stored unit-normalized at build time, so a query is a dot product
with no per-comparison square root, with a cosine fallback for older indexes.

### `agent.py` and `guardrails.py`

The principle made literal: **the model proposes, deterministic code
authorizes.** `agent.py` runs a model -> tool -> observation loop over native
tool calling. `guardrails.py` is five plain-Python rails: input
(prompt-injection and disallowed intent), tool allowlist (least privilege),
parameter (type and bounds), approval (human-in-the-loop for consequential
actions), and output (secret leakage).

The model may request `set_power_limit(150)`. The parameter rail rejects it
because 150 falls outside 80-115, and no amount of model confidence changes
that. Guardrails are deterministic precisely because the thing they guard is
probabilistic.

### `evals.py`

Two suites: RAG groundedness and abstention, and adversarial guardrail
enforcement scored on outcome. Scoring is deterministic (keyword matching,
abstention phrasing, citation validation) so results are reproducible, with no
LLM-judge nondeterminism in the pass/fail path.

`is_grounded` validates that every `[n]` refers to a passage that was actually
retrieved. A `[7]` when four sources were supplied is a fabricated citation, not
grounding, and the raw-model path passes zero sources so it can never score as
grounded.

**Stated limitation:** these checks verify that a cited passage *exists*, not
that it *supports* the claim. Semantic evaluation, whether a human rubric or an
LLM judge as a secondary signal, is the documented next step, with deterministic
checks remaining the hard pass/fail gate.

## Request lifecycle

A benchmark from click to saved report:

```
Browser: New Run, "Run benchmark"
   │  POST /api/runs {model, prompt, options, attachments, [engine]}
   ▼
server.py   builds a config dict, saves a "running" placeholder,
            submits to the queue, returns {id} immediately
   │
   ▼   (background worker, one job at a time)
jobs.py ──► runner.Runner(config).run()
   │          · attachments.prepare(): inject docs, encode images
   │          · per cell: unload → cold start → warm-ups → N timed runs
   │          · each timed run wrapped by the telemetry sampler
   │          · client is OllamaClient or OpenAICompatClient (same contract)
   │          · aggregate into median + spread
   ▼
store.py    writes runs/<id>/{meta.json, result.json, attachments/}
   │
   ▼   meanwhile the browser polls GET /api/jobs/<id> for the live log
   │   on completion it renders metric tiles with verdicts and the answer
   ▼
report.py   on demand: GET /api/runs/<id>/report, self-contained HTML
```

What does not change when the engine changes is the point: swapping Ollama for
TensorRT-LLM alters only the client object. Everything from the sampler upward
is identical.

## Cross-cutting decisions

| Decision | Reason |
|---|---|
| Zero dependencies | Identical code on both machines, which is what makes a cross-system comparison valid |
| Duck-typed engine contract | New engines are cheap and never touch metric code |
| Config dict as run description | Web and CLI cannot drift apart |
| Single-worker queue | Concurrent runs would corrupt each other's timings |
| Honest failure everywhere | Surfaces real bugs instead of hiding them |
| Deterministic guardrails and evals | Reproducibility, and deterministic software should control a probabilistic model |
| Plain English on every surface | The work has to travel to non-experts to be useful |

## Mapping to the local AI stack

The project deliberately walks the entire stack rather than benchmarking a model
in isolation:

```
Hardware  →  driver/runtime  →  engine  →  model  →  serving  →  RAG
telemetry     Metal / CUDA      ollama.py  Qwen etc.  ollama      rag.py
                                backends.py           server

   →  agent + tools  →  guardrails  →  application  →  evaluation
        agent.py         guardrails.py    web UI         evals.py
```
