# Architecture

How llm-bench-lab is put together, and why each decision was made. About 6,500
lines: 5,000 of Python across 18 modules plus a 1,500-line single-page UI, on the
standard library alone.

This document assumes you have read the [README](README.md) and want to know how
it works underneath, or are considering changing something.

## The idea the design serves

A model's speed is not the same thing as a product's behaviour. Quantization,
memory residency, retrieval quality, the harness around the model, the rails
around the harness, and how any of it is evaluated all shape what a person
actually experiences. So this tool measures the whole path rather than one number
at the centre of it, and every layer stays readable enough to defend.

Three constraints follow from that, and they explain most of what is here.

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
  Shared         ┌─────▼┐ ┌▼─────────▼┐ ┌───────▼─────┐ ┌──────────────┐
  services       │config│ │attachments│ │guardrails.py│ │  telemetry   │
                 │  .py │ │+ extract  │ │             │ │ + readiness  │
                 └──────┘ └───────────┘ └─────────────┘ └──────────────┘
                                                    console.py renders
                                                    results + readiness
                                                    for the terminal

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

- **Two first-token clocks, not one.** `ttft_ms` starts on the first token of
  any kind including hidden reasoning (compute latency); `ttfv_ms` starts on the
  first *visible* word (perceived latency). They are identical on non-thinking
  models, verified at 644.8 ms on both clocks. On a reasoning model they diverge
  badly: a measured run started producing tokens at 231 ms but the user saw
  nothing for 2,122 ms, so a single metric understated the felt wait by 9x. A
  model can also spend its whole budget thinking and emit no visible word at
  all, in which case `ttfv_ms` is null and reporting only compute latency would
  be actively misleading.
- Both are wall-clock, not derived from Ollama's internal timings, because these
  are user-experience metrics and should be measured the way a user experiences
  them.
- `gen_tps = eval_count / eval_duration` isolates decode speed from prefill,
  because those are different bottlenecks and a system can be strong at one and
  weak at the other.

`backends.py` does the same job over the OpenAI-compatible
`/v1/chat/completions` SSE protocol, taking token counts from the `usage` chunk.
When a server sends no usage block, it records the stream-chunk count, sets
`approximate_tokens`, and **leaves `gen_tps` unset**: a stream delta is not
necessarily one token, so deriving a rate from delta counts would produce a
number that looks directly comparable to a measured tokens/sec and is not.
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

### `readiness.py`, `fixes.py` and `console.py`

`telemetry.py` answers *what hardware is this*. `readiness.py` answers the
different question *can this machine run a benchmark*, and keeping them apart is
the point: the reason a run cannot start should not be buried among CPU core
counts. It probes the Python version, a reachable engine, installed models, a
GPU backend, writable storage, and on NVIDIA hosts `nvidia-smi` and TensorRT,
grading each `ok` / `warn` / `fail` with the command that fixes it. Both the web
UI (`GET /api/readiness`) and `bench.py doctor` render the same structure, so
the two surfaces cannot disagree about what "ready" means.

`fixes.py` executes the remediation commands the checks suggest, which is the
one place a browser can cause something to run on the machine. Three rules keep
that narrow: the client sends a check *key* and never a command string, so what
runs is always the server's own suggestion for the current state; only
`readiness.RUNNABLE` keys are eligible, which excludes everything wanting sudo,
an admin prompt or a package manager; and the argv is executed directly with no
shell, so metacharacters have nothing to act on. `server.py` adds a same-origin
requirement on that endpoint, since without one any website could POST to
localhost while the user has llmbench open. A useful side effect of deriving the
command from live state: a check that currently passes has no command, so
nothing can be run for it.

`console.py` renders results and readiness for a terminal. It imports its
verdict wording from `report.py` rather than restating it, which is what stops
the CLI and the HTML from drifting apart as the thresholds change. Table columns
are dropped by priority when the terminal is narrow; speed and residency are
never dropped, because those are the two questions a benchmark is run to answer.

### `store.py` and `jobs.py`

`store.py` writes each run as a self-contained folder (`meta.json`,
`result.json`, `attachments/`). Portable by design: copying a folder between
machines merges history, which is the mechanism behind cross-system comparison.

`jobs.py` is a single-worker FIFO queue. The single worker is the point. Two
benchmarks running concurrently would share the GPU and corrupt each other's
timings, so this is a measurement-integrity decision wearing the clothes of a
concurrency detail. Finished jobs are capped at 200 and evicted oldest-first,
since each one retains its full log and result; queued and running jobs are
never evicted, however old, because the UI is still polling them.

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

## Findings that changed the design

Two results that came out of building and running this, kept because they are
the reason for decisions that would otherwise look arbitrary.

### Measuring the client, not the server

Latency is measured client-side with a wall clock, not read from the engine's own
report of itself. That looks like paranoia until it catches something.

An RTX campaign showed 2,071 ms plus or minus 12.9 ms of unattributed latency on
every request, constant across 45 runs regardless of model size or speed.
Ollama's internal timings looked healthy throughout, so throughput was valid
while every latency figure was wrong. The residual was isolated to the Windows
`localhost` connection path: the literal `127.0.0.1` removed it, leaving 21.5 ms.
That is consistent with IPv6 `::1` being tried first and falling back, though no
packet capture was taken, so the mechanism is recorded as unconfirmed rather than
asserted.

Two things follow. The default base URL is `127.0.0.1` rather than `localhost`,
with the reason written where someone might otherwise "tidy" it back
(`config.py`). And a harness that trusted the server's self-report would have
published a campaign whose latency numbers were entirely artifact.

### Injection that arrives inside a document

Injection that arrives inside an indexed document bypasses the input rail
entirely, because the user's question is benign and the payload rides in on a
retrieved chunk. This was measured rather than assumed, and the result was more
interesting than expected.

**Baseline, no rail.** A realistic attack document (a delivery notice with a
buried instruction to run a shell command) was indexed and queried with an
ordinary question. All three models answered the legitimate question correctly
and none obeyed the injection, because the retrieval path exposes no tools and
there was nothing to hijack. But none of them told the user their own document
store contained the instruction either. Silent tolerance is its own weakness,
and the exposure becomes live the moment retrieval feeds a tool-capable context.

**The rail.** `retrieval_rail()` redacts instruction-shaped lines before they
reach the prompt and reports what it found, line by line, so a flagged document
still contributes its legitimate content.

**First attempt failed on false positives.** The naive detector could not
distinguish describing an attack from performing one. A permissions manifest
listing blocked tool names, and a policy line reading "instructions must never
override system policy", were both redacted. Adding negation and
imperative-context handling (`_is_descriptive`) took false positives on the
sample corpus to zero while still catching both real attacks.

**The security notice cost accuracy.** Adding a defensive paragraph to every
grounded prompt regressed two of fourteen eval cases on a 4B model: the extra
instruction competes for attention. Isolated by A/B testing the prompt with the
rail held constant. The fix is to apply the notice *conditionally*, only when
the deterministic rail actually flagged a source, since the payload is already
redacted by then and the paragraph exists to make the model tell the user.
Clean documents pay nothing; accuracy returned to 13/14 with the defense intact.

The general lesson, which is the interesting part: **guardrails are not free.**
They cost latency, tokens, false positives and accuracy, so the useful question
is not "is it safe" but "what did safety cost, and is the trade worth it".

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

## Where a change goes

A map for the most likely reasons to open this code:

| To change | Start in | Watch out for |
|---|---|---|
| How a metric is measured | `ollama.py` (`GenerationResult`) | Every surface reads this dataclass; `backends.py` must map onto it too |
| The methodology (repeats, warm-up, cold start) | `runner.py` | The web UI and CLI share it, so a change lands on both |
| Adding an inference engine | `backends.py` | Satisfy the client contract and nothing downstream needs touching |
| A verdict's wording or threshold | `report.py` | `console.py` imports these, so the terminal follows automatically |
| Retrieval quality | `rag.py` (chunk size, `k`, the grounded prompt) | Chunking is where the known CSV failure lives |
| What the agent may do | `guardrails.py` | Rails are deterministic on purpose; do not move a decision into the prompt |
| An environment check | `readiness.py` | Both the web UI and `bench.py doctor` render whatever you add |
| Making a fix runnable | `readiness.RUNNABLE` | Only add commands that need no elevated rights; the rest stay copy-only |
| The UI | `web/index.html` | One file, no build step; keep it that way |

## What is deliberately not here

Absences worth naming, so they read as decisions rather than oversights:

- **No unit test suite.** Verification has been end-to-end and manual: run the
  campaign, check the numbers against the frozen evidence. For a measurement tool
  this catches the errors that matter more reliably than mocked units would, but
  it is a genuine gap and the honest reason a refactor here is riskier than it
  looks.
- **No semantic citation checking.** `evals.py` verifies that a cited source
  exists, not that it supports the claim. The next step is a human rubric or an
  LLM judge as a secondary signal, with the deterministic checks staying the hard
  pass/fail gate.
- **No database.** Runs are folders of JSON. That is what makes a run portable
  between machines by copying it, which is the mechanism the whole cross-system
  comparison rests on.
- **No async.** A single-worker queue and a threaded stdlib server are sufficient
  when the workload is deliberately serialised, and the concurrency that matters
  here is one background job plus progress polling.
