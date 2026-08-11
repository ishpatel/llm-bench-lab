# llm-bench-lab

A zero-dependency, cross-platform harness for benchmarking local LLM inference
on **NVIDIA RTX** and **Apple Silicon** systems, built around Ollama. It
measures the metrics that actually shape the user experience: time-to-first-token,
generation throughput, prefill throughput, cold-start load time, and memory
residency, under a defensible methodology, and emits a self-contained HTML
report you can hand to anyone.

Nothing but the Python 3.9+ standard library. Same code runs on macOS (Metal)
and Windows/Linux (CUDA); on NVIDIA it additionally samples `nvidia-smi` for
GPU utilization, VRAM, power and temperature.

## What it looks like

The whole thing runs locally in your browser. Every metric is written in plain
English so anyone can read it.

**A benchmark run, scored.** Speed metrics each carry a plain-English verdict,
alongside the model's answer and your own 1&ndash;5 quality rating.

![Run detail with metric verdicts and a four-dimension rating](docs/screenshots/01-run-detail.png)

**Your hardware, explained.** A breakdown of CPU, GPU, NPU, memory and the
software environment, with a note on what each part actually does.

![System hardware panel](docs/screenshots/02-system.png)

**Ask your documents (RAG).** Grounded answers with numbered citations and
per-stage timing (embedding, retrieval, generation).

![Copilot RAG answer with citations](docs/screenshots/03-copilot.png)

**Tools, with guardrails.** The agent proposes an action; deterministic rails
decide whether it runs. Here an approval rail holds a power-limit change.

![Agent guardrail trace](docs/screenshots/04-agent.png)

**Automated trust tests.** Groundedness and safety scored reproducibly: every
adversarial request was prevented.

![Guardrail evaluation scorecard](docs/screenshots/06-evals.png)

**Benchmark history.** Every run is saved, searchable, sortable, rateable, and
selectable for side-by-side comparison.

![Benchmark history with quality scores](docs/screenshots/07-runs.png)

**Shareable HTML report.** Charts and a detail table, each number annotated.

![Generated HTML report](docs/screenshots/05-report.png)

## Why this design

For the full picture (layer map, the two contracts that make it compose, request
lifecycle, and the reasoning behind each module) see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

An LLM's tokens/sec is only one number. This harness treats local inference as
a *system-level experience* question: memory capacity and compute throughput
are independent constraints, quantization trades quality for footprint and
speed, and time-to-first-token vs steady-state decode are different metrics a
system can be independently good or bad at. The report is structured to expose
exactly those distinctions.

## Requirements

- Python 3.9+ (standard library only)
- [Ollama](https://ollama.com) running locally (`ollama serve`)
- Models pulled ahead of time (see each config's `models` list)
- NVIDIA only (optional): `nvidia-smi` on PATH for live GPU telemetry

## Interactive web app (recommended)

```bash
python bench.py serve            # then open http://127.0.0.1:8765
```

A local, zero-dependency web UI for ad-hoc benchmarking:

- **Pick a model** from the ones installed in Ollama (capability-aware: vision /
  thinking / embedding models are labeled, and every field explains itself in
  plain English).
- **Type any prompt** and **attach multimodal reference material**: drag in PDFs,
  Word/PowerPoint/Excel docs, text/code, or images (images need a vision model).
  Documents are auto-extracted to text and injected as context.
- **Submit** → the run is queued (one at a time, so GPU contention never pollutes
  timings) and you watch **live progress**; when it finishes you see the metrics,
  each with a **plain-English verdict** ("feels instant", "spilled to CPU,
  this is the VRAM wall"), plus the model's actual output.
- **Rate every answer** on four dimensions (correct / useful / relevant /
  usable); scores appear in the history so you can find the model that's both
  fast *and* good.
- Every run is **saved** with its own self-contained HTML report.
- A **Runs** master view lets you browse, search, filter by model, sort (incl.
  by score), and read a metric legend written for non-experts.
- Select runs and **Compare** them side-by-side, with the best value per metric
  highlighted.
- A **System details** panel breaks down the local hardware: CPU (core types),
  GPU (cores, compute API), NPU, memory, and the software environment.

An **Advanced** panel exposes the methodology knobs (measured repeats, warm-up,
cold-start, temperature, max tokens, context length). Defaults keep the rigorous
warm-up + 3-repeat + median measurement.

Runs are stored under `runs/<id>/` (meta.json, result.json, attachments/) and are
portable: copy a run folder between machines to merge history into one UI.

## CLI (batch / scripted use)

```bash
# 1. See what the harness detects on this machine
python bench.py info

# 2. Run a benchmark config (writes results/<name>_<system>_<stamp>.json)
python bench.py run configs/quant-sweep.json --label "M3 Max (48GB)"

# 3. Build an HTML report from one or more results files
python bench.py report results/quant-sweep_*.json --out results/report.html
```

## Commands

| Command | What it does |
|---|---|
| `serve` | Launch the interactive web UI (`--host`, `--port`) |
| `info` | Print detected system + Ollama status and models |
| `run CONFIG` | Execute a benchmark config, write a results JSON |
| `report RESULTS...` | Build an HTML report; pass **multiple** files to compare systems |
| `export RUN_ID` | Export a web-UI run to a portable `.llmbench.json` bundle |
| `import BUNDLE` | Import a run bundle from another machine |
| `eval SUITE` | Run an evaluation suite (`rag`, `guardrails`, or a JSON path) |

Useful flags: `run --label "<name>"` (system label shown in the report),
`run --runs N` (override measured repeats), `--base-url` (remote Ollama).

## OpenAI-compatible inference backends (TensorRT-LLM / NIM / vLLM)

By default llmbench benchmarks through Ollama, whose engine is llama.cpp. It can
also route runs through any server that speaks the **OpenAI-compatible API**, so
you can put two engines head-to-head on the same GPU, model and prompt. The
`backends.py` client is a generic OpenAI-compatible client, not tied to one
vendor; **NVIDIA TensorRT-LLM** (via `trtllm-serve` or an NVIDIA NIM container)
is the headline use, and vLLM or even Ollama's own `/v1` work identically.

Ollama has no TensorRT mode, so this is not a flag inside Ollama. TensorRT-LLM
ships its own server that exposes the compatible API, and llmbench connects to
it as an alternate engine.

1. On the RTX machine, serve a model (TensorRT-LLM is Linux-first; use WSL2 on
   Windows):
   ```bash
   pip install tensorrt-llm
   trtllm-serve <hf-model>       # OpenAI-compatible API on port 8000
   ```
   An NVIDIA NIM container or a vLLM server works the same way.
2. In **New Run**, tick **OpenAI-compatible backend**, enter the endpoint URL,
   and press **Check connection**. The model list switches to what the endpoint
   serves, and you name the engine (e.g. "TensorRT-LLM").
3. Run the benchmark. The run is labeled with that engine in its detail view and
   report, so Ollama and TensorRT results never get mixed up.

Notes:
- Speed metrics (TTFT, generation tok/s, token counts) are measured the same way
  as Ollama runs. Cold start is skipped because the external engine manages its
  own model loading, and `ollama ps` residency does not apply; on NVIDIA machines
  the `nvidia-smi` sampler still reports GPU utilization and VRAM.
- The endpoint can be remote: a Mac can drive benchmarks against an RTX machine's
  server over the network.
- `GET /api/backend/status?url=...` reports the honest facts: NVIDIA GPU present,
  python `tensorrt` importable, endpoint reachable and its model list.
- CLI configs use the same engine with
  `"backend": {"type": "openai", "base_url": "http://host:8000", "label": "TensorRT-LLM"}`.
- Model formats differ across engines. TensorRT-LLM builds engines from
  HuggingFace checkpoints, not the GGUF files Ollama uses, so an engine
  head-to-head compares the same architecture and parameter count, **not** an
  identical quantization. State that in any writeup.

### TensorRT-LLM vs TensorRT for RTX (they are different products)

Worth keeping straight, especially in an NVIDIA conversation:

- **TensorRT-LLM** is the LLM-specialized serving/inference stack. It is
  Linux-focused and exposes `trtllm-serve`, OpenAI-compatible serving, and
  KV-cache management. This is what the backend above connects to.
- **TensorRT for RTX** is NVIDIA's lightweight **client** inference runtime for
  RTX PCs (Windows x64, Blackwell included). It is meant to be embedded in an
  application through ONNX Runtime / Windows ML, using portable ahead-of-time
  engines with device-specific JIT optimization on the user's machine. It does
  not ship an LLM server, so it is a separate client-runtime experiment rather
  than a drop-in backend here.

## Cross-system comparison (RTX vs Apple Silicon)

The goal: run the **same model + prompt** on both machines and see where the
8 GB RTX laptop hits the **VRAM wall** while the M3 Max's unified memory does
not. There are two ways to do it.

### Via the web app (recommended)

1. **Run** the model+prompt in the web UI on each machine; the run is
   auto-labeled by the detected GPU (M3 Max vs RTX).
2. On the RTX laptop, **Export** the run (run detail → `Export ⤓`), which
   downloads a portable `<id>.llmbench.json` bundle.
3. Copy the bundle to the Mac and **Import** it (Runs tab → `Import run`).
4. Open the **Cross-system** tab. Any model present on 2+ systems is shown with
   both systems side-by-side; the **VRAM wall is flagged automatically** when
   residency drops off `100% GPU`, with the throughput hit computed for you.
   `Open cross-system report ↗` renders the merged grouped-by-system charts.

Bundles are also scriptable: `python bench.py export <run_id>` /
`python bench.py import <bundle.json>`.

### Via the CLI

Run the **same** config on each machine, then merge the results files into one
report. `configs/mac.json` is the shared cross-system matrix.

```bash
# On the M3 Max:
python bench.py run configs/mac.json --label "M3 Max (48GB)"

# On the RTX 5070 laptop (Windows/Linux, same repo):
python bench.py run configs/mac.json --label "RTX 5070 (8GB)"

# Copy both results JSONs to one machine, then:
python bench.py report results/cross-system_*.json --out results/cross.html
```

Either way, on the 8 GB RTX laptop `gemma3:12b` (~8.1 GB) and `qwen3:4b-fp16`
(~8.1 GB) are where you expect the wall: residency flips from `100% GPU` to a
CPU/GPU split and generation throughput collapses, while both fit comfortably in
the M3 Max's unified memory. That contrast is the centerpiece of the writeup.

## Reference files, documents & images (task-based benchmarking)

To benchmark a model on a *real task* with supporting material, attach `files`
and/or `images` to a prompt. The harness injects text documents into the prompt
and sends images to the model's `images` field, then measures the result and
captures the model's actual output in the report.

```jsonc
"prompts": [
  {
    "key": "spec_qa",
    "text": "Using only the reference material, how much VRAM does this GPU have?",
    "files": ["docs/rtx-spec.md"],      // text docs injected as context
    "images": ["docs/screenshot.png"]    // needs a vision model
  }
]
```

Any injected document inflates `prompt_tokens`, so the prefill metrics and TTFT
**measure the cost of context** directly; it's the on-ramp to RAG. Relative
paths resolve against the config dir, then the prompts dir, then cwd. Cap
injected size with `"max_chars_per_file"` in the config.

**Supported document types** (extracted automatically to text, see
`llmbench/extract.py`):

| Type | How it's extracted | Notes |
|---|---|---|
| `.txt .md .csv .json .py` … | read directly | plain-text family |
| `.pdf` | `pdftotext` if installed, else **built-in stdlib parser** | built-in handles text-based PDFs (FlateDecode + standard encodings); scanned/CID-font PDFs are flagged, so install poppler `pdftotext` or OCR them |
| `.docx .pptx .xlsx .odt` | stdlib `zipfile` + XML | cross-platform, no tools needed |
| `.rtf .doc` | macOS `textutil` if present, else fallback | legacy `.doc` needs `textutil` or conversion |
| `.html .htm` | tag stripping | scripts/styles removed |

Each file's **extraction method and any quality warnings** are shown in the
report next to the attachment, so low-confidence extractions are never silent.

**Images** are base64-encoded and require a **vision model**
(`ollama pull llama3.2-vision` / `qwen2.5vl` / `llava`); text-only models ignore
them. Images ≤ ~1.5 MB are embedded as thumbnails in the report.

The report gains a **"Task outputs & attachments"** section: the task, the files/
images used, measured token counts, and a collapsible view of each model's actual
answer, so you can compare quality *and* speed together. See
`configs/task-with-doc.json` for a grounded-vs-ungrounded demo (the model
hallucinates VRAM without the doc and answers correctly with it).

```bash
# Markdown reference doc (grounded vs ungrounded):
python bench.py run configs/task-with-doc.json --label "M3 Max (48GB)"
# PDF reference doc (exercises the built-in extractor):
python bench.py run configs/task-with-pdf.json --label "M3 Max (48GB)"
python bench.py report results/task-with-*_*.json --out results/task.html
```

## Copilot: local RAG over your documents

The **Copilot** tab is a fully local Retrieval-Augmented Generation pipeline.
Every layer is explicit (no "Enable RAG" black box):

```
documents ─(extract.py)→ text ─(chunk)→ chunks ─(Ollama embed)→ vectors ─→ local JSON index
   question ─→ vector ─(cosine similarity)→ top-k chunks ─→ grounded prompt ─→ LLM ─→ answer + [n] citations
```

- **Build a knowledge base:** drop in documents (PDF/docx/txt/md/…, the same
  extractor as attachments), pick an **embedding model** (`ollama pull
  embeddinggemma`), and it chunks + embeds them into a local index under `kb/`.
- **Ask:** the question is embedded, the most similar chunks are retrieved by
  **pure-Python cosine similarity** (no NumPy), a grounded prompt is built, and
  the answer comes back with **inline `[n]` citations** and the **retrieved
  source chunks** shown beneath it.
- **It's still a benchmark:** every answer reports per-stage timing: embedding,
  retrieval, TTFT, generation tok/s, and token counts.
- **Grounded, not guessing:** when the answer isn't in the sources the model is
  instructed to say so rather than hallucinate (verifiable: ask something the
  documents don't cover).

Scriptable equivalents exist in `llmbench/rag.py` (`build_kb`, `retrieve`,
`build_grounded_prompt`). Knowledge bases live in `kb/<name>/` and are gitignored.

## Agent harness + guardrails (Phases 8–9)

The **Agent** tab is a small agent with two tools and deterministic guardrails
around every action. Core principle: **the model proposes, deterministic code
authorizes**: the LLM may *request* an action, but plain Python decides whether
it runs (never the model).

- Tools: `get_gpu_status` (read-only, runs freely) and `set_power_limit`
  (bounded 80–115 W, schema-validated, **requires human approval**, and even
  then simulated, so no hardware is touched).
- Rails (`llmbench/guardrails.py`), following NeMo Guardrails' categories:
  **input** (prompt-injection / disallowed-intent detection, before the model),
  **tool allowlist** (unregistered tools refused, i.e. least privilege),
  **parameter** (type + bounds validation), **approval** (consequential actions
  gated), and **output** (secret-leak check).
- The tab shows a live **guardrail & tool trace**: which rail fired, what was
  executed, what was blocked.

## Evaluation harness (Phases 7 & 10)

The **Evals** tab turns behaviour into measured evidence: validation applied to
AI. Two suites (`evals/*.json`), runnable in the UI or via
`python bench.py eval rag|guardrails`:

- **RAG suite (Phase 7):** answerable / ambiguous / impossible question buckets,
  scored **with retrieval vs the raw model** on correct answers, groundedness
  (citations), and, critically, **correct abstention** when the corpus lacks
  the answer. A representative result on the sample corpus: RAG **12/12** with
  **0 hallucinations**, raw model **0/12** with **6 hallucinations**, because
  the specs are private, the model *can't* answer them without retrieval.
- **Guardrail suite (Phase 10):** adversarial inputs (injection, forbidden
  actions, out-of-range tool args, approval-gated actions), scored on **outcome**
  (defense in depth: the unsafe action must not happen, whichever rail stops
  it). Sample result: **7/7** prevented.

Scoring is deterministic (keyword + abstention detection), so results are
reproducible rather than dependent on an LLM judge.

## Worked example: the Local AI Life Lab corpus

`corpus/local_ai_life_lab/` is a synthetic personal-data corpus (28 files:
warranties, receipts, transactions, an itinerary, a resume, pantry inventory,
PC telemetry, an application policy, and a deliberately malicious document).
No real personal data. `evals/life_lab_tests.json` maps its ground truth onto
14 scored cases across three buckets.

```bash
python bench.py eval life_lab
```

Result on an M3 Max with `qwen3:4b-q4_K_M` and `embeddinggemma`:

| | Passed | Hallucinations | Grounded | Correct abstentions |
|---|---|---|---|---|
| **With retrieval** | **13/14** | **0** | 11 | 2/2 |
| Raw model | 5/14 | 4 | 0 | 0/2 |

What the cases actually exercise: reasoning across two documents (warranty
length in a PDF, purchase date in a receipt); resolving a conflict where an
itinerary says 3:00 PM and a later email updates check-in to 4:00 PM;
detecting a duplicate subscription charge; and abstaining on facts the corpus
does not contain, including a passport number where the corpus *does* hold a
passport expiration date as bait.

### Findings worth reporting honestly

Running this surfaced more than a score:

- **The one remaining failure is a retrieval limit, not an arithmetic one.**
  On the "total July subscription spend" case the model sums only 5 of 7
  qualifying rows ($105.96 vs $136.94) because at `k=5` with 900-character
  chunks the CSV splits and not every row reaches the context. RAG quality is
  bounded by retrieval, and tabular data is where fixed-size chunking hurts.
- **The corpus ground truth had an off-by-one-cent error** ($136.93 stated,
  $136.94 actual). Found only by running the eval; corrected in the suite with
  the errata recorded inline.
- **A keyword scorer can pass a lucky guess.** On the hotel case the raw model
  answers "typically ranges from 2:00 PM to 4:00 PM" and matches the expected
  "4:00" without knowing anything. The RAG/raw *delta*, hallucination count and
  grounded count are sharper signals than pass rate alone.
- **Abstention detection has to know the policy's own wording.** The model
  echoed the corpus policy phrase "does not establish the answer", which the
  detector did not recognise, scoring a correct abstention as a failure.

## Methodology (what makes the numbers defensible)

- **Warm-up runs** (`warmup`) are executed and discarded before measurement.
- Each cell runs **N measured repeats** (`runs`); the **median** is reported
  with the **min–max spread** shown in the report.
- **Cold-start** load time is measured separately by evicting the model
  (`keep_alive: 0`) and timing the next call, kept out of the warm TTFT.
- **Generation options are held constant** across systems (`temperature`,
  `num_predict`, etc.) so only the intended variable changes per comparison.
- **TTFT** is true wall-clock time to the first streamed token (thinking or
  visible output), not an Ollama-reported internal.

> Apple Silicon uses unified memory, so its residency/memory figures are not
> directly comparable to a discrete-VRAM NVIDIA GPU. Read cross-system results
> as a **system-level experience comparison**, not a raw GPU benchmark.

## Metrics captured per run

| Metric | Meaning |
|---|---|
| `ttft_ms` | Wall-clock time to first token (perceived responsiveness) |
| `gen_tps` | Output tokens/sec = `eval_count / eval_duration` (decode speed) |
| `prompt_tps` | Prefill tokens/sec = `prompt_eval_count / prompt_eval_duration` |
| `load_ms` | Model load portion (cold start) |
| `wall_total_ms` | Full request round-trip |
| `output_tokens` / `prompt_tokens` | Token counts (`eval_count`, `prompt_eval_count`) |
| `residency` | `ollama ps` PROCESSOR split (e.g. `100% GPU`) |
| GPU sample (NVIDIA) | Peak/avg utilization, peak VRAM, peak/avg power, peak temp |

## Config format

```jsonc
{
  "name": "quant-sweep",         // used in output filenames
  "runs": 3,                      // measured repeats (median reported)
  "warmup": 1,                    // discarded warm-up runs
  "measure_cold_start": true,     // evict + time a genuine cold start
  "options": { "temperature": 0, "num_predict": 256 },  // held constant
  "models": ["qwen3:4b-q4_K_M", "qwen3:4b-q8_0"],
  "context_lengths": [null],      // null = model default; or [4096, 8192, 16384]
  "prompts": ["short_qa", "reasoning", "prefill_stress"]  // keys into prompts.json
}
```

Prompts live in `configs/prompts.json` as `{ key: { text, note } }`. For
thinking models (like Qwen3) you can add `"think": false` to `options` to
disable the reasoning stream.

## Included configs

- `smoke.json`: 1 model, 1 prompt, 1 run: fast sanity check (~5 s)
- `quant-sweep.json`: Qwen3 4B at Q4 / Q8 / FP16: same architecture, varying
  precision (Experiment A)
- `mac.json`: the shared cross-system matrix (run on both machines)

## Project layout

```
bench.py                CLI entry point (info / run / report / serve)
llmbench/
  ollama.py             streaming client + per-generation timing
  backends.py           OpenAI-compatible engine client (TensorRT-LLM, NIM)
  telemetry.py          platform detection, ollama ps, nvidia-smi sampler
  extract.py            document text extraction (pdf/docx/pptx/xlsx/…)
  attachments.py        inject reference docs, encode images
  rag.py                Copilot RAG: chunk, embed, cosine retrieval, grounding
  agent.py              agent harness: tools + model→tool→observation loop
  guardrails.py         deterministic rails: input/tool/param/approval/output
  evals.py              evaluation harness: RAG + guardrail scorers
  config.py             config + prompt loading/validation
  runner.py             matrix expansion, methodology, aggregation
  report.py             self-contained HTML + inline SVG charts
  store.py              per-run persistence (runs/<id>/)
  jobs.py               single-worker job queue for the web UI
  server.py             stdlib web server + JSON API
  web/index.html        the single-page web UI
configs/                benchmark configs + prompt set
  docs/                 sample reference documents for task benchmarks
evals/                  evaluation suites (rag_tests.json, guardrails_tests.json)
results/                CLI output JSON + generated HTML reports
runs/                   web-UI run history (one folder per run)
kb/                     Copilot knowledge bases (one folder per KB)
```

## The full local-AI stack, built end to end

This project walks the entire local-AI stack rather than just "an LLM on a GPU":

1. **Benchmarking**: TTFT, throughput, cold-start, memory residency, with a
   defensible methodology.
2. **Task benchmarking**: real prompts with document + image attachments.
3. **Cross-system**: RTX vs Apple Silicon, VRAM-wall detection.
4. **Copilot (RAG)**: chunk → embed → retrieve → grounded, cited answers.
5. **Agent harness**: model + tools + execution loop.
6. **Guardrails**: deterministic input/tool/param/approval/output rails.
7. **Evaluation**: groundedness, correct abstention, and guardrail enforcement,
   scored reproducibly.

The throughline: **the model is one component of an AI product, not the
product.** Model quality, quantization, memory residency, retrieval, harness
design, guardrails, and evaluation all shape the result, and this tool measures
each of them.

## License

Released under the [MIT License](LICENSE).
