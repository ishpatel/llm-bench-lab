# llm-bench-lab

A zero-dependency, cross-platform harness for benchmarking local LLM inference
on **NVIDIA RTX** and **Apple Silicon** systems, built around Ollama. It
measures the metrics that actually shape the user experience — time-to-first-token,
generation throughput, prefill throughput, cold-start load time, and memory
residency — under a defensible methodology, and emits a self-contained HTML
report you can hand to anyone.

Nothing but the Python 3.9+ standard library. Same code runs on macOS (Metal)
and Windows/Linux (CUDA); on NVIDIA it additionally samples `nvidia-smi` for
GPU utilization, VRAM, power and temperature.

## Why this design

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

- **Pick a model** from the ones installed in Ollama.
- **Type any prompt** and **attach multimodal reference material** — drag in PDFs,
  Word/PowerPoint/Excel docs, text/code, or images (images need a vision model).
  Documents are auto-extracted to text and injected as context.
- **Submit** → the run is queued (one at a time, so GPU contention never pollutes
  timings) and you watch **live progress**; when it finishes you see the metrics
  and the model's actual output.
- Every run is **saved** with its own self-contained HTML report.
- A **Runs** master view lets you browse, search, filter by model, and sort.
- Select runs and **Compare** them side-by-side, with the best value per metric
  highlighted.

An **Advanced** panel exposes the methodology knobs (measured repeats, warm-up,
cold-start, temperature, max tokens, context length) — defaults keep the rigorous
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

Useful flags: `run --label "<name>"` (system label shown in the report),
`run --runs N` (override measured repeats), `--base-url` (remote Ollama).

## Cross-system comparison (RTX vs Apple Silicon)

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

The report groups bars by system per configuration. On the 8 GB RTX laptop,
`qwen3:4b-fp16` (~8 GB) and `qwen3:8b-q4_K_M` (~5 GB) are where you expect the
VRAM wall — watch the `residency` column flip from `100% GPU` to a CPU/GPU
split and generation throughput collapse. That contrast against the M3 Max's
unified memory is the centerpiece of the writeup.

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
**measure the cost of context** directly — it's the on-ramp to RAG. Relative
paths resolve against the config dir, then the prompts dir, then cwd. Cap
injected size with `"max_chars_per_file"` in the config.

**Supported document types** (extracted automatically to text — see
`llmbench/extract.py`):

| Type | How it's extracted | Notes |
|---|---|---|
| `.txt .md .csv .json .py` … | read directly | plain-text family |
| `.pdf` | `pdftotext` if installed, else **built-in stdlib parser** | built-in handles text-based PDFs (FlateDecode + standard encodings); scanned/CID-font PDFs are flagged — install poppler `pdftotext` or OCR them |
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
answer — so you can compare quality *and* speed together. See
`configs/task-with-doc.json` for a grounded-vs-ungrounded demo (the model
hallucinates VRAM without the doc and answers correctly with it).

```bash
# Markdown reference doc (grounded vs ungrounded):
python bench.py run configs/task-with-doc.json --label "M3 Max (48GB)"
# PDF reference doc (exercises the built-in extractor):
python bench.py run configs/task-with-pdf.json --label "M3 Max (48GB)"
python bench.py report results/task-with-*_*.json --out results/task.html
```

## Methodology (what makes the numbers defensible)

- **Warm-up runs** (`warmup`) are executed and discarded before measurement.
- Each cell runs **N measured repeats** (`runs`); the **median** is reported
  with the **min–max spread** shown in the report.
- **Cold-start** load time is measured separately by evicting the model
  (`keep_alive: 0`) and timing the next call — kept out of the warm TTFT.
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

- `smoke.json` — 1 model, 1 prompt, 1 run: fast sanity check (~5 s)
- `quant-sweep.json` — Qwen3 4B at Q4 / Q8 / FP16: same architecture, varying
  precision (Experiment A)
- `mac.json` — the shared cross-system matrix (run on both machines)

## Project layout

```
bench.py                CLI entry point (info / run / report / serve)
llmbench/
  ollama.py             streaming client + per-generation timing
  telemetry.py          platform detection, ollama ps, nvidia-smi sampler
  extract.py            document text extraction (pdf/docx/pptx/xlsx/…)
  attachments.py        inject reference docs, encode images
  config.py             config + prompt loading/validation
  runner.py             matrix expansion, methodology, aggregation
  report.py             self-contained HTML + inline SVG charts
  store.py              per-run persistence (runs/<id>/)
  jobs.py               single-worker job queue for the web UI
  server.py             stdlib web server + JSON API
  web/index.html        the single-page web UI
configs/                benchmark configs + prompt set
  docs/                 sample reference documents for task benchmarks
results/                CLI output JSON + generated HTML reports
runs/                   web-UI run history (one folder per run)
```

## Roadmap (next phases)

This is the benchmarking foundation. Planned additions, each building on it:
RAG copilot over local docs, a tool-use harness, deterministic guardrails, and
an evaluation harness that scores task success / groundedness / correct
abstention — turning raw speed numbers into a full local-AI product story.

## License

Released under the [MIT License](LICENSE).
