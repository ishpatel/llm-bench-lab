# Test plan

A structured run order for turning this lab into evidence. Five hypotheses,
each with the exact command, what to watch, and what result would prove the
hypothesis wrong. Record whatever actually happens, including the boring and
the inconvenient.

The rule that makes the whole thing worth doing: **do not adjust a hypothesis
after seeing the data.** Write down what you expect, run it, then report the
gap. A benchmark that only ever confirms its own prediction is marketing.

---

## Before you start: control the machine

Laptop benchmarking measures power and thermal state as much as silicon. Any of
these left uncontrolled will produce differences larger than the effects being
measured.

**On the RTX 5070 laptop:**

- Plugged into AC, not battery.
- Windows power mode set to Best Performance, and the OEM performance profile
  (Legion / Omen / Alienware control panel) set to its performance preset.
- Close other GPU consumers, including browsers with hardware acceleration.
- Note the NVIDIA driver version and keep it fixed across the whole campaign:
  `nvidia-smi --query-gpu=driver_version --format=csv`

**Use 127.0.0.1, never "localhost", on Windows.**

This was measured, not theorised. Pointing the harness at `http://localhost:11434`
on the RTX machine added a constant **2,071 ms plus or minus 12.9 ms** to every
single request: identical across all fifteen cells regardless of model size or
generation speed. The delay was isolated to the Windows `localhost` connection
path, since switching to `127.0.0.1` removed it. That is consistent with IPv6
`::1` resolution and fallback behaviour, though no packet capture was taken to
confirm the exact mechanism.

The reason it is dangerous rather than merely annoying is that the delay happens
*before* the request reaches Ollama, so Ollama's own reported timings look
perfectly healthy while wall-clock latency is wrong by two seconds. Generation
throughput stays valid, because it is derived from server-side counters; every
wall-clock latency number is silently ruined.

The default is now `127.0.0.1`, which measured 21.5 ms of overhead on the same
machine. If you ever override it, override it to an IP address. To check a
result file for the artifact:

```bash
python bench.py summary results/<file>.json
```

which prints the metrics table without building a report, or directly:

```bash
python -c "import json,glob,statistics as st; d=json.load(open(sorted(glob.glob('results/cross-system_*.json'))[-1])); g=[r['ttfv_ms']-(r.get('load_ms') or 0)-(r.get('prompt_eval_ms') or 0) for c in d['cells'] for r in c['runs'] if r.get('ttfv_ms')]; print('overhead median:', round(st.median(g),1), 'ms')"
```

Under about 100 ms is fine. Anything near 2,000 ms means the run is contaminated.

**Precondition to steady state. Do not start cold.**

This one is counterintuitive and matters more than it looks. The runner executes
cells sequentially with no cooldown or shuffling between them, so on a laptop
the first model tested runs on a cool chip at high boost clocks and the last one
runs on a heat-soaked chip at lower clocks. Start from cold and thermal state
becomes correlated with model order, which means a quantization "result" could
partly be a measurement of when in the sequence each model happened to run.

So warm the machine to a stable operating point first, then benchmark:

```bash
python bench.py run configs/smoke.json --label "warmup, discard"
```

Run that two or three times, or leave any GPU workload going, until temperature
and power flatten out. Watch until the numbers stop climbing:

```bash
nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm --format=csv -l 2
```

Then delete those warm-up runs and start the real campaign.

**Order control.** After Phase 1, rerun the quantization sweep with the model
order reversed (edit the `models` list in a copy of the config). If Q4 still
beats Q8 when Q4 runs *last* on a hot machine, the effect is the model rather
than the thermals. If the ordering changes the answer, you have found a thermal
confound and that is the finding to report.

**On the M3 Max:**

- Plugged into AC. macOS High Power Mode if available.
- Quit other heavy apps.

**Both:** same model tags, same quantization, same prompts, same temperature,
same max tokens. The tool holds generation options constant within a run; you
are responsible for holding the machine constant across runs.

Record the environment once per machine:

```bash
python bench.py info
```

---

## Phase 0: setup and smoke test

Do this first on the RTX machine. It takes two minutes and catches setup
problems before you spend an hour on a run that was never going to work.

```bash
git clone https://github.com/ishpatel/llm-bench-lab.git
```

```bash
cd llm-bench-lab
```

```bash
python bench.py info
```

Confirm it reports the RTX 5070, the CUDA backend, dedicated VRAM (not unified),
and lists your pulled models. Then a single fast run:

```bash
python bench.py run configs/smoke.json --label "RTX 5070 (8GB)"
```

If that writes a results file, the harness works. Anything that fails here is a
setup problem, not a finding.

---

## Phase 1: what does quantization actually buy?

**Hypothesis.** Holding architecture and parameter count fixed and varying only
precision (Q4 -> Q8 -> FP16), generation speed falls as precision rises, because
decode is dominated by memory bandwidth and higher precision moves more bytes
per token. Memory footprint rises roughly proportionally. Answer quality is not
expected to change dramatically at these sizes, but is not measured
automatically and must be judged by reading outputs.

**Run it on both machines:**

```bash
python bench.py run configs/quant-sweep.json --label "RTX 5070 (8GB)"
```

9 cells, 45 generations. On the M3 Max this takes roughly 10 minutes; on the
RTX, longer if FP16 spills.

**What to watch.** Generation tok/s across the three precisions. Model placement
on each: `qwen3:4b-fp16` is about 8.1 GB, so on an 8 GB card this row is where
Phase 2 begins whether you intended it or not.

**Reference point.** On the M3 Max this produced roughly 100 / 70 / 42 tok/s for
Q4 / Q8 / FP16, a 2.4x spread from precision alone with no change in
architecture. Expect the RTX to show a *different* shape, not a scaled copy,
because FP16 will not fit.

**What would falsify it.** Q8 or FP16 matching or beating Q4 on tok/s, or the
ratio being roughly flat. That would suggest the workload is compute-bound
rather than bandwidth-bound at this size, which would be a more interesting
finding than the expected result.

**Also record:** rate a few answers in the UI (Runs tab, open a run, star it) so
you can say something about whether the quality cost of Q4 was visible at all.
Speed without a quality judgment is half an answer.

**Interpretation rule, and this is the one that protects your credibility.**
A precision comparison is only clean while every variant stays fully GPU
resident. On the RTX, `qwen3:4b-fp16` is about 8.1 GB and will probably cross
the 8 GB boundary, at which point its slowdown is no longer attributable to
precision alone. It becomes precision *plus* a larger footprint *plus* offload
overhead, three effects at once.

So if the RTX returns something like Q4 85 tok/s at 100% GPU, Q8 57 tok/s at
100% GPU, FP16 9 tok/s on a CPU split, do not write:

> "FP16 is 6x slower because higher precision needs more bandwidth."

Write:

> "Q4 versus Q8 is the clean within-VRAM precision comparison. FP16 crossed the
> memory boundary, so its larger loss combines higher precision with offload
> overhead and cannot be attributed to precision alone."

On the M3 Max all three stay resident, so that machine gives you the clean
three-point precision curve and the RTX gives you the boundary. Use each for
what it can actually support.

---

## Phase 2: the VRAM wall

This is the headline experiment and the reason the RTX machine matters.

**Hypothesis.** On 8 GB of dedicated VRAM, models near or above that size cannot
be held entirely in GPU memory. `gemma3:12b-it-q4_K_M` (about 8.1 GB) and
`qwen3:4b-fp16` (about 8.1 GB) should show model placement changing from
`100% GPU` to a CPU/GPU split, with a large drop in generation speed, while
`qwen3:4b-q4_K_M` (about 2.6 GB) stays fully resident as the control. On the
M3 Max, all three should stay resident because 48 GB of unified memory is not
the binding constraint.

**Run the shared matrix on the RTX machine:**

```bash
python bench.py run configs/cross-system.json --label "RTX 5070 (8GB)"
```

15 cells, 75 generations. Budget 30 to 60 minutes on the RTX; the spilling
models are slow, which is the point.

**And the same file on the Mac:**

```bash
python bench.py run configs/cross-system.json --label "M3 Max (48GB)"
```

**What to watch.**

- **Model placement** is the primary measurement, not speed. `100% GPU` versus
  a `CPU/GPU` split is the actual phenomenon; the tok/s drop is its consequence.
- **Where the boundary falls.** It will probably not be exactly 8.0 GB. Weights
  are only one consumer: the KV cache, activations and runtime workspace all
  compete. A model whose file is smaller than VRAM can still fail to fit.
- **Peak VRAM and utilization** from the `nvidia-smi` sampler.

**What would falsify it.** Both large models staying at `100% GPU` with no
throughput cliff, or the throughput dropping without any residency change (which
would point at thermal throttling instead, and you would check temperature and
clocks before claiming a memory effect).

**The write-up sentence you are trying to earn:** *"the boundary appeared at X,
not at the model's file size, because weights are not the only memory
consumer."*

---

## Phase 3: does context capacity move the boundary?

The subtlest result and the one that best demonstrates understanding. It also
contains a distinction that is easy to get wrong, so it is worth being precise.

**Two different things get called "context".**

1. **Allocated capacity** (`num_ctx`): how large a context window the runtime
   reserves. The KV cache is sized against this, so raising it costs memory
   whether or not you use the space.
2. **Actual input length**: how many tokens the model really has to read. This
   is what drives prefill work and therefore time to first token.

**This experiment tests capacity only.** The prompt is held constant at roughly
490 tokens across all four cells; only `num_ctx` changes from 4K to 32K.

**Hypothesis.** Increasing allocated context capacity increases memory
requirements and may push a previously GPU-resident model across the VRAM
boundary, without changing a single weight and without changing the amount of
text the model reads. Latency should be roughly flat here, because the input did
not grow.

**Separately** (not tested by this config): increasing the actual number of
input tokens increases prefill work and should raise time to first visible
token. If you want that measurement, attach documents of increasing size to a
prompt rather than raising `num_ctx`, and report it as a distinct experiment.

Do not present a latency change from this run as evidence about long-context
cost. If latency does move here, the interesting explanation is memory pressure
or offload, not prefill volume.

```bash
python bench.py run configs/context-scaling.json --label "RTX 5070 (8GB)"
```

4 cells, 16 generations. Fast. Run it on the Mac too:

```bash
python bench.py run configs/context-scaling.json --label "M3 Max (48GB)"
```

**What to watch.** Model placement at each capacity on the RTX. If a model that
was `100% GPU` at 4K goes to a CPU split at 16K or 32K, you have directly
demonstrated that the memory ceiling is a function of configuration, not just of
model size. On the Mac, expect residency to hold throughout, which is the honest
contrast between the two memory architectures.

**What would falsify it.** Residency unchanged across all four capacities on the
RTX, which would mean the KV cache for this model is small relative to your
remaining headroom. Then say so, and note at what capacity you would expect it
to matter.

---

## Phase 4: merge the two machines

Once both machines have run `configs/cross-system.json`, bring them together.

**In the web UI (recommended):** on the RTX machine open each run and press
`Export`, copy the JSON files to the Mac, then Runs tab -> `Import run`. Open the
**Cross-System** tab. Any model benchmarked on both machines appears with the
two side by side, and the VRAM wall is flagged automatically with the
throughput cost computed.

**Via the CLI:** copy both results files into one `results/` directory, then:

```bash
python bench.py report results/cross-system_*.json --out results/cross.html
```

**Framing that matters.** This is a *system-level client experience*
comparison, not a raw GPU benchmark. Different memory architectures, different
runtimes, different thermal envelopes. The interesting question is not "which
GPU is faster" but "at what point does each system stop being able to run the
thing you wanted to run."

---

## Phase 5: was RAG worth it?

**Hypothesis.** On questions whose answers live in a private document corpus,
retrieval converts a model that cannot know the answer into one that answers
correctly and cites its source, at the cost of extra embedding time, retrieval
time and a much larger prompt. The accuracy gain should be large enough to
dominate the latency cost for this class of task.

```bash
python bench.py eval life_lab
```

This runs every question twice, once with retrieval and once against the raw
model, and scores correctness, groundedness, hallucinations and correct
abstentions.

**Reference point from the M3 Max:** RAG 13/14 with 0 hallucinations, raw model
5/14 with 4 hallucinations. The one persistent RAG failure is a spending total
where the source CSV splits across chunks and the model sums 5 of 7 rows, which
is a retrieval limitation rather than an arithmetic one.

**What to watch.** Not just the pass counts. The *hallucination* count is the
sharper signal, because a keyword scorer can pass a lucky guess. Note that the
raw model scored 5/14 partly by guessing plausible values that happened to match
expected strings.

**What would falsify it.** RAG failing to beat the raw model, or the added
latency being large enough that a user would prefer the wrong answer sooner.
Record embedding and retrieval milliseconds from the Copilot tab so you can
quantify that trade rather than asserting it.

---

## Phase 6: safety behaviour

Cheap, fast, and the part most candidates cannot show at all.

```bash
python bench.py eval guardrails
```

```bash
python bench.py eval rag
```

Then the retrieval-injection demonstration by hand, in the Copilot tab: build a
knowledge base that includes `evals/fixtures/shipping_notice.md`, ask an
ordinary question about the delivery, and observe that the answer is correct,
the payload never reaches the model, and the response reports suspicious content
found in the documents.

**The finding already on record:** applying the security notice to every prompt
regressed 2 of 14 eval cases, so it was made conditional on the deterministic
rail actually flagging a source. Guardrails are not free, and the useful question
is what safety cost and whether the trade was worth it.

---

## Phase 7: freeze the evidence

Do this once, at the end. It is what converts "I built a lab" into "I ran
experiments."

```bash
mkdir -p sample-results
```

Copy into `sample-results/`:

- the raw results JSON from each machine,
- one merged cross-system HTML report,
- the eval output from both machines.

Then commit with a deliberately boring message:

```bash
git add sample-results && git commit -m "Add RTX 5070 and M3 Max benchmark results"
```

---

## The everyday-workloads suite (added 2026-08-20)

The original campaign asked hardware questions with synthetic prompts. This
suite asks the question people actually have — *which of my models is enough
for real work* — with tasks that look like a normal Tuesday, each carrying a
checkable definition of "correct" in its prompt note.

**`workloads.json`** — six models from 4B to 27B on five tasks: decline an
email politely (format constraints), compress meeting notes to three owned
bullets, extract a support ticket to strict JSON, fix an off-by-one in Python,
and write a 350-word explainer (decode endurance). Thinking is off and
temperature 0, so outputs are comparable and the JSON case is deterministic.
Hypothesis: the everyday tasks stop improving noticeably well below the
largest model, and the suite shows where.

Quick quality gate, no eyeballing needed:

```bash
python - <<'CHECK'
import json, glob
d = json.load(open(sorted(glob.glob("results/workloads_*.json"))[-1]))
for c in d["cells"]:
    out = c["runs"][0]["response_text"]
    k = c["prompt_key"]; ok = "?"
    if k == "json_extract":
        try:
            j = json.loads(out.strip().strip("`json").strip("`"))
            ok = "PASS" if j.get("severity") in ("high", "medium", "low") else "odd"
        except Exception: ok = "FAIL (not JSON)"
    elif k == "code_bugfix":
        ok = "PASS" if "len(items) - 1" in out or "len(items)-1" in out else "FAIL"
    elif k == "meeting_summary":
        ok = "PASS" if all(n in out for n in ("Priya", "Marcus", "Dana")) else "FAIL"
    elif k == "email_decline":
        ok = "PASS" if "riday" in out and "ubject" in out else "check"
    if ok != "?": print(f"{c['model']:24} {k:16} {ok}")
CHECK
```

**`vision-tasks.json`** — the three vision models read a simple chart (three
bars, one threshold line) and a dense app screenshot. Two things are measured:
whether the answer is right (chart: three bars, yes; screenshot:
qwen3:4b-q4_K_M at 119 tok/s), and what an attached image does to prefill and
first-word latency versus the text-only cells of the same model.

**`thinking-cost-on.json` / `thinking-cost-off.json`** — the same reasoning
prompt through the three thinking models with the reasoning stream on and off,
1,024-token budget. The pair quantifies what thinking actually costs on the
metric users feel (time to the first visible word) so "should I leave thinking
on" gets a measured answer instead of a vibe. Run them back to back; nothing
else on the GPU.

**Measured outcomes (M3 Max, 2026-08-20, frozen in `sample-results/`):**
all 30 workloads cells pass the mechanical checks — including the 4B nano, so
the quality floor for everyday tasks sits far below the largest model and the
real trade is the 5.6x speed spread (103 vs 18.5 tok/s). All six vision
answers were correct; the same screenshot cost ~340 image tokens through both
Gemmas but 4,097 through Qwen3.6, whose vision encoder is far more
token-dense — multimodal prefill cost is a model property, not an image
property. Thinking cost on the same reasoning prompt: +5.5 s to the first
visible word on qwen3:8b (1,648 hidden chars), +7.4 s on gemma4:26b, and
qwen3.6:27b spent the entire 1,024-token budget reasoning without emitting a
visible word at all — at 18 tok/s that is nearly a minute of silence, which is
the measured answer to "should I leave thinking on for interactive use".

Every installed model is exercised by some suite: the 4B quant trio by
`quant-sweep`, nano through 27B by `workloads`, the vision-capable three by
`vision-tasks`, the thinking three by the `thinking-cost` pair, and
`embeddinggemma` by the RAG/life-lab evals, which embed through it.

## What to record as you go

For each phase, in your own notes, not just in the tool:

| Field | Why |
|---|---|
| Machine, driver version, power mode | The controls; without them the numbers are not reproducible |
| Hypothesis, written before the run | Prevents rewriting expectations after the fact |
| The measured result | Including the runs that did nothing interesting |
| Whether it matched, and by how much | The gap is the finding |
| What surprised you | The gap between expectation and result is the finding worth reporting |

## What a completed campaign should let you state

1. Quantization bought *X* on this hardware, and cost *Y* in quality.
2. The 8 GB boundary appeared at *Z*, which was not the model's file size,
   because weights are not the only memory consumer.
3. Unified memory and discrete VRAM failed in different places, so the honest
   comparison is client experience rather than raw GPU throughput.
4. Retrieval changed task success from *A* to *B* and cost *C* milliseconds,
   and here is the one case where it still failed and why.

If a run produces a result that contradicts the hypothesis, that is the most
valuable outcome available. Report it first.
