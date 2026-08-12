# Frozen evidence

Raw output from the completed campaign, 2026-08-11. Kept unmodified so the
findings in the top-level README can be checked against the data that produced
them.

- `cross-system_*.json` — the shared matrix, five models, both machines
- `quant-sweep_*.json` / `quant-reverse_*.json` — precision sweep plus its
  reversed-order control
- `context-scaling_*.json` — one 5.2 GB model across 4K to 32K context capacity
- `eval-*-life_lab.txt` / `eval-*-guardrails.txt` — RAG and safety suite output
- `CROSS-SYSTEM.html` — the generated report merging both machines

Two caveats recorded with the data rather than hidden:

**Unattributed request-path overhead.** Defined here as wall-clock time to first
visible token minus Ollama's own reported load and prompt-evaluation time. It is
a residual, not a measurement of one component: it can contain transport and name
resolution, server-side scheduling, runtime bookkeeping, and some first-token
behaviour. Every file here sits at 3 to 26 ms of it.

An earlier RTX campaign measured 2,071 ms plus or minus 12.9 ms of that residual,
constant across 45 runs regardless of model size or speed. The delay was isolated
to the Windows `localhost` connection path, because switching to `127.0.0.1`
removed it; that is consistent with IPv6 `::1` resolution and fallback, though no
packet capture was taken to confirm the mechanism. Since the delay occurred before
the request reached Ollama, server-side timings looked healthy while every
wall-clock latency was wrong, so that campaign was discarded rather than
published.

**Reproducibility.** M3 Max within-run spread is 0.0 to 2.9%. The RTX laptop's
median is 1.5 to 2.2% but individual cells reach 19.6%, and its Q8 measurement
disagrees between experiments (70.0 versus 79.3 tok/s). The clean precision curve
therefore comes from the Mac; the RTX supplies the memory boundary, where the
effects are large (55 to 57%) relative to that noise.
