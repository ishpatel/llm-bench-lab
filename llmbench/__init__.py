"""llmbench — a zero-dependency local-AI benchmarking harness.

Measures inference performance of models served by Ollama (TTFT, generation
throughput, prompt/prefill throughput, load/cold-start time, latency and
memory residency) across NVIDIA RTX and Apple Silicon systems, using a
defensible methodology (warm-up, N repeats, median + spread, one variable at
a time). Produces a self-contained HTML report.

Runs identically on macOS (Apple Silicon) and Windows/Linux (NVIDIA) with
nothing but the Python 3.9+ standard library.
"""

__version__ = "0.1.0"
