"""Thin Ollama HTTP client + single-generation timing.

Uses only urllib so the harness has no third-party dependencies. Talks to the
Ollama `/api/generate` endpoint in streaming mode so we can measure a true,
wall-clock time-to-first-token in addition to the nanosecond timings Ollama
reports in its final message.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000


@dataclass
class GenerationResult:
    """Metrics from one generation call. Durations normalised to ms; rates to
    tokens/second. `ok=False` with `error` set means the call failed."""

    model: str
    ok: bool = True
    error: Optional[str] = None

    # Wall-clock (what the user actually experiences).
    # Reasoning models stream hidden thinking before any visible words, so the
    # two clocks diverge: ttft_ms is when the model started producing anything
    # (compute latency), ttfv_ms is when the user first saw a word (perceived
    # latency). They are identical on non-thinking models.
    ttft_ms: Optional[float] = None           # first token of any kind
    ttfv_ms: Optional[float] = None           # first *visible* token
    wall_total_ms: Optional[float] = None     # full request round-trip

    # Ollama-reported internals (nanoseconds -> ms)
    load_ms: Optional[float] = None           # model load / cold-start portion
    total_ms: Optional[float] = None          # Ollama's own total_duration
    prompt_eval_ms: Optional[float] = None    # prefill time
    eval_ms: Optional[float] = None           # decode/generation time

    # Token counts
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None       # eval_count: all generated tokens

    # Derived rates
    gen_tps: Optional[float] = None           # output tokens / eval time
    prompt_tps: Optional[float] = None        # prompt tokens / prefill time

    response_text: str = ""
    thinking_chars: int = 0                    # size of separate reasoning stream
    # True when token counts came from counting stream deltas rather than a
    # server-reported usage block. Derived rates are suppressed in that case so
    # an estimate can never be mistaken for a measurement.
    approximate_tokens: bool = False
    stream_chunks: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434",
                 timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._cap_cache: Dict[str, List[str]] = {}

    # -- low-level helpers -------------------------------------------------
    def _post(self, path: str, payload: Dict[str, Any]) -> Any:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path, data=data,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def is_up(self) -> bool:
        try:
            with urllib.request.urlopen(self.base_url + "/api/tags", timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        try:
            with urllib.request.urlopen(self.base_url + "/api/tags", timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return sorted(m.get("name", "") for m in data.get("models", []))
        except Exception:
            return []

    _caps_cache: Dict[str, List[str]] = {}

    def capabilities(self, model: str) -> List[str]:
        if model in self._caps_cache:
            return self._caps_cache[model]
        """Model capabilities via /api/show (e.g. ['completion','vision'] or
        ['embedding']). Cached per model. Empty list if unavailable — callers
        treat unknown as generative so nothing is wrongly blocked."""
        if model in self._cap_cache:
            return self._cap_cache[model]
        caps: List[str] = []
        try:
            data = self._post("/api/show", {"model": model})
            caps = data.get("capabilities", []) or []
        except Exception:
            caps = []
        self._cap_cache[model] = caps
        self._caps_cache[model] = caps
        return caps

    def models_detailed(self) -> List[Dict[str, Any]]:
        """List models with derived capability flags for the UI."""
        out: List[Dict[str, Any]] = []
        for name in self.list_models():
            caps = self.capabilities(name)
            out.append({
                "name": name,
                "capabilities": caps,
                # unknown (no caps reported) -> assume generative, don't block
                "generative": ("completion" in caps) or (not caps),
                "vision": "vision" in caps,
                "embedding": ("embedding" in caps) and ("completion" not in caps),
            })
        return out

    def chat(self, model: str, messages: List[Dict[str, Any]],
             tools: Optional[List[Dict[str, Any]]] = None,
             options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Non-streaming /api/chat, used for tool-calling agent turns. Returns
        {message, tool_calls, wall_ms, eval_ms, output_tokens, gen_tps}."""
        opts = dict(options or {})
        payload: Dict[str, Any] = {"model": model, "messages": messages,
                                   "stream": False}
        if "think" in opts:
            payload["think"] = opts.pop("think")
        payload["options"] = opts
        if tools:
            payload["tools"] = tools
        t0 = time.perf_counter()
        data = self._post("/api/chat", payload)
        wall = (time.perf_counter() - t0) * 1000.0
        msg = data.get("message", {}) or {}
        eval_ms = (data.get("eval_duration") or 0) / NS_PER_MS
        n_out = data.get("eval_count")
        gen_tps = (n_out / (data.get("eval_duration") / NS_PER_S)
                   if n_out and data.get("eval_duration") else None)
        return {
            "message": msg,
            "tool_calls": msg.get("tool_calls") or [],
            "content": msg.get("content", ""),
            "wall_ms": wall,
            "eval_ms": eval_ms,
            "output_tokens": n_out,
            "gen_tps": gen_tps,
        }

    def embed(self, model: str, inputs: List[str]) -> Dict[str, Any]:
        """Embed one or more strings via /api/embed. Returns
        {embeddings: [[...]], total_ms, prompt_tokens}."""
        t0 = time.perf_counter()
        data = self._post("/api/embed", {"model": model, "input": inputs})
        wall_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "embeddings": data.get("embeddings", []),
            "wall_ms": wall_ms,
            "total_ms": (data.get("total_duration") or 0) / NS_PER_MS,
            "prompt_tokens": data.get("prompt_eval_count"),
        }

    def unload(self, model: str) -> None:
        """Ask Ollama to evict a model from memory so the next call measures a
        genuine cold start. Best-effort; ignores failures."""
        try:
            self._post("/api/generate", {"model": model, "keep_alive": 0, "prompt": ""})
        except Exception:
            pass

    # -- the measured call -------------------------------------------------
    def generate(
        self,
        model: str,
        prompt: str,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
        images: Optional[List[str]] = None,
    ) -> GenerationResult:
        """Run one streamed generation and return timing metrics.

        `images` is a list of base64-encoded image strings for vision models."""
        opts = dict(options or {})
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": True,
        }
        # `think` is a top-level flag for thinking models, not a sampler option.
        if "think" in opts:
            payload["think"] = opts.pop("think")
        payload["options"] = opts
        if images:
            payload["images"] = images
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        result = GenerationResult(model=model)
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/api/generate", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )

        chunks: List[str] = []
        think_chars = 0
        t_start = time.perf_counter()
        first_token_at: Optional[float] = None      # any token (compute)
        first_visible_at: Optional[float] = None    # visible token (perceived)
        final: Dict[str, Any] = {}
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw in resp:  # iterates newline-delimited JSON objects
                    line = raw.decode("utf-8").strip()
                    if not line:
                        continue
                    msg = json.loads(line)
                    piece = msg.get("response", "")
                    thinking = msg.get("thinking", "")
                    # Two clocks: any token starts compute latency, only a
                    # visible token starts perceived latency.
                    if (piece or thinking) and first_token_at is None:
                        first_token_at = time.perf_counter()
                    if piece and first_visible_at is None:
                        first_visible_at = time.perf_counter()
                    if piece:
                        chunks.append(piece)
                    if thinking:
                        think_chars += len(thinking)
                    if msg.get("done"):
                        final = msg
        except urllib.error.URLError as e:
            result.ok = False
            result.error = f"{type(e).__name__}: {getattr(e, 'reason', e)}"
            return result
        except Exception as e:  # noqa: BLE001 - surface any transport error
            result.ok = False
            result.error = f"{type(e).__name__}: {e}"
            return result

        t_end = time.perf_counter()
        result.response_text = "".join(chunks)
        result.thinking_chars = think_chars
        result.wall_total_ms = (t_end - t_start) * 1000.0
        if first_token_at is not None:
            result.ttft_ms = (first_token_at - t_start) * 1000.0
        if first_visible_at is not None:
            result.ttfv_ms = (first_visible_at - t_start) * 1000.0

        # Ollama internal timings (nanoseconds)
        def ns_to_ms(key: str) -> Optional[float]:
            v = final.get(key)
            return (v / NS_PER_MS) if isinstance(v, (int, float)) else None

        result.load_ms = ns_to_ms("load_duration")
        result.total_ms = ns_to_ms("total_duration")
        result.prompt_eval_ms = ns_to_ms("prompt_eval_duration")
        result.eval_ms = ns_to_ms("eval_duration")
        result.prompt_tokens = final.get("prompt_eval_count")
        result.output_tokens = final.get("eval_count")

        # Derived throughput rates
        if final.get("eval_count") and final.get("eval_duration"):
            result.gen_tps = final["eval_count"] / (final["eval_duration"] / NS_PER_S)
        if final.get("prompt_eval_count") and final.get("prompt_eval_duration"):
            result.prompt_tps = final["prompt_eval_count"] / (
                final["prompt_eval_duration"] / NS_PER_S
            )
        return result
