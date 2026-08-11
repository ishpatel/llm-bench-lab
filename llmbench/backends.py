"""OpenAI-compatible backend client, used for NVIDIA TensorRT-LLM.

llmbench's default engine is Ollama (llama.cpp under the hood). On NVIDIA RTX
hardware you can benchmark through NVIDIA's TensorRT-LLM stack instead:
`trtllm-serve` and NIM containers expose an OpenAI-compatible HTTP API, so this
client speaks that protocol and maps the streamed response onto the same
GenerationResult the rest of the harness uses. Any server speaking the protocol
works (trtllm-serve, NIM, vLLM, even Ollama's own /v1), which is also how the
code path is verified on machines without NVIDIA hardware.

Notes on metrics for external engines:
* TTFT and total time are wall-clock, same definition as the Ollama path.
* Token counts come from the final `usage` chunk when the server sends one
  (stream_options.include_usage). If it does not, output tokens are estimated
  from the number of streamed deltas and flagged as approximate in the log.
* Generation speed is output tokens divided by the time between the first and
  last streamed token.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .ollama import GenerationResult


class OpenAICompatClient:
    """Minimal client for /v1/models and streaming /v1/chat/completions."""

    def __init__(self, base_url: str, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # -- discovery ---------------------------------------------------------
    def is_up(self) -> bool:
        try:
            with urllib.request.urlopen(self.base_url + "/v1/models",
                                        timeout=4) as resp:
                return resp.status == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        try:
            with urllib.request.urlopen(self.base_url + "/v1/models",
                                        timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return sorted(m.get("id", "") for m in data.get("data", []) if m.get("id"))
        except Exception:
            return []

    def unload(self, model: str) -> None:
        """External engines manage their own model residency; nothing to do."""

    # -- the measured call -------------------------------------------------
    def generate(
        self,
        model: str,
        prompt: str,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
        images: Optional[List[str]] = None,
    ) -> GenerationResult:
        opts = dict(options or {})
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": opts.get("temperature", 0),
            "max_tokens": opts.get("num_predict", 256),
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"}, method="POST")

        result = GenerationResult(model=model)
        chunks: List[str] = []
        think_chars = 0
        n_deltas = 0
        usage: Dict[str, Any] = {}
        t_start = time.perf_counter()
        t_first: Optional[float] = None
        t_first_visible: Optional[float] = None
        t_last_tok: Optional[float] = None
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw in resp:
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        msg = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if msg.get("usage"):
                        usage = msg["usage"]
                    choices = msg.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content") or ""
                    # Reasoning models stream hidden thinking separately; it
                    # still counts as generated work for TTFT purposes.
                    thinking = (delta.get("reasoning")
                                or delta.get("reasoning_content") or "")
                    if piece or thinking:
                        now = time.perf_counter()
                        if t_first is None:
                            t_first = now
                        if piece and t_first_visible is None:
                            t_first_visible = now
                        t_last_tok = now
                        n_deltas += 1
                    if piece:
                        chunks.append(piece)
                    if thinking:
                        think_chars += len(thinking)
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
        result.total_ms = result.wall_total_ms
        if t_first is not None:
            result.ttft_ms = (t_first - t_start) * 1000.0
        if t_first_visible is not None:
            result.ttfv_ms = (t_first_visible - t_start) * 1000.0

        result.prompt_tokens = usage.get("prompt_tokens")
        result.stream_chunks = n_deltas or None
        completion = usage.get("completion_tokens")
        if t_first is not None and t_last_tok is not None:
            result.eval_ms = (t_last_tok - t_first) * 1000.0

        if completion:
            # Server reported real token counts: safe to derive a rate.
            result.output_tokens = completion
            gen_s = (t_last_tok - t_first) if (t_first and t_last_tok) else 0.0
            if gen_s > 0.001:
                result.gen_tps = completion / gen_s
        else:
            # No usage block. A stream delta is not necessarily one token, so
            # deriving tokens/sec from delta counts would produce a number that
            # looks comparable to a measured rate but is not. Report the chunk
            # count, mark the run approximate, and leave gen_tps unset.
            result.approximate_tokens = True
            result.output_tokens = None
            result.gen_tps = None
        return result
