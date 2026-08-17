"""Discovery of local inference engines.

Ollama is one way to run models locally, not the only one. LM Studio,
llama.cpp's llama-server, vLLM, Jan and GPT4All all expose the same
OpenAI-compatible /v1 API on well-known ports, so the bench can find whatever
the user already runs instead of insisting on Ollama.

Discovery is two steps per candidate: does /v1/models answer on the port, and
can a native fingerprint endpoint name the product. Fingerprinting matters
because two products share a default port (vLLM and trtllm-serve on 8000,
llama-server and mlx_lm.server on 8080); when no fingerprint matches, the
honest label is the generic one.

Everything downstream speaks to these through the existing clients
(`OllamaClient` / `OpenAICompatClient`), so discovery adds no new protocol.
"""
from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

PROBE_TIMEOUT = 0.6   # per HTTP call; candidates are probed in parallel

# (default base_url, product that owns the port by convention)
OPENAI_CANDIDATES = [
    ("http://127.0.0.1:1234", "LM Studio"),
    ("http://127.0.0.1:8080", "llama.cpp"),
    ("http://127.0.0.1:8000", "vLLM"),
    ("http://127.0.0.1:1337", "Jan"),
    ("http://127.0.0.1:4891", "GPT4All"),
]

# How to install each, for the readiness fix text. All are GUI-or-installer
# products except llama.cpp, so only Ollama stays a runnable fix.
INSTALL_HINTS = {
    "Ollama": "https://ollama.com",
    "LM Studio": "https://lmstudio.ai",
    "llama.cpp": "brew install llama.cpp && llama-server -m model.gguf",
}


def _get_json(url: str, timeout: float = PROBE_TIMEOUT) -> Optional[Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _fingerprint(base: str, port_owner: str) -> str:
    """Name the product behind an OpenAI-compatible port, if it will say."""
    # LM Studio's native API is unambiguous.
    if _get_json(base + "/api/v0/models") is not None:
        return "LM Studio"
    # llama-server exposes /props with generation settings.
    props = _get_json(base + "/props")
    if isinstance(props, dict) and "default_generation_settings" in props:
        return "llama.cpp"
    # vLLM answers /version.
    ver = _get_json(base + "/version")
    if isinstance(ver, dict) and "version" in ver:
        return "vLLM"
    # The port's conventional owner is a guess; say so rather than assert it.
    return f"OpenAI-compatible ({port_owner}?)"


def _probe_openai(base: str, port_owner: str) -> Optional[Dict[str, Any]]:
    data = _get_json(base + "/v1/models")
    if not isinstance(data, dict):
        return None
    models = [m.get("id", "") for m in data.get("data", []) if m.get("id")]
    return {"kind": "openai", "base_url": base,
            "label": _fingerprint(base, port_owner), "models": models}


def _probe_ollama(base: str) -> Optional[Dict[str, Any]]:
    data = _get_json(base + "/api/tags")
    if not isinstance(data, dict):
        return None
    models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    return {"kind": "ollama", "base_url": base, "label": "Ollama",
            "models": models}


def detect(ollama_base_url: str = "http://127.0.0.1:11434",
           extra_openai: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Probe all known engines in parallel. Ollama first when present, because
    it is the only engine the Copilot/embedding features can use."""
    jobs = [("ollama", ollama_base_url, "")]
    jobs += [("openai", base, owner) for base, owner in OPENAI_CANDIDATES
             if not ollama_base_url.startswith(base)]
    jobs += [("openai", base.rstrip("/"), "custom")
             for base in (extra_openai or [])]

    def run(job):
        kind, base, owner = job
        return _probe_ollama(base) if kind == "ollama" else _probe_openai(base, owner)

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        found = [e for e in pool.map(run, jobs) if e]
    found.sort(key=lambda e: (e["kind"] != "ollama", e["label"]))
    return found
