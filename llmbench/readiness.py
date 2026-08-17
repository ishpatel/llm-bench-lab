"""Can this machine actually run a benchmark, and what is missing if not.

`describe_system_deep` answers "what hardware is this". That is a different
question from "is the bench ready to run", which is about the runtime around
the hardware: an interpreter new enough, a reachable engine, models pulled, a
place to write results. Keeping them apart means the answer to "why can't I
run" is not buried among CPU core counts.

Each check reports one of three states:
    ok      the requirement is met
    warn    the bench runs, but something optional or situational is absent
    fail    a benchmark cannot produce a measurement until this is fixed
"""
from __future__ import annotations

import os
import shutil
import sys
from typing import Any, Dict, List, Optional

from . import engines as engines_mod
from . import telemetry

MIN_PYTHON = (3, 9)

# The checks describe the machine the server is running on, so the fix commands
# are resolved for that machine's OS here rather than shipping all three
# variants to the browser and guessing there.
if sys.platform == "darwin":
    OS_KEY, OS_NAME = "macos", "macOS"
elif sys.platform.startswith("win"):
    OS_KEY, OS_NAME = "windows", "Windows"
else:
    OS_KEY, OS_NAME = "linux", "Linux"


def _for_os(choices: Dict[str, str]) -> str:
    """Pick this machine's variant, falling back to a shared default."""
    return choices.get(OS_KEY, choices.get("all", ""))


# Only these may be executed by the server on request. Everything else is
# copy-only: anything needing sudo, an admin prompt, a package manager or a
# download stays the user's decision to run themselves, in their own shell.
RUNNABLE = {"engine", "models", "embedding"}


def _check(key: str, label: str, status: str, detail: str,
           fix: str = "", why: str = "", cmd: str = "") -> Dict[str, Any]:
    """`fix` is prose for a human; `cmd` is a literal command to copy or run.
    A check may have either, both, or neither."""
    return {"key": key, "label": label, "status": status, "detail": detail,
            "fix": fix, "why": why, "cmd": cmd,
            "runnable": bool(cmd) and key in RUNNABLE, "os": OS_NAME}


def _python_check() -> Dict[str, str]:
    v = sys.version_info
    got = f"{v.major}.{v.minor}.{v.micro}"
    need = ".".join(str(x) for x in MIN_PYTHON)
    if (v.major, v.minor) >= MIN_PYTHON:
        return _check("python", "Python runtime", "ok", f"{got}",
                      why=f"The harness is stdlib-only and needs {need} or newer.")
    return _check("python", "Python runtime", "fail", f"{got}, below {need}",
                  fix=f"Install Python {need} or newer and re-run from it.",
                  cmd=_for_os({
                      "macos": "brew install python@3.12",
                      "windows": "winget install Python.Python.3.12",
                      "linux": "sudo apt install python3",
                  }),
                  why="The harness is stdlib-only, but relies on "
                      f"{need}+ language features.")


def _engine_check(found: List[Dict[str, Any]], base_url: str,
                  version: Optional[str]) -> Dict[str, str]:
    """Any local engine satisfies the bench; Ollama is one of several."""
    why = ("An engine loads the models and serves the tokens every measurement "
           "is taken from. Ollama, LM Studio, llama.cpp's llama-server, vLLM, "
           "Jan and GPT4All are all detected automatically.")
    if not found:
        return _check("engine", "Inference engine", "fail",
                      "none detected (Ollama, LM Studio, llama.cpp, vLLM, "
                      "Jan, GPT4All)",
                      fix="Start whichever you use. If that is Ollama:",
                      cmd=_for_os({
                          # Identical on every platform, and the only engine
                          # with a start command rather than an app to open.
                          "all": "ollama serve",
                      }),
                      why=why)
    parts = []
    for e in found:
        port = e["base_url"].rsplit(":", 1)[-1]
        if e["label"] == "Ollama" and version:
            parts.append(f"Ollama v{version}")
        else:
            parts.append(f"{e['label']} on :{port}")
    return _check("engine", "Inference engine", "ok", " + ".join(parts), why=why)


def _model_checks(models: List[Dict[str, Any]], found: List[Dict[str, Any]]
                  ) -> List[Dict[str, str]]:
    """Generative models are required; an embedding model is only needed by the
    retrieval features, so its absence is a warning rather than a failure.
    `models` is Ollama's capability-tagged list; other engines report ids only,
    which all count as generative because /v1/models cannot say otherwise."""
    if not found:
        return [_check("models", "Models installed", "fail",
                       "cannot tell until an engine is running",
                       fix="Start an engine first.",
                       why="A benchmark needs at least one model that answers "
                           "prompts.")]
    gen = [m for m in models if m.get("generative") and not m.get("embedding")]
    emb = [m for m in models if m.get("embedding")]
    other = [(e["label"], len(e["models"])) for e in found
             if e["kind"] != "ollama" and e["models"]]
    out = []
    if gen or other:
        parts = ([f"{len(gen)} via Ollama"] if gen else []) +                 [f"{n} via {lab}" for lab, n in other]
        out.append(_check("models", "Models installed", "ok",
                          ", ".join(parts) or "0",
                          why="A benchmark needs at least one model that "
                              "answers prompts."))
    else:
        only_emb = " Only embedding models are present." if emb else ""
        out.append(_check("models", "Models installed", "fail",
                          f"none available to benchmark.{only_emb}",
                          fix="Pull or download a model that can answer "
                              "prompts. With Ollama:",
                          cmd="ollama pull qwen3:4b-q4_K_M",
                          why="A benchmark needs at least one model that "
                              "answers prompts."))
    has_ollama = any(e["kind"] == "ollama" for e in found)
    out.append(_check(
        "embedding", "Embedding model", "ok" if emb else "warn",
        f"{emb[0]['name']}" if emb else
        ("none installed" if has_ollama else "needs Ollama (not running)"),
        fix="" if emb else "Pull an embedding model to enable retrieval.",
        cmd="" if emb or not has_ollama else "ollama pull embeddinggemma",
        why="Only the Copilot and Evals tabs need one, to turn documents into "
            "vectors for retrieval; they embed through Ollama specifically. "
            "Benchmarks run without it."))
    return out


def _accel_check(deep: Dict[str, Any]) -> Dict[str, str]:
    env, gpu = deep.get("env") or {}, deep.get("gpu") or {}
    backend = env.get("backend") or ""
    if backend:
        return _check("accel", "GPU acceleration", "ok",
                      backend + (f" · {gpu['name']}" if gpu.get("name") else ""),
                      why="Inference on the GPU is what the benchmark is "
                          "measuring. On CPU alone the numbers describe a "
                          "fallback path, not the hardware.")
    return _check("accel", "GPU acceleration", "warn", "no GPU backend detected",
                  fix="Check that the GPU driver and Ollama are installed.",
                  why="Without a GPU backend, models run on the CPU and the "
                      "results describe a fallback path rather than the "
                      "accelerator.")


def _nvidia_checks(deep: Dict[str, Any]) -> List[Dict[str, str]]:
    """NVIDIA-only extras. Skipped entirely elsewhere: a Mac reporting a
    missing nvidia-smi is noise, not a finding."""
    trt = telemetry.tensorrt_status()
    gpu_name = ((deep.get("gpu") or {}).get("name") or "").lower()
    if not trt.get("nvidia_gpu") and "nvidia" not in gpu_name:
        return []
    out = [_check(
        "nvsmi", "nvidia-smi", "ok" if telemetry.has_nvidia_smi() else "warn",
        "on PATH" if telemetry.has_nvidia_smi() else "not on PATH",
        fix="" if telemetry.has_nvidia_smi() else
            "Install the NVIDIA driver, or put nvidia-smi on PATH.",
        cmd="" if telemetry.has_nvidia_smi() else _for_os({
            "windows": r'setx PATH "%PATH%;C:\Program Files\NVIDIA Corporation\NVSMI"',
            "linux": "sudo apt install nvidia-utils-535",
            "macos": "",
        }),
        why="Supplies live GPU utilisation and VRAM during a run. Timing "
            "measurements do not depend on it.")]
    ver = trt.get("python_tensorrt")
    out.append(_check(
        "tensorrt", "TensorRT", "ok" if ver else "warn",
        f"python package {ver}" if ver else "not installed",
        fix="" if ver else "Optional. Needs an OpenAI-compatible server such "
                           "as trtllm-serve to benchmark against.",
        why="Optional alternate engine. Ollama has no TensorRT mode, so this "
            "is only used through the OpenAI-compatible backend option."))
    return out


def _storage_check(project_root: str) -> Dict[str, str]:
    runs = os.path.join(project_root, "runs")
    target = runs if os.path.isdir(runs) else project_root
    try:
        os.makedirs(runs, exist_ok=True)
        probe = os.path.join(runs, ".write-probe")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
    except OSError as exc:
        return _check("storage", "Results storage", "fail",
                      f"cannot write to {runs} ({exc.strerror or exc})",
                      fix="Grant write permission to the project directory.",
                      cmd=_for_os({
                          "windows": f'icacls "{runs}" /grant "%USERNAME%":(OI)(CI)F',
                          "all": f'chmod u+w "{runs}"',
                      }),
                      why="Every run is saved here; without it a finished "
                          "benchmark cannot be kept.")
    free_gb = None
    try:
        free_gb = round(shutil.disk_usage(target).free / (1024 ** 3), 1)
    except OSError:
        pass
    low = free_gb is not None and free_gb < 2
    return _check("storage", "Results storage", "warn" if low else "ok",
                  (f"writable · {free_gb} GB free" if free_gb is not None
                   else "writable"),
                  fix="Free up disk space." if low else "",
                  why="Runs, their full outputs and any attachments are saved "
                      "to the runs folder.")


def describe_readiness(client=None, base_url: str = "",
                       models: Optional[List[Dict[str, Any]]] = None,
                       project_root: str = ".",
                       deep: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run every check and summarise. `client` and `models` are passed in so a
    caller that has already talked to Ollama does not query it twice."""
    deep = deep or telemetry.describe_system_deep()
    found = engines_mod.detect(ollama_base_url=base_url) if base_url \
        else engines_mod.detect()
    checks: List[Dict[str, str]] = [
        _python_check(),
        _engine_check(found, base_url, (deep.get("env") or {}).get("ollama")),
    ]
    checks += _model_checks(models or [], found)
    checks.append(_accel_check(deep))
    checks.append(_storage_check(project_root))
    checks += _nvidia_checks(deep)

    fails = [c for c in checks if c["status"] == "fail"]
    warns = [c for c in checks if c["status"] == "warn"]
    if fails:
        state, headline = "fail", (
            f"{len(fails)} thing{'s' if len(fails) > 1 else ''} must be fixed "
            "before a benchmark can run")
    elif warns:
        state, headline = "warn", (
            f"Ready to benchmark · {len(warns)} optional item"
            f"{'s' if len(warns) > 1 else ''} unavailable")
    else:
        state, headline = "ok", "Ready to benchmark"
    return {"state": state, "headline": headline, "checks": checks,
            "engines": found,
            "counts": {"ok": len(checks) - len(fails) - len(warns),
                       "warn": len(warns), "fail": len(fails)}}
