"""Benchmark orchestration.

Expands a config into a matrix of cells (model x prompt x context length),
enforces the methodology (optional cold start, warm-up runs, N measured
repeats), samples GPU telemetry around each measured run, and aggregates the
repeats into a median + spread. Emits a structured results dict ready to be
serialised to JSON and fed to the report generator.
"""
from __future__ import annotations

import os
import statistics
import sys
import time
from typing import Any, Callable, Dict, List, Optional

from . import attachments as attach_mod
from . import config as cfg_mod
from . import telemetry
from .ollama import GenerationResult, OllamaClient

# Metrics we aggregate across repeats (attribute name -> higher_is_better)
AGG_METRICS = {
    "gen_tps": True,
    "prompt_tps": True,
    "ttft_ms": False,
    "wall_total_ms": False,
    "load_ms": False,
    "eval_ms": False,
    "prompt_eval_ms": False,
    "output_tokens": True,
    "prompt_tokens": True,
}


def _aggregate(runs: List[GenerationResult]) -> Dict[str, Any]:
    """Median / min / max / stdev for each metric across successful repeats."""
    ok = [r for r in runs if r.ok]
    agg: Dict[str, Any] = {"n_ok": len(ok), "n_total": len(runs)}
    for metric in AGG_METRICS:
        vals = [getattr(r, metric) for r in ok if getattr(r, metric) is not None]
        if not vals:
            agg[metric] = None
            continue
        agg[metric] = {
            "median": statistics.median(vals),
            "min": min(vals),
            "max": max(vals),
            "stdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        }
    return agg


class Runner:
    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Dict[str, str]],
                 log: Optional[Callable[[str], None]] = None,
                 base_dirs: Optional[List[str]] = None):
        self.config = config
        backend = config.get("backend") or None
        if backend and backend.get("type") == "openai":
            from .backends import OpenAICompatClient
            self.client = OpenAICompatClient(backend["base_url"])
            self.engine_label = backend.get("label", "external engine")
        else:
            self.client = OllamaClient(base_url=config["base_url"])
            self.engine_label = None  # local Ollama
        self.prompts = cfg_mod.resolve_prompts(config["prompts"], prompts)
        self.log = log or (lambda m: print(m, file=sys.stderr))
        self.base_dirs = base_dirs or [os.getcwd()]

    # -- a single measured generation, wrapped in telemetry sampling --------
    def _measured(self, model: str, prompt: str, options: Dict[str, Any],
                  keep_alive: Optional[str] = None,
                  images: Optional[List[str]] = None) -> Dict[str, Any]:
        sampler = telemetry.GpuSampler()
        sampler.start()
        res = self.client.generate(model, prompt, options=options,
                                   keep_alive=keep_alive, images=images)
        stats = sampler.stop()
        out = res.as_dict()
        out["gpu"] = stats.__dict__
        return out

    def run(self) -> Dict[str, Any]:
        c = self.config
        system = telemetry.describe_system(c.get("system_label"))
        self.log(f"System: {system.get('label')} "
                 f"[{system.get('accelerator')}] · Ollama models resident checked per cell")

        available = set(self.client.list_models())
        cells: List[Dict[str, Any]] = []

        for model in c["models"]:
            if self.engine_label:
                # External engines may not implement model listing fully;
                # warn instead of refusing.
                if available and model not in available:
                    self.log(f"  ! {model} not listed by the endpoint; trying anyway")
            elif model not in available:
                self.log(f"  ! skipping {model}: not pulled (ollama pull {model})")
                continue
            for ctx in c["context_lengths"]:
                options = dict(c["options"])
                if ctx is not None:
                    options["num_ctx"] = ctx
                for p in self.prompts:
                    label = self._cell_label(model, ctx, p["key"])
                    self.log(f"→ {label}")
                    cell = self._run_cell(model, ctx, options, p)
                    cells.append(cell)

        return {
            "meta": {
                "config_name": c["name"],
                "system": system,
                "options": c["options"],
                "runs": c["runs"],
                "warmup": c["warmup"],
                "measure_cold_start": c["measure_cold_start"],
                "engine": self.engine_label or "Ollama",
                "ollama_version": self._ollama_version(),
            },
            "cells": cells,
        }

    def _run_cell(self, model: str, ctx: Optional[int], options: Dict[str, Any],
                  prompt: Dict[str, Any]) -> Dict[str, Any]:
        c = self.config

        # Assemble the task prompt with any reference documents injected, plus
        # base64 images for vision models. This is done once per cell so the
        # exact same input is used for cold-start, warm-up and measured runs.
        text, images, attach_meta = attach_mod.prepare(
            prompt["text"],
            files=prompt.get("files"),
            images=prompt.get("images"),
            base_dirs=self.base_dirs,
            max_chars_per_file=c.get("max_chars_per_file"),
        )
        if attach_meta["files"] or attach_meta["images"]:
            self.log(f"    attachments: {len(attach_meta['files'])} file(s), "
                     f"{len(attach_meta['images'])} image(s), "
                     f"~{attach_meta['assembled_approx_tokens']} prompt tokens (est.)")
        if images and self.engine_label:
            self.log("    note: image attachments are not sent to external "
                     "engines; running text-only")
            images = []

        cold: Optional[Dict[str, Any]] = None
        if c["measure_cold_start"]:
            if self.engine_label:
                self.log("    external engine manages its own model loading; "
                         "skipping cold-start measurement")
            else:
                self.client.unload(model)
                time.sleep(0.5)  # let the eviction settle
                self.log("    cold-start run…")
                cold = self._measured(model, text, options, images=images)

        for i in range(c["warmup"]):
            self.log(f"    warm-up {i + 1}/{c['warmup']}…")
            self.client.generate(model, text, options=options, images=images)

        runs: List[GenerationResult] = []
        raw_runs: List[Dict[str, Any]] = []
        for i in range(c["runs"]):
            self.log(f"    run {i + 1}/{c['runs']}…")
            m = self._measured(model, text, options, images=images)
            raw_runs.append(m)
            runs.append(_dict_to_result(m))

        # Residency snapshot after the model has been exercised. External
        # engines are not visible to `ollama ps`; label them by engine and let
        # the NVIDIA GPU sampler tell the memory story.
        if self.engine_label:
            residency = f"{self.engine_label} engine"
        else:
            residency = telemetry.ollama_ps().get(model, {}).get("processor", "")

        return {
            "label": self._cell_label(model, ctx, prompt["key"]),
            "model": model,
            "context_length": ctx,
            "prompt_key": prompt["key"],
            "prompt_note": prompt.get("note", ""),
            "prompt_text": prompt["text"],
            "attachments": attach_meta,
            "residency": residency,
            "cold_start": cold,
            "runs": raw_runs,
            "aggregate": _aggregate(runs),
        }

    @staticmethod
    def _cell_label(model: str, ctx: Optional[int], prompt_key: str) -> str:
        ctx_s = f" ctx={ctx}" if ctx is not None else ""
        return f"{model}{ctx_s} · {prompt_key}"

    def _ollama_version(self) -> str:
        return telemetry._run(["ollama", "--version"]) or "unknown"


def _dict_to_result(d: Dict[str, Any]) -> GenerationResult:
    r = GenerationResult(model=d.get("model", ""))
    for k, v in d.items():
        if hasattr(r, k):
            setattr(r, k, v)
    return r
