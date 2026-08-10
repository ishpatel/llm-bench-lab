"""Loading and validation of benchmark configs and prompt sets."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

DEFAULTS: Dict[str, Any] = {
    "name": "benchmark",
    "system_label": None,          # None => auto-detect
    "base_url": "http://localhost:11434",
    "runs": 3,                      # measured repeats per cell (median reported)
    "warmup": 1,                    # discarded warm-up runs per cell
    "measure_cold_start": True,     # unload then time a genuine cold start
    "options": {                    # Ollama generation options (held constant)
        "temperature": 0,
        "num_predict": 256,
    },
    "models": [],
    "context_lengths": [None],      # None => model default; else sets num_ctx
    "prompts": [],                  # keys into the prompt set (or inline dicts)
    "max_chars_per_file": None,     # cap injected document size (None = no cap)
    # Optional alternate engine. Example (TensorRT-LLM via trtllm-serve):
    #   "backend": {"type": "openai", "base_url": "http://localhost:8000",
    #               "label": "TensorRT-LLM"}
    "backend": None,
}


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompts(path: str) -> Dict[str, Dict[str, str]]:
    """Prompt set: {key: {"text": ..., "note": ...}}."""
    data = _load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Prompt file {path} must be a JSON object of key->prompt.")
    return data


def resolve_prompts(
    refs: List[Any], prompt_set: Dict[str, Dict[str, str]]
) -> List[Dict[str, str]]:
    """Turn a list of prompt references into concrete {key,text,note} dicts.

    A ref may be a string key into `prompt_set`, or an inline dict with at
    least a `text` field."""
    resolved: List[Dict[str, Any]] = []
    for ref in refs:
        if isinstance(ref, str):
            if ref not in prompt_set:
                raise KeyError(f"Prompt key '{ref}' not found in prompt set.")
            entry = prompt_set[ref]
            resolved.append(_prompt_entry(ref, entry))
        elif isinstance(ref, dict) and "text" in ref:
            resolved.append(_prompt_entry(ref.get("key", "inline"), ref))
        else:
            raise ValueError(f"Invalid prompt reference: {ref!r}")
    return resolved


def _prompt_entry(key: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise a prompt into {key, text, note, files, images}. `files` and
    `images` are optional lists of paths to reference documents / images."""
    return {
        "key": key,
        "text": entry["text"],
        "note": entry.get("note", ""),
        "files": entry.get("files", []) or [],
        "images": entry.get("images", []) or [],
    }


def load_config(path: str) -> Dict[str, Any]:
    """Load a config file, filling in defaults for any missing keys."""
    cfg = dict(DEFAULTS)
    user = _load_json(path)
    if not isinstance(user, dict):
        raise ValueError(f"Config {path} must be a JSON object.")
    # shallow-merge, but deep-merge the nested options map
    opts = dict(DEFAULTS["options"])
    opts.update(user.get("options", {}))
    cfg.update(user)
    cfg["options"] = opts
    if not cfg["models"]:
        raise ValueError("Config must list at least one model in 'models'.")
    if not cfg["prompts"]:
        raise ValueError("Config must list at least one prompt in 'prompts'.")
    return cfg
