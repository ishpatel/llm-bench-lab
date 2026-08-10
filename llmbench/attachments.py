"""Reference-material handling: inject text documents into a prompt and encode
images for multimodal models.

Two mechanisms, two different things measured:

* Text/document references are read and stitched into the prompt as context.
  This inflates the *prompt* token count, so the harness's prefill metrics
  (`prompt_tokens`, `prompt_tps`, TTFT) directly measure the cost of context.
  It is also the conceptual precursor to RAG.

* Images are base64-encoded and passed to Ollama's `images` field. This only
  works with a vision model (e.g. llama3.2-vision, qwen2.5vl, llava); text-only
  models will ignore or reject them.
"""
from __future__ import annotations

import base64
import os
from typing import Any, Dict, List, Optional, Tuple

from . import extract

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

# Rough rule of thumb; the *measured* count comes from Ollama's prompt_eval_count.
CHARS_PER_TOKEN = 4.0


def approx_tokens(chars: int) -> int:
    return round(chars / CHARS_PER_TOKEN)


def _resolve(path: str, base_dirs: List[str]) -> str:
    """Find a file: absolute as-is, else relative to each base dir in turn."""
    if os.path.isabs(path) and os.path.exists(path):
        return path
    for base in base_dirs:
        cand = os.path.join(base, path)
        if os.path.exists(cand):
            return cand
    # fall back to the raw path so the error message names what was requested
    return path


def encode_image(path: str) -> Tuple[str, int]:
    with open(path, "rb") as f:
        raw = f.read()
    return base64.b64encode(raw).decode("ascii"), len(raw)


def prepare(
    task_text: str,
    files: Optional[List[str]] = None,
    images: Optional[List[str]] = None,
    base_dirs: Optional[List[str]] = None,
    max_chars_per_file: Optional[int] = None,
) -> Tuple[str, List[str], Dict[str, Any]]:
    """Assemble the final prompt (task + injected document context) and the list
    of base64 images. Returns (prompt, images_b64, meta).

    `meta` records which files/images were attached and their sizes so the
    report can show what the model was given alongside the measured cost.
    """
    base_dirs = base_dirs or [os.getcwd()]
    files = files or []
    images = images or []

    file_meta: List[Dict[str, Any]] = []
    blocks: List[str] = []
    for ref in files:
        resolved = _resolve(ref, base_dirs)
        name = os.path.basename(resolved)
        try:
            ex = extract.extract_document(resolved)
        except FileNotFoundError:
            file_meta.append({"name": ref, "error": "not found"})
            continue
        except Exception as e:  # noqa: BLE001
            file_meta.append({"name": ref, "error": f"{type(e).__name__}: {e}"})
            continue
        content = ex.text
        truncated = False
        if max_chars_per_file is not None and len(content) > max_chars_per_file:
            content = content[:max_chars_per_file]
            truncated = True
        chars = len(content)
        file_meta.append({
            "name": name, "path": resolved, "chars": chars,
            "approx_tokens": approx_tokens(chars), "truncated": truncated,
            "method": ex.method, "warnings": ex.warnings,
        })
        blocks.append(
            f"===== REFERENCE FILE: {name} ({chars} chars"
            f"{', TRUNCATED' if truncated else ''}) =====\n"
            f"{content}\n"
            f"===== END {name} ====="
        )

    image_meta: List[Dict[str, Any]] = []
    images_b64: List[str] = []
    for ref in images:
        resolved = _resolve(ref, base_dirs)
        name = os.path.basename(resolved)
        ext = os.path.splitext(resolved)[1].lower()
        try:
            b64, nbytes = encode_image(resolved)
        except FileNotFoundError:
            image_meta.append({"name": ref, "error": "not found"})
            continue
        images_b64.append(b64)
        image_meta.append({
            "name": name, "path": resolved, "bytes": nbytes,
            "ext": ext.lstrip("."),
            # small enough to embed as a thumbnail in the report
            "data_uri": (f"data:image/{ext.lstrip('.') or 'png'};base64,{b64}"
                         if nbytes <= 1_500_000 else None),
        })

    # Build the final prompt.
    parts: List[str] = []
    if blocks:
        parts.append(
            "Use the following reference material to complete the task. "
            "Base your answer on it where relevant.\n"
        )
        parts.extend(blocks)
    if image_meta:
        names = ", ".join(m["name"] for m in image_meta)
        parts.append(f"[The following image(s) are attached: {names}]")
    parts.append(f"TASK:\n{task_text}" if (blocks or image_meta) else task_text)
    prompt = "\n\n".join(parts)

    meta = {
        "files": file_meta,
        "images": image_meta,
        "assembled_chars": len(prompt),
        "assembled_approx_tokens": approx_tokens(len(prompt)),
    }
    return prompt, images_b64, meta
