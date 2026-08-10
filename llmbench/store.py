"""Persistence for benchmark runs.

Each run is a self-contained folder under `runs/`:

    runs/<id>/
        meta.json          summary: model, prompt, options, metrics, output, system
        result.json        full runner output (all repeats, telemetry)
        attachments/       the exact files/images submitted with the run

The master view is just a scan of `meta.json` files, so runs survive restarts
and are trivially portable (copy the folder between machines to merge history).
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
from typing import Any, Dict, List, Optional


def _safe_name(name: str) -> str:
    """Reduce an arbitrary upload filename to a safe basename."""
    name = os.path.basename(name or "file")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "file"
    return name[:120]


class RunStore:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    # -- paths -------------------------------------------------------------
    def run_dir(self, run_id: str) -> str:
        return os.path.join(self.root, run_id)

    def attachments_dir(self, run_id: str) -> str:
        d = os.path.join(self.run_dir(run_id), "attachments")
        os.makedirs(d, exist_ok=True)
        return d

    def make_id(self, stamp: str, seq: int) -> str:
        # stamp is passed in by the caller (server has real time); seq avoids
        # collisions within the same second.
        return f"{stamp}-{seq:03d}"

    # -- writing -----------------------------------------------------------
    def save_attachment_b64(self, run_id: str, filename: str, b64: str) -> str:
        raw = base64.b64decode(b64)
        safe = _safe_name(filename)
        path = os.path.join(self.attachments_dir(run_id), safe)
        # avoid clobbering same-named files
        if os.path.exists(path):
            stem, ext = os.path.splitext(safe)
            i = 2
            while os.path.exists(path):
                path = os.path.join(self.attachments_dir(run_id), f"{stem}_{i}{ext}")
                i += 1
        with open(path, "wb") as f:
            f.write(raw)
        return path

    def save(self, run_id: str, meta: Dict[str, Any],
             result: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(self.run_dir(run_id), exist_ok=True)
        with open(os.path.join(self.run_dir(run_id), "meta.json"), "w",
                  encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        if result is not None:
            with open(os.path.join(self.run_dir(run_id), "result.json"), "w",
                      encoding="utf-8") as f:
                json.dump(result, f, indent=2)

    # -- reading -----------------------------------------------------------
    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        p = os.path.join(self.run_dir(run_id), "meta.json")
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_result(self, run_id: str) -> Optional[Dict[str, Any]]:
        p = os.path.join(self.run_dir(run_id), "result.json")
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def list(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for name in os.listdir(self.root):
            d = os.path.join(self.root, name)
            if not os.path.isdir(d):
                continue
            meta = self.get(name)
            if meta:
                out.append(meta)
        # newest first by id (timestamp-prefixed) then created field
        out.sort(key=lambda m: m.get("created", m.get("id", "")), reverse=True)
        return out

    def delete(self, run_id: str) -> bool:
        d = self.run_dir(run_id)
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
            return True
        return False

    # -- portability: move runs between machines ---------------------------
    def _unique_id(self, base: str) -> str:
        rid, i = base, 2
        while os.path.isdir(self.run_dir(rid)):
            rid = f"{base}-{i}"
            i += 1
        return rid

    def export_bundle(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Self-contained JSON bundle (meta + result + attachments) so a run
        can be carried from one machine to another and re-imported."""
        meta = self.get(run_id)
        if meta is None:
            return None
        atts: List[Dict[str, str]] = []
        adir = os.path.join(self.run_dir(run_id), "attachments")
        if os.path.isdir(adir):
            for name in sorted(os.listdir(adir)):
                p = os.path.join(adir, name)
                if os.path.isfile(p):
                    with open(p, "rb") as f:
                        atts.append({"name": name,
                                     "b64": base64.b64encode(f.read()).decode("ascii")})
        return {
            "format": "llmbench.run.v1",
            "meta": meta,
            "result": self.get_result(run_id),
            "attachments": atts,
        }

    def import_bundle(self, bundle: Dict[str, Any]) -> str:
        """Recreate a run folder from an exported bundle. Assigns a fresh id if
        the original collides with an existing run."""
        meta = dict(bundle.get("meta") or {})
        base = meta.get("id") or "imported"
        run_id = self._unique_id(base)
        meta["id"] = run_id
        for a in bundle.get("attachments") or []:
            self.save_attachment_b64(run_id, a.get("name", "file"), a.get("b64", ""))
        self.save(run_id, meta, bundle.get("result"))
        return run_id
