"""Local web server for interactive benchmarking.

Stdlib-only (http.server). Binds to localhost. Serves a single-page UI and a
small JSON API that drives the existing Runner/attachments/extract stack:

    GET  /                      the UI
    GET  /api/models            locally installed Ollama models + system info
    POST /api/runs              start a benchmark run -> {id}; runs in the queue
    GET  /api/jobs/<id>         live job state + progress log (UI polls this)
    GET  /api/runs              all saved runs (summaries) for the master view
    GET  /api/runs/<id>         one run (full meta incl. output)
    GET  /api/runs/<id>/report  self-contained HTML report for that run
    DELETE /api/runs/<id>       delete a run
"""
from __future__ import annotations

import base64
import datetime
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from . import rag as rag_mod
from . import report as report_mod
from . import telemetry
from .jobs import JobManager
from .ollama import OllamaClient
from .rag import RagStore
from .runner import Runner
from .store import RunStore, _safe_name

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(HERE, "web")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
CTYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript",
          ".css": "text/css", ".json": "application/json", ".png": "image/png",
          ".svg": "image/svg+xml", ".ico": "image/x-icon"}


# --------------------------------------------------------------------------
# Turning a runner result into a compact run summary
# --------------------------------------------------------------------------
def _median(agg: Dict[str, Any], metric: str) -> Optional[float]:
    m = (agg or {}).get(metric)
    return m.get("median") if isinstance(m, dict) else None


def summarize(cell: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    agg = cell.get("aggregate", {})
    runs = cell.get("runs", [])
    first = runs[0] if runs else {}
    metrics = {k: _median(agg, k) for k in (
        "gen_tps", "prompt_tps", "ttft_ms", "load_ms", "wall_total_ms")}
    metrics["prompt_tokens"] = first.get("prompt_tokens")
    metrics["output_tokens"] = first.get("output_tokens")
    metrics["residency"] = cell.get("residency", "")
    cold = cell.get("cold_start") or {}
    metrics["cold_load_ms"] = cold.get("load_ms")
    return {
        "metrics": metrics,
        "output": first.get("response_text", ""),
        "thinking_chars": first.get("thinking_chars", 0),
        "attachments": cell.get("attachments", {}),
    }


class BenchServer:
    """Holds shared state and builds the job function for a submitted run."""

    def __init__(self, base_url: str, project_root: str):
        self.base_url = base_url
        self.client = OllamaClient(base_url=base_url)
        self.store = RunStore(os.path.join(project_root, "runs"))
        self.rag = RagStore(os.path.join(project_root, "kb"))
        self.jobs = JobManager()
        self._seq = 0
        self._seq_lock = threading.Lock()

    def new_id(self) -> str:
        with self._seq_lock:
            self._seq += 1
            seq = self._seq
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        return self.store.make_id(stamp, seq)

    def start_run(self, payload: Dict[str, Any]) -> str:
        run_id = self.new_id()
        # Persist a placeholder immediately so the run shows up as "running".
        created = datetime.datetime.now().isoformat(timespec="seconds")
        placeholder = {
            "id": run_id, "created": created, "state": "running",
            "name": payload.get("name") or "",
            "model": payload.get("model", ""),
            "prompt": {"text": payload.get("prompt", ""), "files": [], "images": []},
        }
        self.store.save(run_id, placeholder)
        self.jobs.submit(run_id, self._make_job_fn(run_id, payload, created))
        return run_id

    def _make_job_fn(self, run_id: str, payload: Dict[str, Any], created: str):
        server = self

        def fn(log):
            # 1. Save attachments to the run folder, split into docs vs images.
            file_names: List[str] = []
            image_names: List[str] = []
            for att in payload.get("attachments", []):
                path = server.store.save_attachment_b64(
                    run_id, att.get("filename", "file"), att.get("b64", ""))
                base = os.path.basename(path)
                ext = os.path.splitext(base)[1].lower()
                (image_names if ext in IMAGE_EXTS else file_names).append(base)
            if file_names or image_names:
                log(f"attachments saved: {len(file_names)} doc(s), "
                    f"{len(image_names)} image(s)")

            # 2. Build an in-memory config and reuse the Runner.
            opts: Dict[str, Any] = {
                "temperature": payload.get("temperature", 0),
                "num_predict": payload.get("num_predict", 256),
            }
            if payload.get("think") is not None:
                opts["think"] = bool(payload["think"])
            num_ctx = payload.get("num_ctx")
            cfg = {
                "name": payload.get("name") or "run",
                "system_label": payload.get("system_label") or None,
                "base_url": server.base_url,
                "runs": int(payload.get("runs", 3)),
                "warmup": int(payload.get("warmup", 1)),
                "measure_cold_start": bool(payload.get("cold_start", True)),
                "options": opts,
                "models": [payload["model"]],
                "context_lengths": [int(num_ctx)] if num_ctx else [None],
                "max_chars_per_file": payload.get("max_chars_per_file") or None,
                "prompts": [{
                    "key": "prompt",
                    "text": payload.get("prompt", ""),
                    "files": file_names,
                    "images": image_names,
                }],
            }
            attach_dir = server.store.attachments_dir(run_id)
            runner = Runner(cfg, {}, log=log, base_dirs=[attach_dir])
            result = runner.run()

            cells = result.get("cells", [])
            if not cells:
                raise RuntimeError(
                    f"No result produced — is '{payload.get('model')}' installed?")
            cell = cells[0]
            summary = summarize(cell, result["meta"])
            meta = {
                "id": run_id,
                "created": created,
                "state": "done",
                "name": payload.get("name") or "",
                "model": payload["model"],
                "system": result["meta"]["system"],
                "options": opts,
                "num_ctx": num_ctx,
                "runs": cfg["runs"],
                "warmup": cfg["warmup"],
                "measure_cold_start": cfg["measure_cold_start"],
                "prompt": {
                    "text": payload.get("prompt", ""),
                    "files": file_names,
                    "images": image_names,
                },
                "metrics": summary["metrics"],
                "attachments": summary["attachments"],
                "output": summary["output"],
                "thinking_chars": summary["thinking_chars"],
                "ollama_version": result["meta"].get("ollama_version", ""),
            }
            server.store.save(run_id, meta, result)
            log("done.")
            return meta

        return fn

    # -- RAG: knowledge-base build (queued) + ask (synchronous) ------------
    def start_kb_build(self, payload: Dict[str, Any]) -> str:
        name = payload["name"]
        job_id = "kb-" + self.new_id()
        self.jobs.submit(job_id, self._kb_job(payload), kind="kb")
        return job_id

    def _kb_job(self, payload: Dict[str, Any]):
        server = self

        def fn(log):
            name = payload["name"]
            embed_model = payload["embed_model"]
            srcdir = os.path.join(server.rag.kb_dir(name), "sources")
            os.makedirs(srcdir, exist_ok=True)
            docs = []
            for att in payload.get("attachments", []):
                raw = base64.b64decode(att.get("b64", ""))
                fname = _safe_name(att.get("filename", "file"))
                path = os.path.join(srcdir, fname)
                with open(path, "wb") as f:
                    f.write(raw)
                docs.append((fname, path))
            log(f"ingesting {len(docs)} document(s) with {embed_model}…")
            kb = rag_mod.build_kb(
                name, docs, embed_model, server.client, log=log,
                chunk_size=int(payload.get("chunk_size", 900)),
                overlap=int(payload.get("overlap", 150)))
            server.rag.save(kb)
            log("knowledge base ready.")
            return server.rag.summary(name)

        return fn

    def ask_kb(self, name: str, question: str, model: str,
               k: int, options: Dict[str, Any]) -> Dict[str, Any]:
        kb = self.rag.load(name)
        if kb is None:
            return {"error": f"knowledge base '{name}' not found"}
        er = self.client.embed(kb["embed_model"], [question])
        qvec = (er.get("embeddings") or [[]])[0]
        t = time.perf_counter()
        hits = rag_mod.retrieve(kb, qvec, k=k)
        retrieve_ms = (time.perf_counter() - t) * 1000.0
        prompt = rag_mod.build_grounded_prompt(question, hits)
        gen = self.client.generate(model, prompt, options=options)
        return {
            "question": question,
            "answer": gen.response_text,
            "sources": hits,
            "model": model,
            "embed_model": kb["embed_model"],
            "timing": {
                "embed_ms": round(er.get("wall_ms", 0.0), 1),
                "retrieve_ms": round(retrieve_ms, 2),
                "ttft_ms": gen.ttft_ms,
                "gen_tps": gen.gen_tps,
                "wall_total_ms": gen.wall_total_ms,
            },
            "prompt_tokens": gen.prompt_tokens,
            "output_tokens": gen.output_tokens,
        }


class Handler(BaseHTTPRequestHandler):
    server_app: BenchServer = None  # set by make_handler

    # -- helpers -----------------------------------------------------------
    def _json(self, obj: Any, code: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _bytes(self, data: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> Dict[str, Any]:
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b"{}"
        return json.loads(raw.decode("utf-8"))

    def log_message(self, *args) -> None:  # silence default logging
        pass

    # -- routing -----------------------------------------------------------
    def do_GET(self) -> None:
        app = self.server_app
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path.startswith("/static/"):
                return self._static(path[len("/static/"):])
            if path == "/api/models":
                up = app.client.is_up()
                return self._json({
                    "ok": up,
                    "models": app.client.models_detailed() if up else [],
                    "system": telemetry.describe_system(),
                    "base_url": app.base_url,
                })
            if path == "/api/runs":
                runs = app.store.list()
                for r in runs:  # keep the list light
                    r.pop("output", None)
                return self._json({"runs": runs})
            if path == "/api/cross-report":
                qs = parse_qs(urlparse(self.path).query)
                ids = [i for i in (qs.get("ids", [""])[0].split(",")) if i]
                return self._cross_report(ids)
            if path == "/api/kb":
                return self._json({"kb": app.rag.list()})
            m = _match(path, "/api/runs/", "/report")
            if m is not None:
                return self._run_report(m)
            m = _match(path, "/api/runs/", "/export")
            if m is not None:
                return self._export(m)
            if path.startswith("/api/runs/"):
                return self._json_or_404(app.store.get(path[len("/api/runs/"):]))
            if path.startswith("/api/jobs/"):
                job = app.jobs.get(path[len("/api/jobs/"):])
                return self._json_or_404(job.snapshot() if job else None)
            self._json({"error": "not found"}, 404)
        except BrokenPipeError:
            pass
        except Exception as e:  # noqa: BLE001
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self) -> None:
        app = self.server_app
        path = urlparse(self.path).path
        try:
            if path == "/api/runs":
                payload = self._read_json()
                if not payload.get("model"):
                    return self._json({"error": "model is required"}, 400)
                if not (payload.get("prompt") or payload.get("attachments")):
                    return self._json({"error": "prompt or attachment required"}, 400)
                run_id = app.start_run(payload)
                return self._json({"id": run_id})
            if path == "/api/import":
                bundle = self._read_json()
                if bundle.get("format") != "llmbench.run.v1":
                    return self._json({"error": "not a run bundle"}, 400)
                rid = app.store.import_bundle(bundle)
                return self._json({"id": rid})
            if path == "/api/kb":
                payload = self._read_json()
                if not payload.get("name"):
                    return self._json({"error": "name is required"}, 400)
                if not payload.get("embed_model"):
                    return self._json({"error": "embed_model is required"}, 400)
                if not payload.get("attachments"):
                    return self._json({"error": "at least one document is required"}, 400)
                return self._json({"job_id": app.start_kb_build(payload)})
            m = _match(path, "/api/kb/", "/ask")
            if m is not None:
                payload = self._read_json()
                opts = {
                    "temperature": payload.get("temperature", 0),
                    "num_predict": payload.get("num_predict", 300),
                    "think": bool(payload.get("think", False)),
                }
                res = app.ask_kb(m, payload.get("question", ""),
                                 payload.get("model", ""),
                                 int(payload.get("k", 4)), opts)
                return self._json(res, 400 if res.get("error") else 200)
            self._json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_DELETE(self) -> None:
        app = self.server_app
        path = urlparse(self.path).path
        if path.startswith("/api/runs/"):
            ok = app.store.delete(path[len("/api/runs/"):])
            return self._json({"ok": ok}, 200 if ok else 404)
        if path.startswith("/api/kb/"):
            ok = app.rag.delete(path[len("/api/kb/"):])
            return self._json({"ok": ok}, 200 if ok else 404)
        self._json({"error": "not found"}, 404)

    # -- endpoint bodies ---------------------------------------------------
    def _static(self, rel: str) -> None:
        rel = rel.replace("..", "")
        full = os.path.join(WEB_DIR, rel)
        if not os.path.isfile(full):
            return self._json({"error": "not found"}, 404)
        with open(full, "rb") as f:
            data = f.read()
        ext = os.path.splitext(full)[1].lower()
        self._bytes(data, CTYPES.get(ext, "application/octet-stream"))

    def _json_or_404(self, obj: Any) -> None:
        if obj is None:
            return self._json({"error": "not found"}, 404)
        self._json(obj)

    def _run_report(self, run_id: str) -> None:
        result = self.server_app.store.get_result(run_id)
        if result is None:
            return self._json({"error": "no result yet"}, 404)
        html = report_mod.build_report([result], title=f"Run {run_id}")
        self._bytes(html.encode("utf-8"), CTYPES[".html"])

    def _export(self, run_id: str) -> None:
        bundle = self.server_app.store.export_bundle(run_id)
        if bundle is None:
            return self._json({"error": "not found"}, 404)
        body = json.dumps(bundle).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Disposition",
                         f'attachment; filename="{run_id}.llmbench.json"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _cross_report(self, ids: List[str]) -> None:
        """Merge several runs' results into one grouped-by-system report."""
        results = []
        for rid in ids:
            r = self.server_app.store.get_result(rid)
            if r is not None:
                results.append(r)
        if not results:
            return self._json({"error": "no results for given ids"}, 404)
        html = report_mod.build_report(
            results, title="Cross-system comparison")
        self._bytes(html.encode("utf-8"), CTYPES[".html"])


def _match(path: str, prefix: str, suffix: str) -> Optional[str]:
    if path.startswith(prefix) and path.endswith(suffix):
        return path[len(prefix):-len(suffix)]
    return None


def make_handler(app: BenchServer):
    return type("BoundHandler", (Handler,), {"server_app": app})


def serve(host: str, port: int, base_url: str, project_root: str) -> None:
    app = BenchServer(base_url=base_url, project_root=project_root)
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    url = f"http://{host}:{port}"
    print(f"llmbench web UI → {url}")
    print(f"Ollama backend  → {base_url}  "
          f"({'reachable' if app.client.is_up() else 'NOT reachable'})")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping…")
        httpd.shutdown()
