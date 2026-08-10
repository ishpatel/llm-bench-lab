"""A tiny single-worker job queue.

Benchmark runs must not overlap — two generations sharing the GPU would ruin
each other's timings — so every submitted run is processed FIFO by one worker
thread. Each job carries a live log buffer that the web UI polls for progress.
"""
from __future__ import annotations

import queue
import threading
import traceback
from typing import Any, Callable, Dict, List, Optional


class Job:
    def __init__(self, job_id: str, kind: str = "run"):
        self.id = job_id
        self.kind = kind
        self.state = "queued"          # queued | running | done | error
        self.log: List[str] = []
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self._lock = threading.Lock()

    def append_log(self, msg: str) -> None:
        with self._lock:
            self.log.append(msg)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "state": self.state,
                "log": list(self.log),
                "error": self.error,
                "result": self.result,
            }


# fn signature: fn(log: Callable[[str], None]) -> dict (the run meta)
JobFn = Callable[[Callable[[str], None]], Dict[str, Any]]


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._fns: Dict[str, JobFn] = {}
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run_loop, daemon=True)
        self._worker.start()

    def submit(self, job_id: str, fn: JobFn, kind: str = "run") -> Job:
        job = Job(job_id, kind=kind)
        with self._lock:
            self._jobs[job_id] = job
            self._fns[job_id] = fn
        self._queue.put(job_id)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def queue_depth(self) -> int:
        return self._queue.qsize()

    def _run_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            with self._lock:
                job = self._jobs.get(job_id)
                fn = self._fns.get(job_id)
            if job is None or fn is None:
                continue
            job.state = "running"
            try:
                job.result = fn(job.append_log)
                job.state = "done"
            except Exception as e:  # noqa: BLE001 - report, never crash the worker
                job.error = f"{type(e).__name__}: {e}"
                job.append_log("ERROR: " + job.error)
                job.append_log(traceback.format_exc())
                job.state = "error"
            finally:
                with self._lock:
                    self._fns.pop(job_id, None)
