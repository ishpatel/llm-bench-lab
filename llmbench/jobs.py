"""A tiny single-worker job queue.

Benchmark runs must not overlap — two generations sharing the GPU would ruin
each other's timings — so every submitted run is processed FIFO by one worker
thread. Each job carries a live log buffer that the web UI polls for progress.
"""
from __future__ import annotations

import queue
import threading
import traceback
from collections import OrderedDict
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


# A finished job keeps its whole log and result (including the model's full
# answer) so the UI can still poll it. Nothing ever removed them, so a long
# campaign grew the registry for the life of the process. Far more than the UI
# will ever ask for again, but bounded.
MAX_FINISHED_JOBS = 200


class JobManager:
    def __init__(self, max_finished: int = MAX_FINISHED_JOBS) -> None:
        self._jobs: "OrderedDict[str, Job]" = OrderedDict()
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._fns: Dict[str, JobFn] = {}
        self._lock = threading.Lock()
        self._max_finished = max_finished
        self._worker = threading.Thread(target=self._run_loop, daemon=True)
        self._worker.start()

    def submit(self, job_id: str, fn: JobFn, kind: str = "run") -> Job:
        job = Job(job_id, kind=kind)
        with self._lock:
            self._jobs[job_id] = job
            self._fns[job_id] = fn
            self._evict_locked()
        self._queue.put(job_id)
        return job

    def _evict_locked(self) -> None:
        """Drop the oldest finished jobs. Queued and running ones are never
        evicted, however old: the UI is still polling those."""
        finished = [jid for jid, j in self._jobs.items()
                    if j.state in ("done", "error")]
        for jid in finished[:max(0, len(finished) - self._max_finished)]:
            self._jobs.pop(jid, None)

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
                # Evict here rather than only on submit: a job is not finished
                # at the moment it is queued, so submit-time eviction never saw
                # anything to drop.
                with self._lock:
                    self._fns.pop(job_id, None)
                    self._evict_locked()
