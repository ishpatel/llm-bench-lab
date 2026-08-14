"""Running the remediation commands that the readiness checks suggest.

The web UI can offer a "Run" button next to a failing check. That means a page
in a browser can cause a command to execute on this machine, so the rules are
deliberately narrow:

1. The browser sends a check *key*, never a command string. The command that
   runs is the one `readiness.py` generated for the current state of this
   machine, looked up server-side.
2. Only keys in `readiness.RUNNABLE` are eligible. Anything needing sudo, an
   administrator prompt, a package manager or a download is copy-only and stays
   the user's decision to run in their own shell.
3. No shell. The command is split into an argv list and executed directly, so
   there is nothing for a metacharacter to do.

`server.py` additionally requires a same-origin request, which is what stops
another website from POSTing here while the user has llmbench open.
"""
from __future__ import annotations

import shlex
import subprocess
import sys
import threading
from collections import OrderedDict, deque
from typing import Any, Deque, Dict, Optional

MAX_OUTPUT_LINES = 300
MAX_KEPT = 40


class FixRun:
    def __init__(self, run_id: str, key: str, cmd: str):
        self.id = run_id
        self.key = key
        self.cmd = cmd
        self.state = "running"          # running | done | error
        self.exit_code: Optional[int] = None
        self.error: Optional[str] = None
        self.lines: Deque[str] = deque(maxlen=MAX_OUTPUT_LINES)
        self._lock = threading.Lock()

    def append(self, line: str) -> None:
        with self._lock:
            self.lines.append(line.rstrip())

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {"id": self.id, "key": self.key, "cmd": self.cmd,
                    "state": self.state, "exit_code": self.exit_code,
                    "error": self.error, "output": list(self.lines)}


class FixRunner:
    def __init__(self) -> None:
        self._runs: "OrderedDict[str, FixRun]" = OrderedDict()
        self._lock = threading.Lock()
        self._seq = 0

    def get(self, run_id: str) -> Optional[FixRun]:
        with self._lock:
            return self._runs.get(run_id)

    def start(self, key: str, cmd: str) -> FixRun:
        """Launch `cmd` detached and stream its output into the returned run."""
        with self._lock:
            self._seq += 1
            run = FixRun(f"fix{self._seq}", key, cmd)
            self._runs[run.id] = run
            while len(self._runs) > MAX_KEPT:
                self._runs.popitem(last=False)

        argv = shlex.split(cmd, posix=(sys.platform != "win32"))
        if not argv:
            run.state, run.error = "error", "empty command"
            return run

        # A new session/process group means a server the user starts this way
        # outlives llmbench itself, which is what someone pressing "Run" on
        # `ollama serve` expects: they are starting Ollama, not lending it ours.
        kwargs: Dict[str, Any] = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kwargs["start_new_session"] = True

        try:
            proc = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, **kwargs)
        except OSError as exc:
            run.state = "error"
            run.error = f"could not start {argv[0]}: {exc.strerror or exc}"
            run.append(run.error)
            return run

        threading.Thread(target=self._pump, args=(run, proc), daemon=True).start()
        return run

    @staticmethod
    def _pump(run: FixRun, proc: "subprocess.Popen[str]") -> None:
        try:
            if proc.stdout is not None:
                for line in proc.stdout:
                    run.append(line)
            proc.wait()
            run.exit_code = proc.returncode
            # A long-running server (`ollama serve`) only reaches here when it
            # exits, so a non-zero code here is a real failure either way.
            run.state = "done" if proc.returncode == 0 else "error"
            if proc.returncode:
                run.error = f"exited with code {proc.returncode}"
        except Exception as exc:  # noqa: BLE001 - report, never kill the thread
            run.state = "error"
            run.error = f"{type(exc).__name__}: {exc}"
            run.append(run.error)
