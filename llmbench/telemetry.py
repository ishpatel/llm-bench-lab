"""Platform detection + GPU/memory telemetry.

Cross-platform and best-effort: on NVIDIA systems it shells out to `nvidia-smi`
for utilization / VRAM / power / temperature and can sample them on a
background thread during a generation. On Apple Silicon there is no per-process
VRAM counter without elevated `powermetrics`, so we report what is freely
available (unified-memory totals, chip name) and Ollama's own residency view.
"""
from __future__ import annotations

import platform
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _run(cmd: List[str], timeout: float = 15.0) -> Optional[str]:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def has_nvidia_smi() -> bool:
    return shutil.which("nvidia-smi") is not None


# --------------------------------------------------------------------------
# Static system description (gathered once per run)
# --------------------------------------------------------------------------
_BASE_CACHE: Optional[Dict[str, str]] = None


def describe_system(label_override: Optional[str] = None) -> Dict[str, str]:
    global _BASE_CACHE
    if _BASE_CACHE is not None:
        info = dict(_BASE_CACHE)
        info["label"] = label_override or info.get(
            "gpu", info.get("machine", info.get("os", "system")))
        return info
    info: Dict[str, str] = {
        "os": platform.system(),
        "os_version": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "accelerator": "unknown",
    }
    sysname = platform.system()

    if has_nvidia_smi():
        raw = _run([
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ])
        if raw:
            parts = [p.strip() for p in raw.splitlines()[0].split(",")]
            if len(parts) >= 3:
                info["gpu"] = parts[0]
                info["vram_mb"] = parts[1]
                info["driver"] = parts[2]
                info["accelerator"] = "nvidia-cuda"

    if sysname == "Darwin":
        info["accelerator"] = "apple-metal"
        chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        # On Apple Silicon the "brand string" is the CPU; the chip name comes
        # from the hardware model / SPHardwareDataType.
        hw = _run(["sysctl", "-n", "hw.model"])
        if hw:
            info["hw_model"] = hw
        mem = _run(["sysctl", "-n", "hw.memsize"])
        if mem and mem.isdigit():
            info["unified_memory_mb"] = str(int(mem) // (1024 * 1024))
        # Chip marketing name (e.g. "Apple M3 Max") via system_profiler.
        sp = _run(["system_profiler", "SPHardwareDataType"], timeout=20)
        if sp:
            m = re.search(r"Chip:\s*(.+)", sp)
            if m:
                info["gpu"] = m.group(1).strip() + " (unified)"
        if "gpu" not in info and chip:
            info["gpu"] = chip

    # Cache the expensive parts (system_profiler / nvidia-smi) without the label.
    base = dict(info)
    base.pop("label", None)
    _BASE_CACHE = base

    if label_override:
        info["label"] = label_override
    else:
        info["label"] = info.get("gpu", info.get("machine", sysname))
    return info


# --------------------------------------------------------------------------
# Ollama residency (`ollama ps`) — where is the model actually running?
# --------------------------------------------------------------------------
def ollama_ps() -> Dict[str, Dict[str, str]]:
    """Parse `ollama ps` into {model: {size, processor}}. The PROCESSOR column
    reveals GPU vs CPU split (e.g. "100% GPU" or "35%/65% CPU/GPU")."""
    out = _run(["ollama", "ps"])
    result: Dict[str, Dict[str, str]] = {}
    if not out:
        return result
    lines = out.splitlines()
    if len(lines) < 2:
        return result
    header = lines[0]
    # Column positions are whitespace-aligned; locate by header keywords.
    try:
        proc_idx = header.index("PROCESSOR")
    except ValueError:
        proc_idx = None
    for line in lines[1:]:
        if not line.strip():
            continue
        name = line.split()[0]
        processor = ""
        if proc_idx is not None and len(line) > proc_idx:
            # Take from PROCESSOR column to the next column boundary if any.
            processor = line[proc_idx:].strip()
            # trim a trailing UNTIL/context column if present
            processor = re.split(r"\s{2,}", processor)[0].strip()
        result[name] = {"processor": processor}
    return result


# --------------------------------------------------------------------------
# Live GPU sampler (NVIDIA only) — captures peaks during a generation.
# --------------------------------------------------------------------------
@dataclass
class SampleStats:
    available: bool = False
    samples: int = 0
    util_peak: Optional[float] = None
    util_avg: Optional[float] = None
    vram_used_peak_mb: Optional[float] = None
    power_peak_w: Optional[float] = None
    power_avg_w: Optional[float] = None
    temp_peak_c: Optional[float] = None


class GpuSampler:
    """Background poller for `nvidia-smi`. No-op on non-NVIDIA systems so the
    same code path runs everywhere."""

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self.enabled = has_nvidia_smi()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._util: List[float] = []
        self._vram: List[float] = []
        self._power: List[float] = []
        self._temp: List[float] = []

    def _poll_once(self) -> None:
        raw = _run([
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ], timeout=5)
        if not raw:
            return
        parts = [p.strip() for p in raw.splitlines()[0].split(",")]
        try:
            self._util.append(float(parts[0]))
            self._vram.append(float(parts[1]))
            self._power.append(float(parts[2]))
            self._temp.append(float(parts[3]))
        except (ValueError, IndexError):
            pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._poll_once()
            self._stop.wait(self.interval)

    def start(self) -> None:
        if not self.enabled:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> SampleStats:
        if not self.enabled:
            return SampleStats(available=False)
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

        def peak(xs: List[float]) -> Optional[float]:
            return max(xs) if xs else None

        def avg(xs: List[float]) -> Optional[float]:
            return (sum(xs) / len(xs)) if xs else None

        return SampleStats(
            available=True,
            samples=len(self._util),
            util_peak=peak(self._util),
            util_avg=avg(self._util),
            vram_used_peak_mb=peak(self._vram),
            power_peak_w=peak(self._power),
            power_avg_w=avg(self._power),
            temp_peak_c=peak(self._temp),
        )
