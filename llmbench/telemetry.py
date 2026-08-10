"""Platform detection + GPU/memory telemetry.

Cross-platform and best-effort: on NVIDIA systems it shells out to `nvidia-smi`
for utilization / VRAM / power / temperature and can sample them on a
background thread during a generation. On Apple Silicon there is no per-process
VRAM counter without elevated `powermetrics`, so we report what is freely
available (unified-memory totals, chip name) and Ollama's own residency view.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
# Deep system breakdown — CPU / GPU / NPU / memory / environment
# --------------------------------------------------------------------------
_DEEP_CACHE: Optional[Dict[str, Any]] = None


def _to_int(s: Optional[str]) -> Optional[int]:
    try:
        return int(str(s).strip())
    except (TypeError, ValueError):
        return None


def describe_system_deep() -> Dict[str, Any]:
    """Structured hardware + environment description for the UI's system panel.
    Cached after the first call (system_profiler is slow)."""
    global _DEEP_CACHE
    if _DEEP_CACHE is not None:
        return _DEEP_CACHE

    sysname = platform.system()
    cpu: Dict[str, Any] = {}
    gpu: Dict[str, Any] = {}
    npu: Dict[str, Any] = {}
    mem: Dict[str, Any] = {}
    env: Dict[str, Any] = {
        "arch": platform.machine(),
        "python": platform.python_version(),
    }
    ov = _run(["ollama", "--version"]) or ""
    mv = re.search(r"(\d+\.\d+[\.\d]*)", ov)
    env["ollama"] = mv.group(1) if mv else (ov or "unknown")

    if sysname == "Darwin":
        prod = _run(["sw_vers", "-productVersion"]) or platform.release()
        env["os"] = f"macOS {prod}"
        env["backend"] = "Apple Metal"

        cpu["name"] = (_run(["sysctl", "-n", "machdep.cpu.brand_string"])
                       or "Apple Silicon")
        cpu["cores_total"] = _to_int(_run(["sysctl", "-n", "hw.ncpu"]))
        cpu["cores_performance"] = _to_int(
            _run(["sysctl", "-n", "hw.perflevel0.physicalcpu"]))
        cpu["cores_efficiency"] = _to_int(
            _run(["sysctl", "-n", "hw.perflevel1.physicalcpu"]))

        memsize = _run(["sysctl", "-n", "hw.memsize"])
        if memsize and memsize.isdigit():
            mem["total_gb"] = round(int(memsize) / 2**30)
        mem["type"] = "Unified — one pool shared by CPU, GPU and Neural Engine"

        sp = _run(["system_profiler", "SPDisplaysDataType"], timeout=25) or ""
        mg = re.search(r"Chipset Model:\s*(.+)", sp)
        gpu["name"] = mg.group(1).strip() if mg else cpu["name"]
        mc = re.search(r"Total Number of Cores:\s*(\d+)", sp)
        if mc:
            gpu["cores"] = int(mc.group(1))
        mm = re.search(r"Metal Support:\s*(.+)", sp)
        gpu["api"] = mm.group(1).strip() if mm else "Metal"
        if mem.get("total_gb"):
            gpu["memory"] = f"shares the {mem['total_gb']} GB unified pool"

        if env["arch"] == "arm64":
            npu = {"name": "Apple Neural Engine", "status": "present",
                   "note": "Not used for these benchmarks — Ollama runs LLM "
                           "inference on the GPU via Metal."}
        else:
            npu = {"name": None, "status": "none detected"}
    else:
        env["os"] = f"{sysname} {platform.release()}"
        cpu["name"] = platform.processor() or platform.machine()
        cpu["cores_total"] = os.cpu_count()

        if has_nvidia_smi():
            env["backend"] = "NVIDIA CUDA"
            raw = _run(["nvidia-smi",
                        "--query-gpu=name,memory.total,driver_version",
                        "--format=csv,noheader,nounits"])
            if raw:
                parts = [p.strip() for p in raw.splitlines()[0].split(",")]
                if len(parts) >= 3:
                    gpu["name"] = parts[0]
                    vram = _to_int(parts[1])
                    if vram:
                        gpu["vram_gb"] = round(vram / 1024, 1)
                        gpu["memory"] = (f"{gpu['vram_gb']} GB dedicated VRAM "
                                         f"(separate from system RAM)")
                    gpu["driver"] = parts[2]
            head = _run(["nvidia-smi"]) or ""
            mcu = re.search(r"CUDA Version:\s*([\d.]+)", head)
            gpu["api"] = f"CUDA {mcu.group(1)}" if mcu else "CUDA"
        else:
            env["backend"] = "CPU only (no NVIDIA GPU detected)"

        if sysname == "Linux":
            try:
                with open("/proc/meminfo", "r", encoding="utf-8") as f:
                    mt = re.search(r"MemTotal:\s*(\d+)\s*kB", f.read())
                if mt:
                    mem["total_gb"] = round(int(mt.group(1)) / 2**20)
            except Exception:
                pass
        elif sysname == "Windows":
            try:
                import ctypes

                class _MEMSTAT(ctypes.Structure):
                    _fields_ = [("dwLength", ctypes.c_ulong),
                                ("dwMemoryLoad", ctypes.c_ulong),
                                ("ullTotalPhys", ctypes.c_ulonglong),
                                ("ullAvailPhys", ctypes.c_ulonglong),
                                ("ullTotalPageFile", ctypes.c_ulonglong),
                                ("ullAvailPageFile", ctypes.c_ulonglong),
                                ("ullTotalVirtual", ctypes.c_ulonglong),
                                ("ullAvailVirtual", ctypes.c_ulonglong),
                                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

                st = _MEMSTAT()
                st.dwLength = ctypes.sizeof(_MEMSTAT)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
                mem["total_gb"] = round(st.ullTotalPhys / 2**30)
            except Exception:
                pass
        mem["type"] = ("System RAM — separate from the GPU's VRAM"
                       if gpu.get("vram_gb") else "System RAM")
        npu = {"name": None, "status": "not detected",
               "note": "Dedicated NPUs (on AI PCs) are not used by Ollama."}

    _DEEP_CACHE = {"cpu": cpu, "gpu": gpu, "npu": npu, "memory": mem,
                   "env": env}
    return _DEEP_CACHE


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
