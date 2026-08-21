"""Platform detection + GPU/memory telemetry for the hardware people have.

Vendor coverage, stated honestly:
- Apple (any Apple Silicon, and Intel Macs): chip/GPU via sysctl and
  system_profiler. No per-process VRAM counter exists without elevated
  `powermetrics`, so residency comes from the engine's own view.
- NVIDIA: `nvidia-smi` for name/VRAM/driver plus live utilization, VRAM,
  power and temperature sampling during a generation.
- AMD: named via PCI ids (`/sys/class/drm`, lspci) or Windows CIM; live
  sampling through `rocm-smi --json` where ROCm is installed (Linux).
- Intel GPUs: named the same ways; no live sampler is claimed, because none
  is reliably present across driver stacks.

Everything is best-effort and degrades to naming less rather than guessing
more: a machine whose GPU cannot be identified is reported that way.
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


def has_rocm_smi() -> bool:
    return shutil.which("rocm-smi") is not None


# PCI vendor ids, the one naming authority that needs no vendor tooling.
_PCI_VENDORS = {"0x10de": "NVIDIA", "0x1002": "AMD", "0x8086": "Intel"}


def _drm_gpus() -> List[Dict[str, str]]:
    """Linux: GPU vendors straight from /sys/class/drm, no tools required."""
    out: List[Dict[str, str]] = []
    try:
        import glob
        for vf in sorted(glob.glob("/sys/class/drm/card[0-9]/device/vendor")):
            try:
                with open(vf, "r", encoding="utf-8") as f:
                    vendor = _PCI_VENDORS.get(f.read().strip().lower())
                if vendor:
                    out.append({"vendor": vendor})
            except OSError:
                continue
    except Exception:
        pass
    return out


def _lspci_gpu_name(vendor: str) -> Optional[str]:
    """Best-effort model name for a vendor from lspci, if lspci exists."""
    raw = _run(["lspci"], timeout=5) or ""
    for line in raw.splitlines():
        if re.search(r"VGA|3D|Display", line) and vendor.lower() in line.lower():
            return re.sub(r"^.*?: ", "", line).strip()
    return None


def _windows_gpus() -> List[Dict[str, Any]]:
    """Windows: name every video controller via CIM. PowerShell ships on every
    supported Windows, so this needs no vendor tooling either."""
    raw = _run(["powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_VideoController | "
                "Select-Object Name,AdapterRAM | ConvertTo-Json"], timeout=20)
    return _parse_windows_gpus(raw)


def _parse_windows_gpus(raw: Optional[str]) -> List[Dict[str, Any]]:
    if not raw:
        return []
    import json as _json
    try:
        data = _json.loads(raw)
    except ValueError:
        return []
    if isinstance(data, dict):
        data = [data]
    out: List[Dict[str, Any]] = []
    for item in data if isinstance(data, list) else []:
        name = str(item.get("Name") or "").strip()
        if not name:
            continue
        vendor = ("NVIDIA" if "nvidia" in name.lower() else
                  "AMD" if re.search(r"amd|radeon", name.lower()) else
                  "Intel" if "intel" in name.lower() else "unknown")
        gpu: Dict[str, Any] = {"vendor": vendor, "name": name}
        ram = item.get("AdapterRAM")
        # AdapterRAM is a 32-bit field: cards with more than 4 GB report a
        # value pinned just under it (classically 4 GB minus 1 MB). Anything
        # near the cap is the lie, not the VRAM, so only clearly-below values
        # are kept. On a tool whose whole thesis is that VRAM capacity decides
        # the experience, a wrong capacity is worse than none.
        if isinstance(ram, (int, float)) and 0 < ram < 4 * 2**30 - 2**26:
            gpu["vram_mb"] = int(ram // 2**20)
        out.append(gpu)
    return out


def gpu_inventory() -> List[Dict[str, Any]]:
    """Non-Apple GPU discovery across vendors. NVIDIA via nvidia-smi (richest),
    then PCI ids / CIM for AMD and Intel. Order: discrete-looking first."""
    gpus: List[Dict[str, Any]] = []
    if has_nvidia_smi():
        raw = _run(["nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits"])
        for line in (raw or "").splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({"vendor": "NVIDIA", "name": parts[0],
                             "vram_mb": _to_int(parts[1]),
                             "driver": parts[2]})
    sysname = platform.system()
    if sysname == "Linux":
        for g in _drm_gpus():
            if g["vendor"] == "NVIDIA" and any(
                    x["vendor"] == "NVIDIA" for x in gpus):
                continue
            g2: Dict[str, Any] = dict(g)
            name = _lspci_gpu_name(g["vendor"])
            if name:
                g2["name"] = name
            gpus.append(g2)
    elif sysname == "Windows" and not gpus:
        gpus = _windows_gpus()
    # Prefer a discrete card as the headline GPU when both are present.
    gpus.sort(key=lambda g: (g.get("vendor") == "Intel" and
                             "arc" not in str(g.get("name", "")).lower()))
    return gpus


def _backend_label(gpus: List[Dict[str, Any]]) -> str:
    """What acceleration this machine plausibly offers. Descriptive, not a
    claim about which runtime the engine actually chose."""
    vendors = [g.get("vendor") for g in gpus]
    if "NVIDIA" in vendors:
        return "NVIDIA CUDA"
    if "AMD" in vendors:
        return ("AMD ROCm" if has_rocm_smi()
                else "AMD GPU (install ROCm for live telemetry)")
    if "Intel" in vendors:
        return "Intel GPU"
    return "CPU only (no GPU detected)"


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

    if sysname != "Darwin":
        gpus = gpu_inventory()
        if gpus:
            g = gpus[0]
            info["gpu"] = g.get("name") or f"{g.get('vendor', '?')} GPU"
            if g.get("vram_mb"):
                info["vram_mb"] = str(g["vram_mb"])
            if g.get("driver"):
                info["driver"] = g["driver"]
            info["accelerator"] = {
                "NVIDIA": "nvidia-cuda",
                "AMD": "amd-rocm" if has_rocm_smi() else "amd",
                "Intel": "intel",
            }.get(g.get("vendor", ""), "unknown")

    if sysname == "Darwin":
        info["accelerator"] = "apple-metal"
        chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        hw = _run(["sysctl", "-n", "hw.model"])
        if hw:
            info["hw_model"] = hw
        mem = _run(["sysctl", "-n", "hw.memsize"])
        if mem and mem.isdigit():
            info["unified_memory_mb"] = str(int(mem) // (1024 * 1024))
        if platform.machine() == "arm64":
            # Any Apple Silicon: the CPU brand string ("Apple M1" ... "Apple M4
            # Max") IS the chip name, so the slow system_profiler call is
            # skipped entirely.
            if chip:
                info["gpu"] = chip + " (unified)"
        else:
            # Intel Mac: the GPU is a separate part (AMD or Intel) and the
            # memory is not unified, so neither claim is made.
            info.pop("unified_memory_mb", None)
            sp = _run(["system_profiler", "SPDisplaysDataType"], timeout=25) or ""
            mg = re.search(r"Chipset Model:\s*(.+)", sp)
            info["gpu"] = mg.group(1).strip() if mg else (chip or "Intel Mac")

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
        mem["type"] = ("Unified memory: one pool shared by CPU, GPU and "
                       "Neural Engine" if env["arch"] == "arm64"
                       else "System RAM (Intel Mac: GPU memory is separate)")

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
                   "note": "Not used for these benchmarks; Ollama runs LLM "
                           "inference on the GPU via Metal."}
        else:
            npu = {"name": None, "status": "none detected"}
    else:
        env["os"] = f"{sysname} {platform.release()}"
        cpu["name"] = platform.processor() or platform.machine()
        cpu["cores_total"] = os.cpu_count()

        gpus = gpu_inventory()
        env["backend"] = _backend_label(gpus)
        if gpus:
            g = gpus[0]
            gpu["vendor"] = g.get("vendor")
            gpu["name"] = g.get("name") or f"{g.get('vendor', '?')} GPU"
            if g.get("vram_mb"):
                gpu["vram_gb"] = round(g["vram_mb"] / 1024, 1)
                gpu["memory"] = (f"{gpu['vram_gb']} GB dedicated VRAM "
                                 f"(separate from system RAM)")
            if g.get("driver"):
                gpu["driver"] = g["driver"]
            if g.get("vendor") == "NVIDIA":
                head = _run(["nvidia-smi"]) or ""
                mcu = re.search(r"CUDA Version:\s*([\d.]+)", head)
                gpu["api"] = f"CUDA {mcu.group(1)}" if mcu else "CUDA"
            elif g.get("vendor") == "AMD":
                gpu["api"] = "ROCm/HIP" if has_rocm_smi() else "Vulkan/DirectML (engine-dependent)"
            elif g.get("vendor") == "Intel":
                gpu["api"] = "Level Zero/SYCL or Vulkan (engine-dependent)"

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
        mem["type"] = ("System RAM, separate from the GPU's VRAM"
                       if gpu.get("vram_gb") else "System RAM")
        npu = {"name": None, "status": "not detected",
               "note": "Dedicated NPUs (on AI PCs) are not used by Ollama."}

    _DEEP_CACHE = {"cpu": cpu, "gpu": gpu, "npu": npu, "memory": mem,
                   "env": env}
    return _DEEP_CACHE


# --------------------------------------------------------------------------
# TensorRT availability (honest facts; the caller decides what to do)
# --------------------------------------------------------------------------
def tensorrt_status() -> Dict[str, Any]:
    """Report what is actually true about TensorRT on this machine: whether an
    NVIDIA GPU is present and whether the Python tensorrt package imports.
    TensorRT requires NVIDIA hardware, so on Apple Silicon both are false and
    the UI says so instead of pretending."""
    status: Dict[str, Any] = {
        "nvidia_gpu": has_nvidia_smi(),
        "python_tensorrt": None,
    }
    try:
        import tensorrt  # type: ignore  # noqa: F401
        status["python_tensorrt"] = getattr(tensorrt, "__version__", "installed")
    except Exception:
        pass
    return status


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
    # GPU extras (NVIDIA): clock behaviour is the thermals story in numbers.
    sm_clock_avg_mhz: Optional[float] = None
    sm_clock_min_mhz: Optional[float] = None
    gpu_throttled: Optional[bool] = None
    # Whole-system signals (Apple, sudoless): thermal pressure and, when the
    # machine is unplugged, the real watts being drawn from the battery.
    thermal_pressure_peak: Optional[str] = None
    on_battery: Optional[bool] = None
    battery_power_avg_w: Optional[float] = None
    battery_power_peak_w: Optional[float] = None


def _read_nvidia() -> Optional[Dict[str, Any]]:
    raw = _run([
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu,"
        "clocks.sm,clocks_throttle_reasons.active",
        "--format=csv,noheader,nounits",
    ], timeout=5)
    if not raw:
        return None
    parts = [p.strip() for p in raw.splitlines()[0].split(",")]
    try:
        out: Dict[str, Any] = {"utilization_pct": float(parts[0]),
                               "vram_used_mb": float(parts[1]),
                               "power_w": float(parts[2]),
                               "temp_c": float(parts[3])}
    except (ValueError, IndexError):
        return None
    # Clock + throttle mask are the thermal story in numbers: a hot GPU keeps
    # its temperature flat and gives up clocks instead. Both fields can be
    # "[N/A]" on some boards, so they are optional.
    if len(parts) >= 5:
        try:
            out["sm_clock_mhz"] = float(parts[4])
        except ValueError:
            pass
    if len(parts) >= 6 and parts[5].startswith("0x"):
        try:
            out["throttled"] = int(parts[5], 16) != 0
        except ValueError:
            pass
    return out


def _parse_rocm_json(raw: Optional[str]) -> Optional[Dict[str, float]]:
    """Pull util/VRAM/power/temp out of `rocm-smi --json`. Key names drift
    between ROCm releases, so match by pattern instead of exact string, and
    return only what was actually found."""
    if not raw:
        return None
    import json as _json
    try:
        data = _json.loads(raw)
    except ValueError:
        return None
    card = next((v for k, v in sorted(data.items())
                 if isinstance(v, dict) and k.lower().startswith("card")), None)
    if not card:
        return None
    out: Dict[str, float] = {}

    def find(pattern: str) -> Optional[float]:
        for k, v in card.items():
            if re.search(pattern, k, re.I):
                try:
                    return float(str(v).replace("%", "").strip())
                except (TypeError, ValueError):
                    continue
        return None

    util = find(r"GPU use")
    power = find(r"Power")
    temp = find(r"Temperature.*(edge|junction)") or find(r"Temperature")
    vram_b = find(r"Used.*Memory|Memory.*Used")
    if util is not None:
        out["utilization_pct"] = util
    if power is not None:
        out["power_w"] = power
    if temp is not None:
        out["temp_c"] = temp
    if vram_b is not None:
        # rocm-smi reports bytes for meminfo; anything that large is bytes.
        out["vram_used_mb"] = round(vram_b / 2**20, 1) if vram_b > 2**20 else vram_b
    return out or None


def _read_rocm() -> Optional[Dict[str, float]]:
    return _parse_rocm_json(_run(
        ["rocm-smi", "--showuse", "--showpower", "--showtemp",
         "--showmeminfo", "vram", "--json"], timeout=5))


_THERMAL_STATES = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}


def _read_apple_system() -> Optional[Dict[str, Any]]:
    """Sudoless macOS signals: thermal pressure via NSProcessInfo (through
    osascript, so no third-party bindings) and battery draw via ioreg. Battery
    watts are only meaningful unplugged; plugged in, only thermal state is
    reported. ~50 ms per sample, so the sampler calls this on a slow tick.
    The fuel gauge refreshes on its own ~20-25 s cadence (measured: it held a
    stale idle value through a short generation, then stepped to the real
    load), so short runs may under-report and sustained runs are truthful."""
    out: Dict[str, Any] = {}
    ts = _run(["osascript", "-l", "JavaScript", "-e",
               'ObjC.import("Foundation"); '
               "$.NSProcessInfo.processInfo.thermalState"], timeout=5)
    if ts is not None and ts.strip().isdigit():
        out["thermal_state"] = int(ts.strip())
    raw = _run(["ioreg", "-rn", "AppleSmartBattery"], timeout=5) or ""
    parsed = _parse_battery_ioreg(raw)
    if parsed:
        out.update(parsed)
    return out or None


def _parse_battery_ioreg(raw: str) -> Optional[Dict[str, Any]]:
    m_ext = re.search(r'"ExternalConnected"\s*=\s*(Yes|No)', raw)
    m_amp = re.search(r'"Amperage"\s*=\s*(\d+)', raw)
    m_volt = re.search(r'"Voltage"\s*=\s*(\d+)', raw)
    if not (m_ext and m_amp and m_volt):
        return None
    on_battery = m_ext.group(1) == "No"
    amp = int(m_amp.group(1))
    if amp >= 2 ** 63:                     # unsigned wrap: discharge is negative
        amp -= 2 ** 64
    watts = abs(amp) * int(m_volt.group(1)) / 1_000_000.0
    out: Dict[str, Any] = {"on_battery": on_battery}
    if on_battery and amp < 0:
        out["battery_w"] = round(watts, 1)
    return out


def _read_powermetrics() -> Optional[Dict[str, float]]:
    """Package power on Apple Silicon, root only. `powermetrics` is Apple's own
    counter but refuses to run unprivileged, so this reader exists only when
    the process is root; the docs say plainly that sudo unlocks it."""
    raw = _run(["powermetrics", "-n", "1", "-i", "300",
                "--samplers", "cpu_power"], timeout=10)
    return _parse_powermetrics(raw)


def _parse_powermetrics(raw: Optional[str]) -> Optional[Dict[str, float]]:
    if not raw:
        return None
    m = re.search(r"Combined Power \(CPU \+ GPU \+ ANE\):\s*([\d.]+)\s*mW", raw)
    if not m:
        return None
    return {"package_power_w": round(float(m.group(1)) / 1000.0, 2)}


def _pick_system_reader():
    """Whole-system sampler for this platform. Apple always has the sudoless
    reader; root additionally unlocks powermetrics package power."""
    if platform.system() != "Darwin":
        return None
    is_root = hasattr(os, "geteuid") and os.geteuid() == 0

    def read() -> Optional[Dict[str, Any]]:
        out = _read_apple_system() or {}
        if is_root:
            pm = _read_powermetrics()
            if pm:
                out.update(pm)
        return out or None

    return read


def _pick_gpu_reader():
    """The sampling tool for this machine: nvidia-smi, rocm-smi, or nothing.
    Intel GPUs are named but not sampled; no CLI is reliably present across
    Intel driver stacks, and pretending otherwise would report noise."""
    if has_nvidia_smi():
        return _read_nvidia
    if has_rocm_smi():
        return _read_rocm
    return None


def sample_gpu_once() -> Dict[str, Any]:
    """One synchronous GPU telemetry read (NVIDIA or AMD/ROCm). Use this for a
    point-in-time status query; GpuSampler is for wrapping a generation over
    time, and starting then immediately stopping it races the polling thread
    for zero or one sample."""
    reader = _pick_gpu_reader()
    sample = reader() if reader else None
    sys_reader = _pick_system_reader()
    sysm = sys_reader() if sys_reader else None
    merged: Dict[str, Any] = {}
    if sample:
        merged.update(sample)
    if sysm:
        if "thermal_state" in sysm:
            merged["thermal_pressure"] = _THERMAL_STATES.get(
                sysm["thermal_state"], str(sysm["thermal_state"]))
        for k in ("on_battery", "battery_w", "package_power_w"):
            if k in sysm:
                merged[k] = sysm[k]
    if not merged:
        return {"available": False}
    return {"available": True, **merged}


class GpuSampler:
    """Background GPU poller (nvidia-smi, or rocm-smi where ROCm is
    installed). No-op elsewhere so the same code path runs everywhere."""

    SYSTEM_EVERY = 8   # thermal pressure and the battery gauge move on
                       # multi-second cadences; sampling them every ~2 s halves
                       # the subprocess spawns without losing information

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self._reader = _pick_gpu_reader()
        self._sys_reader = _pick_system_reader()
        self.enabled = self._reader is not None or self._sys_reader is not None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._util: List[float] = []
        self._vram: List[float] = []
        self._power: List[float] = []
        self._temp: List[float] = []
        self._clock: List[float] = []
        self._throttled = False
        self._throttle_seen = False
        self._thermal: List[int] = []
        self._battery_w: List[float] = []
        self._on_battery: Optional[bool] = None
        self._tick = 0

    def _poll_once(self) -> None:
        sample = self._reader() if self._reader else None
        if sample:
            if "utilization_pct" in sample:
                self._util.append(sample["utilization_pct"])
            if "vram_used_mb" in sample:
                self._vram.append(sample["vram_used_mb"])
            if "power_w" in sample:
                self._power.append(sample["power_w"])
            if "temp_c" in sample:
                self._temp.append(sample["temp_c"])
            if "sm_clock_mhz" in sample:
                self._clock.append(sample["sm_clock_mhz"])
            if "throttled" in sample:
                self._throttle_seen = True
                self._throttled = self._throttled or sample["throttled"]
        if self._sys_reader and self._tick % self.SYSTEM_EVERY == 0:
            sysm = self._sys_reader()
            if sysm:
                if "thermal_state" in sysm:
                    self._thermal.append(sysm["thermal_state"])
                if "on_battery" in sysm:
                    self._on_battery = sysm["on_battery"]
                if "battery_w" in sysm:
                    self._battery_w.append(sysm["battery_w"])
                if "package_power_w" in sysm:
                    self._power.append(sysm["package_power_w"])
        self._tick += 1

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

        n = max(len(self._util), len(self._thermal), len(self._battery_w),
                len(self._power))
        return SampleStats(
            available=n > 0,
            samples=n,
            util_peak=peak(self._util),
            util_avg=avg(self._util),
            vram_used_peak_mb=peak(self._vram),
            power_peak_w=peak(self._power),
            power_avg_w=avg(self._power),
            temp_peak_c=peak(self._temp),
            sm_clock_avg_mhz=avg(self._clock),
            sm_clock_min_mhz=(min(self._clock) if self._clock else None),
            gpu_throttled=(self._throttled if self._throttle_seen else None),
            thermal_pressure_peak=(_THERMAL_STATES.get(max(self._thermal))
                                   if self._thermal else None),
            on_battery=self._on_battery,
            battery_power_avg_w=avg(self._battery_w),
            battery_power_peak_w=peak(self._battery_w),
        )
