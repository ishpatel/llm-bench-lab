"""A small agent harness (Phase 8) with deterministic guardrails (Phase 9).

The harness is the software *around* the model that lets it act: it constructs
context, offers a fixed set of tools, runs the model→tool→observation loop, and
decides when the task is done. Every tool request passes through the guardrails
in `guardrails.py` before anything executes — the model proposes, deterministic
code authorizes.

Two tools demonstrate the spectrum:
  * get_gpu_status  — read-only, safe, runs freely
  * set_power_limit — consequential: arguments are schema-validated to a safe
    range AND require human approval before executing (and even then it is
    simulated — no hardware is touched)
Any tool the model invents that is not in the registry is refused by the tool
rail. This is the principle of least privilege in practice.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from . import guardrails, telemetry

SYSTEM_PROMPT = (
    "You are a local hardware assistant for a benchmarking workstation. You may "
    "use ONLY the provided tools. For ANY question about the current GPU, memory, "
    "utilization, temperature, or hardware status, you MUST call get_gpu_status "
    "instead of answering from memory. To change the power limit, call "
    "set_power_limit. Never claim to have performed an action that a tool did not "
    "confirm. If a request is outside your tools or not permitted, say so plainly."
)


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], str]
    requires_approval: bool = False

    def schema(self) -> Dict[str, Any]:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": self.parameters}}


# --------------------------------------------------------------------------
# Tool implementations
# --------------------------------------------------------------------------
def _get_gpu_status(_args: Dict[str, Any]) -> str:
    sysinfo = telemetry.describe_system()
    resident = telemetry.ollama_ps()
    status = {
        "gpu": sysinfo.get("gpu", sysinfo.get("machine")),
        "accelerator": sysinfo.get("accelerator"),
        "memory": (f"{int(sysinfo['unified_memory_mb']) // 1024} GB unified"
                   if sysinfo.get("unified_memory_mb")
                   else (f"{int(sysinfo['vram_mb']) // 1024} GB VRAM"
                         if sysinfo.get("vram_mb") else "unknown")),
        "resident_models": {k: v.get("processor", "") for k, v in resident.items()}
                           or "none loaded",
    }
    smp = telemetry.GpuSampler()
    smp.start()
    stats = smp.stop()
    if stats.available:
        status["utilization_pct"] = stats.util_peak
        status["vram_used_mb"] = stats.vram_used_peak_mb
        status["power_w"] = stats.power_peak_w
        status["temp_c"] = stats.temp_peak_c
    return json.dumps(status)


def _set_power_limit(args: Dict[str, Any]) -> str:
    watts = int(args["watts"])
    return (f"Power limit set to {watts} W (SIMULATED: approval granted, "
            f"validated within safe range; no hardware was modified).")


def default_tools() -> Dict[str, Tool]:
    return {
        "get_gpu_status": Tool(
            name="get_gpu_status",
            description="Read the current local GPU/accelerator status: name, "
                        "memory, resident models, and (on NVIDIA) utilization, "
                        "VRAM, power and temperature.",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=_get_gpu_status,
        ),
        "set_power_limit": Tool(
            name="set_power_limit",
            description="Set the GPU power limit in watts. Only values within the "
                        "safe range are permitted, and the action requires human "
                        "approval before it takes effect.",
            parameters={"type": "object", "properties": {
                "watts": {"type": "integer", "minimum": 80, "maximum": 115,
                          "description": "target power limit in watts (80-115)"}},
                "required": ["watts"]},
            handler=_set_power_limit,
            requires_approval=True,
        ),
    }


# --------------------------------------------------------------------------
# The harness
# --------------------------------------------------------------------------
@dataclass
class AgentResult:
    answer: str = ""
    blocked: bool = False
    events: List[Dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    wall_ms: float = 0.0
    output_flag: Optional[str] = None


class Agent:
    def __init__(self, client, model: str,
                 tools: Optional[Dict[str, Tool]] = None,
                 max_steps: int = 4):
        self.client = client
        self.model = model
        self.tools = tools or default_tools()
        self.max_steps = max_steps

    def run(self, question: str, approve: bool = False,
            log: Optional[Callable[[str], None]] = None) -> AgentResult:
        log = log or (lambda m: None)
        res = AgentResult()

        # ---- input rail (before the model sees anything) ----
        ir = guardrails.input_rail(question)
        res.events.append({"type": "input_rail", "allowed": ir.allowed,
                           "reason": ir.reason})
        if not ir.allowed:
            log(f"input rail ✗ {ir.reason}")
            res.blocked = True
            res.answer = ("Request refused by the input guardrail; it looked "
                          "like a prompt-injection or disallowed action.")
            return res

        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question}]
        schemas = [t.schema() for t in self.tools.values()]

        for step in range(self.max_steps):
            res.steps = step + 1
            resp = self.client.chat(self.model, messages, tools=schemas,
                                    options={"temperature": 0, "think": False})
            res.wall_ms += resp.get("wall_ms", 0.0)
            calls = resp.get("tool_calls") or []
            if not calls:
                answer = resp.get("content", "")
                orr = guardrails.output_rail(answer)
                res.events.append({"type": "output_rail", "allowed": orr.allowed,
                                   "reason": orr.reason})
                if not orr.allowed:
                    res.output_flag = orr.reason
                    answer = "[response withheld by output guardrail]"
                res.answer = answer
                log(f"answer: {answer[:80]}")
                return res

            messages.append(resp["message"])
            for call in calls:
                fn = call.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", {}) or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                obs = self._authorize_and_run(name, args, approve, res, log)
                messages.append({"role": "tool", "tool_name": name, "content": obs})

        res.answer = "(stopped: reached max reasoning steps)"
        return res

    def _authorize_and_run(self, name: str, args: Dict[str, Any], approve: bool,
                           res: AgentResult, log) -> str:
        """Run the guardrail gauntlet for one tool call. Returns the observation
        text fed back to the model (either the tool result or a rejection)."""
        # 1. tool allowlist
        tr = guardrails.tool_rail(name, self.tools)
        if not tr.allowed:
            self._event(res, name, args, tr, log)
            return f"TOOL REJECTED: {tr.reason}"
        tool = self.tools[name]
        # 2. parameter schema/bounds
        pr = guardrails.param_rail(args, tool.parameters)
        if not pr.allowed:
            self._event(res, name, args, pr, log)
            return f"TOOL REJECTED: {pr.reason}"
        # 3. human approval for consequential tools
        ar = guardrails.approval_rail(tool.requires_approval, approve)
        if not ar.allowed:
            self._event(res, name, args, ar, log)
            return (f"ACTION HELD: {ar.reason}. The action was NOT performed. "
                    f"Tell the user it needs approval.")
        # 4. execute
        observation = tool.handler(args)
        res.events.append({"type": "tool_exec", "tool": name, "args": args,
                           "allowed": True, "observation": observation})
        log(f"tool ✓ {name}({json.dumps(args)}) → {observation[:60]}")
        return observation

    @staticmethod
    def _event(res: AgentResult, name: str, args: Dict[str, Any],
               rail, log) -> None:
        res.events.append({"type": "guardrail_block", "rail": rail.rail,
                           "tool": name, "args": args, "allowed": False,
                           "reason": rail.reason})
        log(f"{rail.rail} rail ✗ {name}: {rail.reason}")
