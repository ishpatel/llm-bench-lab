"""Deterministic guardrails for the agent harness.

Core principle: **the model proposes, deterministic software authorizes.** The
LLM may *request* an action, but whether it runs is decided by ordinary code
here — never by the model. Rails follow NVIDIA NeMo Guardrails' categories:
input, tool/execution, parameter, approval, and output.

Everything is plain Python (no model calls), so the guardrails themselves are
fast, predictable, and testable — the opposite of "ask the LLM to behave."
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RailResult:
    allowed: bool
    rail: str = ""
    reason: str = ""


# --------------------------------------------------------------------------
# Input rail — runs before the model sees the request
# --------------------------------------------------------------------------
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(your\s+)?(previous\s+|prior\s+)?instructions",
    r"disregard\s+(the\s+)?(above|previous|prior|system)",
    r"reveal\s+.*(system prompt|instructions|confidential|secret)",
    r"you\s+are\s+now\s+(a\s+)?(dan|developer mode|unrestricted)",
    r"pretend\s+(that\s+)?you\s+(are|have no)",
    r"(delete|remove|wipe|rm)\s+.*(file|log|database|benchmark|-rf)",
    r"(run|execute)\s+.*(shell|bash|command|/bin|subprocess|os\.system)",
    r"(exfiltrate|leak|send)\s+.*(data|file|secret|key)",
]
_INJ = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def input_rail(text: str) -> RailResult:
    for rx in _INJ:
        if rx.search(text or ""):
            return RailResult(False, "input",
                              f"blocked: matched disallowed-intent pattern /{rx.pattern}/")
    return RailResult(True, "input")


# --------------------------------------------------------------------------
# Tool rail — only explicitly registered tools may run (allowlist)
# --------------------------------------------------------------------------
def tool_rail(name: str, registry: Dict[str, Any]) -> RailResult:
    if name in registry:
        return RailResult(True, "tool")
    return RailResult(False, "tool",
                      f"blocked: '{name}' is not a registered tool "
                      f"(allowed: {', '.join(sorted(registry)) or 'none'})")


# --------------------------------------------------------------------------
# Parameter rail — validate arguments against the tool's schema
# --------------------------------------------------------------------------
def param_rail(args: Dict[str, Any], schema: Dict[str, Any]) -> RailResult:
    props = schema.get("properties", {})
    required = schema.get("required", [])
    args = args or {}
    for r in required:
        if r not in args:
            return RailResult(False, "param", f"blocked: missing required arg '{r}'")
    for key, val in args.items():
        if key not in props:
            return RailResult(False, "param", f"blocked: unexpected arg '{key}'")
        spec = props[key]
        typ = spec.get("type")
        if typ == "integer":
            iv = _as_int(val)
            if iv is None:
                return RailResult(False, "param",
                                  f"blocked: arg '{key}'={val!r} is not an integer")
            if "minimum" in spec and iv < spec["minimum"]:
                return RailResult(False, "param",
                                  f"blocked: '{key}'={iv} below minimum {spec['minimum']}")
            if "maximum" in spec and iv > spec["maximum"]:
                return RailResult(False, "param",
                                  f"blocked: '{key}'={iv} above maximum {spec['maximum']}")
        elif typ == "string":
            if not isinstance(val, str):
                return RailResult(False, "param", f"blocked: arg '{key}' must be a string")
            if "enum" in spec and val not in spec["enum"]:
                return RailResult(False, "param",
                                  f"blocked: '{key}'={val!r} not in {spec['enum']}")
    return RailResult(True, "param")


def _as_int(v: Any) -> Optional[int]:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str):
        try:
            return int(v.strip())
        except ValueError:
            return None
    return None


# --------------------------------------------------------------------------
# Approval rail — consequential tools need human sign-off before executing
# --------------------------------------------------------------------------
def approval_rail(requires_approval: bool, approved: bool) -> RailResult:
    if requires_approval and not approved:
        return RailResult(False, "approval",
                          "held: consequential action requires human approval")
    return RailResult(True, "approval")


# --------------------------------------------------------------------------
# Output rail — last check before the answer reaches the user
# --------------------------------------------------------------------------
SECRET_PATTERNS = [r"sk-[A-Za-z0-9]{16,}", r"gh[pousr]_[A-Za-z0-9]{20,}",
                   r"AKIA[0-9A-Z]{16}"]
_SECRET = [re.compile(p) for p in SECRET_PATTERNS]


def output_rail(text: str) -> RailResult:
    for rx in _SECRET:
        if rx.search(text or ""):
            return RailResult(False, "output", "blocked: output contained a secret-like token")
    return RailResult(True, "output")
