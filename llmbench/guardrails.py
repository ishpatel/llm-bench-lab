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
# Retrieval rail — retrieved document text is DATA, never instructions
# --------------------------------------------------------------------------
# Injection that arrives inside an indexed document bypasses the input rail
# entirely: the user's question is benign, the payload rides in on a retrieved
# chunk. Measured behaviour without this rail: models correctly declined to act
# on a planted instruction (the retrieval path exposes no tools, so there is
# nothing to hijack) but they also never told the user their own document store
# contained one. Silent tolerance is its own weakness, and the exposure becomes
# live the moment retrieval feeds a tool-capable context.
#
# So the rail does two things: redact instruction-shaped lines before they reach
# the prompt, and report what it found so the answer can carry a notice.
RETRIEVAL_INJECTION_PATTERNS = [
    r"(?:assistant|system)\s+instructions?\s*:",
    r"ignore\s+(all\s+)?(your\s+|the\s+)?(previous\s+|prior\s+)?(policies|instructions|rules)",
    r"system\s+override\s*:",
    r"\boverride\s+(all\s+)?(the\s+)?(system|application\s+)?polic(y|ies)",
    r"before\s+(answering|responding)[^.\n]{0,40}\bcall\b",
    # Tool names only count when something is telling the model to invoke them.
    # A permissions file that *lists* the same names is documentation, not an
    # attack, so a bare mention must not trip the rail.
    r"\b(call|invoke|execute|run|use)\s+[`\"']?(run_arbitrary_shell_command"
    r"|delete_file|change_bios_settings|change_gpu_power_limit)",
    r"\brm\s+-rf\b",
    r"do\s+not\s+(mention|tell|inform)\s+(the\s+)?user",
    r"never\s+tell\s+(the\s+)?user",
    r"claim\s+the\s+operation\s+succeeded",
]
_RETR = [re.compile(p, re.IGNORECASE) for p in RETRIEVAL_INJECTION_PATTERNS]

# A line that prohibits or describes an action is not the action. Security
# policies, permission manifests and documentation all talk *about* attacks;
# redacting them destroys exactly the content a user needs to ask about their
# own security posture. Measured on the sample corpus, omitting this check
# produced false positives on two legitimate documents.
_DESCRIPTIVE = re.compile(
    r"\b(must\s+not|must\s+never|may\s+not|should\s+not|shall\s+not|cannot|"
    r"can\s?not|never\s+override|do\s+not\s+(allow|permit)|is\s+blocked|"
    r"are\s+blocked|blocked\b|forbidden|denied|not\s+permitted|not\s+allowed|"
    r"disallowed|prohibited|\"blocked\"|'blocked')",
    re.IGNORECASE)


def _is_descriptive(line: str) -> bool:
    """True when the line prohibits or documents the behaviour rather than
    commanding it."""
    return bool(_DESCRIPTIVE.search(line))

REDACTION = "[redacted by retrieval guardrail: instruction-shaped text]"


@dataclass
class RetrievalScan:
    chunks: List[Dict[str, Any]]          # sanitized chunks, safe to prompt with
    flagged: List[Dict[str, Any]]         # {doc, patterns, lines} per hit
    clean: bool = True


def retrieval_rail(retrieved: List[Dict[str, Any]]) -> RetrievalScan:
    """Scan retrieved chunks for instruction-shaped content, redact those lines,
    and report what was found. Operates line by line so a flagged document still
    contributes its legitimate content instead of being dropped wholesale."""
    safe: List[Dict[str, Any]] = []
    flagged: List[Dict[str, Any]] = []
    for chunk in retrieved or []:
        text = chunk.get("text", "")
        hits: List[str] = []
        out_lines: List[str] = []
        for line in text.splitlines():
            matched = ([] if _is_descriptive(line)
                       else [rx.pattern for rx in _RETR if rx.search(line)])
            if matched:
                hits.extend(matched)
                out_lines.append(REDACTION)
            else:
                out_lines.append(line)
        if hits:
            flagged.append({
                "doc": chunk.get("doc", "?"),
                "patterns": sorted(set(hits)),
                "redacted_lines": sum(1 for l in out_lines if l == REDACTION),
            })
            safe.append({**chunk, "text": "\n".join(out_lines), "sanitized": True})
        else:
            safe.append(chunk)
    return RetrievalScan(chunks=safe, flagged=flagged, clean=not flagged)


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
