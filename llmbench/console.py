"""Plain-text benchmark summaries for the terminal.

A CLI run used to end by printing the path to a JSON file, so the numbers were
only readable in the web UI or the HTML report. This renders the headline
metrics inline, which is where the person who started the run is already
looking. Verdict wording is imported from `report` rather than restated, so the
terminal and the HTML never drift apart.
"""
from __future__ import annotations

import os
import shutil
import sys
import textwrap
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .report import _median, _spread, residency_verdict, speed_verdict

# A spread wider than this fraction of the median means the repeats disagreed
# enough that the ratio between cells should be read as indicative, not exact.
NOISY_SPREAD = 0.10

_STYLES = {"good": "32", "ok": "33", "bad": "31", "dim": "2",
           "bold": "1", "head": "1;4"}


# --------------------------------------------------------------------------
# Formatting primitives
# --------------------------------------------------------------------------
def use_color(stream=None) -> bool:
    """Colour when writing to a terminal, honouring NO_COLOR / FORCE_COLOR."""
    stream = stream or sys.stdout
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return bool(getattr(stream, "isatty", lambda: False)())


def _paint(text: str, style: str, color: bool) -> str:
    code = _STYLES.get(style)
    return f"\033[{code}m{text}\033[0m" if (color and code) else text


def _num(v: Optional[float], nd: int = 1) -> str:
    if v is None:
        return "-"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{nd}f}"


def _int(v: Optional[float]) -> str:
    return "-" if v is None else f"{int(round(v)):,}"


def _trunc(text: str, width: int) -> str:
    """Drop from the middle. A run label is `model · prompt`, and cutting the
    tail would leave several cells of the same model looking identical."""
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    head = (width - 1) // 2
    return text[:head] + "…" + text[len(text) - (width - 1 - head):]


def _spread_frac(agg: Dict[str, Any], metric: str) -> Optional[float]:
    """Width of the min..max band as a fraction of the median."""
    span, med = _spread(agg, metric), _median(agg, metric)
    if not span or not med:
        return None
    return (span[1] - span[0]) / med


# --------------------------------------------------------------------------
# Table
# --------------------------------------------------------------------------
def _render_table(headers: Sequence[str],
                  rows: Sequence[Sequence[Tuple[str, str]]],
                  color: bool) -> List[str]:
    """Rows are sequences of (text, style). Padding is applied to the raw text
    so escape codes never count toward column width."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, (text, _) in enumerate(row):
            widths[i] = max(widths[i], len(text))

    # First column is left-aligned (the label); metrics read better right-aligned.
    def line(cells: Sequence[Tuple[str, str]]) -> str:
        out = []
        for i, (text, style) in enumerate(cells):
            pad = text.ljust(widths[i]) if i == 0 else text.rjust(widths[i])
            out.append(_paint(pad, style, color) if style else pad)
        return "  ".join(out).rstrip()

    lines = [line([(h, "head") for h in headers])]
    lines += [line(r) for r in rows]
    return lines


def _fit_labels(cells: List[Dict[str, Any]], budget: int) -> List[str]:
    """Shorten cell labels to `budget` columns, dropping the shared prefix
    first so what actually differs between cells stays visible."""
    labels = [str(c.get("label") or c.get("model") or "?") for c in cells]
    if not labels or max(len(x) for x in labels) <= budget:
        return labels
    parts = [x.split(" · ") for x in labels]
    if len(labels) > 1 and all(len(p) > 1 for p in parts):
        heads = {p[0] for p in parts}
        if len(heads) == 1:  # every cell shares the model/context prefix
            labels = [" · ".join(p[1:]) for p in parts]
    return [_trunc(x, budget) for x in labels]


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
def render_summary(results: Dict[str, Any], color: Optional[bool] = None,
                   width: Optional[int] = None) -> str:
    """Render one results document (meta + cells) as a terminal report."""
    if color is None:
        color = use_color()
    if width is None:
        width = shutil.get_terminal_size((100, 24)).columns
    meta = results.get("meta") or {}
    cells = results.get("cells") or []
    out: List[str] = []

    def add(text: str = "", style: str = "") -> None:
        out.append(_paint(text, style, color) if style else text)

    system = (meta.get("system") or {}).get("label", "unknown system")
    opts = meta.get("options") or {}
    add()
    add(f"{meta.get('config_name', 'benchmark')} on {system}", "bold")
    bits = [f"engine: {meta.get('engine', 'Ollama')}",
            f"{meta.get('runs', '?')} timed runs (+{meta.get('warmup', 0)} warm-up)"]
    if opts.get("temperature") is not None:
        bits.append(f"temperature {opts['temperature']}")
    if opts.get("num_predict") is not None:
        bits.append(f"max {opts['num_predict']} tokens")
    add("\n".join(textwrap.wrap("  ".join(bits), max(30, width))), "dim")
    add("\n".join(textwrap.wrap(
        "Median of the timed runs. Higher speed is better; lower first-word is "
        "better.", max(30, width))), "dim")
    add()

    if not cells:
        add("No cells were measured.", "bad")
        return "\n".join(out)

    full_labels = _fit_labels(cells, 10 ** 6)   # untruncated, for the prose below

    def cold_of(cell: Dict[str, Any]) -> Optional[float]:
        return (cell.get("cold_start") or {}).get("load_ms")

    def first_of(agg: Dict[str, Any]) -> Optional[float]:
        v = _median(agg, "ttfv_ms")
        return _median(agg, "ttft_ms") if v is None else v

    # `drop` orders the sacrifice when the terminal is too narrow: the columns
    # that answer "is it fast" and "did it fit" are never dropped, because those
    # are the two questions a benchmark is run to answer.
    columns: List[Dict[str, Any]] = [
        {"h": "Speed", "drop": 0,
         "v": lambda c, a: (_num(_median(a, "gen_tps")),
                            speed_verdict(_median(a, "gen_tps"))[1] or "")},
        {"h": "First word", "drop": 0,
         "v": lambda c, a: (_num(first_of(a), 0) if first_of(a) is not None else "-", "")},
        {"h": "Reading", "drop": 2,
         "v": lambda c, a: (_num(_median(a, "prompt_tps"), 0), "")},
        {"h": "Cold start", "drop": 1,
         "v": lambda c, a: (_num(cold_of(c), 0) if cold_of(c) is not None else "-", "")},
        {"h": "In / Out", "drop": 3,
         "v": lambda c, a: (f"{_int(_median(a, 'prompt_tokens'))} / "
                            f"{_int(_median(a, 'output_tokens'))}", "dim")},
        {"h": "Placement", "drop": 0,
         "v": lambda c, a: (c.get("residency") or "-",
                            "" if residency_verdict(c.get("residency") or "")[1]
                            == "neutral" else residency_verdict(c.get("residency") or "")[1])},
    ]
    if not any(cold_of(c) for c in cells):
        columns = [col for col in columns if col["h"] != "Cold start"]

    for col in columns:
        col["cells"] = [col["v"](c, c.get("aggregate") or {}) for c in cells]
        col["w"] = max([len(col["h"])] + [len(t) for t, _ in col["cells"]])

    # Give the label whatever the metric columns leave, and drop optional
    # columns while that would squeeze the label below a readable width.
    def label_room() -> int:
        return width - sum(c["w"] + 2 for c in columns)

    while label_room() < 24:
        droppable = [c for c in columns if c["drop"]]
        if not droppable:
            break
        columns.remove(max(droppable, key=lambda c: c["drop"]))

    labels = [_trunc(x, max(18, min(46, label_room()))) for x in full_labels]

    headers = ["Run"] + [c["h"] for c in columns]
    rows: List[List[Tuple[str, str]]] = []
    for i, (label, cell) in enumerate(zip(labels, cells)):
        if not ((cell.get("aggregate") or {}).get("n_ok") or 0):
            rows.append([(label, "bad"), ("failed", "bad")]
                        + [("-", "dim")] * (len(headers) - 2))
            continue
        rows.append([(label, "")] + [col["cells"][i] for col in columns])

    out.extend(_render_table(headers, rows, color))
    add()
    units = ["Speed", "Reading"] if any(c["h"] == "Reading" for c in columns) else ["Speed"]
    ms = [c["h"] for c in columns if c["h"] in ("First word", "Cold start")]
    legend = (f"{' and '.join(units)} {'are' if len(units) > 1 else 'is'} tokens/sec"
              + (" (answering, then reading the prompt)" if len(units) > 1 else "")
              + (f". {' and '.join(ms)} {'are' if len(ms) > 1 else 'is'} milliseconds."
                 if ms else "."))
    add("\n".join(textwrap.wrap(legend, max(30, width))), "dim")

    notes = _notes(cells, full_labels)
    if notes:
        add()
        add("What this means", "bold")
        for text, style in notes:
            body = textwrap.wrap(text, max(30, width - 4)) or [text]
            marker = "!" if style == "bad" else "-"
            block = "\n".join([f"  {marker} {body[0]}"]
                              + [f"    {ln}" for ln in body[1:]])
            add(block, style if style in ("bad", "ok") else "")
    return "\n".join(out)


MAX_VARIANCE_NOTES = 3


def _notes(cells: List[Dict[str, Any]], labels: List[str]) -> List[Tuple[str, str]]:
    """Plain-English read of the table: the comparison, then anything that
    undermines it."""
    notes: List[Tuple[str, str]] = []
    ranked = [(lab, c, _median(c.get("aggregate") or {}, "gen_tps"))
              for lab, c in zip(labels, cells)]

    # Headline comparison, taken from the largest group of cells that share a
    # prompt. Ranking across different prompts would compare cells that differ
    # in two variables at once, which is exactly what the configs control for.
    groups: Dict[str, List[Tuple[str, Dict[str, Any], float]]] = {}
    for item in ranked:
        if item[2] and (item[1].get("aggregate") or {}).get("n_ok"):
            groups.setdefault(item[1].get("prompt_key") or "", []).append(item)
    if groups:
        key = max(groups, key=lambda k: (len(groups[k]),
                                         max(x[2] for x in groups[k])))
        grp = sorted(groups[key], key=lambda x: x[2], reverse=True)
        fast_l, _, fast_v = grp[0]
        if len(grp) > 1:
            slow_l, _, slow_v = grp[-1]
            held = " (prompt held constant)" if len(groups) > 1 else ""
            notes.append((f"Fastest{held}: {fast_l} at {_num(fast_v)} tok/s, "
                          f"{fast_v / slow_v:.1f}x {slow_l} at "
                          f"{_num(slow_v)} tok/s.", ""))
        else:
            notes.append((f"Fastest: {fast_l} at {_num(fast_v)} tok/s.", ""))
        verdict, style = speed_verdict(fast_v)
        if verdict:
            notes.append((f"{verdict}.", style))

    # Failures are per cell, since the error text differs.
    for lab, cell, _ in ranked:
        agg = cell.get("aggregate") or {}
        n_ok, n_total = agg.get("n_ok") or 0, agg.get("n_total") or 0
        if n_total and n_ok < n_total:
            err = next((r.get("error") for r in (cell.get("runs") or [])
                        if r.get("error")), "")
            detail = f": {str(err)[:70]}" if err else ""
            notes.append((f"{lab}: only {n_ok} of {n_total} runs succeeded"
                          f"{detail}.", "bad"))

    # Offload is a property of the model, not of each prompt, so report it once
    # per model instead of repeating an identical line for every cell.
    seen: List[str] = []
    for _, cell, _ in ranked:
        res = cell.get("residency") or ""
        model = cell.get("model") or "model"
        if "cpu" in res.lower() and model not in seen:
            seen.append(model)
            n = sum(1 for _, c, _ in ranked if c.get("model") == model
                    and "cpu" in (c.get("residency") or "").lower())
            where = f" in all {n} of its runs" if n > 1 else ""
            notes.append((f"{model} did not fit entirely on the GPU ({res})"
                          f"{where}. Layers served from system memory are the "
                          "usual cause of a large speed drop.", "bad"))

    noisy = []
    for lab, cell, _ in ranked:
        frac = _spread_frac(cell.get("aggregate") or {}, "gen_tps")
        if frac is not None and frac > NOISY_SPREAD:
            noisy.append((lab, frac, _spread(cell.get("aggregate") or {}, "gen_tps")))
    for lab, frac, span in noisy[:MAX_VARIANCE_NOTES]:
        notes.append((f"{lab} varied by {frac * 100:.0f}% across repeats "
                      f"({_num(span[0])}-{_num(span[1])} tok/s), so treat its "
                      "ratio as indicative rather than exact.", "ok"))
    if len(noisy) > MAX_VARIANCE_NOTES:
        notes.append((f"{len(noisy) - MAX_VARIANCE_NOTES} further cells varied by "
                      f"more than {NOISY_SPREAD * 100:.0f}% across repeats.", "ok"))

    approx = sorted({c.get("model") or lab for lab, c, _ in ranked
                     if any(r.get("approximate_tokens") for r in (c.get("runs") or []))})
    if approx:
        notes.append((f"{', '.join(approx)}: the engine reported no token counts, "
                      "so speed is unavailable and lengths are approximate.", "ok"))
    return notes


def print_summary(results: Dict[str, Any], stream=None) -> None:
    stream = stream or sys.stdout
    print(render_summary(results, color=use_color(stream)), file=stream)


# --------------------------------------------------------------------------
# Readiness
# --------------------------------------------------------------------------
# Same three words the web UI badges use, so a check reads identically
# whichever surface someone is looking at.
_STATUS_WORD = {"ok": "ready", "warn": "optional", "fail": "blocking"}
_STATUS_STYLE = {"ok": "good", "warn": "ok", "fail": "bad"}


def render_readiness(report: Dict[str, Any], system_label: str = "",
                     verbose: bool = False, color: Optional[bool] = None,
                     width: Optional[int] = None) -> str:
    """Render the output of `readiness.describe_readiness` for a terminal."""
    if color is None:
        color = use_color()
    if width is None:
        width = shutil.get_terminal_size((100, 24)).columns
    checks = report.get("checks") or []
    out: List[str] = []

    def add(text: str = "", style: str = "") -> None:
        out.append(_paint(text, style, color) if style else text)

    def wrapped(text: str, indent: int, style: str = "") -> None:
        pad = " " * indent
        for line in textwrap.wrap(text, max(30, width - indent)) or [text]:
            add(pad + line, style)

    add()
    add("Bench readiness" + (f" on {system_label}" if system_label else ""), "bold")
    wrapped("Whether this machine can run a benchmark: the software the harness "
            "depends on, the models it needs, and somewhere to save results.",
            0, "dim")
    add()

    # +3 covers the two brackets and a trailing space, so the widest tag
    # ([blocking]) still clears the label column.
    tag_w = max([len(w) for w in _STATUS_WORD.values()] or [8]) + 3
    label_w = max([len(c.get("label", "")) for c in checks] or [0])
    detail_col = 2 + tag_w + label_w + 2
    for c in checks:
        status = c.get("status", "warn")
        tag = f"[{_STATUS_WORD.get(status, status)}]".ljust(tag_w)
        # Wrap the detail under itself rather than under the tag, so the status
        # column stays scannable. A long unbroken value (a URL) still overflows,
        # which beats truncating something meant to be copied.
        detail = textwrap.wrap(c.get("detail", ""),
                               max(20, width - detail_col)) or [""]
        add(f"  {_paint(tag, _STATUS_STYLE.get(status, ''), color)}"
            f"{c.get('label', '').ljust(label_w)}  {detail[0]}")
        for line in detail[1:]:
            add(" " * detail_col + line)
        # An explanation earns its place where something is wrong; on a healthy
        # machine it is six paragraphs nobody reads.
        if c.get("why") and (verbose or status != "ok"):
            wrapped(c["why"], tag_w + 2, "dim")
        if c.get("fix"):
            wrapped(f"Fix: {c['fix']}", tag_w + 2, _STATUS_STYLE.get(status, ""))

    add()
    add(report.get("headline", ""), _STATUS_STYLE.get(report.get("state"), ""))
    return "\n".join(out)


def print_readiness(report: Dict[str, Any], system_label: str = "",
                    verbose: bool = False, stream=None) -> None:
    stream = stream or sys.stdout
    print(render_readiness(report, system_label=system_label, verbose=verbose,
                           color=use_color(stream)), file=stream)
