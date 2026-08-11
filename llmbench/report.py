"""Self-contained HTML report generation.

Takes one or more results dicts (from the runner or results JSON files), groups
cells by system, and renders a single dependency-free HTML file with inline SVG
charts plus plain-English interpretation, so a non-expert reader can tell what
the numbers mean and whether they are good.
"""
from __future__ import annotations

import datetime
import html
import json
from typing import Any, Dict, List, Optional, Tuple

# NVIDIA green first, Apple graphite second, then fallbacks.
PALETTE = ["#76b900", "#8e8e93", "#0a84ff", "#ff9f0a", "#bf5af2", "#ff375f"]


# --------------------------------------------------------------------------
# Plain-English interpretation of a measurement
# --------------------------------------------------------------------------
def speed_verdict(v: Optional[float]) -> Tuple[str, str]:
    if v is None:
        return ("", "")
    if v >= 80:
        return ("Feels instant, outpaces reading speed", "good")
    if v >= 30:
        return ("Smooth, a comfortable reading pace", "good")
    if v >= 10:
        return ("Workable, but you wait on it", "ok")
    return ("Sluggish; the model may not fit on the GPU", "bad")


def residency_verdict(res: str) -> Tuple[str, str]:
    if not res:
        return ("", "")
    low = res.lower()
    if "100%" in low and "gpu" in low:
        return ("Fully on the GPU, the fastest path", "good")
    if "cpu" in low:
        return ("Spilled to CPU, this is the VRAM wall", "bad")
    return (res, "neutral")


# --------------------------------------------------------------------------
# Data shaping
# --------------------------------------------------------------------------
def _median(agg: Dict[str, Any], metric: str) -> Optional[float]:
    m = (agg or {}).get(metric)
    if isinstance(m, dict):
        return m.get("median")
    return None


def _spread(agg: Dict[str, Any], metric: str) -> Optional[Tuple[float, float]]:
    m = (agg or {}).get(metric)
    if isinstance(m, dict) and m.get("min") is not None:
        return (m["min"], m["max"])
    return None


def _collect(results: List[Dict[str, Any]]):
    """Return (systems, categories, table[system][cell]=aggregate)."""
    systems: List[str] = []
    table: Dict[str, Dict[str, Dict[str, Any]]] = {}
    system_meta: Dict[str, Dict[str, Any]] = {}
    categories: List[str] = []
    seen_cat = set()
    for res in results:
        meta = res.get("meta", {})
        label = meta.get("system", {}).get("label", "system")
        # de-dupe identical labels across files by suffixing
        base = label
        n = 2
        while label in systems and system_meta.get(label) != meta.get("system"):
            label = f"{base} #{n}"
            n += 1
        if label not in systems:
            systems.append(label)
            system_meta[label] = meta.get("system", {})
            table[label] = {}
            system_meta[label]["_meta"] = meta
        for cell in res.get("cells", []):
            c = cell["label"]
            if c not in seen_cat:
                seen_cat.add(c)
                categories.append(c)
            table[label][c] = cell
    return systems, categories, table, system_meta


# --------------------------------------------------------------------------
# SVG horizontal grouped bar chart
# --------------------------------------------------------------------------
def _svg_bars(title: str, unit: str, categories: List[str],
              series: List[Dict[str, Any]], higher_better: bool) -> str:
    if not categories:
        return ""
    n_series = max(1, len(series))
    bar_h = 16
    grp_gap = 14
    row_h = n_series * bar_h + grp_gap
    left = 230
    right = 70
    top = 44
    width = 900
    plot_w = width - left - right
    height = top + row_h * len(categories) + 20

    all_vals = [v for s in series for v in s["values"] if isinstance(v, (int, float))]
    vmax = max(all_vals) if all_vals else 1.0
    if vmax <= 0:
        vmax = 1.0

    def x(val: float) -> float:
        return left + (val / vmax) * plot_w

    parts: List[str] = []
    parts.append(f'<svg viewBox="0 0 {width} {height}" role="img" '
                 f'aria-label="{html.escape(title)}" class="chart">')
    arrow = "▲ higher is better" if higher_better else "▼ lower is better"
    parts.append(f'<text x="0" y="20" class="chart-title">{html.escape(title)} '
                 f'<tspan class="chart-unit">({html.escape(unit)}, {arrow})</tspan></text>')

    # gridlines
    for frac in (0.25, 0.5, 0.75, 1.0):
        gx = left + frac * plot_w
        parts.append(f'<line x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" y2="{height - 20}" '
                     f'class="grid"/>')
        parts.append(f'<text x="{gx:.1f}" y="{top - 6}" class="tick">'
                     f'{vmax * frac:.0f}</text>')

    for ci, cat in enumerate(categories):
        gy = top + ci * row_h
        parts.append(f'<text x="{left - 10}" y="{gy + n_series * bar_h / 2 + 4:.1f}" '
                     f'class="ylabel">{html.escape(cat)}</text>')
        for si, s in enumerate(series):
            val = s["values"][ci] if ci < len(s["values"]) else None
            by = gy + si * bar_h
            if isinstance(val, (int, float)):
                bw = max(1.0, x(val) - left)
                parts.append(f'<rect x="{left}" y="{by + 1:.1f}" width="{bw:.1f}" '
                             f'height="{bar_h - 3}" rx="2" fill="{s["color"]}"/>')
                parts.append(f'<text x="{left + bw + 5:.1f}" y="{by + bar_h - 4:.1f}" '
                             f'class="val">{_fmt(val)}</text>')
            else:
                parts.append(f'<text x="{left + 4}" y="{by + bar_h - 4:.1f}" '
                             f'class="val na">n/a</text>')
    parts.append("</svg>")
    return "".join(parts)


def _fmt(v: float) -> str:
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def _legend(series: List[Dict[str, Any]]) -> str:
    items = "".join(
        f'<span class="lg"><i style="background:{s["color"]}"></i>{html.escape(s["name"])}</span>'
        for s in series
    )
    return f'<div class="legend">{items}</div>'


# --------------------------------------------------------------------------
# HTML assembly
# --------------------------------------------------------------------------
def build_report(results: List[Dict[str, Any]], title: str = "Local AI Benchmark") -> str:
    systems, categories, table, system_meta = _collect(results)
    color_of = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(systems)}

    def series_for(metric: str) -> List[Dict[str, Any]]:
        out = []
        for s in systems:
            vals = [_median(table[s].get(c, {}).get("aggregate", {}), metric)
                    for c in categories]
            out.append({"name": s, "color": color_of[s], "values": vals})
        return out

    charts = []
    for metric, unit, hib, cname, blurb in [
        ("gen_tps", "tokens/sec", True, "Generation speed",
         "How fast the answer streams out. One token is about ¾ of a word, so "
         "80 tok/s is roughly 60 words a second, faster than anyone reads."),
        ("ttfv_ms", "ms", False, "Time to first visible word",
         "How long the reader stares at nothing. Reasoning models think "
         "privately first, so this can be far longer than the moment the model "
         "started working. Under 300 ms feels instant."),
        ("ttft_ms", "ms", False, "Time to first token (compute)",
         "When the model produced its first token of any kind, including hidden "
         "reasoning. Identical to the visible figure on non-thinking models."),
        ("prompt_tps", "tokens/sec", True, "Prompt reading speed",
         "How fast the model ingests the prompt and any attachments. This only "
         "matters much when you feed it long documents."),
    ]:
        series = series_for(metric)
        svg = _svg_bars(cname, unit, categories, series, hib)
        charts.append(f'<section class="card"><p class="blurb">{html.escape(blurb)}</p>'
                      f'{svg}{_legend(series)}</section>')

    # System info cards
    sys_cards = []
    for s in systems:
        m = system_meta[s]
        rows = []
        for k in ("gpu", "vram_mb", "unified_memory_mb", "accelerator", "os",
                  "os_version", "driver", "machine"):
            if m.get(k):
                v = m[k]
                if k in ("vram_mb", "unified_memory_mb"):
                    v = f"{int(v) / 1024:.1f} GB"
                rows.append(f'<tr><td>{html.escape(_pretty(k))}</td>'
                            f'<td>{html.escape(str(v))}</td></tr>')
        meta = m.get("_meta", {})
        engine = meta.get("engine")
        if engine and engine != "Ollama":
            rows.append(f'<tr><td>Engine</td><td>{html.escape(str(engine))}</td></tr>')
        rows.append(f'<tr><td>Ollama</td><td>{html.escape(str(meta.get("ollama_version","?")))}</td></tr>')
        opts = meta.get("options", {})
        rows.append(f'<tr><td>Gen options</td><td>{html.escape(json.dumps(opts))}</td></tr>')
        rows.append(f'<tr><td>Repeats</td><td>{meta.get("runs","?")} measured + '
                    f'{meta.get("warmup","?")} warm-up (median reported)</td></tr>')
        sys_cards.append(
            f'<div class="syscard"><h3><span class="dot" style="background:{color_of[s]}"></span>'
            f'{html.escape(s)}</h3><table class="kv">{"".join(rows)}</table></div>'
        )

    # Detail table
    detail = _detail_table(systems, categories, table, color_of)
    outputs = _outputs_section(systems, categories, table)

    generated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return _PAGE.format(
        title=html.escape(title),
        generated=generated,
        n_sys=len(systems),
        n_cells=len(categories),
        syscards="".join(sys_cards),
        charts="".join(charts),
        detail=detail,
        outputs=outputs,
    )


def _detail_table(systems, categories, table, color_of) -> str:
    head = "".join(f'<th>{html.escape(s)}</th>' for s in systems)
    body = []
    for c in categories:
        cells = []
        for s in systems:
            cell = table[s].get(c)
            if not cell:
                cells.append('<td class="na">not run</td>')
                continue
            agg = cell.get("aggregate", {})
            gen = _median(agg, "gen_tps")
            ttft = _median(agg, "ttft_ms")
            ttfv = _median(agg, "ttfv_ms") or ttft
            sp = _spread(agg, "gen_tps")
            res = cell.get("residency", "")
            gv, gcls = speed_verdict(gen)
            spread_s = (f' <span class="sp">(range {_fmt(sp[0])} to {_fmt(sp[1])})</span>'
                        if sp else "")
            verdict = f'<div class="verdict {gcls}">{html.escape(gv)}</div>' if gv else ""
            cells.append(
                f'<td><b class="sv-{gcls}">{_fmt(gen) if gen else "not run"}</b> tok/s{spread_s}<br>'
                f'<span class="muted">first word {_fmt(ttfv) if ttfv else "—"} ms · '
                f'{html.escape(res or "unknown")}</span>{verdict}</td>'
            )
        body.append(f'<tr><td class="rowlabel">{html.escape(c)}</td>{"".join(cells)}</tr>')
    return (f'<table class="detail"><thead><tr><th>Configuration</th>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table>')


def _first_response(cell: Dict[str, Any]) -> Dict[str, Any]:
    runs = cell.get("runs", [])
    r = runs[0] if runs else {}
    return {
        "text": r.get("response_text", ""),
        "thinking_chars": r.get("thinking_chars", 0),
        "prompt_tokens": r.get("prompt_tokens"),
        "output_tokens": r.get("output_tokens"),
    }


def _attachment_html(attach: Dict[str, Any]) -> str:
    if not attach:
        return ""
    bits: List[str] = []
    files = attach.get("files", [])
    if files:
        rows = []
        for f in files:
            if f.get("error"):
                rows.append(f'<li class="att-err">{html.escape(f["name"])}: '
                            f'{html.escape(f["error"])}</li>')
            else:
                trunc = " · truncated" if f.get("truncated") else ""
                method = f.get("method", "")
                method_s = (f' · <span class="method">{html.escape(method)}</span>'
                            if method else "")
                warns = f.get("warnings") or []
                warn_s = "".join(
                    f'<div class="att-warn">⚠ {html.escape(w)}</div>' for w in warns)
                rows.append(
                    f'<li><b>{html.escape(f["name"])}</b>: {f["chars"]:,} chars '
                    f'(~{f["approx_tokens"]:,} tokens{trunc}){method_s}{warn_s}</li>')
        bits.append(f'<div class="att"><span class="att-h">Reference files</span>'
                    f'<ul>{"".join(rows)}</ul></div>')
    images = attach.get("images", [])
    if images:
        thumbs = []
        for im in images:
            if im.get("error"):
                thumbs.append(f'<span class="att-err">{html.escape(im["name"])}: '
                              f'{html.escape(im["error"])}</span>')
            elif im.get("data_uri"):
                thumbs.append(
                    f'<figure><img src="{im["data_uri"]}" alt="{html.escape(im["name"])}"/>'
                    f'<figcaption>{html.escape(im["name"])} '
                    f'({im["bytes"] // 1024} KB)</figcaption></figure>')
            else:
                thumbs.append(f'<span>{html.escape(im["name"])} '
                              f'({im.get("bytes",0)//1024} KB)</span>')
        bits.append(f'<div class="att"><span class="att-h">Images</span>'
                    f'<div class="thumbs">{"".join(thumbs)}</div></div>')
    return "".join(bits)


def _outputs_section(systems, categories, table) -> str:
    cards = []
    for cat in categories:
        # prompt text + attachments are identical across systems for a config
        sample = next((table[s][cat] for s in systems if cat in table[s]), None)
        if not sample:
            continue
        task = sample.get("prompt_text", "")
        note = sample.get("prompt_note", "")
        att_html = _attachment_html(sample.get("attachments", {}))

        responses = []
        for s in systems:
            cell = table[s].get(cat)
            if not cell:
                continue
            r = _first_response(cell)
            text = r["text"] or ""
            meta = []
            if r["prompt_tokens"] is not None:
                meta.append(f'{r["prompt_tokens"]:,} prompt tok')
            if r["output_tokens"] is not None:
                meta.append(f'{r["output_tokens"]:,} output tok')
            if r["thinking_chars"]:
                meta.append(f'{r["thinking_chars"]:,} thinking chars')
            meta_s = " · ".join(meta)
            if not text.strip():
                body = ('<em class="muted">No visible output: the model spent its '
                        'token budget on reasoning. Raise the answer-length cap or '
                        'turn off thinking to capture a final answer.</em>')
            else:
                body = f'<pre class="resp">{html.escape(text)}</pre>'
            responses.append(
                f'<details><summary>{html.escape(s)} '
                f'<span class="muted">· {html.escape(meta_s)}</span></summary>'
                f'{body}</details>')

        cards.append(
            f'<div class="outcard"><h3 class="rowlabel">{html.escape(cat)}</h3>'
            f'<div class="task"><span class="att-h">Task</span>'
            f'<pre class="taskpre">{html.escape(task)}</pre>'
            f'{("<div class=muted>" + html.escape(note) + "</div>") if note else ""}</div>'
            f'{att_html}{"".join(responses)}</div>')
    if not cards:
        return ""
    return '<h2>Task outputs &amp; attachments</h2>' + "".join(cards)


def _pretty(k: str) -> str:
    return {
        "gpu": "GPU / Chip", "vram_mb": "VRAM", "unified_memory_mb": "Unified memory",
        "accelerator": "Accelerator", "os": "OS", "os_version": "OS version",
        "driver": "Driver", "machine": "Arch",
    }.get(k, k)


_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{
    --bg:#ffffff; --fg:#1a1a1a; --muted:#6b7280; --card:#f7f7f8;
    --border:#e5e7eb; --grid:#e5e7eb; --accent:#76b900;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#0f1115; --fg:#e6e6e6; --muted:#9aa0a6; --card:#181b21;
             --border:#2a2f38; --grid:#2a2f38; }}
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
    font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  .wrap {{ max-width:1000px; margin:0 auto; padding:32px 20px 80px; }}
  header h1 {{ margin:0 0 4px; font-size:26px; }}
  header .sub {{ color:var(--muted); margin-bottom:24px; }}
  .grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
    gap:16px; margin-bottom:28px; }}
  .syscard {{ background:var(--card); border:1px solid var(--border);
    border-radius:12px; padding:16px; }}
  .syscard h3 {{ margin:0 0 10px; font-size:16px; display:flex; align-items:center; gap:8px; }}
  .dot {{ width:12px; height:12px; border-radius:3px; display:inline-block; }}
  table.kv {{ width:100%; border-collapse:collapse; font-size:13px; }}
  table.kv td {{ padding:3px 0; vertical-align:top; }}
  table.kv td:first-child {{ color:var(--muted); width:42%; }}
  .card {{ background:var(--card); border:1px solid var(--border);
    border-radius:12px; padding:16px 18px; margin-bottom:18px; overflow-x:auto; }}
  .blurb {{ margin:0 0 12px; color:var(--muted); font-size:13px; max-width:820px; }}
  .lead {{ color:var(--muted); font-size:14px; max-width:820px; margin:0 0 22px; }}
  .glossary {{ background:var(--card); border:1px solid var(--border);
    border-radius:12px; padding:14px 18px; margin-bottom:24px; }}
  .glossary summary {{ cursor:pointer; font-weight:600; font-size:14px; }}
  .glossary dl {{ margin:12px 0 0; display:grid; grid-template-columns:180px 1fr; gap:7px 16px; font-size:13px; }}
  .glossary dt {{ font-weight:600; }}
  .glossary dd {{ margin:0; color:var(--muted); }}
  @media (max-width:640px) {{ .glossary dl {{ grid-template-columns:1fr; }} .glossary dt {{ margin-top:6px; }} }}
  .verdict {{ font-size:11px; font-weight:600; margin-top:3px; }}
  .verdict.good {{ color:#2f9e44; }} .verdict.ok {{ color:#b07d00; }}
  .verdict.bad {{ color:#d64545; }} .verdict.neutral {{ color:var(--muted); font-weight:400; }}
  .sv-good {{ color:#2f9e44; }} .sv-ok {{ color:#b07d00; }} .sv-bad {{ color:#d64545; }}
  .chart {{ width:100%; height:auto; }}
  .chart-title {{ font-size:15px; font-weight:600; fill:var(--fg); }}
  .chart-unit {{ font-weight:400; fill:var(--muted); font-size:12px; }}
  .grid {{ stroke:var(--grid); stroke-width:1; }}
  .tick {{ fill:var(--muted); font-size:10px; text-anchor:middle; }}
  .ylabel {{ fill:var(--fg); font-size:11px; text-anchor:end; }}
  .val {{ fill:var(--fg); font-size:11px; font-weight:600; }}
  .val.na {{ fill:var(--muted); font-weight:400; }}
  .legend {{ display:flex; gap:16px; flex-wrap:wrap; margin-top:6px; font-size:12px;
    color:var(--muted); }}
  .lg i {{ width:11px; height:11px; border-radius:3px; display:inline-block;
    margin-right:5px; vertical-align:-1px; }}
  h2 {{ font-size:18px; margin:32px 0 12px; }}
  table.detail {{ width:100%; border-collapse:collapse; font-size:13px;
    background:var(--card); border:1px solid var(--border); border-radius:12px; overflow:hidden; }}
  table.detail th, table.detail td {{ padding:10px 12px; text-align:left;
    border-bottom:1px solid var(--border); vertical-align:top; }}
  table.detail th {{ background:rgba(127,127,127,.06); font-weight:600; }}
  .rowlabel {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }}
  .muted {{ color:var(--muted); font-size:11px; }}
  .sp {{ color:var(--muted); font-weight:400; font-size:11px; }}
  .na {{ color:var(--muted); }}
  .outcard {{ background:var(--card); border:1px solid var(--border);
    border-radius:12px; padding:16px 18px; margin-bottom:16px; }}
  .outcard h3 {{ margin:0 0 12px; font-size:14px; }}
  .att-h {{ display:block; font-size:11px; text-transform:uppercase;
    letter-spacing:.04em; color:var(--muted); margin-bottom:4px; }}
  .task, .att {{ margin-bottom:12px; }}
  .taskpre {{ white-space:pre-wrap; margin:0; font-size:12px;
    font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
    max-height:140px; overflow:auto; background:rgba(127,127,127,.06);
    padding:8px 10px; border-radius:6px; }}
  .att ul {{ margin:0; padding-left:18px; font-size:13px; }}
  .att-err {{ color:#ff453a; }}
  .method {{ color:var(--muted); font-size:11px;
    font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
  .att-warn {{ color:#e6a700; font-size:11px; margin:2px 0 0; }}
  .thumbs {{ display:flex; gap:12px; flex-wrap:wrap; }}
  .thumbs figure {{ margin:0; font-size:11px; color:var(--muted); text-align:center; }}
  .thumbs img {{ max-width:180px; max-height:180px; border-radius:8px;
    border:1px solid var(--border); display:block; }}
  details {{ border-top:1px solid var(--border); padding:8px 0 2px; }}
  summary {{ cursor:pointer; font-size:13px; font-weight:600; }}
  pre.resp {{ white-space:pre-wrap; font-size:13px; line-height:1.5;
    background:rgba(127,127,127,.06); padding:10px 12px; border-radius:6px;
    margin:8px 0 4px; max-height:420px; overflow:auto;
    font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
  footer {{ color:var(--muted); font-size:12px; margin-top:32px;
    border-top:1px solid var(--border); padding-top:16px; }}
</style></head>
<body><div class="wrap">
  <header>
    <h1>{title}</h1>
    <div class="sub">{n_sys} system(s) · {n_cells} configuration(s) · generated {generated}</div>
  </header>
  <p class="lead">This report benchmarks local AI models: how fast they respond, and where
    they run. Everything was measured on the hardware listed below, entirely on-device.
    Each number carries a plain-English reading so you can tell good from bad at a glance.</p>
  <details class="glossary">
    <summary>How to read these numbers</summary>
    <dl>
      <dt>Generation speed</dt><dd>How fast the answer streams out, in tokens per second
        (a token is about ¾ of a word). Over 30 feels smooth; over 80 is faster than reading; under 10 drags.</dd>
      <dt>Time to first visible word</dt><dd>The wait before the reader sees anything. Under 300 ms feels
        instant; over 3 seconds feels broken. On reasoning models this is the honest experience number,
        because the model may think privately for seconds before writing a word.</dd>
      <dt>Time to first token</dt><dd>When the model produced its first token of any kind, including hidden
        reasoning. Measures compute latency rather than perceived latency.</dd>
      <dt>Prompt reading speed</dt><dd>How quickly the model takes in your prompt and attachments.
        Only matters much for long documents.</dd>
      <dt>Model placement</dt><dd>Where the model ran. 100% GPU means it fit in GPU memory (fastest).
        A CPU split means it overflowed into slower system memory, the "VRAM wall".</dd>
      <dt>Range</dt><dd>The spread between the fastest and slowest of the repeated runs. A tight range
        means the measurement is trustworthy.</dd>
    </dl>
  </details>
  <div class="grid2">{syscards}</div>
  <h2>Performance</h2>
  {charts}
  <h2>Every configuration in detail</h2>
  {detail}
  {outputs}
  <footer>
    How this was measured: each configuration runs warm-up passes (thrown away), then several
    timed repeats. The median is charted, with the fastest-to-slowest range shown alongside.
    Generation settings are held constant so comparisons are fair. Time to first visible word is
    measured wall-clock on warm runs; compute time to first token separately records when the
    model began producing any token, including hidden reasoning, and the two are identical on
    models that do not reason privately. The one-off cold-start load is
    timed separately. Note that Apple Silicon shares one memory pool between CPU and GPU, so its
    memory figures are not directly comparable to a discrete NVIDIA GPU with separate VRAM. Read
    a cross-system comparison as a real-world experience comparison, not a raw GPU benchmark.
  </footer>
</div></body></html>
"""
