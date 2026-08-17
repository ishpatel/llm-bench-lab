#!/usr/bin/env python3
"""llmbench: local-AI benchmarking harness CLI.

Commands
    info                 Show detected system + Ollama status/models
    doctor               Check dependencies + environment before benchmarking
    run   CONFIG         Run a benchmark config, write a results JSON
    report RESULTS...    Build an HTML report from one or more results files

Zero third-party dependencies; Python 3.9+. Runs identically on Apple Silicon
(macOS) and NVIDIA (Windows/Linux). Point it at different machines, then merge
the results files into one cross-system report.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
from typing import Any, Dict

from llmbench import config as cfg_mod
from llmbench import console as console_mod
from llmbench import report as report_mod
from llmbench import server as server_mod
from llmbench import telemetry
from llmbench.ollama import OllamaClient
from llmbench.runner import Runner

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_PROMPTS = os.path.join(HERE, "configs", "prompts.json")


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def cmd_info(args: argparse.Namespace) -> int:
    sysinfo = telemetry.describe_system()
    print("System")
    for k, v in sysinfo.items():
        print(f"  {k:20} {v}")
    client = OllamaClient(base_url=args.base_url)
    print(f"\nOllama @ {args.base_url}")
    if not client.is_up():
        print("  server: NOT reachable (is `ollama serve` running?)")
        return 1
    print("  server: up")
    models = client.list_models()
    print(f"  models ({len(models)}):")
    for m in models:
        print(f"    - {m}")
    ps = telemetry.ollama_ps()
    if ps:
        print("  resident now:")
        for name, meta in ps.items():
            print(f"    - {name}: {meta.get('processor','')}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check the environment before a campaign. Exits non-zero when something
    blocks a benchmark, so it can gate a run: `bench.py doctor && bench.py run`."""
    from llmbench import readiness as readiness_mod
    client = OllamaClient(base_url=args.base_url)
    up = client.is_up()
    deep = telemetry.describe_system_deep()
    report = readiness_mod.describe_readiness(
        client=client, base_url=args.base_url,
        models=client.models_detailed() if up else [],
        project_root=HERE, deep=deep)
    if args.json:
        report["system"] = telemetry.describe_system().get("label", "")
        print(json.dumps(report, indent=2))
    else:
        console_mod.print_readiness(
            report, system_label=telemetry.describe_system().get("label", ""),
            verbose=args.verbose)
    return 1 if report["state"] == "fail" else 0


def _config_error(exc: Exception, args: argparse.Namespace) -> int:
    """Turn a config or prompt-set problem into an explanation. These are the
    first errors a new machine hits, so a stack trace is the wrong answer."""
    if isinstance(exc, FileNotFoundError):
        missing = getattr(exc, "filename", "") or ""
        what = "prompt set" if missing == args.prompts else "config"
        print(f"error: {what} not found: {missing}", file=sys.stderr)
        if what == "config":
            here = os.path.join(HERE, "configs")
            names = sorted(f for f in os.listdir(here) if f.endswith(".json")
                           ) if os.path.isdir(here) else []
            if names:
                print(f"       available configs: {', '.join(names)}",
                      file=sys.stderr)
    elif isinstance(exc, json.JSONDecodeError):
        print(f"error: {args.config} is not valid JSON ({exc}).", file=sys.stderr)
    elif isinstance(exc, KeyError):
        print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        print(f"       prompt keys live in {args.prompts}", file=sys.stderr)
    else:
        print(f"error: {exc}", file=sys.stderr)
    return 1


def _check_models(cfg: Dict[str, Any], client: "OllamaClient") -> int:
    """Report every missing model at once, with the command that fixes it.
    Returns non-zero only when nothing in the config could run."""
    available = set(client.list_models())
    wanted = list(dict.fromkeys(cfg["models"]))
    missing = [m for m in wanted if m not in available]
    if not missing:
        return 0

    for m in missing:
        print(f"warning: {m} is not pulled on this machine", file=sys.stderr)
    print(f"To install: {'; '.join('ollama pull ' + m for m in missing)}",
          file=sys.stderr)

    if len(missing) < len(wanted):
        runnable = [m for m in wanted if m in available]
        print(f"Continuing with the {len(runnable)} model(s) present: "
              f"{', '.join(runnable)}", file=sys.stderr)
        return 0

    print(f"error: none of the {len(wanted)} model(s) in this config are "
          f"installed, so there is nothing to measure.", file=sys.stderr)
    if available:
        print(f"       installed here: {', '.join(sorted(available))}",
              file=sys.stderr)
    else:
        print("       no models are installed at all "
              "(`ollama pull qwen3:4b-q4_K_M` is a good first one).",
              file=sys.stderr)
    return 1


def _adhoc_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Build a config dict straight from -m/-p flags. Third front door to the
    same contract the JSON configs and the web UI already use: a developer can
    benchmark a model without writing a file first."""
    cfg = dict(cfg_mod.DEFAULTS)
    cfg["options"] = dict(cfg_mod.DEFAULTS["options"])
    cfg["name"] = "adhoc"
    cfg["models"] = list(dict.fromkeys(args.model))
    if args.prompt:
        cfg["prompts"] = [{"key": "adhoc", "text": args.prompt}]
    else:
        cfg["prompts"] = ["short_qa"]   # sensible default from the prompt set
    if args.quick:
        cfg["runs"], cfg["warmup"] = 1, 0
        cfg["measure_cold_start"] = False
    return cfg


def cmd_run(args: argparse.Namespace) -> int:
    if bool(args.config) == bool(args.model):
        print("error: pass either a config file or -m/--model (with an "
              "optional -p/--prompt), not both.\n"
              "  quick check:   bench.py run -m qwen3:4b-q4_K_M --quick\n"
              "  full config:   bench.py run configs/cross-system.json",
              file=sys.stderr)
        return 2
    try:
        cfg = _adhoc_config(args) if args.model else cfg_mod.load_config(args.config)
        if args.base_url:
            cfg["base_url"] = args.base_url
        if args.label:
            cfg["system_label"] = args.label
        if args.runs:
            cfg["runs"] = args.runs
        prompts = cfg_mod.load_prompts(args.prompts)
        cfg_mod.resolve_prompts(cfg["prompts"], prompts)   # fail before measuring
    except (OSError, ValueError, KeyError) as exc:
        return _config_error(exc, args)

    # Validate the engine this config actually uses. A config pointing at an
    # external OpenAI-compatible endpoint (trtllm-serve, NIM, vLLM) must not
    # require Ollama to be running as well: on the RTX machine the TensorRT-LLM
    # server may be the only engine present.
    backend = cfg.get("backend") or None
    if backend and backend.get("type") == "openai":
        from llmbench.backends import OpenAICompatClient
        url = backend.get("base_url", "")
        if not OpenAICompatClient(url).is_up():
            print(f"error: {backend.get('label', 'endpoint')} not reachable at "
                  f"{url} (start the server, e.g. `trtllm-serve <model>`).",
                  file=sys.stderr)
            return 1
    else:
        client = OllamaClient(base_url=cfg["base_url"])
        if not client.is_up():
            print(f"error: Ollama not reachable at {cfg['base_url']} "
                  f"(start it with `ollama serve`).", file=sys.stderr)
            return 1
        # Check the models up front. Discovering this per cell means a long
        # config can spend minutes measuring before revealing that the model
        # it was pointed at was never pulled.
        rc = _check_models(cfg, client)
        if rc:
            return rc

    # Resolve relative attachment paths against the config dir, the prompts
    # dir, and the current working directory (in that order).
    base_dirs = [
        os.path.dirname(os.path.abspath(args.config)) if args.config else HERE,
        os.path.dirname(os.path.abspath(args.prompts)),
        os.getcwd(),
    ]
    runner = Runner(cfg, prompts, base_dirs=base_dirs)
    results = runner.run()

    # An empty result is a failure, not an empty success. Writing the file and
    # printing "Wrote results" would read as if the benchmark had worked.
    if not results.get("cells"):
        print("error: no cells were measured, so no results were written. "
              "The log above says why each one was skipped.", file=sys.stderr)
        return 1

    os.makedirs(args.outdir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    sys_slug = _slug(results["meta"]["system"].get("label", "system"))
    out = args.out or os.path.join(
        args.outdir, f"{_slug(cfg['name'])}_{sys_slug}_{stamp}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    console_mod.print_summary(results)
    print(f"\nWrote results → {out}")
    print(f"Next: python bench.py report {out}   (or: python bench.py summary {out})")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    server_mod.serve(host=args.host, port=args.port,
                     base_url=args.base_url, project_root=HERE)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from llmbench.store import RunStore
    store = RunStore(os.path.join(HERE, "runs"))
    bundle = store.export_bundle(args.run_id)
    if bundle is None:
        print(f"error: run '{args.run_id}' not found", file=sys.stderr)
        return 1
    out = args.out or f"{args.run_id}.llmbench.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(bundle, f)
    print(f"Exported run → {out}  (copy to the other machine, then: "
          f"python bench.py import {os.path.basename(out)})")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    from llmbench.store import RunStore
    store = RunStore(os.path.join(HERE, "runs"))
    with open(args.bundle, "r", encoding="utf-8") as f:
        bundle = json.load(f)
    if bundle.get("format") != "llmbench.run.v1":
        print("error: not an llmbench run bundle", file=sys.stderr)
        return 1
    rid = store.import_bundle(bundle)
    print(f"Imported run → {rid}  (now visible in the web UI / runs list)")
    return 0


def cmd_adopt(args: argparse.Namespace) -> int:
    """Turn CLI results files into web-UI runs.

    A CLI batch writes one file containing many cells; the web UI stores one
    folder per run. This splits each cell into its own run so batch results
    show up in the Runs, Compare and Cross-System views alongside interactive
    ones, without re-running anything.
    """
    from llmbench.server import summarize
    from llmbench.store import RunStore

    store = RunStore(os.path.join(HERE, "runs"))
    existing = {r.get("source_cell") for r in store.list() if r.get("source_cell")}
    adopted = skipped = 0

    for path in sorted(args.results):
        with open(path, "r", encoding="utf-8") as f:
            result = json.load(f)
        meta_in = result.get("meta", {})
        if not result.get("cells"):
            print(f"  ! {os.path.basename(path)}: no cells, skipping", file=sys.stderr)
            continue
        stamp = datetime.datetime.fromtimestamp(
            os.path.getmtime(path)).isoformat(timespec="seconds")

        for i, cell in enumerate(result["cells"], 1):
            # Stable identity so re-adopting the same file does not duplicate.
            fingerprint = f"{os.path.basename(path)}#{cell['label']}"
            if fingerprint in existing:
                skipped += 1
                continue
            run_id = store._unique_id(
                f"{_slug(meta_in.get('config_name','run'))}-"
                f"{_slug(meta_in.get('system',{}).get('label','sys'))}-{i:02d}")
            summary = summarize(cell, meta_in)
            store.save(run_id, {
                "id": run_id,
                "created": stamp,
                "state": "done",
                "name": f"{meta_in.get('config_name','run')} · {cell['label']}",
                "model": cell.get("model", ""),
                "engine": meta_in.get("engine", "Ollama"),
                "system": meta_in.get("system", {}),
                "options": meta_in.get("options", {}),
                "num_ctx": cell.get("context_length"),
                "runs": meta_in.get("runs"),
                "warmup": meta_in.get("warmup"),
                "measure_cold_start": meta_in.get("measure_cold_start"),
                "prompt": {"text": cell.get("prompt_text", ""),
                           "files": [], "images": []},
                "metrics": summary["metrics"],
                "attachments": cell.get("attachments", {}),
                "output": summary["output"],
                "thinking_chars": summary["thinking_chars"],
                "ollama_version": meta_in.get("ollama_version", ""),
                "source_cell": fingerprint,
                "adopted_from": os.path.basename(path),
            # The per-run report needs a results-shaped document, so give it
            # this cell alone under the original run's metadata.
            }, {"meta": meta_in, "cells": [cell]})
            adopted += 1
        print(f"  {os.path.basename(path)}: {len(result['cells'])} cell(s)")

    print(f"\nAdopted {adopted} run(s)" +
          (f", skipped {skipped} already present" if skipped else ""))
    print("Open the web UI to browse them: python bench.py serve")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from llmbench import evals
    from llmbench.agent import Agent
    from llmbench.ollama import OllamaClient
    client = OllamaClient(base_url=args.base_url)
    if not client.is_up():
        print("error: Ollama not reachable", file=sys.stderr)
        return 1
    path = args.suite if os.path.isfile(args.suite) else os.path.join(
        HERE, "evals", f"{args.suite}_tests.json")
    with open(path, "r", encoding="utf-8") as f:
        suite = json.load(f)
    if args.model:
        suite["answer_model"] = args.model

    if "guardrail" in os.path.basename(path):
        agent = Agent(client, suite.get("answer_model", "qwen3:4b-q4_K_M"))
        r = evals.run_guardrail_suite(agent, suite, log=lambda m: print(m, file=sys.stderr))
        print(f"\nGuardrails: {r['summary']['passed']}/{r['summary']['total']} passed")
        for row in r["rows"]:
            print(f"  [{'PASS' if row['pass'] else 'FAIL'}] {row['id']:16} ({row['expect']})")
    else:
        r = evals.run_rag_suite(client, suite, base_dirs=[HERE],
                                log=lambda m: print(m, file=sys.stderr))
        s = r["summary"]
        print(f"\nRAG  {s['rag']['passed']}/{s['rag']['total']} passed · "
              f"{s['rag']['hallucinations']} hallucinations · {s['rag']['grounded']} grounded")
        print(f"RAW  {s['raw']['passed']}/{s['raw']['total']} passed · "
              f"{s['raw']['hallucinations']} hallucinations · {s['raw']['grounded']} grounded")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    results = []
    for path in args.results:
        with open(path, "r", encoding="utf-8") as f:
            results.append(json.load(f))
    html = report_mod.build_report(results, title=args.title)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    for res in results:
        console_mod.print_summary(res)
    print(f"\nWrote report → {args.out}")
    return 0


def _summary_records(results: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one results document into jq-friendly per-cell records: medians
    up front, no digging through the aggregate schema."""
    meta = results.get("meta") or {}

    def med(agg, k):
        m = (agg or {}).get(k)
        return m.get("median") if isinstance(m, dict) else None

    cells = []
    for c in results.get("cells") or []:
        agg = c.get("aggregate") or {}
        cells.append({
            "label": c.get("label"), "model": c.get("model"),
            "prompt_key": c.get("prompt_key"),
            "context_length": c.get("context_length"),
            "gen_tps": med(agg, "gen_tps"), "prompt_tps": med(agg, "prompt_tps"),
            "ttfv_ms": med(agg, "ttfv_ms"), "ttft_ms": med(agg, "ttft_ms"),
            "cold_load_ms": (c.get("cold_start") or {}).get("load_ms"),
            "prompt_tokens": med(agg, "prompt_tokens"),
            "output_tokens": med(agg, "output_tokens"),
            "residency": c.get("residency"),
            "n_ok": agg.get("n_ok"), "n_total": agg.get("n_total"),
        })
    return {"config": meta.get("config_name"),
            "system": (meta.get("system") or {}).get("label"),
            "engine": meta.get("engine", "Ollama"), "cells": cells}


def cmd_summary(args: argparse.Namespace) -> int:
    """Re-print the terminal summary for results saved earlier."""
    docs = []
    for path in args.results:
        with open(path, "r", encoding="utf-8") as f:
            docs.append(json.load(f))
    if args.json:
        print(json.dumps([_summary_records(d) for d in docs], indent=2))
    else:
        for d in docs:
            console_mod.print_summary(d)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="bench.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  bench.py doctor                                   is this machine ready?
  bench.py run -m qwen3:4b-q4_K_M --quick           ad-hoc look at one model
  bench.py run -m MODEL -p "Summarize RAG in a sentence."
  bench.py run configs/cross-system.json --label "RTX 5070 (8GB)"
  bench.py summary results/*.json --json | jq '.[].cells[] | {model, gen_tps}'
  bench.py report results/cross-system_*.json --out results/cross.html
  bench.py serve                                    the web UI, for everyone else""")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL,
                   help="Ollama base URL (default: %(default)s). Accepted "
                        "before or after the subcommand.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def with_base_url(parser):
        """Accept --base-url after the subcommand too, which is where it reads
        naturally. SUPPRESS keeps an unused flag here from overwriting a value
        given before the subcommand; passing it in both places lets the later,
        more specific one win."""
        # No %(default)s here: with SUPPRESS there is no default to expand and
        # argparse raises KeyError when rendering --help.
        parser.add_argument("--base-url", default=argparse.SUPPRESS,
                            help=f"Ollama base URL (default: {DEFAULT_BASE_URL})")
        return parser


    pi = sub.add_parser("info", help="show system + Ollama status")
    with_base_url(pi)
    pi.set_defaults(func=cmd_info)

    pdoc = sub.add_parser("doctor",
                          help="check dependencies + environment for this machine")
    with_base_url(pdoc)
    pdoc.add_argument("--verbose", action="store_true",
                      help="explain every check, not only the ones needing action")
    pdoc.add_argument("--json", action="store_true",
                      help="machine-readable report (exit code is unchanged)")
    pdoc.set_defaults(func=cmd_doctor)

    pr = sub.add_parser("run", help="run a benchmark (config file, or -m MODEL for ad-hoc)")
    with_base_url(pr)
    pr.add_argument("config", nargs="?", default=None,
                    help="path to a benchmark config JSON (or use -m instead)")
    pr.add_argument("-m", "--model", action="append", default=None, metavar="MODEL",
                    help="benchmark this model ad hoc, no config file needed "
                         "(repeat for several)")
    pr.add_argument("-p", "--prompt", default=None,
                    help="inline prompt for an ad-hoc run (default: the "
                         "short_qa prompt from the prompt set)")
    pr.add_argument("--quick", action="store_true",
                    help="1 repeat, no warm-up, no cold start: a fast look, "
                         "not a publishable number")
    pr.add_argument("--prompts", default=DEFAULT_PROMPTS, help="prompt set JSON")
    pr.add_argument("--label", default=None, help="override system label")
    pr.add_argument("--runs", type=int, default=None, help="override measured repeats")
    pr.add_argument("--outdir", default=os.path.join(HERE, "results"))
    pr.add_argument("--out", default=None, help="explicit output path")
    pr.set_defaults(func=cmd_run)

    psv = sub.add_parser("serve", help="launch the interactive web UI")
    with_base_url(psv)
    psv.add_argument("--host", default="127.0.0.1")
    psv.add_argument("--port", type=int, default=8765)
    psv.set_defaults(func=cmd_serve)

    pex = sub.add_parser("export", help="export a run to a portable bundle")
    pex.add_argument("run_id")
    pex.add_argument("--out", default=None)
    pex.set_defaults(func=cmd_export)

    pim = sub.add_parser("import", help="import a run bundle from another machine")
    pim.add_argument("bundle")
    pim.set_defaults(func=cmd_import)

    pad = sub.add_parser("adopt", help="import CLI results files into the web-UI run history")
    pad.add_argument("results", nargs="+", help="one or more results JSON files")
    pad.set_defaults(func=cmd_adopt)

    pev = sub.add_parser("eval", help="run an evaluation suite (rag | guardrails)")
    with_base_url(pev)
    pev.add_argument("suite", help="suite name matching evals/<name>_tests.json "
                                   "('rag', 'guardrails', 'life_lab'), or a path "
                                   "to a suite JSON")
    pev.add_argument("--model", default=None, help="override answer model")
    pev.set_defaults(func=cmd_eval)

    psum = sub.add_parser("summary", help="print metrics for saved results JSON")
    psum.add_argument("results", nargs="+", help="one or more results JSON files")
    psum.add_argument("--json", action="store_true",
                      help="flat per-cell records for jq/scripts instead of the table")
    psum.set_defaults(func=cmd_summary)

    prep = sub.add_parser("report", help="build HTML report from results JSON")
    prep.add_argument("results", nargs="+", help="one or more results JSON files")
    prep.add_argument("--out", default=os.path.join(HERE, "results", "report.html"))
    prep.add_argument("--title", default="Local AI Benchmark: RTX vs Apple Silicon")
    prep.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
