#!/usr/bin/env python3
"""llmbench — local-AI benchmarking harness CLI.

Commands
    info                 Show detected system + Ollama status/models
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

from llmbench import config as cfg_mod
from llmbench import report as report_mod
from llmbench import server as server_mod
from llmbench import telemetry
from llmbench.ollama import OllamaClient
from llmbench.runner import Runner

HERE = os.path.dirname(os.path.abspath(__file__))
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


def cmd_run(args: argparse.Namespace) -> int:
    cfg = cfg_mod.load_config(args.config)
    if args.base_url:
        cfg["base_url"] = args.base_url
    if args.label:
        cfg["system_label"] = args.label
    if args.runs:
        cfg["runs"] = args.runs
    prompts = cfg_mod.load_prompts(args.prompts)

    client = OllamaClient(base_url=cfg["base_url"])
    if not client.is_up():
        print(f"error: Ollama not reachable at {cfg['base_url']} "
              f"(start it with `ollama serve`).", file=sys.stderr)
        return 1

    # Resolve relative attachment paths against the config dir, the prompts
    # dir, and the current working directory (in that order).
    base_dirs = [
        os.path.dirname(os.path.abspath(args.config)),
        os.path.dirname(os.path.abspath(args.prompts)),
        os.getcwd(),
    ]
    runner = Runner(cfg, prompts, base_dirs=base_dirs)
    results = runner.run()

    os.makedirs(args.outdir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    sys_slug = _slug(results["meta"]["system"].get("label", "system"))
    out = args.out or os.path.join(
        args.outdir, f"{_slug(cfg['name'])}_{sys_slug}_{stamp}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote results → {out}")
    print(f"Next: python bench.py report {out}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    server_mod.serve(host=args.host, port=args.port,
                     base_url=args.base_url, project_root=HERE)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    results = []
    for path in args.results:
        with open(path, "r", encoding="utf-8") as f:
            results.append(json.load(f))
    html = report_mod.build_report(results, title=args.title)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote report → {args.out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="bench.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:11434",
                   help="Ollama base URL (default: %(default)s)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("info", help="show system + Ollama status")
    pi.set_defaults(func=cmd_info)

    pr = sub.add_parser("run", help="run a benchmark config")
    pr.add_argument("config", help="path to a benchmark config JSON")
    pr.add_argument("--prompts", default=DEFAULT_PROMPTS, help="prompt set JSON")
    pr.add_argument("--label", default=None, help="override system label")
    pr.add_argument("--runs", type=int, default=None, help="override measured repeats")
    pr.add_argument("--outdir", default=os.path.join(HERE, "results"))
    pr.add_argument("--out", default=None, help="explicit output path")
    pr.set_defaults(func=cmd_run)

    psv = sub.add_parser("serve", help="launch the interactive web UI")
    psv.add_argument("--host", default="127.0.0.1")
    psv.add_argument("--port", type=int, default=8765)
    psv.set_defaults(func=cmd_serve)

    prep = sub.add_parser("report", help="build HTML report from results JSON")
    prep.add_argument("results", nargs="+", help="one or more results JSON files")
    prep.add_argument("--out", default=os.path.join(HERE, "results", "report.html"))
    prep.add_argument("--title", default="Local AI Benchmark — RTX vs Apple Silicon")
    prep.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
