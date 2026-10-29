#!/usr/bin/env python3
"""v2 arm comparison driver, host side: plain python3 + this repo, ResultsDB for history/caching.

Real runs:  scripts/arms/run_arms.py --design ariane133 --arms default,random,tpe,llm --reps 3 --budget-evals 24 \
                --batch 4 --stage cts --db v2.sqlite --out v2.json
Dry run:    ... --dry-run   (FlowMockEvaluator instead of run_arm.sh; the LLM arm still calls the real provider
                             unless --llm-provider fake)

The Gemini key is read from ~/work/secrets/gemini.env (KEY=VALUE lines) into the environment; it is never printed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pnr_node.db import ResultsDB  # noqa: E402
from pnr_node.experiment import run_experiment_v2, summarize_v2  # noqa: E402
from pnr_node.llm import CostMeter, gemini_list_models, make_provider, pick_newest_pro  # noqa: E402

SECRETS = os.path.expanduser("~/work/secrets/gemini.env")


def load_env_file(path: str, environ=os.environ) -> list:
    """Load KEY=VALUE lines (comments, blank lines, `export ` prefix and surrounding quotes tolerated) into
    `environ` without overriding variables already set. Returns the key NAMES loaded (never the values)."""
    names = []
    try:
        lines = open(path).read().splitlines()
    except OSError:
        return names
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        if k and k not in environ:
            environ[k] = v
            names.append(k)
    return names


def _ints(s: str):
    return tuple(int(x) for x in s.split(",") if x.strip() != "")


def orfs_var_args(kvs):
    """["K=V", ...] -> run_arm.sh args ["--orfs-var", "K=V", ...] (rejects anything that is not K=V)."""
    out = []
    for kv in kvs:
        if "=" not in kv or not kv.split("=", 1)[0]:
            raise SystemExit(f"--orfs-var expects K=V, got {kv!r}")
        out += ["--orfs-var", kv]
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", default="ariane133")
    ap.add_argument("--arms", default="default,random,tpe,llm")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--budget-evals", type=int, default=24, help="evaluations per searching arm-run (all count)")
    ap.add_argument("--batch", type=int, default=4, help="configs proposed per round")
    ap.add_argument("--stage", default="cts", choices=["proxy", "grt", "route", "cts"])
    ap.add_argument("--seeds", default="0", help="search seeds per proposed config")
    ap.add_argument("--promote-top", type=int, default=2)
    ap.add_argument("--promote-seeds", default="1")
    ap.add_argument("--default-seeds", default="0,1,2,3", help="seeds for the default arm (the seed lottery)")
    ap.add_argument("--validate-seeds", default="101,102", help="held-out seeds (tinyRocket: 101,102,103,104)")
    ap.add_argument("--validate-stage", default="route")
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--workers", type=int, default=3, help="max concurrent evaluations (run_arm.sh docker runs)")
    ap.add_argument("--arm-workers", type=int, default=None, help="concurrent arm-runs (default: --workers)")
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline-hours", type=float, default=None, help="stop starting new search rounds after this")
    ap.add_argument("--llm-provider", default="gemini_rest", help="gemini_rest (default) | fake | any make_provider kind")
    ap.add_argument("--llm-model", default="auto", help="'auto' = newest pro Gemini from GET /v1beta/models")
    ap.add_argument("--llm-budget-usd", type=float, default=50.0, help="hard cap on the LLM cost estimate")
    ap.add_argument("--llm-price-in", type=float, default=2.50, help="USD per 1M input tokens (estimate)")
    ap.add_argument("--llm-price-out", type=float, default=15.0, help="USD per 1M output (+thinking) tokens")
    ap.add_argument("--secrets", default=SECRETS)
    ap.add_argument("--timeout-s", type=float, default=None, help="per-eval timeout for run_arm.sh")
    ap.add_argument("--orfs-var", action="append", default=[], metavar="K=V",
                    help="ORFS make variable for every eval (run_arm.sh --orfs-var), e.g. OPENROAD_HIERARCHICAL=0")
    ap.add_argument("--dry-run", action="store_true", help="mock evaluator (FlowMockEvaluator); runs in seconds")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    arms = [x for x in a.arms.split(",") if x]
    if "llm_diag" in arms:  # hypotheses JSONL next to --out
        from pnr_node.diag_proposer import register_llm_diag; register_llm_diag(log_path=os.path.splitext(a.out)[0] + ".llm_diag_hypotheses.jsonl")  # noqa: E702
    if a.dry_run:
        from pnr_node.mock_eval import FlowMockEvaluator

        evaluator = FlowMockEvaluator()
    else:
        from pnr_node.arm_eval import ArmRunEvaluator

        evaluator = ArmRunEvaluator(workers=a.workers, timeout_s=a.timeout_s, extra_args=orfs_var_args(a.orfs_var))

    meter = CostMeter(a.llm_budget_usd, a.llm_price_in, a.llm_price_out)
    factory, model = None, None
    if any(x.startswith("llm") for x in arms):
        if a.llm_provider == "fake":
            factory, model = (lambda: make_provider({"provider": "fake"})), "fake"
        else:
            loaded = load_env_file(a.secrets)
            print(f"secrets: loaded {len(loaded)} variable(s) from {a.secrets}", flush=True)
            model = a.llm_model
            if model == "auto":
                model = pick_newest_pro(gemini_list_models())
                if not model:
                    print("no pro Gemini model found; pass --llm-model", file=sys.stderr)
                    return 2
            print(f"LLM: provider={a.llm_provider} model={model} cap=${a.llm_budget_usd}", flush=True)
            factory = lambda: make_provider({"provider": a.llm_provider, "model": model, "meter": meter})

    deadline = time.time() + a.deadline_hours * 3600 if a.deadline_hours is not None else None
    db = ResultsDB(a.db)
    t0 = time.time()
    res = run_experiment_v2(
        evaluator, db, a.design, arms=arms, reps=a.reps, budget_evals=a.budget_evals, batch=a.batch,
        stage=a.stage, seeds=_ints(a.seeds), promote_top=a.promote_top, promote_seeds=_ints(a.promote_seeds),
        default_seeds=_ints(a.default_seeds), validate_seeds=_ints(a.validate_seeds), validate_stage=a.validate_stage,
        workers=a.arm_workers or a.workers, deadline=deadline, provider_factory=factory, validate=not a.no_validate,
        log=lambda m: print(m, flush=True))
    res["llm"] = dict(meter.summary(), provider=a.llm_provider if factory else None, model=model)
    res["wall_s"] = round(time.time() - t0, 1)
    res["dry_run"] = a.dry_run
    res["orfs_vars"] = list(a.orfs_var)
    from pnr_node.evidence import atomic_write_json

    atomic_write_json(a.out, res)
    print(summarize_v2(res))
    print("LLM usage: " + json.dumps(res["llm"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
