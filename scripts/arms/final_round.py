#!/usr/bin/env python3
"""Routed final round over a finished run_arms.py search (pnr_node.experiment.run_final_round_v2).

The search stage (CTS) is only a filter: the top --top distinct configs it found (pooled across every arm-run in --db)
and the default flow are each run through the full flow on --select-seeds; the lowest routed score wins, and the
default is kept unless a finalist beats it. The pick is then reported against the default on --validate-seeds.

usage: final_round.py --design swerv_wrapper --db X.sqlite --out X_final.json [--top 3] [--select-seeds 8,9]
                      [--validate-seeds 101,102,103,104] [--orfs-var K=V ...] [--workers 4] [--dry-run]

--orfs-var must match the search's, so the evaluator identity (and the cache) is the same.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from pnr_node.experiment import run_final_round_v2, summarize_final_round  # noqa: E402
from run_arms import _ints, orfs_var_args  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--design", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="cts", help="the search stage the finalists are pooled from")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--min-seeds", type=int, default=2, help="finalists need a search score over >= this many seeds")
    ap.add_argument("--select-seeds", default="8,9")
    ap.add_argument("--select-stage", default="route")
    ap.add_argument("--min-gain", default="auto",
                    help="a finalist must beat the default's routed mean by more than this; 'auto' (default) = "
                         "2*sd*sqrt(2/n_select seeds) with sd = the default flow's search-stage seed sd; 0 = any gain")
    ap.add_argument("--validate-seeds", default="101,102")
    ap.add_argument("--validate-stage", default="route")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout-s", type=float, default=None)
    ap.add_argument("--orfs-var", action="append", default=[], metavar="K=V")
    ap.add_argument("--dry-run", action="store_true", help="mock evaluator")
    a = ap.parse_args(argv)
    if set(_ints(a.select_seeds)) & set(_ints(a.validate_seeds)):
        raise SystemExit("--select-seeds and --validate-seeds must not overlap (the held-out seeds stay held out)")
    if a.dry_run:
        from pnr_node.mock_eval import FlowMockEvaluator
        evaluator = FlowMockEvaluator()
    else:
        from pnr_node.arm_eval import ArmRunEvaluator
        evaluator = ArmRunEvaluator(workers=a.workers, timeout_s=a.timeout_s, extra_args=orfs_var_args(a.orfs_var))
    from pnr_node.db import ResultsDB
    from pnr_node.evidence import atomic_write_json

    min_gain = a.min_gain if a.min_gain == "auto" else float(a.min_gain)
    res = run_final_round_v2(evaluator, ResultsDB(a.db), a.design, stage=a.stage, top=a.top,
                             select_seeds=_ints(a.select_seeds), select_stage=a.select_stage, min_gain=min_gain,
                             validate_seeds=_ints(a.validate_seeds), validate_stage=a.validate_stage,
                             min_seeds=a.min_seeds)
    res["orfs_vars"] = list(a.orfs_var)
    res["dry_run"] = a.dry_run
    atomic_write_json(a.out, res)
    print(summarize_final_round(res), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
