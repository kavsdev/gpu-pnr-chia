#!/usr/bin/env python3
"""Seed a run_arms.py ResultsDB with flow runs made outside it (e.g. the default-flow noise-floor runs from
scripts/gap/jobs_*_cts_default.txt), so the experiment replays them as cache hits instead of re-running.

usage: import_runs.py --db X.sqlite --design D --stage cts [--flow-cfg JSON] SEED=LOG [SEED=LOG ...]

Each LOG is a run_arm.sh log (last JSON line = the EvalResult). The request is built exactly as run_arm_v2 builds
it (ArmRunEvaluator's host tool_id, default placer_cfg, clamped flow_cfg), so its key matches the experiment's.
Rows go under arm "import" (cache only; no arm-run's statistics read them). Only ok results are imported.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from pnr_node.arm_eval import ArmRunEvaluator, last_json_line  # noqa: E402
from pnr_node.contract import FLOW_KNOBS, EvalRequest, EvalResult, clamp_config, default_config, score_timing  # noqa: E402
from pnr_node.db import ResultsDB  # noqa: E402
from run_arms import orfs_var_args  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", required=True)
    ap.add_argument("--design", required=True)
    ap.add_argument("--stage", default="cts")
    ap.add_argument("--tech", default="nangate45")
    ap.add_argument("--flow-cfg", default="{}")
    ap.add_argument("--orfs-var", action="append", default=[], metavar="K=V",
                    help="as given to run_arms.py (part of the evaluator identity, so keys match)")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args(argv)
    db = ResultsDB(a.db)
    tool_id = ArmRunEvaluator(extra_args=orfs_var_args(a.orfs_var)).tool_id
    fc = clamp_config(json.loads(a.flow_cfg), FLOW_KNOBS)
    n = 0
    for spec in a.runs:
        seed, log = spec.split("=", 1)
        d = last_json_line(open(log).read())
        if d is None:
            print(f"skip {log}: no result JSON")
            continue
        res = EvalResult.from_json(json.dumps(d))
        if not res.ok or res.stage != a.stage:
            print(f"skip {log}: ok={res.ok} stage={res.stage}")
            continue
        req = EvalRequest(a.design, a.tech, default_config(), int(seed), a.stage, tool_id, flow_cfg=fc)
        if db.lookup(req.key()) is not None:
            print(f"have seed {seed} ({req.key()})")
            continue
        res.provenance = dict(res.provenance or {}, imported_from=log)
        db.record_eval("import", 0, req, res, "imported", score_fn=score_timing)
        n += 1
        print(f"seed {seed}: wns {res.wns} tns {res.tns} score {score_timing(res):.3f} key {req.key()}")
    print(f"imported {n}")


if __name__ == "__main__":
    main()
