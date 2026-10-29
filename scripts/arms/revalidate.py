#!/usr/bin/env python3
"""Re-run the missing held-out validations of a finished run_arms.py experiment and redo the comparison.

usage: revalidate.py --db X.sqlite --out X_v2.json [--workers 3] [--write X_v2_final.json]

Validation evals that failed for infrastructure reasons (e.g. --timeout-s expiring while run_arm.sh waited for a
global run slot) are retried: first from their run_arm.sh log if the container finished after the driver gave up,
otherwise by running them again. Successful validations are replayed from the ResultsDB (CachedEvaluator), so only
the missing (config, seed) pairs cost anything. No timeout is applied.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from pnr_node.arm_eval import ArmRunEvaluator, last_json_line  # noqa: E402
from pnr_node.contract import EvalRequest, EvalResult, score_timing  # noqa: E402
from pnr_node.db import CachedEvaluator, ResultsDB  # noqa: E402
from pnr_node.experiment import compare_v2, summarize_v2, validate_v2  # noqa: E402


def recover_late(db: ResultsDB, stage: str) -> int:
    """Failed `stage` evals whose run_arm.sh log now ends in a result JSON: record that result (same request)."""
    n = 0
    rows = db.con.execute("SELECT arm, round, req_json, result_json FROM evals WHERE stage=? AND score IS NULL",
                          (stage,)).fetchall()
    for arm, rnd, rq, rs in rows:
        log = (json.loads(rs).get("provenance") or {}).get("arm_log")
        if not log or not os.path.exists(log):
            continue
        d = last_json_line(open(log).read())
        if d is None:
            continue
        req = EvalRequest(**json.loads(rq))
        if db.lookup(req.key()) is not None:
            continue
        res = EvalResult.from_json(json.dumps(d))
        res.provenance = dict(res.provenance or {}, arm_log=log, recovered_late=True)
        db.record_eval(arm, rnd, req, res, "recovered_late", score_fn=score_timing)
        n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True, help="the run_arms.py result JSON")
    ap.add_argument("--write", default=None, help="where to write the updated result (default: <out>_final.json)")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args(argv)
    res = json.load(open(a.out))
    s = res["settings"]
    db = ResultsDB(a.db)
    print(f"recovered from logs: {recover_late(db, s['validate_stage'])}", flush=True)
    ev = CachedEvaluator(ArmRunEvaluator(workers=a.workers).for_arm("validate"), db)
    val = validate_v2(ev, db, res["design"], res["search"], [int(x) for x in s["validate_seeds"]],
                      s["validate_stage"])
    res["comparison"] = compare_v2(res["search"], val)
    for v in val.values():
        v.pop("_results", None)
    res["validation"] = val
    res["revalidated"] = {"cache_hits": ev.hits}
    out = a.write or os.path.splitext(a.out)[0] + "_final.json"
    json.dump(res, open(out, "w"), indent=1, default=str)
    print(summarize_v2(res))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
