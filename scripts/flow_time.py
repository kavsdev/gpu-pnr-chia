#!/usr/bin/env python3
"""Placement -> detailed-route time per run, from the ORFS stage logs (speed-up plan, Ariane end-to-end target).

usage: flow_time.py [--json OUT] LABEL=DIR [LABEL=DIR ...]

DIR is a stock WORK_HOME or a raw_handoff run dir (`<design>/<req_key>`), as for stage_timing.py. Synthesis and
floorplan (1_*, 2_*, 3_1, 3_2) are identical for both flows and are excluded. Per stage: wall seconds and CPU seconds
(user + sys) from the ORFS "Elapsed time: ... CPU time: user U sys S" line. A raw_handoff run adds the pre-3_3
DREAMPlace phase (`provenance.dp_runtime_s` in result.json: GPs, rebuffer repair stages, STA), as wall time only.
Wall times are only comparable between runs made on an otherwise idle machine; CPU times are the screening proxy
under contention.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_timing import find_logs  # noqa: E402

STAGES = ["3_3_place_gp", "3_4_place_resized", "3_5_place_dp", "4_1_cts", "5_1_grt", "5_2_route", "5_3_fillcell",
          "6_1_fill", "6_1_merge", "6_report"]
_EL = re.compile(r"Elapsed time: (?:(\d+):)?(\d+):([0-9.]+)\[h:\]min:sec\. CPU time: user ([0-9.]+) sys ([0-9.]+)")


def stage_time(logs: str, stem: str):
    p = os.path.join(logs, stem + ".log")
    if not os.path.exists(p):
        return None
    m = _EL.findall(open(p, errors="replace").read())
    if not m:
        return None
    h, mi, s, u, sy = m[-1]
    return {"wall": int(h or 0) * 3600 + int(mi) * 60 + float(s), "cpu": float(u) + float(sy)}


def run_time(d: str):
    logs = find_logs(d)
    out = {"dir": logs, "stages": {}, "dp_wall": None}
    for st in STAGES:
        t = stage_time(logs, st)
        if t:
            out["stages"][st] = t
    # raw_handoff: <run>/orfs/logs/<platform>/<design>/base -> <run>/result.json
    rj = os.path.join(logs, "..", "..", "..", "..", "..", "result.json")
    if os.path.exists(rj):
        out["dp_wall"] = json.load(open(rj)).get("provenance", {}).get("dp_runtime_s")
    s = out["stages"]
    place = ["3_3_place_gp", "3_4_place_resized", "3_5_place_dp"]
    route = ["5_2_route"]
    out["place_wall"] = sum(s[x]["wall"] for x in place if x in s) + (out["dp_wall"] or 0.0)
    out["drt_wall"] = sum(s[x]["wall"] for x in route if x in s)
    out["drt_cpu"] = sum(s[x]["cpu"] for x in route if x in s)
    out["total_wall"] = sum(v["wall"] for v in s.values()) + (out["dp_wall"] or 0.0)
    out["total_cpu_orfs"] = sum(v["cpu"] for v in s.values())
    out["complete"] = "6_report" in s
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    groups = {}
    for spec in a.runs:
        lb, d = spec.split("=", 1)
        groups.setdefault(lb, []).append(run_time(d))
    cols = [("dp_wall", "DREAMPlace phase wall"), ("place_wall", "placement wall (dp+3_3..3_5)"),
            ("drt_wall", "detailed route wall"), ("drt_cpu", "detailed route CPU"),
            ("total_wall", "3_3->6 total wall"), ("total_cpu_orfs", "3_3->6 ORFS CPU")]
    print("| metric (s) | " + " | ".join(f"{lb} (n={len(g)})" for lb, g in groups.items()) + " |")
    print("|---" * (len(groups) + 1) + "|")
    for k, name in cols:
        cells = []
        for g in groups.values():
            v = [r[k] for r in g if r[k] is not None]
            cells.append("—" if not v else (f"{v[0]:.0f}" if len(v) == 1 else
                                             f"{statistics.fmean(v):.0f} [{min(v):.0f}‥{max(v):.0f}]"))
        print(f"| {name} | " + " | ".join(cells) + " |")
    for lb, g in groups.items():
        for r in g:
            if not r["complete"]:
                print(f"note: {lb} {r['dir']} has no 6_report (incomplete run)")
    if a.json:
        json.dump(groups, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
