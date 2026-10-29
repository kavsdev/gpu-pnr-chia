#!/usr/bin/env python3
"""Per-stage timing/buffering/congestion table for ORFS runs.

usage: stage_timing.py [--json OUT] [--stages 3_3,3_4,...] LABEL=DIR [LABEL=DIR ...]

DIR is an ORFS `logs/<platform>/<design>/base` dir, or any dir above one (a stock WORK_HOME, a raw_handoff run dir
`<design>/<req_key>` or its `orfs/`). Several DIRs with the same LABEL are one group (e.g. seeds) and are reported
as mean and min..max. Every number comes from the stage's metrics JSON or its log; nothing is recomputed.

Per stage (3_3 GP, 3_4 resize, 3_5 OpenDP, 4_1 CTS, 5_1 grt, 6 final): setup WNS/TNS and violating endpoints,
instance count and area, buffers inserted/removed and instances resized by the resizer in that stage, OpenDP mean
displacement, grt / estimated wirelength, grt overflow and vias. Buffer totals by class (`timing_repair_buffer`)
exist only at 6_report.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys
from typing import Dict, List, Optional

# stage -> (metrics json / log stem, metric prefix)
STAGES = {
    "3_3": ("3_3_place_gp", "globalplace"),
    "3_4": ("3_4_place_resized", "placeopt"),
    "3_5": ("3_5_place_dp", "detailedplace"),
    "4_1": ("4_1_cts", "cts"),
    "5_1": ("5_1_grt", "globalroute"),
    "6": ("6_report", "finish"),
}

_INS = re.compile(r"RSZ-00(?:27|28|38|40|45)\] Inserted (\d+) (?:BUF_X\d+ (?:input|output) )?buffers")
_REM = re.compile(r"RSZ-00(?:26|59)\] Removed (\d+) buffers")
_RSZ = re.compile(r"RSZ-00(?:39|51)\] Resized (\d+) instances")
_GRT_TOTAL = re.compile(r"^Total\s+(\d+)\s+(\d+)\s+([\d.]+)%\s+(\d+)\s*/\s*(\d+)\s*/\s*(\d+)", re.M)
_VIAS = re.compile(r"GRT-0111\] Final number of vias: (\d+)")


def find_logs(d: str) -> str:
    d = os.path.abspath(d)
    if glob.glob(os.path.join(d, "3_3_place_gp.*")) or glob.glob(os.path.join(d, "5_1_grt.*")):
        return d
    hits = sorted(glob.glob(os.path.join(d, "**", "logs", "*", "*", "base"), recursive=True))
    hits = [h for h in hits if os.path.exists(os.path.join(h, "3_3_place_gp.log"))]
    if not hits:
        raise SystemExit(f"no ORFS logs dir under {d}")
    return min(hits, key=len)  # the run's own logs, not a nested STA work dir (netwt/orfs)


def _num(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def stage_metrics(logs: str, stem: str, pre: str) -> Optional[Dict]:
    jp, lp = os.path.join(logs, stem + ".json"), os.path.join(logs, stem + ".log")
    if not os.path.exists(jp) and not os.path.exists(lp):
        return None
    j = json.load(open(jp)) if os.path.exists(jp) else {}
    log = open(lp, errors="replace").read() if os.path.exists(lp) else ""

    def g(k):
        return _num(j.get(f"{pre}__{k}"))

    m = {"wns": g("timing__setup__ws"), "tns": g("timing__setup__tns"),
         "viol": g("timing__drv__setup_violation_count"),
         "inst": g("design__instance__count"), "area": g("design__instance__area"),
         "buf_ins": sum(int(x) for x in _INS.findall(log)), "buf_rem": sum(int(x) for x in _REM.findall(log)),
         "resized": sum(int(x) for x in _RSZ.findall(log))}
    if pre == "detailedplace":
        m["disp_mean"] = g("design__instance__displacement__mean")
        m["disp_max"] = g("design__instance__displacement__max")
    if pre in ("globalplace", "detailedplace", "cts", "globalroute"):
        m["est_wl"] = g("route__wirelength__estimated")
    if pre == "globalplace":
        m["gpl_congestion"] = g("gpl__routability__congestion")
        m["gpl_td_area_pct"] = g("gpl__area__timing_delta__percent")
        m["gpl_rd_inflation_pct"] = g("gpl__area__routability_inflation__percent")
    if pre == "globalroute":
        m["grt_wl"] = g("global_route__wirelength")
        t = _GRT_TOTAL.findall(log)
        if t:
            res, dem, use, oh, ov, otot = t[-1]
            m.update(grt_usage_pct=float(use), grt_overflow=int(otot))
        v = _VIAS.findall(log)
        m["grt_vias"] = int(v[-1]) if v else None
    if pre == "finish":
        m["timing_repair_buffers"] = _num(j.get("finish__design__instance__count__class:timing_repair_buffer"))
        m["timing_repair_buffer_area"] = _num(j.get("finish__design__instance__area__class:timing_repair_buffer"))
        m["drc"] = None
        rj = os.path.join(logs, "5_2_route.json")
        if os.path.exists(rj):
            r = json.load(open(rj))
            m["drc"] = _num(r.get("detailedroute__route__drc_errors"))
            m["routed_wl"] = _num(r.get("detailedroute__route__wirelength"))
    return m


def run_metrics(d: str, stages: List[str]) -> Dict:
    logs = find_logs(d)
    out = {"dir": logs, "stages": {}}
    for s in stages:
        stem, pre = STAGES[s]
        m = stage_metrics(logs, stem, pre)
        if m is not None:
            out["stages"][s] = m
    return out


def aggregate(runs: List[Dict], stages: List[str]) -> Dict:
    agg = {}
    for s in stages:
        vals: Dict[str, List[float]] = {}
        for r in runs:
            for k, v in r["stages"].get(s, {}).items():
                if isinstance(v, (int, float)) and v is not None:
                    vals.setdefault(k, []).append(float(v))
        if vals:
            agg[s] = {k: {"mean": statistics.fmean(v), "min": min(v), "max": max(v), "n": len(v),
                          "sd": statistics.stdev(v) if len(v) > 1 else 0.0} for k, v in vals.items()}
    return agg


def _fmt(a: Optional[Dict], k: str, nd: int) -> str:
    if not a or k not in a:
        return "—"
    x = a[k]
    if x["n"] == 1:
        return f"{x['mean']:.{nd}f}"
    return f"{x['mean']:.{nd}f} [{x['min']:.{nd}f}‥{x['max']:.{nd}f}]"


def markdown(groups: Dict[str, Dict], stages: List[str]) -> str:
    labels = list(groups)
    ns = {lb: len(groups[lb]["runs"]) for lb in labels}
    hdr = "| stage | " + " | ".join(f"{lb} (n={ns[lb]})" for lb in labels) + " |"
    sep = "|---" * (len(labels) + 1) + "|"
    out = ["Setup WNS ns / TNS ns / violating endpoints (mean [min‥max] over the group):", "", hdr, sep]
    for s in stages:
        cells = []
        for lb in labels:
            a = groups[lb]["agg"].get(s)
            cells.append("—" if not a else f"{_fmt(a, 'wns', 3)} / {_fmt(a, 'tns', 1)} / {_fmt(a, 'viol', 0)}")
        out.append(f"| {s} | " + " | ".join(cells) + " |")
    rows = [("3_3", "inst", 0, "instances"), ("3_3", "buf_ins", 0, "buffers inserted"),
            ("3_3", "buf_rem", 0, "buffers removed"), ("3_3", "resized", 0, "resized"),
            ("3_3", "gpl_td_area_pct", 2, "gpl TD delta area %"),
            ("3_3", "gpl_rd_inflation_pct", 2, "gpl RD inflation %"), ("3_3", "est_wl", 0, "est. WL"),
            ("3_4", "buf_ins", 0, "buffers inserted"), ("3_4", "resized", 0, "resized"),
            ("3_4", "area", 0, "instance area"),
            ("3_5", "disp_mean", 3, "OpenDP mean displacement µm"), ("3_5", "est_wl", 0, "est. WL"),
            ("4_1", "buf_ins", 0, "buffers inserted"), ("4_1", "resized", 0, "resized"),
            ("5_1", "buf_ins", 0, "buffers inserted"), ("5_1", "grt_wl", 0, "grt WL"),
            ("5_1", "est_wl", 0, "est. WL"), ("5_1", "grt_vias", 0, "grt vias"),
            ("5_1", "grt_overflow", 0, "grt overflow"), ("5_1", "grt_usage_pct", 2, "grt usage %"),
            ("5_1", "area", 0, "instance area"),
            ("6", "timing_repair_buffers", 0, "timing_repair_buffer count"),
            ("6", "timing_repair_buffer_area", 0, "timing_repair_buffer area"),
            ("6", "routed_wl", 0, "routed WL"), ("6", "drc", 0, "DRC")]
    out += ["", "Placement, buffering and congestion:", "", "| stage | metric | " + " | ".join(labels) + " |",
            "|---|---" + "|---" * len(labels) + "|"]
    for s, k, nd, name in rows:
        if s not in stages:
            continue
        cells = [_fmt(groups[lb]["agg"].get(s), k, nd) for lb in labels]
        if all(c == "—" for c in cells):
            continue
        out.append(f"| {s} | {name} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="LABEL=DIR (repeat a LABEL to group seeds)")
    ap.add_argument("--json", default=None, help="write the per-run and aggregated metrics here")
    ap.add_argument("--md", default=None, help="write the markdown tables here (also printed)")
    ap.add_argument("--stages", default=",".join(STAGES), help="comma-separated subset of " + ",".join(STAGES))
    a = ap.parse_args(argv)
    stages = [s for s in a.stages.split(",") if s]
    for s in stages:
        if s not in STAGES:
            raise SystemExit(f"unknown stage {s}")
    groups: Dict[str, Dict] = {}
    for spec in a.runs:
        if "=" not in spec:
            raise SystemExit(f"expected LABEL=DIR, got {spec!r}")
        lb, d = spec.split("=", 1)
        groups.setdefault(lb, {"runs": []})["runs"].append(run_metrics(d, stages))
    for g in groups.values():
        g["agg"] = aggregate(g["runs"], stages)
    md = markdown(groups, stages)
    sys.stdout.write(md)
    if a.md:
        open(a.md, "w").write(md)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(groups, f, indent=1)


if __name__ == "__main__":
    main()
