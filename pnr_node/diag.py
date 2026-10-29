"""Evidence packer for the `llm_diag` arm: what a finished eval *showed*, not just its score.

`evidence(result_json, run_dir, dp_dir=None)` turns one EvalResult (dict or object) plus its raw_handoff work dir
(`<work>/<design>/<req_key>`, ORFS tree under `orfs/`) into a small, deterministic dict; `render(ev)` turns that into
a compact text block for a prompt (<= ~1.5k tokens, chars/4). Everything is best effort: a missing file, an
unparsable log or a pruned intermediate drops that field, it never raises.

Sources (as found in real run dirs, ~/work/netwt/{gcd,ariane133}/<req_key>):
  result provenance       stage_wall_s, dp_runtime_s/iterations/overflow, rebuf.stages (or rebuf.trace.stages)
  logs/<p>/<d>/base/*.json   ORFS metrics: <prefix>__timing__setup__ws/tns per stage, instance count, utilization,
                             setup buffers, globalroute wirelength, detailedroute drc_errors(+per-iter)
  logs/<p>/<d>/base/*.log    "Elapsed time" per stage (fallback wall), RSZ repair counts, repair_timing final row,
                             "Took N seconds: <cmd>" (slowest steps), 5_1_grt final congestion table
  reports/<p>/<d>/base/*.rpt latest stage report (6_finish > 5_global_route > 4_cts_final > 3_*): max-delay paths,
                             clock skew; congestion-<n>.rpt violation counts
  <dp_dir>/rebuf/r<k>.slack  per-net slack dump of the last kept repair: highest-fanout nets with their slack
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

# ORFS stage logs in flow order, with their metrics prefix
STAGE_FILES = (
    ("2_1_floorplan", "floorplan"), ("3_3_place_gp", "globalplace"), ("3_4_place_resized", "placeopt"),
    ("3_5_place_dp", "detailedplace"), ("4_1_cts", "cts"), ("5_1_grt", "globalroute"),
    ("5_2_route", "detailedroute"), ("6_report", "finish"),
)
SHORT = {"2_1_floorplan": "fp", "3_3_place_gp": "gp", "3_4_place_resized": "rsz", "3_5_place_dp": "dpl",
         "4_1_cts": "cts", "5_1_grt": "grt", "5_2_route": "drt", "6_report": "fin"}
# latest-first: the path report closest to the end of the run
REPORTS = ("6_finish.rpt", "5_global_route.rpt", "4_cts_final.rpt", "3_detailed_place.rpt", "3_resizer.rpt",
           "3_global_place.rpt")


def _r(x: Any, nd: int = 3) -> Any:
    """Round floats for a deterministic, compact rendering; pass everything else through."""
    if isinstance(x, bool) or x is None:
        return x
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        if abs(x) >= 1000:
            return round(x)
        return round(x, nd)
    return x


def _short(s: str, n: int) -> str:
    """Keep the informative tail of a long hierarchical name."""
    return s if len(s) <= n else "~" + s[-(n - 1):]


def _read(path: str, limit: int = 64 << 20) -> Optional[str]:
    try:
        if os.path.getsize(path) > limit:
            return None
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return None


def _json(path: str) -> Optional[dict]:
    t = _read(path)
    if t is None:
        return None
    try:
        d = json.loads(t)
        return d if isinstance(d, dict) else None
    except ValueError:
        return None


def _as_dict(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        return result
    try:
        from dataclasses import asdict
        return asdict(result)
    except Exception:
        return {}


def _base_dirs(run_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """(logs base, reports base) = orfs/{logs,reports}/<platform>/<design>/base, whatever platform/design exist."""
    out = []
    for kind in ("logs", "reports"):
        hits = sorted(glob.glob(os.path.join(run_dir, "orfs", kind, "*", "*", "base")))
        out.append(hits[0] if hits else None)
    return out[0], out[1]


# ---------------------------------------------------------------------------------------------------------------------
# result / provenance
# ---------------------------------------------------------------------------------------------------------------------

def _from_result(d: Dict[str, Any], ev: Dict[str, Any]) -> None:
    ev["ok"] = bool(d.get("ok"))
    ev["stage"] = d.get("stage")
    for k in ("wns", "tns", "grt_wl", "routed_wl", "drc_violations", "power", "area"):
        if d.get(k) is not None:
            ev[k] = _r(d[k])
    if d.get("runtime_s"):
        ev["eval_wall_s"] = _r(float(d["runtime_s"]), 0)
    err = (d.get("error") or "").strip()
    if err:
        ev["rejected"] = err.splitlines()[0][:160]
    p = d.get("provenance") or {}
    dp = {k[3:]: _r(p[k]) for k in ("dp_iterations", "dp_overflow", "dp_runtime_s") if p.get(k) is not None}
    if dp:
        ev["dp"] = dp
    sw = p.get("stage_wall_s")
    if isinstance(sw, dict) and sw:
        ev["stage_wall_s"] = {SHORT.get(k, k): _r(float(v), 0) for k, v in sorted(sw.items())}
    rb = p.get("rebuf") or {}
    stages = rb.get("stages") or (rb.get("trace") or {}).get("stages") or []
    tr = []
    for st in stages:
        try:
            row: Dict[str, Any] = {"k": st.get("stage")}
            gp = st.get("gp") or {}
            for a, b in (("overflow", "ovf"), ("hpwl", "hpwl"), ("iterations", "it")):
                if gp.get(a) is not None:
                    row[b] = _r(gp[a])
            rp = st.get("repair") or {}
            if rp:
                row["bufs"], row["resized"] = rp.get("buffers"), rp.get("resized")
                rri = rp.get("rebuffer_removed_inserted")
                if rri:
                    row["rebuf_rm_ins"] = rri[-1]
            sta = st.get("sta") or {}
            if sta:
                row["sta_wns"], row["sta_tns"], row["sta_viol"] = _r(sta.get("wns")), _r(sta.get("tns")), sta.get("viol")
            w = st.get("weights") or {}
            if w.get("weighted") is not None:
                row["weighted_nets"] = w["weighted"]
            tr.append({k: v for k, v in row.items() if v is not None})
        except AttributeError:
            continue
    if tr:
        ev["rebuf_trace"] = tr
    tc = p.get("toolchain") or {}
    if tc.get("dp_extra"):
        ev["dp_extra"] = tc["dp_extra"]


# ---------------------------------------------------------------------------------------------------------------------
# ORFS logs / metrics
# ---------------------------------------------------------------------------------------------------------------------

_ELAPSED = re.compile(r"Elapsed time: (?:(\d+):)?(\d+):(\d+(?:\.\d+)?)\[h:\]min:sec")
_TOOK = re.compile(r"^Took (\d+) seconds: (\S+)", re.M)


def _elapsed(text: str) -> Optional[float]:
    m = list(_ELAPSED.finditer(text))
    if not m:
        return None
    h, mi, s = m[-1].groups()
    return int(h or 0) * 3600 + int(mi) * 60 + float(s)


def _repairs(text: str) -> Dict[str, Any]:
    """Resizer activity in one ORFS stage log (repair_design + repair_timing)."""
    out: Dict[str, Any] = {}
    ins = [int(x) for x in re.findall(r"RSZ-0038\] Inserted (\d+) buffers", text)]
    ins += [int(x) for x in re.findall(r"RSZ-0048\] Inserted (\d+) buffers", text)]
    if ins:
        out["drv_bufs"] = sum(ins)
    t_ins = [int(x) for x in re.findall(r"RSZ-00(?:45|40)\] Inserted (\d+) buffers", text)]
    if t_ins:
        out["timing_bufs"] = sum(t_ins)
    rs = [int(x) for x in re.findall(r"RSZ-00(?:39|51)\] Resized (\d+) instances", text)]
    if rs:
        out["resized"] = sum(rs)
    m = re.findall(r"RSZ-0094\] Found (\d+) endpoints with setup violations", text)
    if m:
        out["viol_endpoints_in"] = int(m[-1])
    fin = re.findall(r"^\s*final \|.*\|\s*(-?[\d.]+)\s*\|\s*(-?[\d.]+)\s*\|\s*(-?[\d.]+)\s*\|\s*(\d+)\s*\|\s*(\S+)\s*$",
                     text, re.M)
    if fin:
        w, _st, en, viol, worst = fin[-1]
        out["repair_timing_final"] = {"wns": float(w), "tns": float(en), "viol": int(viol), "worst_ep": worst[:80]}
    if "RSZ-0062" in text:
        out["setup_unrepaired"] = True
    return out


def _congestion(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    tables = re.findall(r"^Total\s+(\d+)\s+(\d+)\s+([\d.]+)%\s+(\d+)\s*/\s*(\d+)\s*/\s*(\d+)", text, re.M)
    if tables:
        first, last = tables[0], tables[-1]
        out["usage_pct"] = float(last[2])
        out["overflow_initial"] = int(first[5])
        out["overflow_final"] = int(last[5])
    extra = re.findall(r"Start extra iteration (\d+)/(\d+)", text)
    if extra:
        out["extra_iters"] = f"{max(int(a) for a, _ in extra)}/{extra[-1][1]}"
    m = re.findall(r"GRT-0018\] Total wirelength: (\d+)", text)
    if m:
        out["grt_wl"] = int(m[0])
    layers = re.findall(r"^(metal\d+)\s+\d+\s+\d+\s+([\d.]+)%\s+\d+\s*/", text, re.M)
    if layers:  # the final table's busiest layers
        n = len([1 for _ in re.finditer(r"^metal1\s", text, re.M)])
        last = layers[-(len(layers) // max(n, 1)):] if n else layers
        top = sorted(last, key=lambda x: -float(x[1]))[:2]
        out["busiest_layers"] = [f"{l}:{float(u):.0f}%" for l, u in top]
    return out


def _from_logs(logs: str, ev: Dict[str, Any]) -> None:
    series, wall, rep, slow = [], {}, {}, []
    last_metrics: Dict[str, Any] = {}
    for stem, pre in STAGE_FILES:
        m = _json(os.path.join(logs, stem + ".json")) or {}
        ws, tns = m.get(f"{pre}__timing__setup__ws"), m.get(f"{pre}__timing__setup__tns")
        if ws is not None or tns is not None:
            series.append([SHORT[stem], _r(ws), _r(tns)])
        if m:
            for k in ("design__instance__count", "design__instance__utilization", "design__instance__count__setup_buffer",
                      "timing__drv__setup_violation_count", "clock__skew__setup"):
                if f"{pre}__{k}" in m:
                    last_metrics[k] = (SHORT[stem], m[f"{pre}__{k}"])
        if pre == "detailedroute" and m:
            drc = m.get("detailedroute__route__drc_errors")
            it0 = m.get("detailedroute__route__drc_errors__iter:0")
            n_it = len([k for k in m if k.startswith("detailedroute__route__drc_errors__iter:")])
            ev["drt"] = {k: v for k, v in (("drc", drc), ("drc_iter0", it0), ("iters", n_it or None)) if v is not None}
        text = _read(os.path.join(logs, stem + ".log"))
        if text is None:
            continue
        e = _elapsed(text)
        if e is not None:
            wall[SHORT[stem]] = _r(e, 0)
        if stem == "2_1_floorplan":  # fixed for every config (synth/floorplan are not searched)
            continue
        r = _repairs(text)
        if r:
            rep[SHORT[stem]] = r
        for s, cmd in _TOOK.findall(text):
            slow.append((int(s), SHORT[stem], cmd))
        if stem == "5_1_grt":
            c = _congestion(text)
            if c:
                ev["grt_congestion"] = c
    if series:
        ev["setup_series"] = series  # [stage, ws, tns]
    if wall and "stage_wall_s" not in ev:
        ev["stage_wall_s"] = wall
    if rep:
        ev["orfs_repairs"] = rep
    if slow:
        slow.sort(key=lambda x: (-x[0], x[1], x[2]))
        ev["slowest_steps"] = [f"{c}@{st}:{s}s" for s, st, c in slow[:4]]
    if last_metrics:
        ev["design"] = {k.replace("design__instance__", "inst_").replace("timing__drv__", ""): [st, _r(v)]
                        for k, (st, v) in sorted(last_metrics.items())}


# ---------------------------------------------------------------------------------------------------------------------
# ORFS timing report: worst max-delay paths
# ---------------------------------------------------------------------------------------------------------------------

_OUT_ROW = re.compile(r"^\s*(\d+)\s+[\d.]+\s+[\d.]+\s+(-?[\d.]+)\s+-?[\d.]+\s+[\^v]\s+(\S+)\s+\((\w+)\)")
_IN_ROW = re.compile(r"^\s+[\d.]+\s+(-?[\d.]+)\s+-?[\d.]+\s+[\^v]\s+(\S+)\s+\((\w+|in|out)\)\s*$")


def _parse_path(block: str) -> Optional[Dict[str, Any]]:
    sp = re.search(r"^Startpoint: (\S+)(?: \(([^)]*)\))?", block, re.M)
    epm = re.search(r"^Endpoint: (\S+)", block, re.M)
    sl = re.search(r"^\s*(-?[\d.]+)\s+slack \((MET|VIOLATED)\)", block, re.M)
    if not (sp and epm and sl):
        return None
    if not re.search(r"^Path Type: max", block, re.M):
        return None
    start, end = sp.group(1), epm.group(1)
    grp = re.search(r"^Path Group: (\S+)", block, re.M)
    p: Dict[str, Any] = {"from": start, "to": end, "slack": float(sl.group(1)),
                         "group": grp.group(1) if grp else None}
    arrival = block.split("data arrival time", 1)[0]
    data, cells, wire, cell_d, fo_max, bufs, worst = False, 0, 0.0, 0.0, 0, 0, None
    for line in arrival.splitlines():
        mo = _OUT_ROW.match(line)
        if mo:
            fo, d, pin, typ = int(mo.group(1)), float(mo.group(2)), mo.group(3), mo.group(4)
            if not data:
                if pin.startswith(start + "/") or pin == start:
                    data = True
                else:
                    continue
            cells += 1
            cell_d += d
            fo_max = max(fo_max, fo)
            if typ.startswith("BUF") or typ.startswith("CLKBUF"):
                bufs += 1
            if worst is None or d > worst[1]:
                worst = (typ, d, fo)
            continue
        mi = _IN_ROW.match(line)
        if mi and data:
            wire += float(mi.group(1))
    if cells:
        p.update({"depth": cells, "cell_ns": round(cell_d, 3), "wire_ns": round(wire, 3), "max_fanout": fo_max,
                  "bufs": bufs})
        if worst:
            p["worst_cell"] = f"{worst[0]} {worst[1]:.3f}ns fo{worst[2]}"
    return p


def _from_report(reports: str, ev: Dict[str, Any], top: int = 5) -> None:
    for name in REPORTS:
        text = _read(os.path.join(reports, name))
        if not text:
            continue
        paths: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for blk in re.split(r"(?=^Startpoint: )", text, flags=re.M)[1:]:
            p = _parse_path(blk)
            if p and (p["from"], p["to"]) not in paths:
                paths[(p["from"], p["to"])] = p
        sk = re.search(r"report_clock_skew\n-+\n(.*?)\n=====", text, re.S)
        if sk:
            m = re.search(r"Clock core_clock.*?\n\s*(-?[\d.]+) setup skew", sk.group(1), re.S) or \
                re.search(r"(-?[\d.]+) setup skew", sk.group(1))
            if m:
                ev["clock_skew_setup"] = float(m.group(1))
        if paths:
            ev["paths_report"] = name
            ev["worst_paths"] = sorted(paths.values(), key=lambda p: (p["slack"], p["from"], p["to"]))[:top]
        return
    return


def _from_congestion_rpts(reports: str, ev: Dict[str, Any]) -> None:
    out = {}
    for f in sorted(glob.glob(os.path.join(reports, "congestion-*.rpt"))):
        t = _read(f)
        if t is None:
            continue
        m = re.search(r"congestion-(\d+)\.rpt$", f)
        out[f"iter{m.group(1) if m else '?'}"] = t.count("violation type:")
    if out:
        ev.setdefault("grt_congestion", {})["hotspots"] = dict(sorted(out.items(), key=lambda kv: kv[0]))


# ---------------------------------------------------------------------------------------------------------------------
# DREAMPlace / rebuffer dir: high-fanout nets
# ---------------------------------------------------------------------------------------------------------------------

def find_dp_dir(run_dir: str, result: Dict[str, Any]) -> Optional[str]:
    """The dp-<hash> sibling of `run_dir` whose dp_result.json runtime matches this result's dp_runtime_s (the
    dp-dir hash needs the full request, which a result does not carry). None if not found."""
    p = (result or {}).get("provenance") or {}
    rt = p.get("dp_runtime_s")
    if rt is None:
        return None
    for d in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(run_dir)), "dp-*"))):
        j = _json(os.path.join(d, "dp_result.json"))
        if j and j.get("runtime_s") == rt:
            return d
    return None


def _from_dp_dir(dp_dir: str, ev: Dict[str, Any], top: int = 3) -> None:
    dumps = sorted(glob.glob(os.path.join(dp_dir, "rebuf", "r*.slack")),
                   key=lambda f: int(re.sub(r"\D", "", os.path.basename(f)) or 0))
    if not dumps:
        return
    f = dumps[-1]
    rows, neg = [], 0
    try:
        with open(f, errors="replace") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    s = float(parts[1])
                    s = s if math.isfinite(s) else None
                    deg = int(parts[2])
                except ValueError:
                    s, deg = None, int(parts[2]) if parts[2].isdigit() else 0
                flag = parts[3] if len(parts) > 3 else ""
                if s is not None and s < 0 and flag != "clk":
                    neg += 1
                if flag != "clk":  # clock nets are CTS's job, not a placement knob's
                    rows.append((deg, parts[0], s, flag))
    except OSError:
        return
    if not rows:
        return
    rows.sort(key=lambda r: (-r[0], r[1]))
    timed = [r for r in rows if r[2] is not None]
    ev["high_fanout_nets"] = {"dump": os.path.basename(f), "neg_slack_nets": neg, "nets": len(rows),
                              "untimed_high_degree": sum(1 for r in rows if r[3] == "hideg"),
                              "top": [f"{_short(n, 40)} fo{d} slack={_r(s)}" for d, n, s, _fl in timed[:top]]}


# ---------------------------------------------------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------------------------------------------------

def run_dir_for(result: Any, design: Optional[str] = None, work_root: Optional[str] = None) -> Optional[str]:
    """`<work>/<design>/<provenance.req_key>`; work = `work_root`, else derived from provenance.arm_log
    (<work>/logs/arm_*.log, set by ArmRunEvaluator), else $NETWT_WORK, else ~/work/netwt. None if absent."""
    d = _as_dict(result)
    p = d.get("provenance") or {}
    key = p.get("req_key")
    if not key:
        return None
    roots = [work_root] if work_root else []
    if p.get("arm_log"):
        roots.append(os.path.dirname(os.path.dirname(os.path.abspath(p["arm_log"]))))
    roots += [os.environ.get("NETWT_WORK"), os.path.expanduser("~/work/netwt")]
    designs = [design] if design else []
    lbl = os.path.basename(p.get("arm_log") or "")
    m = re.match(r"arm_([^_]+)_", lbl)
    if m:
        designs.append(m.group(1))
    for r in roots:
        if not r:
            continue
        for dz in designs:
            cand = os.path.join(r, dz, key)
            if os.path.isdir(cand):
                return cand
    return None


def evidence(result_json: Any, run_dir: Optional[str] = None, dp_dir: Optional[str] = None) -> Dict[str, Any]:
    """Compact, deterministic summary of one finished eval. Never raises; missing pieces are simply omitted."""
    ev: Dict[str, Any] = {}
    d = _as_dict(result_json)
    steps = [lambda: _from_result(d, ev)]
    if run_dir and os.path.isdir(run_dir):
        logs, reports = _base_dirs(run_dir)
        if logs:
            steps.append(lambda: _from_logs(logs, ev))
        if reports:
            steps.append(lambda: _from_report(reports, ev))
            steps.append(lambda: _from_congestion_rpts(reports, ev))
        if dp_dir is None:
            steps.append(lambda: _from_dp_dir(find_dp_dir(run_dir, d) or "/nonexistent", ev))
    if dp_dir:
        steps.append(lambda: _from_dp_dir(dp_dir, ev))
    for s in steps:
        try:
            s()
        except Exception as e:  # best effort by contract; note it so a silent gap is visible in the prompt
            ev.setdefault("_parse_errors", []).append(type(e).__name__)
    return ev


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_fmt(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ",".join(f"{k}:{_fmt(x)}" for k, x in v.items()) + "}"
    return str(v)


def render(ev: Dict[str, Any], brief: bool = False, max_paths: int = 5) -> str:
    """Text block for a prompt. brief=True: one line (final timing, convergence, top path, congestion)."""
    if not ev:
        return "(no evidence)"
    head = [f"ok={ev.get('ok')}", f"stage={ev.get('stage')}"]
    for k in ("wns", "tns", "routed_wl", "grt_wl", "drc_violations"):
        if k in ev:
            head.append(f"{k}={_fmt(ev[k])}")
    if "rejected" in ev:
        head.append(f"rejected={ev['rejected']!r}")
    if brief:
        dp = ev.get("dp") or {}
        if dp:
            head.append(f"dp_it={dp.get('iterations')} dp_ovf={dp.get('overflow')}")
        tr = ev.get("rebuf_trace") or []
        bufs = [t.get("bufs") for t in tr if t.get("bufs") is not None]
        if bufs:
            head.append(f"rebuf_bufs={_fmt(bufs)}")
        if ev.get("worst_paths"):
            p = ev["worst_paths"][0]
            head.append(f"worst={p['slack']:g} depth={p.get('depth')} wire={p.get('wire_ns')}")
        c = ev.get("grt_congestion") or {}
        if "overflow_initial" in c:
            head.append(f"grt_ovf0={c['overflow_initial']}")
        return " ".join(head)
    lines = [" ".join(head)]
    if ev.get("setup_series"):
        lines.append("setup ws/tns by stage: " + " ".join(f"{s}:{_fmt(w)}/{_fmt(t)}" for s, w, t in ev["setup_series"]))
    if ev.get("dp"):
        lines.append("DREAMPlace final GP: " + _fmt(ev["dp"]))
    if ev.get("rebuf_trace"):
        lines.append("rebuffer trace (k: GP ovf/hpwl/it, kept repair bufs/resized, STA after repair wns/tns/viol):")
        for t in ev["rebuf_trace"]:
            lines.append("  " + " ".join(f"{k}={_fmt(v)}" for k, v in t.items()))
    if ev.get("orfs_repairs"):
        lines.append("ORFS resizer: " + "; ".join(f"{s} {_fmt(r)}" for s, r in ev["orfs_repairs"].items()))
    if ev.get("stage_wall_s"):
        lines.append("stage wall s: " + _fmt(ev["stage_wall_s"]))
    if ev.get("slowest_steps"):
        lines.append("slowest steps: " + ", ".join(ev["slowest_steps"]))
    if ev.get("design"):
        lines.append("design: " + _fmt(ev["design"]))
    if "clock_skew_setup" in ev:
        lines.append(f"clock setup skew: {ev['clock_skew_setup']:g} ns")
    if ev.get("worst_paths"):
        lines.append(f"worst setup paths ({ev.get('paths_report')}):")
        for p in ev["worst_paths"][:max_paths]:
            extra = " ".join(f"{k}={_fmt(p[k])}" for k in ("depth", "cell_ns", "wire_ns", "max_fanout", "bufs",
                                                          "worst_cell") if k in p)
            lines.append(f"  {p['slack']:g} [{p.get('group')}] {_short(p['from'], 48)} -> {_short(p['to'], 48)} {extra}")
    if ev.get("grt_congestion"):
        lines.append("global route congestion: " + _fmt(ev["grt_congestion"]))
    if ev.get("drt"):
        lines.append("detailed route: " + _fmt(ev["drt"]))
    if ev.get("high_fanout_nets"):
        h = ev["high_fanout_nets"]
        lines.append(f"highest-fanout timed nets ({h['dump']}, {h['neg_slack_nets']}/{h['nets']} nets neg slack, "
                     f"{h.get('untimed_high_degree', 0)} untimed high-degree): "
                     + "; ".join(h["top"]))
    if ev.get("_parse_errors"):
        lines.append("(evidence parse gaps: " + ",".join(ev["_parse_errors"]) + ")")
    return "\n".join(lines)


def tokens(text: str) -> int:
    """Token estimate used for the prompt budget: chars / 4."""
    return (len(text) + 3) // 4
