#!/usr/bin/env python3
"""tinyRocket local pair: Run A (DREAMPlace GPU global+legal+detail handoff) vs Run B (stock ORFS, RePlAce).

  python collect.py --copy      # copy logs/JSONs from ~/work/trpair/{A,B,Ap} into ./A, ./B, ./Ap, then write compare.md
  python collect.py             # regenerate compare.md from the copies already in ./A and ./B (and ./Ap if present)

Run A′ (./Ap, optional) is Run A with --dp-depth global: DREAMPlace GP only, so OpenDP does all legalization/detail.

Layout of the copies:
  A/base/      ORFS logs+JSONs of the evaluator's base run (1_* .. 3_2, plus the 3_3 DEF-dump pass)
  A/cand/      ORFS logs+JSONs of the candidate run (3_3 import onward; 1_*..3_2 there are copies of base)
  A/dp/        dreamplace.log, dreamplace.json
  A/           run.log, gpu.csv, start_epoch, end_epoch, evidence/ (EvalResult)
  B/logs/      stock ORFS logs+JSONs
  B/           run.log, start_epoch, end_epoch
  Ap/          same layout as A/
"""
import argparse, csv, glob, json, os, re, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
PLAT = "nangate45/tinyRocket/base"
STAGES = ["1_1_yosys_canonicalize", "1_2_yosys", "1_synth", "2_1_floorplan", "2_2_floorplan_macro",
          "2_3_floorplan_tapcell", "2_4_floorplan_pdn", "3_1_place_gp_skip_io", "3_2_place_iop", "3_3_place_gp",
          "3_4_place_resized", "3_5_place_dp", "4_1_cts", "5_1_grt", "5_2_route", "5_3_fillcell", "6_1_fill", "6_1_merge",
          "6_report"]
PRE = STAGES[:STAGES.index("3_3_place_gp")]


# ---------------------------------------------------------------- copy
def _copy_logs(src, dst):
    os.makedirs(dst, exist_ok=True)
    for f in glob.glob(os.path.join(src, "*.log")) + glob.glob(os.path.join(src, "*.json")):
        shutil.copy2(f, dst)


def copy_handoff(work_a, a):
    _copy_logs(os.path.join(work_a, "tinyRocket/base/logs", PLAT), os.path.join(a, "base"))
    cands = [d for d in glob.glob(os.path.join(work_a, "tinyRocket/*/orfs")) if os.path.isdir(d)]
    assert len(cands) == 1, f"expected one candidate dir, got {cands}"
    _copy_logs(os.path.join(cands[0], "logs", PLAT), os.path.join(a, "cand"))
    for rpt in glob.glob(os.path.join(cands[0], "reports", PLAT, "*")):
        if os.path.isfile(rpt):
            os.makedirs(os.path.join(a, "cand/reports"), exist_ok=True)
            shutil.copy2(rpt, os.path.join(a, "cand/reports"))
    dps = glob.glob(os.path.join(work_a, "tinyRocket/dp-*/dp_out"))
    assert len(dps) == 1, dps
    os.makedirs(os.path.join(a, "dp"), exist_ok=True)
    for n in ("dreamplace.log", "dreamplace.json"):
        shutil.copy2(os.path.join(dps[0], n), os.path.join(a, "dp"))
    for n in ("run.log", "gpu.csv", "start_epoch", "end_epoch"):
        if os.path.exists(os.path.join(work_a, n)):
            shutil.copy2(os.path.join(work_a, n), a)
    if os.path.isdir(os.path.join(work_a, "evidence")):
        shutil.copytree(os.path.join(work_a, "evidence"), os.path.join(a, "evidence"), dirs_exist_ok=True)


def copy(work_a, work_b, work_ap=None):
    copy_handoff(work_a, os.path.join(HERE, "A"))
    if work_ap and os.path.isdir(work_ap) and os.path.exists(os.path.join(work_ap, "end_epoch")):
        copy_handoff(work_ap, os.path.join(HERE, "Ap"))
    b = os.path.join(HERE, "B")
    _copy_logs(os.path.join(work_b, "logs", PLAT), os.path.join(b, "logs"))
    for rpt in glob.glob(os.path.join(work_b, "reports", PLAT, "*")):
        if os.path.isfile(rpt):
            os.makedirs(os.path.join(b, "reports"), exist_ok=True)
            shutil.copy2(rpt, os.path.join(b, "reports"))
    for n in ("run.log", "start_epoch", "end_epoch"):
        if os.path.exists(os.path.join(work_b, n)):
            shutil.copy2(os.path.join(work_b, n), b)


# ---------------------------------------------------------------- parse
_FOOT = re.compile(r"Elapsed time: (?:(\d+):)?(\d+):([\d.]+)\[h:\]min:sec\. CPU time: user ([\d.]+) sys ([\d.]+) "
                   r"\((\d+)%\)\. Peak memory: (\d+)KB")


def stage_time(logdir, stage):
    """(elapsed_s, cpu_s, peak_MB) from the stage log's last footer, or None."""
    p = os.path.join(logdir, stage + ".log")
    if not os.path.exists(p):
        return None
    m = None
    for m in _FOOT.finditer(open(p, errors="replace").read()):
        pass
    if not m:
        return None
    h, mi, s, u, sy, _, kb = m.groups()
    return (int(h or 0) * 3600 + int(mi) * 60 + float(s), float(u) + float(sy), int(kb) / 1024)


def jload(logdir, stage):
    p = os.path.join(logdir, stage + ".json")
    return json.load(open(p)) if os.path.exists(p) else {}


def grep1(path, rx, cast=float):
    if not os.path.exists(path):
        return None
    m = re.search(rx, open(path, errors="replace").read())
    return cast(m.group(1)) if m else None


def epoch_wall(d):
    try:
        return int(open(os.path.join(d, "end_epoch")).read()) - int(open(os.path.join(d, "start_epoch")).read())
    except (OSError, ValueError):
        return None


def dp_breakdown(dpdir):
    log = open(os.path.join(dpdir, "dreamplace.log"), errors="replace").read()
    out = {}
    for key, rx in [("read_db_s", r"reading database takes ([\d.]+)"),
                    ("nlp_init_s", r"non-linear placement initialization takes ([\d.]+)"),
                    ("macro_lg_ms", r"Macro legalization takes ([\d.]+) ms"),
                    ("greedy_lg_ms", r"Greedy legalization takes ([\d.]+) ms"),
                    ("abacus_lg_ms", r"Abacus legalization takes ([\d.]+) ms"),
                    ("legalization_s", r"DREAMPlace - legalization takes ([\d.]+)"),
                    ("detailed_place_s", r"detailed placement takes ([\d.]+)"),
                    ("nlp_total_s", r"non-linear placement takes ([\d.]+)"),
                    ("placement_total_s", r"DREAMPlace - placement takes ([\d.]+)")]:
        m = re.search(rx, log)
        out[key] = float(m.group(1)) if m else None
    its = re.findall(r"iteration +(\d+), \( *\d+, *\d+, *\d+\), Obj \S+, DensityWeight \S+, wHPWL ([^,\s]+), "
                     r"Overflow ([^,\s]+)", log)
    if its:
        out["gp_iterations"] = int(its[-1][0])
        out["gp_whpwl"] = float(its[-1][1])
        out["gp_overflow"] = float(its[-1][2])
    # wHPWL after LG+DP: the last "iteration N, wHPWL" line without the (a,b,c) tuple
    post = re.findall(r"iteration +\d+, wHPWL (\S+), time", log)
    out["post_lg_dp_whpwl"] = float(post[-1]) if post else None
    hp = re.findall(r"iteration (\d+), target hpwl (\S+), delta", log)
    out["abcd_target_hpwl_first_last"] = (float(hp[0][1]), float(hp[-1][1])) if hp else None
    out["legality_check"] = "Legality check takes" in log
    # Placer.py subprocess wall: the runner writes dreamplace.json right before spawning it and dreamplace.log right
    # after it exits (copy2 keeps the mtimes), so this includes Python/torch start-up and DB read. git does not keep
    # mtimes, so the originals are frozen in dp/mtimes.json and read from there when present.
    frozen = os.path.join(dpdir, "mtimes.json")
    mt = (json.load(open(frozen)) if os.path.exists(frozen) else
          {f: os.path.getmtime(os.path.join(dpdir, f)) for f in ("dreamplace.json", "dreamplace.log")})
    out["subprocess_wall_s"] = mt["dreamplace.log"] - mt["dreamplace.json"]
    out["wrote_def"] = (re.findall(r"writing placement to (\S+)", log) or [None])[-1]
    out["_order_dp_before_write"] = (log.find("detailed placement takes") < log.find("writing to") and
                                     log.find("detailed placement takes") >= 0)
    return out


def gpu_stats(path):
    if not os.path.exists(path):
        return None
    util, mem = [], []
    with open(path) as f:
        for row in csv.reader(f):
            try:
                util.append(float(row[1].strip().rstrip(" %")))
                mem.append(float(row[2].strip().split()[0]))
            except (ValueError, IndexError):
                continue
    if not util:
        return None
    base = min(mem)
    busy = [u for u in util if u > 0]
    return {"samples": len(util), "peak_mem_MiB": max(mem), "idle_mem_MiB": base,
            "peak_mem_above_idle_MiB": max(mem) - base, "mean_util_all": sum(util) / len(util),
            "mean_util_nonzero": (sum(busy) / len(busy)) if busy else 0.0, "nonzero_samples": len(busy)}


def eval_result(adir):
    """EvalResult JSON: from evidence/ if present, else the last JSON object printed to run.log."""
    for p in sorted(glob.glob(os.path.join(adir, "evidence", "**", "*.json"), recursive=True)):
        try:
            d = json.load(open(p))
        except ValueError:
            continue
        if isinstance(d, dict) and "provenance" in d:
            return d, os.path.relpath(p, adir)
    p = os.path.join(adir, "run.log")
    if os.path.exists(p):
        for line in reversed(open(p, errors="replace").read().splitlines()):
            line = line.strip()
            if line.startswith("{") and "provenance" in line:
                try:
                    return json.loads(line), "run.log"
                except ValueError:
                    pass
    return None, None


def quality(logdir):
    g, r, f = jload(logdir, "5_1_grt"), jload(logdir, "5_2_route"), jload(logdir, "6_report")
    s, rz = jload(logdir, "1_synth"), os.path.join(logdir, "3_4_place_resized.log")
    dpl = os.path.join(logdir, "3_5_place_dp.log")
    drc_traj = [(int(k.split(":")[1]), v) for k, v in r.items() if k.startswith("detailedroute__route__drc_errors__iter:")]
    drc_traj.sort()
    return {
        "synth_instances": s.get("synth__design__instance__count"),
        "routed_WL": r.get("detailedroute__route__wirelength"),
        "DRC_final": r.get("detailedroute__route__drc_errors"),
        "DRC_trajectory": [v for _, v in drc_traj],
        "drt_iterations": len(drc_traj),
        "vias": r.get("detailedroute__route__vias"),
        "antenna_nets": r.get("detailedroute__antenna__violating__nets"),
        "antenna_diodes": r.get("detailedroute__antenna_diodes_count"),
        "setup_WNS_ns(finish)": f.get("finish__timing__setup__ws"),
        "setup_TNS_ns(finish)": f.get("finish__timing__setup__tns"),
        "setup_viol_endpoints(finish)": f.get("finish__timing__drv__setup_violation_count"),
        "fmax_MHz(finish)": (f["finish__timing__fmax"] / 1e6) if f.get("finish__timing__fmax") else None,
        "hold_WNS_ns(finish)": f.get("finish__timing__hold__ws"),
        "hold_TNS_ns(finish)": f.get("finish__timing__hold__tns"),
        "setup_WNS_ns(grt)": g.get("globalroute__timing__setup__ws"),
        "setup_TNS_ns(grt)": g.get("globalroute__timing__setup__tns"),
        "fmax_MHz(grt)": (g["globalroute__timing__fmax"] / 1e6) if g.get("globalroute__timing__fmax") else None,
        "grt_WL": g.get("globalroute__global_route__wirelength"),
        "power_W(finish)": f.get("finish__power__total"),
        "stdcell_area_um2(finish)": f.get("finish__design__instance__area__stdcell"),
        "utilization(finish)": f.get("finish__design__instance__utilization"),
        "stdcell_count(finish)": f.get("finish__design__instance__count__stdcell"),
        "timing_repair_buffers(finish)": f.get("finish__design__instance__count__class:timing_repair_buffer"),
        "clock_buffers(finish)": f.get("finish__design__instance__count__class:clock_buffer"),
        # repair_design prints RSZ-0039 only when it resized something
        "3_4_resized": grep1(rz, r"RSZ-0039\] Resized (\d+) instances", int) or (0 if os.path.exists(rz) else None),
        "3_4_buffers_inserted": grep1(rz, r"RSZ-0038\] Inserted (\d+) buffers", int),
        "3_5_OpenDP_avg_disp_um": grep1(dpl, r"average displacement\s+([\d.]+)"),
        "3_5_OpenDP_max_disp_um": grep1(dpl, r"max displacement\s+([\d.]+) u"),
        "3_5_OpenDP_total_disp_um": grep1(dpl, r"total displacement\s+([\d.]+)"),
    }


# ---------------------------------------------------------------- report
def fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:,.{nd}f}" if abs(v) < 1e5 else f"{v:,.0f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def delta(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and b and a is not None:
        return f"{(a - b) / abs(b) * 100:+.1f}%"
    return ""


def ap_section(w, A, APD, B, qa, qb, dp):
    """Run A′ = A with --dp-depth global: isolates what DREAMPlace LG + ABCDPlace change."""
    ev, _ = eval_result(APD)
    prov = (ev or {}).get("provenance", {}) or {}
    dpp = dp_breakdown(os.path.join(APD, "dp"))
    dpj = json.load(open(os.path.join(APD, "dp", "dreamplace.json")))
    c_ap, c_a, b_logs = os.path.join(APD, "cand"), os.path.join(A, "cand"), os.path.join(B, "logs")
    w("## Run A′: GP-only handoff (`--dp-depth global`)\n")
    w("Same as A, except DREAMPlace does global placement only, and ORFS's OpenDP (3_5) does all legalization and "
      "detailed placement. A′ − A isolates what DREAMPlace legalization + ABCDPlace change. n = 1, and the same "
      "reading rules apply.\n")
    w("| gate / item | A′ | A |\n|---|---|---|")
    for k, va, vb in [
        ("dreamplace.json legalize_flag / detailed_place_flag",
         f"{dpj.get('legalize_flag')} / {dpj.get('detailed_place_flag')}", "1 / 1"),
        ("log has `detailed placement takes`", dpp["detailed_place_s"] is not None, dp["detailed_place_s"] is not None),
        ("EvalResult ok", (ev or {}).get("ok"), True),
        ("gpl_lines_in_3_3 / handoff_read_def_in_3_3",
         f"{prov.get('gpl_lines_in_3_3')} / {prov.get('handoff_read_def_in_3_3')}", "0 / True"),
        ("DREAMPlace GP iterations / overflow", f"{fmt(dpp.get('gp_iterations'))} / {fmt(dpp.get('gp_overflow'))}",
         f"{fmt(dp.get('gp_iterations'))} / {fmt(dp.get('gp_overflow'))}"),
        ("DREAMPlace wHPWL at GP end / handed off", f"{fmt(dpp.get('gp_whpwl'))} / {fmt(dpp.get('post_lg_dp_whpwl') or dpp.get('gp_whpwl'))}",
         f"{fmt(dp.get('gp_whpwl'))} / {fmt(dp.get('post_lg_dp_whpwl'))}"),
        ("DREAMPlace non-linear placement (s)", fmt(dpp.get("nlp_total_s"), 2), fmt(dp.get("nlp_total_s"), 2)),
        ("DREAMPlace subprocess wall (s)", fmt(dpp["subprocess_wall_s"], 1), fmt(dp["subprocess_wall_s"], 1)),
        ("end-to-end wall (s)", fmt(epoch_wall(APD)), fmt(epoch_wall(A))),
    ]:
        w(f"| {k} | {fmt(va)} | {fmt(vb)} |")
    w("\n| stage (elapsed s) | A′ | A | B | A′/A |\n|---|---|---|---|---|")
    for st in STAGES[STAGES.index("3_3_place_gp"):]:
        t = [stage_time(d, st) for d in (c_ap, c_a, b_logs)]
        if not any(t):
            continue
        r = f"{t[0][0] / t[1][0]:.2f}×" if t[0] and t[1] and t[1][0] > 0 else ""
        w(f"| {st} | " + " | ".join("—" if x is None else f"{x[0]:.1f}" for x in t) + f" | {r} |")
    qap = quality(c_ap)
    w("\n| metric | A′ (GP only) | A (GP+LG+DP) | B (stock) | A vs A′ |\n|---|---|---|---|---|")
    for k in qap:
        w(f"| {k} | {fmt(qap[k])} | {fmt(qa.get(k))} | {fmt(qb.get(k))} | {delta(qa.get(k), qap[k])} |")
    w("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", action="store_true")
    ap.add_argument("--work-a", default=os.path.expanduser("~/work/trpair/A"))
    ap.add_argument("--work-b", default=os.path.expanduser("~/work/trpair/B"))
    ap.add_argument("--work-ap", default=os.path.expanduser("~/work/trpair/Ap"))
    args = ap.parse_args()
    if args.copy:
        copy(args.work_a, args.work_b, args.work_ap)
    A, B = os.path.join(HERE, "A"), os.path.join(HERE, "B")
    a_base, a_cand, b_logs = os.path.join(A, "base"), os.path.join(A, "cand"), os.path.join(B, "logs")
    ev, ev_src = eval_result(A)
    dp = dp_breakdown(os.path.join(A, "dp"))
    dpj = json.load(open(os.path.join(A, "dp", "dreamplace.json")))
    gpu = gpu_stats(os.path.join(A, "gpu.csv"))
    L = []
    w = L.append
    w("# tinyRocket local pair: DREAMPlace GPU handoff (A) vs stock ORFS (B)\n")
    w("Generated by `collect.py` from the copies in `A/` and `B/`.\n")
    w("Machine: Ryzen 7 5800H (16 thr), 27 GB, RTX 3060 Laptop 6 GB (sm_86). Image `chia-pnr:dp`, `NUM_CORES=16`, "
      "`LEC_CHECK=0`, runs sequential, fresh work dirs. **n = 1 per arm.** Laptop runtimes are not comparable with the VM numbers.\n")
    w("- **A**: ORFS 1_*..3_2 → DREAMPlace GPU GP (default knobs, seed 0, **timing off**) + DREAMPlace legalization "
      "(greedy+Abacus) + ABCDPlace detailed placement → components-only DEF `read_def -incremental` in 3_3 → stock ORFS 3_4..6.")
    w("- **B**: stock ORFS end to end (RePlAce, `GPL_TIMING_DRIVEN=1`, `GPL_ROUTABILITY_DRIVEN=1`).\n")

    # ---- gates
    w("## Gates\n")
    prov = (ev or {}).get("provenance", {}) or {}
    evid = prov.get("_evidence", {}) if isinstance(prov.get("_evidence"), dict) else {}
    gpl = prov.get("gpl_lines_in_3_3", evid.get("gpl_lines_in_3_3"))
    rdd = prov.get("handoff_read_def_in_3_3", evid.get("handoff_read_def_in_3_3"))
    rows = [
        ("A1 `dreamplace.json` detailed_place_flag / legalize_flag", f"{dpj.get('detailed_place_flag')} / {dpj.get('legalize_flag')}"),
        ("A1 stale key `detail_place_flag` absent", "detail_place_flag" not in dpj),
        ("A1 log has `legalization takes`", dp["legalization_s"] is not None),
        ("A1 log has `detailed placement takes`", dp["detailed_place_s"] is not None),
        ("A1 handoff DEF written after detailed placement", f"{dp['_order_dp_before_write']} (`{os.path.basename(dp['wrote_def'] or '')}`)"),
        ("A1 DREAMPlace legality check ran", dp["legality_check"]),
        ("A2 EvalResult ok", (ev or {}).get("ok")),
        ("A2 `gpl_lines_in_3_3 == 0`", gpl),
        ("A2 `handoff_read_def_in_3_3`", rdd),
        ("A2 `drc_violations` present", (ev or {}).get("drc_violations")),
        ("B `5_2_route.json` / `6_report.json` exist", f"{bool(jload(b_logs, '5_2_route'))} / {bool(jload(b_logs, '6_report'))}"),
    ]
    w("| gate | value |\n|---|---|")
    for k, v in rows:
        w(f"| {k} | {fmt(v)} |")
    w(f"\nEvalResult source: `{ev_src}`.\n")
    w("Note: this DREAMPlace build (6627f33) writes **one** output file, `<design>.gp.def`, at the end of `Placer.py`, "
      "i.e. after legalization and ABCDPlace. There is no separate `.lg.def`/`.dp.def`. The gate checks the log order.\n")

    # ---- runtime
    w("## Per-stage runtime (elapsed s / CPU s / peak MB)\n")
    w("| stage | A | B | A/B elapsed |\n|---|---|---|---|")
    tot = {"A": 0.0, "B": 0.0}

    def cell(t):
        return "—" if t is None else f"{t[0]:.1f} / {t[1]:.1f} / {t[2]:.0f}"

    for st in STAGES:
        ta = stage_time(a_base if st in PRE else a_cand, st)
        tb = stage_time(b_logs, st)
        if ta is None and tb is None:
            continue
        if ta: tot["A"] += ta[0]
        if tb: tot["B"] += tb[0]
        ratio = f"{ta[0] / tb[0]:.2f}×" if ta and tb and tb[0] > 0 else ""
        label = st + (" (A: DEF import)" if st == "3_3_place_gp" else "")
        w(f"| {label} | {cell(ta)} | {cell(tb)} | {ratio} |")
    dump = stage_time(a_base, "3_3_place_gp")
    dpw = dp["subprocess_wall_s"]
    w(f"| **A handoff overhead**: 3_3 dump pass (base) | {cell(dump)} | | |")
    w(f"| **A**: DREAMPlace `Placer.py` subprocess wall (GP+LG+DP incl. start-up; GPU) | {fmt(dpw, 1)} | | |")
    if dump: tot["A"] += dump[0]
    if isinstance(dpw, (int, float)): tot["A"] += dpw
    w(f"| **sum of the above** | {tot['A']:.1f} | {tot['B']:.1f} | {tot['A'] / tot['B']:.2f}× |" if tot["B"] else "")
    wa, wb = epoch_wall(A), epoch_wall(B)
    w(f"| **end-to-end wall** (end_epoch − start_epoch) | {fmt(wa)} | {fmt(wb)} | "
      f"{(f'{wa / wb:.2f}×' if wa and wb else '')} |")
    w(f"\nA EvalResult `runtime_s`: {fmt((ev or {}).get('runtime_s'), 1)}. "
      "A's end-to-end wall also includes the evaluator's own overhead (cache/hash, DEF emit, file copies).\n")
    imp, b33 = stage_time(a_cand, "3_3_place_gp"), stage_time(b_logs, "3_3_place_gp")
    if dump and imp and b33:
        a_pl = dump[0] + dpw + imp[0]
        w(f"**Placement only**: A = dump {dump[0]:.1f} + DREAMPlace {dpw:.1f} + import {imp[0]:.1f} = **{a_pl:.1f} s** vs "
          f"B RePlAce 3_3 **{b33[0]:.1f} s** ({a_pl / b33[0]:.2f}×). DREAMPlace's own compute: non-linear placement "
          f"{dp['nlp_total_s']:.2f} s, of which ABCDPlace {dp['detailed_place_s']:.2f} s and legalization "
          f"{dp['legalization_s']:.3f} s.\n")
    later = [st for st in STAGES[STAGES.index("3_4_place_resized"):]]
    la = sum((stage_time(a_cand, st) or (0,))[0] for st in later)
    lb = sum((stage_time(b_logs, st) or (0,))[0] for st in later)
    w(f"**Downstream (3_4 → 6_report)**: A {la:.1f} s vs B {lb:.1f} s ({la / lb:.2f}×; Δ {la - lb:+.1f} s).\n")

    w("### DREAMPlace breakdown (A)\n")
    w("| item | value |\n|---|---|")
    for k in ("read_db_s", "nlp_init_s", "gp_iterations", "gp_overflow", "gp_whpwl", "macro_lg_ms", "greedy_lg_ms",
              "abacus_lg_ms", "legalization_s", "detailed_place_s", "post_lg_dp_whpwl", "abcd_target_hpwl_first_last",
              "nlp_total_s", "placement_total_s"):
        w(f"| {k} | {fmt(dp.get(k))} |")
    w(f"| subprocess_wall_s (mtimes) | {fmt(dp['subprocess_wall_s'], 2)} |")
    w(f"| provenance dp_runtime_s† / dp_iterations / dp_overflow | {fmt(prov.get('dp_runtime_s'), 3)} / "
      f"{fmt(prov.get('dp_iterations'))} / {fmt(prov.get('dp_overflow'))} |")
    if gpu:
        w(f"| GPU samples (1 Hz, whole run) | {gpu['samples']} |")
        w(f"| GPU peak memory (MiB; idle baseline {gpu['idle_mem_MiB']:.0f}) | {gpu['peak_mem_MiB']:.0f} "
          f"(+{gpu['peak_mem_above_idle_MiB']:.0f}) |")
        w(f"| GPU mean util, non-zero samples ({gpu['nonzero_samples']}) | {gpu['mean_util_nonzero']:.1f}% |")
    w("\n† `provenance.dp_runtime_s` in this run is **ABCDPlace's time only**: the runner's regex took the first "
      "`placement takes` line, which became `detailed placement takes` once detailed placement was on. It's fixed in the "
      "worktree (`non-linear placement takes`, test added). Use `nlp_total_s` / `subprocess_wall_s` instead.\n")
    w("`gp_whpwl` / `post_lg_dp_whpwl` are DREAMPlace wHPWL in DB units. `abcd_target_hpwl_first_last` is ABCDPlace's "
      "exact HPWL at its first and last reported iteration. The GPU samples cover the whole run, including CPU-only ORFS stages, "
      "and the laptop GPU also drives the display, so the idle baseline isn't zero.\n")

    # ---- quality
    qa, qb = quality(a_cand), quality(b_logs)
    vm = (qb["routed_WL"] == 510731 and qb["DRC_final"] == 0 and qb["setup_WNS_ns(finish)"] is not None
          and abs(qb["setup_WNS_ns(finish)"] - (-0.122308)) < 1e-6)
    w(f"B reproduces the VM stock run (WL 510,731, DRC 0, WNS −0.122308): **{vm}**.\n")
    qa["synth_instances(A base)"] = jload(a_base, "1_synth").get("synth__design__instance__count")
    qb["synth_instances(A base)"] = None
    w("## Quality (from 5_1_grt / 5_2_route / 6_report / 3_4 / 3_5)\n")
    w("| metric | A (handoff) | B (stock) | A vs B |\n|---|---|---|---|")
    for k in qa:
        w(f"| {k} | {fmt(qa[k])} | {fmt(qb.get(k))} | {delta(qa[k], qb.get(k))} |")
    w("\nNetlist identity: `synth_instances` must match between A's base and B.\n")
    w("VM stock reference (`results/figures_data/raw/autodmp_ours_stock_bars/stock_netlist_ctrl_*`): DRC 0, "
      "WL 510,731, WNS −0.122 ns, TNS −20.18 ns, power 0.0880 W.\n")
    APD = os.path.join(HERE, "Ap")
    if os.path.isdir(APD):
        ap_section(w, A, APD, B, qa, qb, dp)
    w("## Reading rules (fixed before the numbers were seen)\n")
    w("- n = 1 per arm. A WL difference under ~5% or a WNS difference under ~10–15% is a direction, not a finding.")
    w("- Runtime is the solid part of the comparison.")
    w("- A is not timing-driven and B is, so a timing loss for A is expected (paper §6).\n")
    out = os.path.join(HERE, "compare.md")
    open(out, "w").write("\n".join(L) + "\n")
    print(out)


if __name__ == "__main__":
    main()
