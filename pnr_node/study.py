"""Proxy-vs-route study: does DREAMPlace's cheap cost rank configurations the way the full route does?

The two-tier funnel routes only the proxy-best candidates, so it is justified only if the proxy ranking predicts
the routed ranking well enough. This measures that.

  place    (GPU machine)  sample configs, run DREAMPlace, keep every placement DEF + proxy cost
  route    (CPU machine)  route each shipped DEF through stock ORFS, several in parallel
  analyze                  Spearman rank correlation, top-k recall, and how much noise limits the answer
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

from .contract import CORE_KNOBS as KNOBS, EvalRequest, score


# The original wide ranges (before the knob space was narrowed to the validated region); ~80% of samples are infeasible.
WIDE = {"target_density": (0.4, 1.0), "density_weight": (1e-6, 1e-2), "gamma": (1.0, 20.0),
        "stop_overflow": (0.05, 0.3), "learning_rate": (0.001, 0.05)}


def sample_configs(n: int, seed: int = 0, ranges: Optional[Dict[str, tuple]] = None) -> List[Dict[str, float]]:
    """Latin-hypercube sample over the knob ranges (log-uniform where the knob is log-scaled)."""
    rng = random.Random(seed)
    cols = {}
    for name, kb in KNOBS.items():
        lo, hi = (ranges or {}).get(name, (kb.lo, kb.hi))
        strata = [(i + rng.random()) / n for i in range(n)]
        rng.shuffle(strata)
        cols[name] = [
            kb.clamp(math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))) if (kb.log and lo > 0)
                     else lo + u * (hi - lo))
            for u in strata
        ]
    return [{k: cols[k][i] for k in KNOBS} for i in range(n)]


def _ranks(v: Sequence[float]) -> List[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1  # average rank for ties
        i = j + 1
    return r


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def topk_recall(proxy: Sequence[float], routed: Sequence[float], k: int) -> float:
    """Fraction of the routed-best k that the proxy-best k also contains (1.0 = the funnel loses nothing)."""
    pk = set(sorted(range(len(proxy)), key=lambda i: proxy[i])[:k])
    rk = set(sorted(range(len(routed)), key=lambda i: routed[i])[:k])
    return len(pk & rk) / k


def analyze(rows: List[Dict[str, Any]], noise_sigma_pct: float = 3.0) -> Dict[str, Any]:
    ok = [r for r in rows if r.get("routed_wl") is not None and r.get("proxy_hpwl") is not None]
    if len(ok) < 4:
        return {"n_routed": len(ok), "error": "too few routed points"}
    px, wl = [r["proxy_hpwl"] for r in ok], [r["routed_wl"] for r in ok]
    sc = [r["score"] for r in ok]
    spread = (max(wl) - min(wl)) / (sum(wl) / len(wl)) * 100
    return {
        "n_total": len(rows), "n_routed": len(ok), "n_rejected_or_failed": len(rows) - len(ok),
        "spearman_proxy_vs_routed_wl": spearman(px, wl),
        "spearman_proxy_vs_score": spearman(px, sc),
        "top3_recall": topk_recall(px, sc, 3) if len(ok) >= 6 else None,
        "top5_recall": topk_recall(px, sc, 5) if len(ok) >= 10 else None,
        "routed_wl_range_pct_of_mean": round(spread, 1),
        "noise_sigma_pct_assumed": noise_sigma_pct,
        "signal_to_noise": round(spread / (4 * noise_sigma_pct), 2),  # range vs ~ +-2 sigma band
    }


def _cmd_place(a) -> None:
    from .raw_handoff import RawHandoffEvaluator

    ev = RawHandoffEvaluator(a.work, a.dp_install, python=a.python, native=a.native)
    os.makedirs(os.path.join(a.out, "defs"), exist_ok=True)
    rows = []
    for i, cfg in enumerate(sample_configs(a.n, a.seed, WIDE if a.wide else None)):
        req = EvalRequest(a.design, placer_cfg=cfg, seed=a.run_seed, stage="proxy", tool_id=ev.tool_id)
        res = ev.evaluate(req)
        row = {"idx": i, "cfg": req.placer_cfg, "seed": a.run_seed, "ok": res.ok, "proxy_hpwl": res.proxy_hpwl,
               "error": res.error, "dp_runtime_s": res.provenance.get("dp_runtime_s"),
               "dp_iterations": res.provenance.get("dp_iterations"), "dp_overflow": res.provenance.get("dp_overflow"),
               "rejected": bool(res.provenance.get("rejected")), "def": None}
        run_dp = os.path.join(a.work, a.design, req.key(), "dp_out")
        gp = [os.path.join(r, n) for r, _, ns in os.walk(run_dp) for n in ns if n.endswith(".gp.def")] if os.path.isdir(run_dp) else []
        if res.ok and gp:
            dst = os.path.join(a.out, "defs", f"{i:03d}.gp.def")
            shutil.copy(gp[0], dst)
            row["def"] = os.path.relpath(dst, a.out)
        rows.append(row)
        print(f"[{i + 1}/{a.n}] ok={res.ok} hpwl={res.proxy_hpwl} {res.error[:80]}", flush=True)
    from .evidence import atomic_write_json
    atomic_write_json(os.path.join(a.out, "place.json"), {"design": a.design, "rows": rows})


def _cmd_route(a) -> None:
    from .raw_handoff import RawHandoffEvaluator

    data = json.load(open(os.path.join(a.indir, "place.json")))
    ev = RawHandoffEvaluator(a.work, a.dp_install, python=a.python, native=a.native, docker_cpus=a.docker_cpus)
    base = ev.prepare(data["design"])
    todo = [r for r in data["rows"] if r["ok"] and r["def"]]

    def one(r):
        t0 = time.time()
        req = EvalRequest(data["design"], placer_cfg=r["cfg"], seed=r["seed"], stage="route")
        run = os.path.join(a.work, data["design"], f"study-{r['idx']:03d}")
        os.makedirs(run, exist_ok=True)
        try:
            res = ev.route_from_def(req, base, run, os.path.join(a.indir, r["def"]), r["proxy_hpwl"], {}, t0)
        except Exception as e:
            return dict(r, routed_wl=None, error=f"{type(e).__name__}: {e}")
        return dict(r, routed_wl=res.routed_wl, wns=res.wns, tns=res.tns, power=res.power, area=res.area,
                    drc=res.drc_violations, score=score(res), route_runtime_s=res.runtime_s, error=res.error)

    with ThreadPoolExecutor(a.workers) as ex:
        out = []
        for row in ex.map(one, todo):
            out.append(row)
            print(f"routed {row['idx']:03d} wl={row.get('routed_wl')} wns={row.get('wns')}", flush=True)
    routed = {r["idx"]: r for r in out}
    rows = [routed.get(r["idx"], r) for r in data["rows"]]
    from .evidence import atomic_write_json
    atomic_write_json(os.path.join(a.indir, "routed.json"), {"design": data["design"], "rows": rows})


def _cmd_analyze(a) -> None:
    data = json.load(open(os.path.join(a.indir, "routed.json")))
    out = analyze(data["rows"], a.noise_sigma_pct)
    from .evidence import atomic_write_json
    print(json.dumps(out, indent=1))
    atomic_write_json(os.path.join(a.indir, "analysis.json"), out)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("place", "route"):
        p = sub.add_parser(name)
        p.add_argument("--design", default="gcd")
        p.add_argument("--work", default=os.path.expanduser("~/work/study"))
        p.add_argument("--dp-install", default=os.path.expanduser("~/work/dreamplace/install2"))
        p.add_argument("--python", default="/usr/bin/python3")
        p.add_argument("--native", action="store_true")
        if name == "place":
            p.add_argument("--n", type=int, default=40)
            p.add_argument("--seed", type=int, default=0, help="sampling seed")
            p.add_argument("--run-seed", type=int, default=0, help="DREAMPlace seed for every config")
            p.add_argument("--wide", action="store_true", help="sample the original wide ranges instead of the validated box")
            p.add_argument("--out", required=True)
        else:
            p.add_argument("--indir", required=True)
            p.add_argument("--workers", type=int, default=4)
            p.add_argument("--docker-cpus", type=float, default=2)
    p = sub.add_parser("analyze")
    p.add_argument("--indir", required=True)
    p.add_argument("--noise-sigma-pct", type=float, default=3.0)
    a = ap.parse_args()
    {"place": _cmd_place, "route": _cmd_route, "analyze": _cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
