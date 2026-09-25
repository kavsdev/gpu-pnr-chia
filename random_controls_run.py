"""Budget-matched random-search controls for the tinyRocket 5-arm experiment (results/tinyrocket/tinyrocket_random_controls.md).

Runs on chia-dev against a COPY of tinyRocket_main.db (the original is never opened for writing):
  run      preflight (toolchain + cluster), then concurrently
             A. held-out backfill: default_seeds#r0 + llm_grt#r0-2 best configs on all 4 held-out seeds
             B. random_grt#r0-2 search (same settings as the LLM arms, wall-clock capped), each rep validated
                on the 4 held-out seeds as soon as its search ends
           then C. budget matching: if the random reps completed K < 4 rounds, the LLM reps' best config within
           rounds <= K (and any random rep with more than K rounds) is validated too.
  analyze  recomputes everything from raw DB rows only (never from the run JSON) and prints the comparison.

Held-out rows are recorded under "<label>|heldout", separate from the original run's "|validate" rows, where seed
101 failed on every arm-run. Only successful results are replayed from cache, so the failed seeds re-run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from pnr_node.contract import EvalRequest, score
from pnr_node.db import CachedEvaluator, ResultsDB
from pnr_node.evidence import atomic_write_json
from pnr_node.experiment import INFRA_MARKERS, run_arm

DESIGN = "tinyRocket"
SEEDS = (101, 102, 103, 104)
EXPECTED_TOOL_ID = "d356e84aeccb"  # every row of the reported default/llm arm-runs used this toolchain
SETTINGS = dict(rounds=4, batch=8, route_top=3, seed=0)  # identical to tinyRocket_main.json "settings"
BASE_RUNS = ["default_seeds#r0"]
LLM_RUNS = ["llm_grt#r0", "llm_grt#r1", "llm_grt#r2"]
LLM_PROXY_RUNS = ["llm_proxy#r0", "llm_proxy#r1", "llm_proxy#r2"]


def pinned_split_evaluator(cfg: Dict[str, Any]):
    """ChiaEvaluator in split mode with place_remote and route_remote pinned to the SAME VM.

    Predates node.py's built-in split-mode pinning (`ChiaEvaluator._split_slots`); kept as it was run. Before that,
    stock split mode let Ray put route_remote on any node, but the DEF handoff is a file on the placing VM's local
    disk, so on a multi-VM cluster most routes failed with FileNotFoundError (measured 2026-09-24). Here each request
    is mapped to one VM by its grt identity (so proxy/grt/route of the same candidate share a VM and its work dir),
    weighted by the VM's pnr slots. Worker-side code is unchanged; only driver-side dispatch differs.
    """
    import ray
    from pnr_node.node import ChiaEvaluator, place_remote, route_remote

    class PinnedSplit(ChiaEvaluator):
        def _vm_for(self, r: EvalRequest) -> str:
            return slots[int(self._grt_key(r), 16) % len(slots)]

        def _handle_place(self, r: EvalRequest):
            return place_remote.options(resources={"pnr": 1, f"vm_{self._vm_for(r)}": 0.001})

        def _handle_route(self, r: EvalRequest):
            return route_remote.options(resources={"pnr": 1, f"vm_{self._vm_for(r)}": 0.001})

    ev = PinnedSplit(dict(cfg, split=True))  # constructing it connects to the cluster (toolchain_remote)
    slots: List[str] = []
    for n in ray.nodes():
        res = n["Resources"] if n["Alive"] else {}
        vms = [k[3:] for k in res if k.startswith("vm_")]
        if vms and res.get("pnr") and res.get("GPU"):
            slots += [vms[0]] * int(res["pnr"])
    if not slots:
        sys.exit("NO-GO: no alive node advertises vm_*, pnr and GPU")
    log(f"pinned split: VM slot weights { {v: slots.count(v) for v in sorted(set(slots))} }")
    return ev


def log(msg: str) -> None:
    print(f"{dt.datetime.utcnow():%H:%M:%S} {msg}", flush=True)


def utc_today(hhmm: str) -> float:
    h, m = map(int, hhmm.split(":"))
    now = dt.datetime.utcnow()
    return now.replace(hour=h, minute=m, second=0, microsecond=0).replace(tzinfo=dt.timezone.utc).timestamp()


def best_cfg(db: ResultsDB, arm: str, max_round: Optional[int] = None) -> Optional[Dict[str, float]]:
    rows = [r for r in db.evals(arm, DESIGN, "route")
            if r["score"] is not None and (max_round is None or r["round"] <= max_round)]
    return min(rows, key=lambda r: r["score"])["cfg"] if rows else None


def rounds_done(db: ResultsDB, arm: str) -> int:
    """Rounds that produced at least one detailed route (a round cut off before routing does not count)."""
    v = db.con.execute("SELECT MAX(round) FROM evals WHERE arm=? AND design=? AND stage='route'", (arm, DESIGN)).fetchone()[0]
    return int(v or 0)


def heldout(ev, db: ResultsDB, items: List[Tuple[str, Dict[str, float]]], hard_stop: float) -> None:
    """Route every (label, cfg) on every held-out seed in ONE parallel batch; retry infra failures once."""
    todo = [(lbl, cfg, s) for lbl, cfg in items for s in SEEDS]
    for attempt in (1, 2):
        if not todo:
            return
        if time.time() >= hard_stop:
            log(f"hard stop reached, {len(todo)} held-out routes NOT launched: {sorted({t[0] for t in todo})}")
            return
        log(f"held-out attempt {attempt}: {len(todo)} routes for {sorted({t[0] for t in todo})}")
        reqs = [EvalRequest(DESIGN, placer_cfg=cfg, seed=s, stage="route", tool_id=ev.tool_id) for _, cfg, s in todo]
        res = ev.evaluate_many(reqs)
        retry = []
        for (lbl, cfg, s), r, x in zip(todo, reqs, res):
            db.record_eval(lbl + "|heldout", 1, r, x, "cache_hit" if x.provenance.get("cache_hit") else "")
            if not x.ok and any(m in (x.error or "") for m in INFRA_MARKERS):
                retry.append((lbl, cfg, s))
        todo = retry


def run(a) -> None:
    if os.path.exists(a.db) and not a.resume:
        sys.exit(f"{a.db} exists; pass --resume to continue it, or choose a new path")
    if not os.path.exists(a.db):
        shutil.copy2(a.src_db, a.db)
        db = ResultsDB(a.db)
        # Park the old random-arm rows: run_search resumes from max(round) of the arm name, and those rows (all
        # infra failures or an older toolchain) must neither extend nor pollute the new reps. Successful rows stay
        # replayable from cache under the new name.
        for pat in ("random_grt#%", "random_proxy#%"):
            db.con.execute("UPDATE evals SET arm='_old|'||arm WHERE arm LIKE ?", (pat,))
        db.con.commit()
    db = ResultsDB(a.db)

    if a.mock:
        from pnr_node.mock_eval import MockEvaluator
        inner = MockEvaluator()
    else:
        from pnr_node.node import ChiaEvaluator
        cfg = {"work_root": a.work_root, "evidence_dir": os.path.join(a.work_root, "evidence"), "gpu": True,
               "vm_affinity": a.vm_affinity}
        inner = pinned_split_evaluator(cfg) if a.split else ChiaEvaluator(cfg)
        if inner.tool_id != EXPECTED_TOOL_ID:
            sys.exit(f"NO-GO: toolchain {inner.tool_id} != {EXPECTED_TOOL_ID} used by the LLM arms; results would not be comparable")
        try:
            import ray
            log(f"cluster resources: {ray.cluster_resources()}")
        except Exception as e:  # informational only
            log(f"could not read cluster resources: {e}")
    ev = CachedEvaluator(inner, db)
    log(f"toolchain {ev.tool_id} OK")

    search_deadline, hard_stop = utc_today(a.search_deadline_utc), utc_today(a.hard_stop_utc)
    reps = [f"random_grt#r{r}" for r in range(3)] + ([f"random_proxy#r{r}" for r in range(3)] if a.random_proxy else [])
    out: Dict[str, Any] = {"tool_id": ev.tool_id, "src_db": a.src_db, "settings": SETTINGS, "validate_seeds": list(SEEDS),
                           "search_deadline_utc": a.search_deadline_utc, "split": a.split, "hard_stop_utc": a.hard_stop_utc, "search": []}

    def backfill():
        runs = BASE_RUNS + LLM_RUNS + (LLM_PROXY_RUNS if a.llm_proxy else [])
        heldout(ev, db, [(n, best_cfg(db, n)) for n in runs if best_cfg(db, n)], hard_stop)
        log("backfill done")

    def search(name):
        arm, rep = name.split("#r")
        log(f"start {name}")
        res = run_arm(ev, db, arm, int(rep), DESIGN, deadline=search_deadline, **SETTINGS)
        res["rounds_with_routes"] = rounds_done(db, name)
        log(f"done  {name} rounds={res['rounds_with_routes']} routes={res.get('routes_used')} best={res.get('best_score')}")
        cfg = best_cfg(db, name)
        if cfg:
            heldout(ev, db, [(name, cfg)], hard_stop)
        return res

    with ThreadPoolExecutor(len(reps) + 1) as ex:
        fb = ex.submit(backfill)
        out["search"] = list(ex.map(search, reps))
        fb.result()
    atomic_write_json(a.out, out)

    # C. budget matching at K = the fewest routed rounds any random rep reached
    ks = [s["rounds_with_routes"] for s in out["search"] if s["arm"].startswith("random_grt")]
    k = min(ks) if ks else 0
    out["budget_matched_K"] = k
    if 0 < k < SETTINGS["rounds"]:
        items = []
        for n in LLM_RUNS + [s["arm"] for s in out["search"] if s["arm"].startswith("random_grt")]:
            cfg_k, cfg_all = best_cfg(db, n, k), best_cfg(db, n)
            if cfg_k:
                items.append((f"{n}@{k}", cfg_k))  # identical cfg to the final best -> pure cache hits
        heldout(ev, db, items, hard_stop)
    out["cache_hits"] = ev.hits
    atomic_write_json(a.out, out)
    log(f"run complete; K={k}; now run: python random_controls_run.py analyze --db {a.db}")


def analyze(a) -> None:
    """Independent recompute from raw rows only."""
    db = ResultsDB(a.db)
    rows = db.con.execute("SELECT arm, req_json, result_json, note, req_key FROM evals WHERE arm LIKE '%|heldout' ORDER BY id").fetchall()
    # Two labels can hold the SAME request (e.g. an LLM rep whose best cfg is the default config): dispatched in one
    # batch they race in one work dir and one fails. Identical identities are what the cache replays, so a failed row
    # is resolved from a successful row with the same req_key.
    ok_by_key: Dict[str, str] = {}
    for _a, _q, rj, _n, key in rows:
        if json.loads(rj)["ok"]:
            ok_by_key.setdefault(key, rj)
    rows = [(arm, rq, rj if json.loads(rj)["ok"] else ok_by_key.get(key, rj), note) for arm, rq, rj, note, key in rows]
    per: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for arm, rq, rj, note in rows:
        q, r = json.loads(rq), json.loads(rj)
        lbl = arm[: -len("|heldout")]
        seed = q["seed"]
        prev = per.setdefault(lbl, {}).get(seed)
        if prev and prev["ok"]:
            continue  # keep the first success per seed
        per[lbl][seed] = {"ok": r["ok"], "wl": r.get("routed_wl"), "wns": r.get("wns"), "drc": r.get("drc_violations"),
                          "mock": bool(r.get("provenance", {}).get("mock")), "tool_id": q.get("tool_id"),
                          "err": (r.get("error") or "")[:80]}
    problems = []
    print("| label | seeds ok | WL per seed (101..104) | mean WL | DRC | WNS mean |\n|---|---|---|---|---|---|")
    means: Dict[str, float] = {}
    for lbl in sorted(per):
        s = per[lbl]
        ok = [s[x] for x in SEEDS if x in s and s[x]["ok"]]
        wl = [x["wl"] for x in ok]
        if len(ok) == len(SEEDS):
            means[lbl] = st.mean(wl)
        else:
            problems.append(f"{lbl}: only {len(ok)}/4 held-out seeds ok -> excluded from the comparison")
        for x in s.values():
            if x["mock"]:
                problems.append(f"{lbl}: MOCK row")
            if x["tool_id"] != EXPECTED_TOOL_ID:
                problems.append(f"{lbl}: tool_id {x['tool_id']}")
        cells = " / ".join(f"{s[x]['wl']:.0f}" if x in s and s[x]["ok"] else "FAIL" for x in SEEDS)
        wns = [x["wns"] for x in ok if x["wns"] is not None]
        print(f"| {lbl} | {len(ok)}/4 | {cells} | {st.mean(wl):.0f} | {sum(x['drc'] or 0 for x in ok)} | "
              f"{st.mean(wns):.3f} |" if ok else f"| {lbl} | 0/4 | {cells} | - | - | - |")

    def group(arm: str, k: Optional[str]) -> List[float]:
        # A rep that stopped at exactly K rounds has no "@K" label: its final result IS its K-round result.
        vals = [means.get(f"{arm}#r{r}@{k}", means.get(f"{arm}#r{r}")) if k else means.get(f"{arm}#r{r}")
                for r in range(3)]
        return [v for v in vals if v is not None]

    ks = sorted({lbl.split("@")[1] for lbl in means if "@" in lbl})
    print()
    for tag, k in [("final budget (random may have fewer rounds than LLM)", None)] + [(f"budget-matched K={k}", k) for k in ks]:
        llm, rnd = (group("llm_grt", k) if not k else [means[f"llm_grt#r{r}@{k}"] for r in range(3)
                                                          if f"llm_grt#r{r}@{k}" in means]), group("random_grt", k)
        base = means.get("default_seeds#r0")
        print(f"## {tag}: llm_grt n={len(llm)} {['%.0f' % x for x in llm]}  random_grt n={len(rnd)} {['%.0f' % x for x in rnd]}"
              + (f"  default={base:.0f}" if base else ""))
        if len(llm) >= 2 and len(rnd) >= 2:
            d = st.mean(llm) - st.mean(rnd)
            se = (st.variance(llm) / len(llm) + st.variance(rnd) / len(rnd)) ** 0.5
            verdict = ("LLM better" if d < -2 * se else "random better" if d > 2 * se else "no detectable difference")
            print(f"   llm - random = {d:+.0f} um ({100 * d / st.mean(rnd):+.2f}%), 2*SE = {2 * se:.0f} um -> {verdict}")
        else:
            print("   fewer than 2 complete reps per side -> no comparison")
    problems = list(dict.fromkeys(problems))
    print("\nPROBLEMS:" if problems else "\nno integrity problems")
    for p in problems:
        print("  -", p)
    n_search = db.con.execute("SELECT arm, COUNT(*), SUM(CASE WHEN result_json LIKE '%\"ok\": false%' THEN 1 ELSE 0 END) "
                              "FROM evals WHERE arm LIKE 'random_%' AND arm NOT LIKE '%|%' GROUP BY arm").fetchall()
    print("\nrandom search rows (arm, evals, failed):", [tuple(x) for x in n_search])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--src-db", required=True)
    r.add_argument("--db", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--work-root", default="/pnr_work/exp")
    r.add_argument("--search-deadline-utc", required=True, help="HH:MM UTC today: no new search round starts after this")
    r.add_argument("--hard-stop-utc", required=True, help="HH:MM UTC today: no new held-out batch starts after this")
    r.add_argument("--vm-affinity", action="store_true")
    r.add_argument("--split", action="store_true", help="GPU only for placement, route on CPU pnr slots (throughput)")
    r.add_argument("--random-proxy", action="store_true", help="also run random_proxy#r0-2 (only if GPUs allow)")
    r.add_argument("--llm-proxy", action="store_true", help="also backfill llm_proxy#r0-2 held-out seeds")
    r.add_argument("--resume", action="store_true")
    r.add_argument("--mock", action="store_true", help="dry run with MockEvaluator (skips the toolchain gate)")
    z = sp.add_parser("analyze")
    z.add_argument("--db", required=True)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else analyze(a)
