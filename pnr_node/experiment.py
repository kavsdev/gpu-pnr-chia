"""The fixed-budget arm comparison, with a held-out-seed validation step.

Protocol (why it looks like this): routed results move by ~3% (sigma) between seeds even for the same config, so the best
of N routed candidates is biased low ("winner's curse": a seed lottery). We therefore
  1. SEARCH: every arm gets the same number of detailed routes and selects its best config on evaluation seed `seed`;
  2. VALIDATE: each arm-run's best config is re-routed on K held-out seeds; arms are compared on the validated mean;
  3. BASELINE: the default config routed on many seeds gives the seed-lottery distribution to beat.

Arms (all with the same detailed-route budget = rounds * route_top):
  default_seeds  default knobs, one route per seed (the noise floor / seed lottery)
  random_proxy   random candidates, the DREAMPlace proxy picks who is routed (the original two-tier funnel)
  random_grt     random candidates, the global-route estimate picks who is routed
  llm_proxy      LLM sees proxy numbers only, proxy picks who is routed (the "proxy-only" pipeline)
  llm_grt        LLM sees global-route and routed results, the global-route estimate picks who is routed
"""
from __future__ import annotations

import json
import statistics as st
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence

from .contract import CORE_KNOBS, KNOBS, Constraints, EvalRequest, default_config, score
from .db import CachedEvaluator, ResultsDB
from .search import LLMProposer, RandomProposer, run_search

# "llm_grt_expanded": same funnel + budget as llm_grt, but the proposer tunes the full KNOBS action space and sees
# fuller routed context (tns/power/area + DREAMPlace's dp_iterations/dp_overflow); llm_grt is restricted to
# CORE_KNOBS and wl/wns/drc. The two differ only in action space + context, so a same-budget A/B isolates that.
ARMS = ("default_seeds", "random_proxy", "random_grt", "llm_proxy", "llm_grt", "llm_grt_expanded")


def _arm_name(arm: str, rep: int) -> str:
    return f"{arm}#r{rep}"


def run_arm(evaluator, db: ResultsDB, arm: str, rep: int, design: str, *, rounds: int, batch: int, route_top: int,
            seed: int, provider_factory: Optional[Callable[[], Any]] = None,
            deadline: Optional[float] = None, constraints: Optional[Constraints] = None) -> Dict[str, Any]:
    name = _arm_name(arm, rep)
    if arm == "default_seeds":
        if deadline is not None and time.time() >= deadline:  # budget already exhausted before this arm-run started
            return {"arm": name, "routes_used": 0, "best_score": None, "best_cfg": None, "best_seed": None,
                    "stopped_early": True}
        # one route per seed; the searching seed is included so this arm sees the same budget
        tool_id = getattr(evaluator, "tool_id", "")
        constraints_id = constraints.id if constraints else ""
        reqs = [EvalRequest(design, placer_cfg=default_config(), seed=seed + i, stage="route", tool_id=tool_id,
                            constraints_id=constraints_id)
                for i in range(rounds * route_top)]
        res = evaluator.evaluate_many(reqs) if hasattr(evaluator, "evaluate_many") else [evaluator.evaluate(r) for r in reqs]
        for r, x in zip(reqs, res):
            db.record_eval(name, 1, r, x, "cache_hit" if x.provenance.get("cache_hit") else "")
        best = db.best_routed(name, design)
        return {"arm": name, "routes_used": len(reqs), "best_score": best["score"] if best else None,
                "best_cfg": best["cfg"] if best else None, "best_seed": best["result"].provenance.get("seed") if best else None,
                "stopped_early": False}
    if arm.startswith("random"):
        proposer = RandomProposer(1000 * rep + 7, knobs=CORE_KNOBS)
    elif arm == "llm_grt_expanded":  # full action space + fuller routed context, otherwise identical to llm_grt
        proposer = LLMProposer(provider_factory(), db, arm=name, show_routed=True, knobs=KNOBS, expanded_context=True,
                               constraints=constraints)
    else:  # llm_grt (routed context, grt funnel) / llm_proxy (proxy-only), both restricted to the core 5 knobs
        proposer = LLMProposer(provider_factory(), db, arm=name, show_routed=(arm == "llm_grt"), knobs=CORE_KNOBS,
                               constraints=constraints)
    grt_top = batch if (arm.endswith("_grt") or arm == "llm_grt_expanded") else None
    out = run_search(evaluator, proposer, db, name, design, rounds=rounds, batch=batch, route_top=route_top,
                     seed=seed, grt_top=grt_top, deadline=deadline, constraints=constraints)
    out["invalid_llm_answers"] = getattr(proposer, "invalid", None)
    return out


def validate(evaluator, db: ResultsDB, design: str, arm_names: Sequence[str], seeds: Sequence[int]) -> Dict[str, Any]:
    """Re-route each arm-run's best config on held-out seeds. Returns per-arm-run validated statistics."""
    tool_id = getattr(evaluator, "tool_id", "")
    out: Dict[str, Any] = {}
    for name in arm_names:
        best = db.best_routed(name, design)
        if not best:
            out[name] = {"error": "no routed result"}
            continue
        reqs = [EvalRequest(design, placer_cfg=best["cfg"], seed=s, stage="route", tool_id=tool_id) for s in seeds]
        res = evaluator.evaluate_many(reqs) if hasattr(evaluator, "evaluate_many") else [evaluator.evaluate(r) for r in reqs]
        vname = name + "|validate"
        for r, x in zip(reqs, res):
            db.record_eval(vname, 1, r, x, "cache_hit" if x.provenance.get("cache_hit") else "")
        sc = [score(x) for x in res if x.ok]
        wl = [x.routed_wl for x in res if x.ok and x.routed_wl is not None]
        out[name] = {"selection_score": best["score"], "validated_scores": sc, "validated_mean": st.mean(sc) if sc else None,
                     "validated_std": st.pstdev(sc) if len(sc) > 1 else None, "validated_wl_mean": st.mean(wl) if wl else None,
                     "n_ok": len(sc), "cfg": best["cfg"]}
    return out


def run_experiment(evaluator, db: ResultsDB, design: str, *, arms: Sequence[str] = ARMS, repeats: int = 3, rounds: int = 4,
                   batch: int = 8, route_top: int = 3, seed: int = 0, validate_seeds: Sequence[int] = (101, 102, 103, 104),
                   workers: int = 4, provider_factory: Optional[Callable[[], Any]] = None,
                   log: Callable[[str], None] = print, jobs: Optional[Sequence[tuple]] = None,
                   max_wall_s: Optional[float] = None, constraints: Optional[Constraints] = None) -> Dict[str, Any]:
    """`max_wall_s` (unset by default: unbounded) bounds the whole experiment's wall-clock, checked between search
    rounds only -- an in-flight route always finishes."""
    ev = CachedEvaluator(evaluator, db)
    deadline = time.time() + max_wall_s if max_wall_s is not None else None
    if jobs is None:
        jobs = [(a, r) for a in arms for r in (range(1) if a == "default_seeds" else range(repeats))]

    def one(job):
        a, r = job
        log(f"start {_arm_name(a, r)}")
        res = run_arm(ev, db, a, r, design, rounds=rounds, batch=batch, route_top=route_top, seed=seed,
                      provider_factory=provider_factory, deadline=deadline, constraints=constraints)
        log(f"done  {_arm_name(a, r)} best={res.get('best_score')}" + (" [STOPPED EARLY: wall-clock budget]"
                                                                        if res.get("stopped_early") else ""))
        return res

    with ThreadPoolExecutor(workers) as ex:
        searched = list(ex.map(one, jobs))
    names = [x["arm"] for x in searched]
    val = validate(ev, db, design, names, validate_seeds)
    return {"design": design, "settings": {"rounds": rounds, "batch": batch, "route_top": route_top, "repeats": repeats,
                                           "seed": seed, "validate_seeds": list(validate_seeds), "max_wall_s": max_wall_s},
            "search": searched, "validation": val, "cache_hits": ev.hits,
            "any_stopped_early": any(x.get("stopped_early") for x in searched)}


# Errors that mean the infrastructure failed, not the candidate: an arm-run that saw one is not a fair sample.
INFRA_MARKERS = ("NVIDIA driver", "worker:", "WorkerCrashed", "ActorDied", "No CUDA GPUs", "ORFS to 3_2 failed",
                 "pre_gp.def dump failed", "could not extract ORFS scripts", "OSError", "ConnectionError",
                 "TimeoutExpired")


def infra_failed_arm_runs(db: ResultsDB) -> List[str]:
    """Arm-runs (search arms, not validation rows) containing at least one infrastructure failure."""
    bad = set()
    for arm, rj in db.con.execute("SELECT arm, result_json FROM evals").fetchall():
        r = json.loads(rj)
        if not r["ok"] and any(m in (r.get("error") or "") for m in INFRA_MARKERS):
            bad.add(arm.split("|")[0])
    return sorted(bad)


def purge_arm_runs(db: ResultsDB, arms: Sequence[str]) -> int:
    """Delete every eval and LLM call of the given arm-runs (and their validation rows). Successful evaluations stay
    cached under other rows' identities only if another arm-run also produced them; the reruns recompute the rest."""
    n = 0
    for a in arms:
        for name in (a, a + "|validate"):
            n += db.con.execute("DELETE FROM evals WHERE arm=?", (name,)).rowcount
            db.con.execute("DELETE FROM llm_calls WHERE arm=?", (name,))
    db.con.commit()
    return n


def repair(evaluator, db: ResultsDB, design: str, previous: Dict[str, Any], **kw) -> Dict[str, Any]:
    """Rerun only the arm-runs that hit infrastructure failures and merge them into the previous result JSON."""
    bad = infra_failed_arm_runs(db)
    if not bad:
        return previous
    # Keep the successful evaluations reusable: copy them under a neutral arm name before purging.
    for a in bad:
        db.con.execute("UPDATE evals SET arm=? WHERE arm=? AND result_json LIKE '%\"ok\": true%'", ("_cache|" + a, a))
        db.con.execute("UPDATE evals SET arm=? WHERE arm=? AND result_json LIKE '%\"ok\": true%'", ("_cache|" + a + "|validate", a + "|validate"))
    db.con.commit()
    purge_arm_runs(db, bad)
    jobs = []
    for a in bad:
        arm, rep = a.split("#r")
        jobs.append((arm, int(rep)))
    new = run_experiment(evaluator, db, design, jobs=jobs, **kw)
    keep = [s for s in previous["search"] if s["arm"] not in bad]
    merged = dict(previous, search=keep + new["search"], validation={**{k: v for k, v in previous["validation"].items()
                                                                        if k not in bad}, **new["validation"]},
                  repaired=bad, cache_hits=previous.get("cache_hits", 0) + new["cache_hits"])
    return merged


def summarize(result: Dict[str, Any]) -> str:
    """Markdown table: per arm, selection-seed best vs held-out validated mean (the number to trust)."""
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for s in result["search"]:
        by_arm.setdefault(s["arm"].split("#")[0], []).append(s)
    lines = ["| arm | runs | selection best (mean) | validated mean | validated sd (within run) | routes/run |", "|---|---|---|---|---|---|"]
    def fmt(xs: List[float]) -> str:
        # an arm can have zero usable entries (e.g. all infra failures); st.mean([]) would raise
        return f"{st.mean(xs):.0f}" if xs else "n/a"

    for arm, runs in by_arm.items():
        sel = [r["best_score"] for r in runs if r.get("best_score") is not None]
        vals = [result["validation"][r["arm"]] for r in runs if result["validation"].get(r["arm"], {}).get("validated_mean")]
        vm = [v["validated_mean"] for v in vals]
        vs = [v["validated_std"] for v in vals if v.get("validated_std") is not None]
        lines.append(f"| {arm} | {len(runs)} | {fmt(sel)} | {fmt(vm)} | {fmt(vs)} | {runs[0].get('routes_used')} |")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import os

    ap = argparse.ArgumentParser(description="Run the arm comparison against the CHIA cluster (or the mock)")
    ap.add_argument("--design", default="gcd")
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--route-top", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--work-root", default="/pnr_work/exp")
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--provider", default="vertex",
                    help="LLM provider for the llm_* arms. Default 'vertex' (Gemini on GCP). Off-cloud options: "
                         "'gemini_api' (Gemini Developer API, GEMINI_API_KEY/GOOGLE_API_KEY), "
                         "'anthropic'/'claude' (ANTHROPIC_API_KEY, pass a Claude --model), "
                         "'openai_compat'/'vllm'/'ollama' (pass --base-url).")
    ap.add_argument("--base-url", default=None, help="base URL for --provider openai_compat/vllm/ollama")
    ap.add_argument("--repair", action="store_true", help="rerun only arm-runs that hit infrastructure failures")
    ap.add_argument("--vm-affinity", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="DREAMPlace on CPU for every worker (device is part of the identity)")
    ap.add_argument("--max-wall-s", type=float, default=None,
                    help="stop launching new search rounds past this many wall-clock seconds (unset: unbounded)")
    a = ap.parse_args()
    if a.mock:
        from .mock_eval import MockEvaluator

        evaluator = MockEvaluator()
    else:
        from .node import ChiaEvaluator

        evaluator = ChiaEvaluator({"work_root": a.work_root, "evidence_dir": os.path.join(a.work_root, "evidence"),
                                   "gpu": not a.cpu, "vm_affinity": a.vm_affinity})
    from .llm import make_provider

    # `location` is only read by the vertex provider (others take **_), so this spec is unchanged for the default.
    _spec = {"provider": a.provider, "model": a.model, "location": "global"}
    if a.base_url:
        _spec["base_url"] = a.base_url
    factory = lambda: make_provider(_spec)
    db = ResultsDB(a.db)
    kw = dict(repeats=a.repeats, rounds=a.rounds, batch=a.batch, route_top=a.route_top, workers=a.workers,
              provider_factory=factory, max_wall_s=a.max_wall_s)
    from .evidence import atomic_write_json
    if a.repair:
        res = repair(evaluator, db, a.design, json.load(open(a.out)), **kw)
    else:
        res = run_experiment(evaluator, db, a.design, arms=a.arms.split(","), **kw)
    atomic_write_json(a.out, res)
    print(summarize(res))
