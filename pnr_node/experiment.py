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
import math
import statistics as st
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Sequence

from .contract import (CORE_KNOBS, FLOW_KNOBS, KNOBS, Constraints, EvalRequest, EvalResult, clamp_config,
                       default_config, passes, score, score_timing)
from .db import CachedEvaluator, ResultsDB
from .search import (LLMProposer, RandomProposer, TPEProposer, _eval_many, config_stats, run_search,
                     run_search_v2)

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


# ---------------------------------------------------------------------------------------------------------------------
# v2 arms: flow-level knobs (FLOW_KNOBS), timing-first objective (score_timing), budget in evaluations,
# multi-seed promotion, held-out validation at `route`, paired comparison against the default flow.
# ---------------------------------------------------------------------------------------------------------------------

ARMS_V2 = ("default", "random", "tpe", "llm", "llm_diag")

# arm -> factory(name, rep, db, provider_factory) -> proposer. "llm_diag" is registered by
# pnr_node.diag_proposer (evidence-rich prompt) via register_arm_v2; until then requesting it raises a clear error.
PROPOSERS_V2: Dict[str, Callable[..., Any]] = {}


def register_arm_v2(arm: str, factory: Callable[..., Any]) -> None:
    """Hook for new searching arms (e.g. llm_diag): factory(name, rep, db, provider_factory) -> proposer."""
    PROPOSERS_V2[arm] = factory


def _llm_proposer_v2(name, rep, db, provider_factory):
    if provider_factory is None:
        raise ValueError("the llm arm needs a provider_factory")
    return LLMProposer(provider_factory(), db, arm=name, knobs=FLOW_KNOBS, objective="timing",
                       fallback=RandomProposer(1000 * rep + 13, FLOW_KNOBS))


register_arm_v2("random", lambda name, rep, db, pf: RandomProposer(1000 * rep + 7, knobs=FLOW_KNOBS))
register_arm_v2("tpe", lambda name, rep, db, pf: TPEProposer(1000 * rep + 11, knobs=FLOW_KNOBS))
register_arm_v2("llm", _llm_proposer_v2)


def _v2_name(arm: str, rep: int) -> str:
    return f"v2_{arm}#r{rep}"


def default_flow_cfg() -> Dict[str, Any]:
    return clamp_config({}, FLOW_KNOBS)


def run_arm_v2(evaluator, db: ResultsDB, arm: str, rep: int, design: str, *, budget_evals: int, batch: int,
               stage: str = "cts", seeds: Sequence[int] = (0,), promote_top: int = 2,
               promote_seeds: Sequence[int] = (1,), default_seeds: Optional[Sequence[int]] = None,
               deadline: Optional[float] = None, provider_factory: Optional[Callable[[], Any]] = None,
               tech: str = "nangate45", log: Optional[Callable[[str], None]] = None,
               prior: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    name = _v2_name(arm, rep)
    if arm == "default":  # the default flow on seeds 0..k-1: the seed lottery / noise floor. Always runs (one round).
        ds = tuple(default_seeds) if default_seeds is not None else tuple(range(4))
        tool_id = getattr(evaluator, "tool_id", "")
        reqs = [EvalRequest(design, tech, default_config(), s, stage, tool_id, flow_cfg=default_flow_cfg()) for s in ds]
        for q, x in zip(reqs, _eval_many(evaluator, reqs)):
            db.record_eval(name, 1, q, x, "default" + (",cache_hit" if x.provenance.get("cache_hit") else ""),
                           score_fn=score_timing)
        st_ = config_stats(db, name, design, stage)
        row = st_[0] if st_ else None
        sc = [x for x in (row["seed_scores"] if row else []) if x != float("inf")]
        return {"arm": name, "design": design, "stage": stage, "evals_used": len(reqs), "rounds": 1,
                "best_cfg": default_flow_cfg(), "best_score": st.mean(sc) if sc else None,
                "best_n_seeds": len(sc), "best_seed_scores": row["seed_scores"] if row else None,
                "seed_lottery_sd": st.stdev(sc) if len(sc) > 1 else None,
                "seed_lottery_min": min(sc) if sc else None, "stopped_early": False, "stop_reason": "budget"}
    if arm not in PROPOSERS_V2:
        raise ValueError(f"arm {arm!r} has no registered proposer (register_arm_v2); known: {sorted(PROPOSERS_V2)}")
    proposer = PROPOSERS_V2[arm](name, rep, db, provider_factory)
    return run_search_v2(evaluator, proposer, db, name, design, budget_evals=budget_evals, batch=batch, stage=stage,
                         seeds=seeds, promote_top=promote_top, promote_seeds=promote_seeds, deadline=deadline,
                         tech=tech, log=log, prior=prior)


def incumbent_prior(db: ResultsDB, design: str, stage: str, seeds: Sequence[int]) -> List[Dict[str, Any]]:
    """The default arm-run's config_stats row restricted to `seeds` (the searching arms' search + promotion seeds),
    so an arm compares its candidates with the default on the same seeds. [] if the default has none of them."""
    rows = [h for h in db.evals(_v2_name("default", 0), design, stage)
            if json.loads(h["req_json"]).get("seed") in set(seeds)]
    if not rows:
        return []
    by_seed = {}
    for h in rows:
        by_seed.setdefault(json.loads(h["req_json"]).get("seed"), h["result"])
    res = list(by_seed.values())
    sc = [score_timing(r) for r in res]
    failed = any(x == float("inf") for x in sc)
    m = lambda v: st.mean(v) if v else None  # noqa: E731
    return [{"cfg": default_flow_cfg(), "design": design, "stage": stage, "seeds": sorted(by_seed, key=str),
             "n_seeds": len(res), "seed_scores": sc, "failed": failed, "score": None if failed else m(sc),
             "wns": m([r.wns for r in res if r.wns is not None]), "tns": m([r.tns for r in res if r.tns is not None]),
             "result": res[0], "rounds": [1]}]


def validate_v2(evaluator, db: ResultsDB, design: str, searched: Sequence[Dict[str, Any]], seeds: Sequence[int],
                stage: str = "route", tech: str = "nangate45") -> Dict[str, Any]:
    """Each arm-run's best flow config at `stage` (route) on held-out seeds; per-seed score_timing kept for pairing."""
    tool_id = getattr(evaluator, "tool_id", "")
    jobs, out = [], {}
    for s in searched:
        if not s.get("best_cfg"):
            out[s["arm"]] = {"error": "no successful config"}
            continue
        jobs += [(s["arm"], EvalRequest(design, tech, default_config(), sd, stage, tool_id, flow_cfg=s["best_cfg"]))
                 for sd in seeds]
    # seed-major: every config's run on seed k is started before any run on seed k+1, so an interrupted or partly
    # finished validation still has complete pairs
    order = {sd: i for i, sd in enumerate(seeds)}
    jobs.sort(key=lambda j: order[j[1].seed])
    names = {id(q): name for name, q in jobs}
    res = _eval_recording(evaluator, [q for _, q in jobs], lambda q, x: db.record_eval(
        names[id(q)] + "|validate", 1, q, x, "cache_hit" if x.provenance.get("cache_hit") else "", score_fn=score_timing))
    per: Dict[str, Dict[str, Any]] = {}
    for (name, q), x in zip(jobs, res):
        d = per.setdefault(name, {"cfg": q.flow_cfg, "by_seed": {}, "wns": [], "tns": [], "wl": [], "results": []})
        d["by_seed"][q.seed] = score_timing(x)
        d["results"].append(x)
        if x.ok:
            d["wns"] += [x.wns] if x.wns is not None else []
            d["tns"] += [x.tns] if x.tns is not None else []
            d["wl"] += [x.routed_wl] if x.routed_wl is not None else []
    for name, d in per.items():
        fin = [v for v in d["by_seed"].values() if v != float("inf")]
        out[name] = {"cfg": d["cfg"], "seed_scores": {str(k): (None if v == float("inf") else v)
                                                      for k, v in d["by_seed"].items()},
                     "validated_mean": st.mean(fin) if fin else None, "n_ok": len(fin), "n": len(d["by_seed"]),
                     "wns_mean": st.mean(d["wns"]) if d["wns"] else None,
                     "tns_mean": st.mean(d["tns"]) if d["tns"] else None,
                     "wl_mean": st.mean(d["wl"]) if d["wl"] else None, "_results": d["results"]}
    return out


def compare_v2(searched: Sequence[Dict[str, Any]], validation: Dict[str, Any]) -> Dict[str, Any]:
    """Per arm: paired (same held-out seed) score differences vs the default flow at the validation stage.
    diff < 0 means better than default. The unit of replication is the arm-run: each run's diff is its mean over the
    held-out seeds, and diff_2se / beats_default_2se use the SE over runs (seeds within a run are not independent
    replicates of the search). The pooled (arm-run, seed) SE is kept as diff_2se_pooled for reference. Also reports
    and whether the improvement clears 2 sigma of the default seed lottery (sigma = sd of the default's validated
    per-seed scores when it has >= 2, else of its search-stage seed lottery -- flagged by `sigma_source`)."""
    dname = next((s["arm"] for s in searched if s["arm"].startswith("v2_default#")), None)
    dval = validation.get(dname, {}) if dname else {}
    dseeds = {k: v for k, v in (dval.get("seed_scores") or {}).items() if v is not None}
    ref = {"wns": dval.get("wns_mean"), "tns": dval.get("tns_mean"), "wl": dval.get("wl_mean")}
    if len(dseeds) >= 2:
        sigma, src = st.stdev(dseeds.values()), "default_validation"
    else:
        dsearch = next((s for s in searched if s["arm"] == dname), {})
        sigma, src = dsearch.get("seed_lottery_sd"), "default_search_stage"
    arms: Dict[str, List[Dict[str, Any]]] = {}
    for s in searched:
        arms.setdefault(s["arm"].split("#")[0][len("v2_"):], []).append(s)
    out: Dict[str, Any] = {"default_arm_run": dname, "default_validated_mean": dval.get("validated_mean"),
                           "sigma_default": sigma, "sigma_source": src, "ref": ref, "arms": {}}
    for arm, runs in arms.items():
        diffs, run_diffs, vmeans, npass, nres = [], [], [], 0, 0
        for r in runs:
            v = validation.get(r["arm"], {})
            if v.get("validated_mean") is not None:
                vmeans.append(v["validated_mean"])
            rd = [sc - dseeds[sd] for sd, sc in (v.get("seed_scores") or {}).items() if sc is not None and sd in dseeds]
            diffs += rd
            if rd:
                run_diffs.append(st.mean(rd))
            if ref["wns"] is not None and ref["tns"] is not None:
                for x in v.get("_results", []):
                    nres += 1
                    npass += bool(passes(x, ref))
        row: Dict[str, Any] = {"runs": len(runs), "evals_used_mean": st.mean([r.get("evals_used", 0) for r in runs]),
                               "selection_mean": (st.mean([r["best_score"] for r in runs if r.get("best_score") is not None])
                                                  if any(r.get("best_score") is not None for r in runs) else None),
                               "validated_mean": st.mean(vmeans) if vmeans else None, "n_pairs": len(diffs),
                               "pass_rate_vs_default": (npass / nres) if nres else None,
                               "stopped_early": sum(1 for r in runs if r.get("stopped_early"))}
        if diffs:
            m = st.mean(run_diffs)
            se = st.stdev(run_diffs) / math.sqrt(len(run_diffs)) if len(run_diffs) > 1 else None
            sep = st.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else None
            row.update(diff_mean=m, diff_2se=2 * se if se is not None else None, run_diffs=run_diffs,
                       diff_2se_pooled=2 * sep if sep is not None else None,
                       beats_default_2se=(se is not None and m + 2 * se < 0),
                       clears_2sigma_lottery=(sigma is not None and -m > 2 * sigma))
        out["arms"][arm] = row
    return out


SELECT_ARM = "v2_select#r0"


def pool_candidates(db: ResultsDB, design: str, stage: str = "cts", top: int = 3,
                    min_seeds: int = 2) -> List[Dict[str, Any]]:
    """The `top` distinct flow configs with the best mean search-stage score over >= `min_seeds` seeds, pooled across
    every searching arm-run in `db` (v2_<arm>#r<k>, not the default, imports or validation rows). Ties and configs
    proposed by several arm-runs count once; each row keeps the arm-runs that found it."""
    arms = [r[0] for r in db.con.execute("SELECT DISTINCT arm FROM evals WHERE design=? AND stage=?", (design, stage))]
    arms = [a for a in arms if a.startswith("v2_") and "|" not in a and not a.startswith("v2_default#")
            and a != SELECT_ARM]
    best: Dict[str, Dict[str, Any]] = {}
    for a in sorted(arms):
        for c in config_stats(db, a, design, stage):
            if c["failed"] or c["n_seeds"] < min_seeds:
                continue
            cid = json.dumps(c["cfg"], sort_keys=True, default=str)
            if cid not in best or c["score"] < best[cid]["score"]:
                best[cid] = dict(c, found_by=sorted(set(best.get(cid, {}).get("found_by", [])) | {a}))
            else:
                best[cid]["found_by"] = sorted(set(best[cid]["found_by"]) | {a})
    rows = sorted(best.values(), key=lambda c: (c["score"], json.dumps(c["cfg"], sort_keys=True, default=str)))
    return [{"cfg": c["cfg"], "search_score": c["score"], "search_seeds": c["seeds"], "found_by": c["found_by"]}
            for c in rows[:top]]


def _eval_recording(evaluator, reqs: Sequence[EvalRequest], record: Callable[[EvalRequest, EvalResult], Any]):
    """Evaluate `reqs` concurrently (the evaluator's own semaphore caps real runs) and `record` each result in this
    thread the moment it finishes, so an interrupted round loses only the runs still in flight (a re-run replays the
    rest from the ResultsDB). Returns results in request order."""
    if not reqs:
        return []
    out: List[Any] = [None] * len(reqs)
    with ThreadPoolExecutor(len(reqs)) as ex:
        futs = {ex.submit(evaluator.evaluate, q): i for i, q in enumerate(reqs)}
        for f in as_completed(futs):
            i = futs[f]
            out[i] = f.result()
            record(reqs[i], out[i])
    return out


def route_select_v2(evaluator, db: ResultsDB, design: str, candidates: Sequence[Dict[str, Any]],
                    seeds: Sequence[int], stage: str = "route", tech: str = "nangate45",
                    min_gain: float = 0.0) -> Dict[str, Any]:
    """The final selection round: the pooled search-stage finalists AND the default flow, each run through the full
    flow (`stage` = route) on `seeds` (selection seeds: not the search seeds, not the held-out ones). The winner is
    the lowest mean routed score_timing; a finalist must beat the default's mean by more than `min_gain` (and finish
    on every seed), otherwise the answer is the default flow. So the search optimises the end of the flow, not the
    stage it searched at, and never hands back a config that routed worse than the default on the selection seeds."""
    tool_id = getattr(evaluator, "tool_id", "")
    dflt = default_flow_cfg()
    cands = [{"cfg": dflt, "is_default": True}] + [dict(c, is_default=False) for c in candidates
                                                     if c["cfg"] != dflt]
    reqs = [(i, EvalRequest(design, tech, default_config(), s, stage, tool_id, flow_cfg=c["cfg"]))
            for i, c in enumerate(cands) for s in seeds]
    rec = lambda q, x: db.record_eval(SELECT_ARM, 1, q, x, "select" + (",cache_hit" if x.provenance.get("cache_hit")  # noqa: E731
                                                                     else ""), score_fn=score_timing)
    res = _eval_recording(evaluator, [q for _, q in reqs], rec)
    per: Dict[int, Dict[str, Any]] = {}
    for (i, q), x in zip(reqs, res):
        d = per.setdefault(i, {"by_seed": {}, "wns": [], "tns": [], "wl": []})
        d["by_seed"][str(q.seed)] = score_timing(x)
        if x.ok:
            d["wns"] += [x.wns] if x.wns is not None else []
            d["tns"] += [x.tns] if x.tns is not None else []
            d["wl"] += [x.routed_wl] if x.routed_wl is not None else []
    m = lambda v: st.mean(v) if v else None  # noqa: E731
    rows = []
    for i, c in enumerate(cands):
        d = per.get(i, {"by_seed": {}, "wns": [], "tns": [], "wl": []})
        sc = list(d["by_seed"].values())
        ok = bool(sc) and all(v != float("inf") for v in sc)
        rows.append(dict(c, routed_scores={k: (None if v == float("inf") else v) for k, v in d["by_seed"].items()},
                         routed_mean=st.mean(sc) if ok else None, complete=ok,
                         wns_mean=m(d["wns"]), tns_mean=m(d["tns"]), wl_mean=m(d["wl"])))
    base = rows[0]["routed_mean"]
    ok_rows = [r for r in rows[1:] if r["complete"]]
    top = min(ok_rows, key=lambda r: r["routed_mean"]) if ok_rows else None
    beats = top is not None and (base is None or top["routed_mean"] < base - min_gain)
    winner = top if beats else rows[0]
    return {"seeds": list(seeds), "stage": stage, "min_gain": min_gain, "rows": rows, "default_mean": base,
            "winner_cfg": winner["cfg"], "winner_is_default": winner["is_default"],
            "winner_mean": winner["routed_mean"]}


def auto_min_gain(db: ResultsDB, design: str, stage: str, n_select: int) -> float:
    """Noise-aware threshold for route_select_v2: 2 standard errors of the difference of two n_select-seed means,
    using the default flow's seed-to-seed sd at the search stage (its seed lottery, measured before any selection):
    2 * sd * sqrt(2 / n_select). 0.0 if the default has fewer than 2 finished seeds there."""
    rows = config_stats(db, _v2_name("default", 0), design, stage)
    sc = [x for x in (rows[0]["seed_scores"] if rows else []) if x != float("inf")]
    if len(sc) < 2 or n_select < 1:
        return 0.0
    return 2 * st.stdev(sc) * math.sqrt(2 / n_select)


def run_final_round_v2(evaluator, db: ResultsDB, design: str, *, stage: str = "cts", top: int = 3,
                       select_seeds: Sequence[int] = (8, 9), select_stage: str = "route", min_gain: Any = 0.0,
                       validate_seeds: Sequence[int] = (101, 102), validate_stage: str = "route",
                       tech: str = "nangate45", min_seeds: int = 2) -> Dict[str, Any]:
    """Pool finalists from every arm-run already in `db`, pick at `select_stage` (route_select_v2), then report the
    pick against the default flow on held-out seeds (validate_v2 + compare_v2: arm "select" vs "default")."""
    def ev_for(name: str):
        inner = evaluator.for_arm(name) if hasattr(evaluator, "for_arm") else evaluator
        return CachedEvaluator(inner, db)

    min_gain_src = "fixed"
    if min_gain == "auto":
        min_gain, min_gain_src = auto_min_gain(db, design, stage, len(select_seeds)), "auto"
    cands = pool_candidates(db, design, stage, top, min_seeds)
    sel = route_select_v2(ev_for("select"), db, design, cands, select_seeds, select_stage, tech, min_gain)
    searched = [{"arm": "v2_default#r0", "best_cfg": default_flow_cfg(), "best_score": sel["default_mean"]},
                {"arm": SELECT_ARM, "best_cfg": sel["winner_cfg"], "best_score": sel["winner_mean"]}]
    validation = validate_v2(ev_for("validate"), db, design, searched, validate_seeds, validate_stage, tech)
    comparison = compare_v2(searched, validation)
    for v in validation.values():
        v.pop("_results", None)
    return {"design": design, "version": 2, "final_round": True, "candidates": cands, "selection": sel,
            "settings": {"stage": stage, "top": top, "select_seeds": list(select_seeds),
                         "select_stage": select_stage, "min_gain": min_gain, "min_gain_source": min_gain_src, "validate_seeds": list(validate_seeds),
                         "validate_stage": validate_stage},
            "validation": validation, "comparison": comparison}


def summarize_final_round(result: Dict[str, Any]) -> str:
    """Markdown: the finalists' search-stage vs routed scores, the pick, and the pick vs default on held-out seeds."""
    f = lambda x: "n/a" if x is None else f"{x:.2f}"  # noqa: E731
    s = result["selection"]
    lines = [f"Final round at {s['stage']} on seeds {s['seeds']} (lower is better):",
             "| finalist | found by | search score | routed mean | routed per seed | WNS | TNS | WL |",
             "|---|---|---|---|---|---|---|---|"]
    for r in s["rows"]:
        name = "default" if r["is_default"] else json.dumps({k: v for k, v in r["cfg"].items()
                                                             if v != default_flow_cfg().get(k)}, sort_keys=True)
        lines.append(f"| {name} | {','.join(r.get('found_by', [])) or '-'} | {f(r.get('search_score'))} | "
                     f"{f(r['routed_mean'])} | {r['routed_scores']} | {f(r['wns_mean'])} | {f(r['tns_mean'])} | "
                     f"{f(r['wl_mean'])} |")
    lines.append(f"pick: {'default flow' if s['winner_is_default'] else 'finalist'} "
                 f"(routed mean {f(s['winner_mean'])} vs default {f(s['default_mean'])})")
    c = result["comparison"]["arms"].get("select", {})
    v = result["validation"]
    lines.append(f"held-out {result['settings']['validate_seeds']}: pick {f(v.get(SELECT_ARM, {}).get('validated_mean'))}"
                 f" vs default {f(v.get('v2_default#r0', {}).get('validated_mean'))}; paired diff "
                 f"{f(c.get('diff_mean'))} (pooled 2SE {f(c.get('diff_2se_pooled'))})")
    return "\n".join(lines)


def run_experiment_v2(evaluator, db: ResultsDB, design: str, *, arms: Sequence[str] = ("default", "random", "tpe", "llm"),
                      reps: int = 3, budget_evals: int = 24, batch: int = 4, stage: str = "cts",
                      seeds: Sequence[int] = (0,), promote_top: int = 2, promote_seeds: Sequence[int] = (1,),
                      default_seeds: Optional[Sequence[int]] = None, validate_seeds: Sequence[int] = (101, 102),
                      validate_stage: str = "route", workers: int = 4, deadline: Optional[float] = None,
                      provider_factory: Optional[Callable[[], Any]] = None, validate: bool = True,
                      tech: str = "nangate45", log: Callable[[str], None] = print) -> Dict[str, Any]:
    """All v2 arm-runs (default once, every searching arm `reps` times, same budget/loop), then held-out validation of
    each arm-run's best config at `validate_stage` (validation always runs; `deadline` only gates search rounds)."""
    jobs = [(a, r) for a in arms for r in (range(1) if a == "default" else range(reps))]
    hits = {"n": 0}
    prior_box: Dict[str, Any] = {"rows": []}

    def ev_for(name: str):
        inner = evaluator.for_arm(name.replace("#", "-")) if hasattr(evaluator, "for_arm") else evaluator
        return CachedEvaluator(inner, db)

    def one(job):
        a, r = job
        name = _v2_name(a, r)
        log(f"start {name}")
        ev = ev_for(name)
        res = run_arm_v2(ev, db, a, r, design, budget_evals=budget_evals, batch=batch, stage=stage, seeds=seeds,
                         promote_top=promote_top, promote_seeds=promote_seeds, default_seeds=default_seeds,
                         deadline=deadline, provider_factory=provider_factory, tech=tech, log=log,
                         prior=prior_box["rows"] or None)
        hits["n"] += ev.hits
        log(f"done  {name} best={res.get('best_score')} evals={res.get('evals_used')} stop={res.get('stop_reason')}")
        return res

    # The default arm runs first: its result on the search seeds is every searching arm's incumbent (free history).
    first = [j for j in jobs if j[0] == "default"]
    rest = [j for j in jobs if j[0] != "default"]
    prior_box["rows"] = []
    searched = [one(j) for j in first]
    if first:
        prior_box["rows"] = incumbent_prior(db, design, stage, tuple(seeds) + tuple(promote_seeds))
        if not prior_box["rows"]:
            log("note: default arm has no evals on the search seeds; searching arms get no incumbent")
    with ThreadPoolExecutor(max(1, workers)) as ex:
        searched += list(ex.map(one, rest))
    validation: Dict[str, Any] = {}
    if validate:
        vev = ev_for("validate")
        validation = validate_v2(vev, db, design, searched, validate_seeds, validate_stage, tech)
        hits["n"] += vev.hits
    comparison = compare_v2(searched, validation)
    for v in validation.values():
        v.pop("_results", None)
    return {"design": design, "version": 2,
            "settings": {"arms": list(arms), "reps": reps, "budget_evals": budget_evals, "batch": batch, "stage": stage,
                         "seeds": list(seeds), "promote_top": promote_top, "promote_seeds": list(promote_seeds),
                         "default_seeds": list(default_seeds) if default_seeds is not None else None,
                         "validate_seeds": list(validate_seeds), "validate_stage": validate_stage},
            "search": searched, "validation": validation, "comparison": comparison, "cache_hits": hits["n"],
            "any_stopped_early": any(s.get("stopped_early") for s in searched)}


def summarize_v2(result: Dict[str, Any]) -> str:
    """Markdown: per arm, search-stage selection, held-out validated mean, paired diff vs default (lower = better)."""
    c = result["comparison"]

    def f(x, p=3):
        return "n/a" if x is None else (f"{x:.{p}f}" if isinstance(x, float) else str(x))

    lines = [f"Default seed-lottery sigma ({c['sigma_source']}): {f(c['sigma_default'])}; "
             f"default validated mean: {f(c['default_validated_mean'])}",
             "| arm | runs | evals/run | selection mean | validated mean | diff vs default | 2SE | beats (2SE) | "
             "clears 2sigma | pass rate | stopped early |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, r in c["arms"].items():
        lines.append(f"| {arm} | {r['runs']} | {f(r['evals_used_mean'], 1)} | {f(r['selection_mean'])} | "
                     f"{f(r['validated_mean'])} | {f(r.get('diff_mean'))} | {f(r.get('diff_2se'))} | "
                     f"{r.get('beats_default_2se', 'n/a')} | {r.get('clears_2sigma_lottery', 'n/a')} | "
                     f"{f(r['pass_rate_vs_default'], 2)} | {r['stopped_early']} |")
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
