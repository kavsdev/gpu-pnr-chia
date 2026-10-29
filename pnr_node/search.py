"""Two-tier search: cheap proxy screens every candidate, only the leaders pay for a route.

The arms built on this loop are listed in experiment.py.
"""
from __future__ import annotations

import json
import math
import random
import time
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

from .contract import (FLOW_KNOBS, KNOBS, Knob, EvalRequest, EvalResult, PromptConfig, Constraints, clamp_config,
                       default_config, score, score_timing, failure_signature)
from .db import ResultsDB
from .llm import LLMBudgetExceeded, LLMError, LLMProvider, Msg, extract_json


class Evaluator(Protocol):
    def evaluate(self, req: EvalRequest) -> EvalResult: ...


class RandomProposer:
    def __init__(self, seed: int = 0, knobs: Optional[Dict[str, Knob]] = None):
        self.rng = random.Random(seed)
        self.knobs = knobs or KNOBS  # the action space this arm samples (e.g. CORE_KNOBS for the baseline arm)

    def propose(self, history: List[Dict[str, Any]], n: int) -> List[Dict[str, float]]:
        out = []
        for _ in range(n):
            cfg = {}
            for k, kb in self.knobs.items():
                if kb.choices is not None:  # categorical: uniform over the choices
                    cfg[k] = kb.clamp(self.rng.choice(kb.choices))
                    continue
                if kb.log:
                    v = math.exp(self.rng.uniform(math.log(kb.lo), math.log(kb.hi)))
                else:
                    v = self.rng.uniform(kb.lo, kb.hi)
                cfg[k] = kb.clamp(v)
            out.append(cfg)
        return out


_SYSTEM = (
    "You tune the global-placement parameters of DREAMPlace for a chip design. Each candidate is "
    "evaluated by a cheap proxy (HPWL) and the best few by a full route that reports routed wirelength, "
    "worst negative slack (wns, ns, negative is bad), and DRC violations. Lower score is better. "
    "Propose new candidates that improve on the history, use what routing revealed about failures, and "
    "stay inside the knob ranges. Reply with JSON only."
)


_SYSTEM_TIMING = (
    "You tune flow-level knobs of a DREAMPlace-inside-OpenROAD (ORFS) placement flow for a chip design. Each "
    "candidate config is run through clock-tree synthesis (or further) on one or more placement seeds; the objective "
    "is timing: score = -TNS + 100 * -WNS (ns; lower is better; failed runs or DRC violations score +inf). Results "
    "move a few percent between seeds even for the same config, so prefer consistent improvements over single lucky "
    "runs. Propose new candidates that improve on the history, avoid regions that failed, and stay inside the knob "
    "ranges; categorical knobs must take one of the listed values exactly. Reply with JSON only."
)


def _knob_row(k: str, v: Knob) -> str:
    if v.choices is not None:
        return f"- {k}: one of {json.dumps(list(v.choices))} default {json.dumps(v.default)} (categorical)"
    return (f"- {k}: [{v.lo}, {v.hi}] default {v.default}{' (int)' if v.integer else ''}"
            f"{' (log scale)' if v.log else ''}")


def _knob_table(knobs: Optional[Dict[str, Knob]] = None) -> str:
    return "\n".join(_knob_row(k, v) for k, v in (knobs or KNOBS).items())


def _rv(v: Any) -> Any:
    """Round floats for a prompt; leave categoricals (str/bool/int) as given."""
    return round(v, 6) if isinstance(v, float) else v


class LLMProposer:
    def __init__(self, provider: LLMProvider, db: Optional[ResultsDB] = None, arm: str = "llm",
                 show_routed: bool = True, top_k: int = 12, fallback: Optional[RandomProposer] = None,
                 max_failed: int = 8, knobs: Optional[Dict[str, Knob]] = None, expanded_context: bool = False,
                 prompt_cfg: Optional[PromptConfig] = None, constraints: Optional[Constraints] = None,
                 objective: str = "ppa"):
        self.p, self.db, self.arm = provider, db, arm
        # objective="timing": v2 arms -- history rows are per-config seed means of score_timing at
        # the search stage (built by run_search_v2), rendered with wns/tns; the system prompt is _SYSTEM_TIMING.
        self.objective = objective
        self.show_routed, self.top_k, self.max_failed = show_routed, top_k, max_failed
        # `knobs`: the action space this proposer may tune (CORE_KNOBS for the baseline llm_grt arm, the full KNOBS
        # for llm_grt_expanded). `expanded_context`: also show the LLM tns/power/area and DREAMPlace's own
        # convergence (dp_iterations/dp_overflow) for routed candidates, not just wl/wns/drc.
        self.knobs = knobs or KNOBS
        self.expanded_context = expanded_context
        self.prompt_cfg = prompt_cfg or PromptConfig()
        self.constraints = constraints
        if self.constraints and self.constraints.allowed_knobs:
            # narrow KNOBS to the allowed subset
            self.knobs = {k: self.knobs[k] for k in self.constraints.allowed_knobs if k in self.knobs}

        self.fallback = fallback or RandomProposer(1, self.knobs)
        self.round = 0
        self.invalid = 0  # count of unusable LLM answers, reported in results

    def _history_text_timing(self, history: List[Dict[str, Any]]) -> str:
        good = sorted([h for h in history if not h.get("failed") and h.get("score") is not None],
                      key=lambda h: h["score"])[: self.top_k]
        bad = [h for h in history if h.get("failed")][-self.max_failed:]
        lines = []
        for h in good:
            row = {"cfg": {k: _rv(v) for k, v in h["cfg"].items() if k in self.knobs}, "score": round(h["score"], 3),
                   "wns": _rv(h.get("wns")), "tns": _rv(h.get("tns")), "n_seeds": h.get("n_seeds", 1)}
            if h.get("is_default"):
                row["note"] = "default flow"
            lines.append(json.dumps(row))
        text = "\n".join(lines) or "(no history yet)"
        if bad:
            text += ("\n\nFailed candidates (avoid these regions):\n"
                     + "\n".join(json.dumps({"cfg": {k: _rv(v) for k, v in h["cfg"].items() if k in self.knobs},
                                             "why": (h["result"].error or "DRC violations or no timing")[:140]})
                                 for h in bad))
        return text

    def _history_text(self, history: List[Dict[str, Any]]) -> str:
        if self.objective == "timing":
            return self._history_text_timing(history)
        # The proxy-only ablation must not learn anything from routing, not even through the ordering.
        key = (lambda h: h["score"]) if self.show_routed else (lambda h: h["result"].proxy_hpwl or float("inf"))
        good = [h for h in history if not h.get("failed")]
        bad = [h for h in history if h.get("failed") and (self.show_routed or h["stage"] == "proxy")][-self.max_failed:]

        if self.show_routed:  # measured routes first, then global-route estimates (different scale, listed separately)
            top_routed = sorted([h for h in good if h["stage"] == "route"], key=key)[: self.top_k]

            # Pareto-optimal routed results not already in the top-k (capped at 8 to bound the prompt)
            pareto_rows = []
            if self.db and top_routed:
                # history is built from db.evals(arm, design), so design is known in every row.
                design = history[0].get("design") if history else None
                if design:
                    p_results = self.db.pareto(self.arm, design)
                    seen_cfgs = {json.dumps(h["cfg"], sort_keys=True) for h in top_routed}
                    for p in p_results:
                        p_cfg_json = json.dumps(p["cfg"], sort_keys=True)
                        if p_cfg_json not in seen_cfgs:
                            if len(pareto_rows) < 8:
                                pareto_rows.append(p)

            rows = (top_routed
                    + [dict(p, is_pareto=True) for p in pareto_rows]
                    + sorted([h for h in good if h["stage"] == "grt"], key=key)[: max(self.top_k // 2, 1)]
                    + sorted([h for h in good if h["stage"] == "proxy"], key=key)[: 3])
        else:
            rows = sorted(good, key=key)[: self.top_k]

        lines = []
        for h in rows:
            r: EvalResult = h["result"]
            # Show only the knobs this arm can actually tune.
            cfg = {k: _rv(v) for k, v in h["cfg"].items() if k in self.knobs}
            if self.show_routed and r.stage == "route":
                row = {"cfg": cfg, "score": round(h["score"], 2), "routed_wl": r.routed_wl,
                       "wns": r.wns, "drc": r.drc_violations}
                if h.get("is_pareto"):
                    row["note"] = "Pareto-optimal (trade-off surface)"
                if self.expanded_context:
                    row["tns"] = r.tns
                    row["power"] = r.power
                    row["area"] = r.area
                    row["dp_iterations"] = r.provenance.get("dp_iterations")
                    row["dp_overflow"] = r.provenance.get("dp_overflow")
                lines.append(json.dumps(row))
            elif self.show_routed and r.stage == "grt":
                lines.append(json.dumps({"cfg": cfg, "grt_wl_estimate": r.grt_wl, "wns_at_grt": r.wns,
                                         "note": "global-route estimate; tracks final routed WL closely, not final"}))
            else:
                lines.append(json.dumps({"cfg": cfg, "proxy_hpwl": r.proxy_hpwl}))
        text = "\n".join(lines) or "(no history yet)"
        if bad:  # failures teach the feasible region; showing them avoids re-proposing the same dead ends
            text += ("\n\nRejected or failed candidates (they cost no route; avoid these regions):\n"
                     + "\n".join(json.dumps({"cfg": {k: _rv(v) for k, v in h["cfg"].items() if k in self.knobs},
                                             "why": h["result"].error[:140]}) for h in bad))
        return text

    def _recall_hints(self, history: List[Dict[str, Any]]) -> str:
        """Signature-keyed memory. Surface hints from past runs when current failures recur."""
        if not self.db:
            return ""
        design = None
        for h in history:
            if h.get("design"):
                design = h["design"]
                break
        if not design:
            return ""

        current_sigs = set()
        for h in history:
            sig = failure_signature(h["result"])
            if sig:
                current_sigs.add(sig)
        if not current_sigs:
            return ""

        all_past = self.db.all_evals(design)
        hints = []
        for sig in current_sigs:
            # Find past occurrences of this signature across any arm
            matches = [i for i, h in enumerate(all_past) if failure_signature(h["result"]) == sig]
            for m_idx in matches:
                m_h = all_past[m_idx]
                if not self.show_routed and m_h["stage"] == "route":
                    continue  # The proxy-only arm must not learn from routing failures
                m_score = score(m_h["result"])
                # Look for a later improvement in the same arm's run
                for f_idx in range(m_idx + 1, len(all_past)):
                    f_h = all_past[f_idx]
                    if f_h["arm"] != m_h["arm"]:
                        continue
                    if not self.show_routed and f_h["stage"] == "route":
                        continue  # The proxy-only arm must not learn from routing successes
                    f_score = score(f_h["result"])
                    if f_score < m_score * 0.95:  # Meaningful improvement
                        changes = []
                        for k in self.knobs:
                            v1, v2 = m_h["cfg"].get(k), f_h["cfg"].get(k)
                            if v1 is not None and v2 is not None and abs(v1 - v2) > 1e-9:
                                direction = "higher" if v2 > v1 else "lower"
                                changes.append(f"{k} -> {direction}")
                        if changes:
                            hints.append(f"Failures matching {sig} were previously improved by: {', '.join(changes)}")
                        break
        unique_hints = sorted(list(set(hints)))[:4]
        if unique_hints:
            return "\n\nHistorical Recall (patterns found in past runs):\n- " + "\n- ".join(unique_hints)
        return ""

    def propose(self, history: List[Dict[str, Any]], n: int, stage: str = "route") -> List[Dict[str, float]]:
        self.round += 1
        base_sys = _SYSTEM_TIMING if self.objective == "timing" else _SYSTEM
        system = self.prompt_cfg.system_prompts.get(stage, self.prompt_cfg.system_prompts.get("default", base_sys))

        # Fold the constraint summary into the system prompt so the LLM knows the priorities/budget
        if self.constraints:
            summary = f"\n\nConstraints (ID: {self.constraints.id}):"
            if self.constraints.target_period:
                summary += f"\n- Target clock period: {self.constraints.target_period}ns"
            if self.constraints.utilization_limit:
                summary += f"\n- Utilization limit: {self.constraints.utilization_limit}"
            if self.constraints.priority:
                summary += f"\n- Priorities: {', '.join(self.constraints.priority)}"
            system += summary

        hist_text = self._history_text(history)
        recall_text = self._recall_hints(history) if self.prompt_cfg.use_memory else ""

        default_template = (f"Knobs:\n{_knob_table(self.knobs)}\n\nBest results so far:\n{{history}}\n\n"
                            'Return {{"reasoning": "<2-4 sentences>", "candidates": [{{...knob values...}}, ...]}} '
                            f"with exactly {n} candidates.")
        template = self.prompt_cfg.templates.get(stage, self.prompt_cfg.templates.get("default", default_template))
        prompt = template.format(history=hist_text + recall_text)

        if self.prompt_cfg.feedback_mode == "two_call":
            critique_sys = self.prompt_cfg.critique_system_prompt or (
                "You are reviewing DREAMPlace tuning history; identify patterns in what improved or hurt results, "
                "and what the failures have in common."
            )
            critique_prompt = f"Analyze the following tuning history:\n\n{hist_text}\n\nWhat patterns do you see?"
            critique_resp = self.p.complete(critique_sys, [Msg("user", critique_prompt)], json_mode=False, max_tokens=1024)
            if self.db:
                self.db.record_llm(self.arm, self.round, critique_sys, critique_prompt, critique_resp, phase="critique")

            propose_prompt = f"Analysis of prior results: {critique_resp.text}\n\n{prompt}"
            resp = self.p.complete(system, [Msg("user", propose_prompt)], json_mode=True, max_tokens=4096)
            if self.db:
                self.db.record_llm(self.arm, self.round, system, propose_prompt, resp, phase="propose")
        else:
            resp = self.p.complete(system, [Msg("user", prompt)], json_mode=True, max_tokens=4096)
            if self.db:
                self.db.record_llm(self.arm, self.round, system, prompt, resp, phase="propose")

        cands: List[Dict[str, float]] = []
        try:
            data = extract_json(resp.text)
            raw = data["candidates"] if isinstance(data, dict) else data
            cands = [clamp_config(c, self.knobs) for c in raw if isinstance(c, dict)]
        except Exception:
            self.invalid += 1
        if len(cands) < n:  # never let a bad answer stall the budgeted search
            self.invalid += 1 if cands else 0
            cands += self.fallback.propose(history, n - len(cands))
        return cands[:n]


def _history(db: ResultsDB, arm: str, design: str) -> List[Dict[str, Any]]:
    """What a proposer may learn from: every routed result plus proxy-only candidates never routed."""
    routed = [h for h in db.evals(arm, design, "route") if h["score"] is not None]
    seen = [h["cfg"] for h in routed]
    grt = [h for h in db.evals(arm, design, "grt") if h["score"] is not None and h["cfg"] not in seen]
    seen += [h["cfg"] for h in grt]
    proxy = [h for h in db.evals(arm, design, "proxy")
             if h["result"].ok and h["score"] is not None and h["cfg"] not in seen]
    # Failed / rejected candidates. The proxy-only arm sees only proxy-tier failures (DREAMPlace itself), never
    # anything that came from a route.
    failed = [dict(h, failed=True, score=None) for h in db.evals(arm, design)
              if not h["result"].ok]
    return routed + grt + proxy + failed


def _eval_many(evaluator, reqs: List[EvalRequest]) -> List[EvalResult]:
    """Use the evaluator's batch path (parallel on a cluster) when it has one."""
    if hasattr(evaluator, "evaluate_many"):
        return list(evaluator.evaluate_many(reqs))
    return [evaluator.evaluate(r) for r in reqs]


def _note(res: EvalResult) -> str:
    return "cache_hit" if res.provenance.get("cache_hit") else ""


def run_search(evaluator: Evaluator, proposer, db: ResultsDB, arm: str, design: str, *, rounds: int,
               batch: int, route_top: int, seed: int = 0, tech: str = "nangate45",
               resume: bool = True, grt_top: Optional[int] = None, deadline: Optional[float] = None,
               constraints: Optional[Constraints] = None) -> Dict[str, Any]:
    """Fixed-budget loop. New detailed-route budget = rounds * route_top. `rounds` counts additional rounds, so a
    run killed on a preemptible worker resumes from the DB (history and round numbers) instead of restarting.

    Funnel:
      grt_top is None   DREAMPlace proxy screens every candidate; the proxy-best `route_top` are routed. This is the
                        original design; on gcd the proxy is anti-correlated with routed quality (Spearman -0.29).
      grt_top = k       DREAMPlace only checks feasibility (non-converged candidates are rejected free); up to k
                        feasible candidates go through global route; the best `route_top` by the global-route
                        wirelength estimate (Spearman +0.99 vs final) pay for detailed route.

    `deadline` (a `time.time()`-comparable wall-clock cutoff, unset by default): checked only between rounds, never
    mid-route, so a route already dispatched always finishes and is never wasted.
    """
    tool_id = getattr(evaluator, "tool_id", "")
    constraints_id = constraints.id if constraints else ""
    history = _history(db, arm, design)
    start = db.max_round(arm, design) if resume else 0
    stopped_early = False
    for rnd in range(start + 1, start + rounds + 1):
        if deadline is not None and time.time() >= deadline:
            stopped_early = True
            break
        # Proposer can optionally take a stage hint; fallback to 2-arg if it doesn't.
        try:
            cfgs = proposer.propose(history, batch, stage="proxy")
        except TypeError:
            cfgs = proposer.propose(history, batch)

        reqs = [EvalRequest(design, tech, cfg, seed, "proxy", tool_id, constraints_id) for cfg in cfgs]
        proxied = []
        for req, res in zip(reqs, _eval_many(evaluator, reqs)):
            db.record_eval(arm, rnd, req, res, _note(res))
            proxied.append((res.proxy_hpwl if res.ok else float("inf"), req.placer_cfg))
        feasible = [(hp, cfg) for hp, cfg in proxied if hp != float("inf")]  # failed/rejected never pay for a route
        if grt_top is None:
            chosen = [cfg for _, cfg in sorted(feasible, key=lambda t: t[0])[:route_top]]
        else:
            greqs = [EvalRequest(design, tech, cfg, seed, "grt", tool_id, constraints_id) for _, cfg in feasible[:grt_top]]
            ranked = []
            for req, res in zip(greqs, _eval_many(evaluator, greqs)):
                db.record_eval(arm, rnd, req, res, _note(res))
                if res.ok and res.grt_wl is not None:
                    ranked.append((res.grt_wl, req.placer_cfg))
            chosen = [cfg for _, cfg in sorted(ranked, key=lambda t: t[0])[:route_top]]
        rreqs = [EvalRequest(design, tech, cfg, seed, "route", tool_id, constraints_id) for cfg in chosen]
        for req, res in zip(rreqs, _eval_many(evaluator, rreqs)):
            db.record_eval(arm, rnd, req, res, _note(res))
        history = _history(db, arm, design)
    best = db.best_routed(arm, design)
    return {"arm": arm, "design": design, "routes_used": db.n_routes(arm, design),
            "routes_executed": db.n_routes(arm, design, executed_only=True),
            "grt_runs": db.n_stage(arm, design, "grt"),
            "best_score": best["score"] if best else None, "best_cfg": best["cfg"] if best else None,
            "stopped_early": stopped_early}


def run_default(evaluator: Evaluator, db: ResultsDB, design: str, arm: str = "default", seed: int = 0,
                tech: str = "nangate45", constraints: Optional[Constraints] = None) -> Dict[str, Any]:
    req = EvalRequest(design, tech, default_config(), seed, "route", getattr(evaluator, "tool_id", ""),
                      constraints.id if constraints else "")
    res = evaluator.evaluate(req)
    s = db.record_eval(arm, 0, req, res, _note(res))
    return {"arm": arm, "design": design, "routes_used": 1, "best_score": s, "best_cfg": req.placer_cfg}


# ---------------------------------------------------------------------------------------------------------------------
# v2: flow-level knobs, timing-first objective, budget in evaluations, multi-seed promotion.
# ---------------------------------------------------------------------------------------------------------------------

class TPEProposer:
    """Small Tree-structured Parzen Estimator (no optuna/scipy; stdlib only, deterministic given `seed`).

    Observations = history rows with a `cfg` and a `score` (lower is better; failed rows count as +inf, i.e. "bad").
    Split at gamma=0.25 into good/bad; per numeric knob a 1-D Gaussian Parzen mixture in (log-)space plus a uniform
    prior component; per categorical knob the choice frequencies with prior +1. `n_candidates` samples are drawn
    from the good model and the one maximising l(x)/g(x) is proposed. The first `n_startup` proposals are random.
    """

    def __init__(self, seed: int = 0, knobs: Optional[Dict[str, Knob]] = None, gamma: float = 0.25,
                 n_candidates: int = 24, n_startup: int = 6):
        self.rng = random.Random(seed)
        self.knobs = knobs or FLOW_KNOBS
        self.gamma, self.n_candidates, self.n_startup = gamma, n_candidates, n_startup
        self.random = RandomProposer(seed + 1, self.knobs)
        self.n_proposed = 0

    # --- numeric helpers: work in [0, 1] of the (log-)range
    @staticmethod
    def _to_u(kb: Knob, v: float) -> float:
        lo, hi, v = (math.log(kb.lo), math.log(kb.hi), math.log(max(float(v), kb.lo))) if kb.log else \
            (kb.lo, kb.hi, float(v))
        return min(max((v - lo) / (hi - lo), 0.0), 1.0) if hi > lo else 0.5

    @staticmethod
    def _from_u(kb: Knob, u: float) -> Any:
        u = min(max(u, 0.0), 1.0)
        v = math.exp(math.log(kb.lo) + u * (math.log(kb.hi) - math.log(kb.lo))) if kb.log else kb.lo + u * (kb.hi - kb.lo)
        return kb.clamp(v)

    @staticmethod
    def _bw(n: int) -> float:
        return max(0.02, 0.25 * max(n, 1) ** (-0.2))

    @classmethod
    def _logpdf_num(cls, u: float, pts: List[float]) -> float:
        bw = cls._bw(len(pts))
        w = 1.0 / (len(pts) + 1)  # uniform prior component on [0, 1] has density 1
        dens = w * 1.0 + sum(w * math.exp(-0.5 * ((u - p) / bw) ** 2) / (bw * math.sqrt(2 * math.pi)) for p in pts)
        return math.log(max(dens, 1e-300))

    @staticmethod
    def _cat_probs(kb: Knob, vals: List[Any]) -> List[float]:
        counts = [1.0 + sum(1 for v in vals if kb.clamp(v) == c and type(kb.clamp(v)) is type(c)) for c in kb.choices]
        tot = sum(counts)
        return [c / tot for c in counts]

    def _split(self, history: List[Dict[str, Any]]):
        obs = []
        for h in history:
            if "cfg" not in h:
                continue
            sc = h.get("score")
            sc = float("inf") if (sc is None or h.get("failed")) else float(sc)
            obs.append((sc, json.dumps(h["cfg"], sort_keys=True, default=str), h["cfg"]))
        obs.sort(key=lambda t: (t[0], t[1]))  # deterministic order regardless of history ordering
        finite = [o for o in obs if o[0] != float("inf")]
        if not finite:
            return [], [o[2] for o in obs]
        n_good = max(1, int(math.ceil(self.gamma * len(obs))))
        n_good = min(n_good, len(finite))
        return [o[2] for o in obs[:n_good]], [o[2] for o in obs[n_good:]]

    def _sample_good(self, good: List[Dict[str, Any]]) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        for k, kb in self.knobs.items():
            vals = [g[k] for g in good if k in g]
            if kb.choices is not None:
                probs = self._cat_probs(kb, vals)
                cfg[k] = self.rng.choices(list(kb.choices), weights=probs, k=1)[0]
            else:
                pts = [self._to_u(kb, v) for v in vals]
                # mixture sample: pick the prior with weight 1/(n+1), otherwise a kernel centred on a good point
                if not pts or self.rng.random() < 1.0 / (len(pts) + 1):
                    u = self.rng.random()
                else:
                    u = self.rng.gauss(self.rng.choice(pts), self._bw(len(pts)))
                cfg[k] = self._from_u(kb, u)
        return cfg

    def _log_ratio(self, cfg: Dict[str, Any], good, bad) -> float:
        t = 0.0
        for k, kb in self.knobs.items():
            gv, bv = [g[k] for g in good if k in g], [b[k] for b in bad if k in b]
            if kb.choices is not None:
                i = list(kb.choices).index(cfg[k])
                t += math.log(self._cat_probs(kb, gv)[i]) - math.log(self._cat_probs(kb, bv)[i])
            else:
                u = self._to_u(kb, cfg[k])
                t += (self._logpdf_num(u, [self._to_u(kb, v) for v in gv])
                      - self._logpdf_num(u, [self._to_u(kb, v) for v in bv]))
        return t

    def propose(self, history: List[Dict[str, Any]], n: int, stage: str = "cts") -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        good, bad = self._split(history)
        for _ in range(n):
            if self.n_proposed < self.n_startup or not good:
                out.append(self.random.propose(history, 1)[0])
            else:
                cands = [self._sample_good(good) for _ in range(self.n_candidates)]
                best = max(cands, key=lambda c: self._log_ratio(c, good, bad))
                out.append(clamp_config(best, self.knobs))
            self.n_proposed += 1
        return out


def _cfg_id(cfg: Dict[str, Any]) -> str:
    return json.dumps(cfg, sort_keys=True, default=str)


def _mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def config_stats(db: ResultsDB, arm: str, design: str, stage: str) -> List[Dict[str, Any]]:
    """Per-config aggregate of an arm-run's evaluations at `stage` (one row per distinct flow_cfg, first-seen order).

    `score` = mean score_timing over its seeds; +inf (reported as failed, score None) if any seed failed. Repeated
    evaluations of the same (config, seed) -- cache hits -- count once."""
    groups: Dict[str, Dict[str, Any]] = {}
    for h in db.evals(arm, design, stage):
        cid = _cfg_id(h["flow_cfg"])
        g = groups.setdefault(cid, {"cfg": h["flow_cfg"], "by_seed": {}, "rounds": set(), "result": h["result"]})
        seed = json.loads(h["req_json"]).get("seed")
        g["by_seed"].setdefault(seed, h["result"])
        g["rounds"].add(h["round"])
    out = []
    for g in groups.values():
        res = list(g["by_seed"].values())
        sc = [score_timing(r) for r in res]
        failed = any(x == float("inf") for x in sc)
        bad = next((r for r in res if score_timing(r) == float("inf")), None)
        out.append({"cfg": g["cfg"], "design": design, "stage": stage, "seeds": sorted(g["by_seed"], key=str),
                    "n_seeds": len(res), "seed_scores": sc, "failed": failed,
                    "score": None if failed else _mean(sc),
                    "wns": _mean([r.wns for r in res if r.wns is not None]),
                    "tns": _mean([r.tns for r in res if r.tns is not None]),
                    "result": bad or g["result"], "rounds": sorted(g["rounds"])})
    return out


def run_search_v2(evaluator: Evaluator, proposer, db: ResultsDB, arm: str, design: str, *, budget_evals: int,
                  batch: int, stage: str = "cts", seeds: Sequence[int] = (0,), promote_top: int = 2,
                  promote_seeds: Sequence[int] = (1,), deadline: Optional[float] = None, tech: str = "nangate45",
                  resume: bool = True, log: Optional[Callable[[str], None]] = None,
                  prior: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """v2 search loop. Each round: propose `batch` flow configs, evaluate each on `seeds` at `stage`, then
    re-evaluate the round's top-`promote_top` (by mean score_timing) on `promote_seeds`. Budget = evaluations (every
    requested eval counts: search, promotion, failures, cache hits); a round is truncated to fit what is left.
    `deadline` is checked only between rounds and the first round always runs, so an arm-run never ends with 0
    completed evals because of the deadline. Best config = best mean over >= 2 seeds (single-seed fallback flagged).
    An LLM proposer whose provider hits its cost cap (LLMBudgetExceeded) ends the arm-run.

    `prior`: config_stats rows measured outside this arm-run (the default flow on the search seeds, the incumbent).
    They are shown to the proposer as free history (flagged `is_default`) and never re-evaluated: a proposed config
    already in the history is replaced by a random one. They are not candidates for this arm-run's best config."""
    tool_id = getattr(evaluator, "tool_id", "")
    seeds, promote_seeds = tuple(seeds), tuple(promote_seeds)
    rnd = db.max_round(arm, design) if resume else 0
    used = db.n_stage(arm, design, stage) if resume else 0
    stop_reason, rounds_run, llm_errors = "budget", 0, 0
    say = log or (lambda _m: None)

    def run(reqs: List[EvalRequest], note: str) -> List[EvalResult]:
        res = _eval_many(evaluator, reqs)
        for q, x in zip(reqs, res):
            db.record_eval(arm, rnd, q, x, ",".join(filter(None, (note, _note(x)))), score_fn=score_timing)
        return res

    while used < budget_evals:
        if rounds_run > 0 and deadline is not None and time.time() >= deadline:
            stop_reason = "deadline"
            break
        remaining = budget_evals - used
        n = max(1, min(batch, remaining // max(len(seeds), 1)))
        own = config_stats(db, arm, design, stage)
        history = [dict(h, is_default=True) for h in (prior or [])] + own
        try:
            try:
                cfgs = proposer.propose(history, n, stage=stage)
            except TypeError:
                cfgs = proposer.propose(history, n)
        except LLMBudgetExceeded as e:
            stop_reason = "llm_budget"
            say(f"{arm}: LLM budget reached, ending arm-run ({e})")
            break
        except LLMError as e:  # transient provider failure: keep the budgeted loop going, but report it
            llm_errors += 1
            say(f"{arm}: LLM error, random fallback this round ({e})")
            fb = getattr(proposer, "fallback", None) or RandomProposer(rnd, FLOW_KNOBS)
            cfgs = fb.propose(history, n)
        rnd += 1
        rounds_run += 1
        cfgs = [clamp_config(c, FLOW_KNOBS) for c in cfgs]
        seen = {_cfg_id(h["cfg"]) for h in history}
        dedup = RandomProposer(10_000 + rnd, FLOW_KNOBS)
        for i, c in enumerate(cfgs):
            while _cfg_id(c) in seen:
                c = clamp_config(dedup.propose(history, 1)[0], FLOW_KNOBS)
            cfgs[i] = c
            seen.add(_cfg_id(c))
        reqs = [EvalRequest(design, tech, default_config(), s, stage, tool_id, flow_cfg=c)
                for c in cfgs for s in seeds][:remaining]
        res = run(reqs, "search")
        used += len(reqs)
        # promotion: this round's top-k configs (finite mean over the search seeds) on the promotion seeds
        per: Dict[str, List[float]] = {}
        cfg_of: Dict[str, Dict[str, Any]] = {}
        for q, x in zip(reqs, res):
            per.setdefault(_cfg_id(q.flow_cfg), []).append(score_timing(x))
            cfg_of[_cfg_id(q.flow_cfg)] = q.flow_cfg
        ranked = sorted(((sum(v) / len(v), cid) for cid, v in per.items() if all(s != float("inf") for s in v)))
        top = [cfg_of[cid] for _, cid in ranked[:promote_top]]
        if top and promote_seeds and used < budget_evals:
            preqs = [EvalRequest(design, tech, default_config(), s, stage, tool_id, flow_cfg=c)
                     for c in top for s in promote_seeds][: budget_evals - used]
            run(preqs, "promote")
            used += len(preqs)
        say(f"{arm}: round {rnd} done, {used}/{budget_evals} evals")
    stats = config_stats(db, arm, design, stage)
    multi = [c for c in stats if not c["failed"] and c["n_seeds"] >= 2]
    pool = multi or [c for c in stats if not c["failed"]]
    best = min(pool, key=lambda c: (c["score"], _cfg_id(c["cfg"]))) if pool else None
    return {"arm": arm, "design": design, "stage": stage, "evals_used": used, "rounds": rounds_run,
            "configs": len(stats), "failed_configs": sum(1 for c in stats if c["failed"]),
            "best_cfg": best["cfg"] if best else None, "best_score": best["score"] if best else None,
            "best_n_seeds": best["n_seeds"] if best else 0, "best_seed_scores": best["seed_scores"] if best else None,
            "best_single_seed_fallback": bool(best) and not multi,
            "stopped_early": stop_reason not in ("budget",), "stop_reason": stop_reason, "llm_errors": llm_errors,
            "invalid_llm_answers": getattr(proposer, "invalid", None),
            "incumbent_score": prior[0]["score"] if prior else None}
