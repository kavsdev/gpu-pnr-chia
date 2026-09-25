"""Two-tier search: cheap proxy screens every candidate, only the leaders pay for a route.

The arms built on this loop are listed in experiment.py.
"""
from __future__ import annotations

import json
import math
import random
import time
from typing import Any, Dict, List, Optional, Protocol

from .contract import KNOBS, Knob, EvalRequest, EvalResult, PromptConfig, Constraints, clamp_config, default_config, score, failure_signature
from .db import ResultsDB
from .llm import LLMProvider, Msg, extract_json


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


def _knob_table(knobs: Optional[Dict[str, Knob]] = None) -> str:
    return "\n".join(f"- {k}: [{v.lo}, {v.hi}] default {v.default}{' (int)' if v.integer else ''}"
                     f"{' (log scale)' if v.log else ''}" for k, v in (knobs or KNOBS).items())


class LLMProposer:
    def __init__(self, provider: LLMProvider, db: Optional[ResultsDB] = None, arm: str = "llm",
                 show_routed: bool = True, top_k: int = 12, fallback: Optional[RandomProposer] = None,
                 max_failed: int = 8, knobs: Optional[Dict[str, Knob]] = None, expanded_context: bool = False,
                 prompt_cfg: Optional[PromptConfig] = None, constraints: Optional[Constraints] = None):
        self.p, self.db, self.arm = provider, db, arm
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

    def _history_text(self, history: List[Dict[str, Any]]) -> str:
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
            cfg = {k: round(v, 6) for k, v in h["cfg"].items() if k in self.knobs}
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
                     + "\n".join(json.dumps({"cfg": {k: round(v, 6) for k, v in h["cfg"].items() if k in self.knobs},
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
        system = self.prompt_cfg.system_prompts.get(stage, self.prompt_cfg.system_prompts.get("default", _SYSTEM))

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
