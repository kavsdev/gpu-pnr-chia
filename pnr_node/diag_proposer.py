"""`llm_diag` arm: an LLM proposer that sees per-eval *evidence* and states a hypothesis.

Same budget and loop as the other v2 arms (run_search_v2); only the prompt differs from `LLMProposer(objective=
"timing")`: besides scores, the model sees for the best / worst configs (and the default-flow reference) what each
run showed -- per-stage setup WNS/TNS, the rebuffer trace (GP overflow, kept-repair buffers, STA after each repair),
ORFS resizer activity, worst setup paths, congestion, stage wall time, high-fanout nets (pnr_node/diag.py) -- and
must answer with a hypothesis and an expected effect per proposal. Hypotheses are logged to JSONL for the write-up.

History format (what run_search_v2 passes = search.config_stats rows, one per distinct flow config):
    {"cfg": flow_cfg dict, "design": str, "stage": str, "seeds": [...], "n_seeds": int, "seed_scores": [...],
     "failed": bool, "score": float | None (mean score_timing; None if any seed failed), "wns": float, "tns": float,
     "result": EvalResult (a failing seed's result if any, else the first), "rounds": [...]}
An optional "evidence" dict per row is used as-is; otherwise it is built from row["result"] with diag.evidence and the
run dir <work>/<design>/<provenance.req_key> (diag.run_dir_for), cached per req_key.

Registration: `register_llm_diag(log_path=...)` puts the arm into experiment.PROPOSERS_V2 (run_arms.py calls it).
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence

from . import diag
from .contract import FLOW_KNOBS, Knob, EvalResult, clamp_config, default_config, score_timing
from .llm import Msg, extract_json
from .search import RandomProposer

# What each FLOW_KNOB physically does in this flow (raw_handoff.flow_kwargs / RawHandoffEvaluator._place_rebuf).
KNOB_DOC: Dict[str, str] = {
    "target_density": "DREAMPlace target bin density for every global-placement (GP) pass. Lower spreads cells: more "
                      "whitespace for repair buffers/upsizing and less detailed-route work, but longer nets on some "
                      "paths. Ariane133 screen (seed 0, routed TNS): 0.33 -511, 0.45 -295, 0.5 -384, 0.6 -313, 0.8 -335.",
    "rebuf_stops": "GP overflow values at which placement pauses for a KEPT ORFS repair_design (buffering/resizing on "
                   "the partial placement, like RePlAce's in-GP repair at ~0.6); after the last stop a warm-started GP "
                   "runs to overflow 0.1. After each kept repair, OpenSTA times the repaired netlist and the next GP "
                   "gets net weights from it. More stops = more repair+reweight rounds (more runtime); an early stop "
                   "(0.7) repairs on a coarse placement, a late one (0.2/0.4) on a near-final one.",
    "netwt_pct": "Percent of non-clock nets (worst slack first) that get a net weight > 1 in each reweight "
                 "(RePlAce gpl timing-driven scheme; its default is 10).",
    "netwt_max": "Weight of the single worst-slack net; weights fall linearly with slack to 1.0 at the netwt_pct-th "
                 "worst net. Higher pulls critical nets shorter at the expense of other wirelength/density.",
    "rebuf_full": "Also run rsz fully_rebuffer after each kept repair_design (as RePlAce's kept timing-driven repair "
                  "does): rebuilds buffer trees for timing, not just for slew/cap/fanout. Costs runtime.",
    "dp_routability": "DREAMPlace routability mode: RUDY + pin-utilization cell inflation with NanGate45 track "
                      "capacities. Spreads congested regions; can fail to converge at low density (0.33 + routability "
                      "ended at overflow 0.21 and was rejected).",
    "density_weight": "DREAMPlace initial density-penalty weight versus wirelength. Higher spreads earlier (fewer "
                      "iterations, more wirelength); lower is wirelength-driven, more iterations, risk of hitting the "
                      "iteration cap before the overflow target.",
    "gamma": "Wirelength-model smoothing (weighted-average/LSE gamma, scaled by bin size). Larger = smoother but less "
             "accurate wirelength gradient; smaller = sharper, noisier optimisation.",
    "learning_rate": "Nesterov step size of DREAMPlace GP. Too large oscillates/diverges (rejected run); too small "
                     "converges slowly and may stop at the iteration cap.",
}

FLOW_DOC = (
    "Flow per eval: ORFS synth/floorplan (fixed) -> DREAMPlace GP, paused at each rebuf_stops overflow for a kept "
    "repair_design (+fully_rebuffer) and an OpenSTA reweight, then a final GP to overflow 0.1 -> import into ORFS 3_3 "
    "-> 3_4 repair_design -> 3_5 detailed placement -> 4_1 CTS + repair_timing (the search stage; its setup WNS/TNS "
    "ranks final routed TNS with Spearman 0.86) [-> 5_1 global route -> 5_2 detailed route]. Fixed: net weights "
    "static gpl scheme, nets >2000 loads pre-split with BUF_X4 before repair 1 (speed only), final stop_overflow 0.1."
)

_SYSTEM_DIAG = (
    "You are a physical-design engineer diagnosing and tuning a DREAMPlace-inside-OpenROAD (ORFS) placement flow. "
    "For every evaluated config you get the score AND the evidence of what happened in the run (per-stage timing, "
    "repair activity, convergence, worst paths, congestion, runtime). Objective: score = -TNS + 100 * -WNS (ns) at "
    "the search stage; lower is better; failed runs or DRC violations are +inf. Seed noise is a few percent, so only "
    "trust consistent effects. First diagnose what limits timing in the evidence (e.g. missing or wasted buffering, "
    "repairs on a too-coarse placement, a dominating high-fanout net, density/congestion, non-converged GP, long "
    "wires vs deep logic on the worst paths), then propose configs that test a specific hypothesis each. Stay inside "
    "the knob ranges; categorical knobs must take one listed value exactly. Reply with JSON only."
)


def knob_table(knobs: Dict[str, Knob]) -> str:
    rows = []
    for k, v in knobs.items():
        if v.choices is not None:
            rng = "one of " + json.dumps(list(v.choices))
        else:
            rng = f"[{v.lo:g}, {v.hi:g}]" + (" log-scale" if v.log else "")
        rows.append(f"- {k}: {rng}, default {json.dumps(v.default)}. {KNOB_DOC.get(k, '')}".rstrip())
    return "\n".join(rows)


def _norm(v: Any) -> Any:
    return float(f"{v:.6g}") if isinstance(v, float) else v


def cfg_key(cfg: Dict[str, Any], knobs: Dict[str, Knob] = FLOW_KNOBS) -> str:
    c = clamp_config(cfg or {}, knobs)
    return json.dumps({k: _norm(v) for k, v in c.items()}, sort_keys=True)


def cfg_diff(cfg: Dict[str, Any], knobs: Dict[str, Knob] = FLOW_KNOBS) -> str:
    """Non-default knobs only, e.g. 'target_density=0.6 rebuf_full=false' ('default flow' if none)."""
    c, d = clamp_config(cfg or {}, knobs), default_config(knobs)
    parts = [f"{k}={json.dumps(_norm(v))}" for k, v in c.items() if _norm(v) != _norm(d[k])]
    return " ".join(parts) or "default flow"


def _fmt_score(s: Optional[float]) -> str:
    return "FAILED" if s is None or (isinstance(s, float) and math.isinf(s)) else f"{s:.2f}"


class DiagProposer:
    """propose(history, n, stage="cts") -> n clamped FLOW_KNOBS configs (LLM, deduped, random fill)."""

    def __init__(self, provider, knobs: Optional[Dict[str, Knob]] = None, design: Optional[str] = None,
                 ref: Optional[Dict[str, Any]] = None, db=None, arm: str = "llm_diag",
                 fallback: Optional[RandomProposer] = None, log_path: Optional[str] = None,
                 work_root: Optional[str] = None, n_best: int = 2, n_worst: int = 1, n_hyp: int = 8,
                 max_prompt_tokens: int = 7500, evidence_fn=None):
        self.p, self.knobs, self.design, self.ref = provider, knobs or FLOW_KNOBS, design, ref
        self.db, self.arm, self.log_path, self.work_root = db, arm, log_path, work_root
        self.fallback = fallback or RandomProposer(17, self.knobs)
        self.n_best, self.n_worst, self.n_hyp, self.max_prompt_tokens = n_best, n_worst, n_hyp, max_prompt_tokens
        self.evidence_fn = evidence_fn  # (result, design) -> evidence dict; default diag.evidence on the run dir
        self.round = 0
        self.invalid = 0  # unusable / short LLM answers (reported by run_search_v2)
        self.hypotheses: List[Dict[str, Any]] = []  # this arm-run's proposals with hypotheses (for later prompts)
        self._ev_cache: Dict[str, Dict[str, Any]] = {}

    # ---------------------------------------------------------------- evidence
    def _evidence(self, h: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(h.get("evidence"), dict):
            return h["evidence"]
        res = h.get("result")
        if res is None:
            return {}
        d = diag._as_dict(res)
        key = (d.get("provenance") or {}).get("req_key") or json.dumps(d, sort_keys=True, default=str)[:512]
        if key not in self._ev_cache:
            try:
                if self.evidence_fn:
                    ev = self.evidence_fn(res, h.get("design") or self.design)
                else:
                    rd = diag.run_dir_for(d, h.get("design") or self.design, self.work_root)
                    ev = diag.evidence(d, rd)
            except Exception as e:  # never let evidence break a budgeted round
                ev = {"_parse_errors": [type(e).__name__]}
            self._ev_cache[key] = ev
        return self._ev_cache[key]

    def _default_ref(self, design: Optional[str], stage: str) -> Optional[Dict[str, Any]]:
        """The default flow at `stage`: self.ref if given, else the v2 default arm's rows in the DB (mean wns/tns,
        evidence of its first ok seed). None until the default arm has results."""
        if self.ref is not None:
            return self.ref
        if self.db is None or not design:
            return None
        try:
            dflt = cfg_key({}, self.knobs)
            rows = [r for r in self.db.all_evals(design) if str(r.get("arm", "")).startswith("v2_default")
                    and r.get("stage") == stage and cfg_key(r.get("flow_cfg") or {}, self.knobs) == dflt]
        except Exception:
            return None
        ok = [r for r in rows if score_timing(r["result"]) != float("inf")]
        if not ok:
            return None
        ws = [r["result"].wns for r in ok]
        ts = [r["result"].tns for r in ok]
        return {"wns": sum(ws) / len(ws), "tns": sum(ts) / len(ts), "n": len(ok),
                "seed_scores": [round(score_timing(r["result"]), 2) for r in ok],
                "result": ok[0]["result"], "design": design}

    # ---------------------------------------------------------------- prompt
    def _row_head(self, h: Dict[str, Any]) -> str:
        sc = h.get("score")
        if h.get("failed"):
            sc = None
        extra = ""
        if h.get("seed_scores") and len(h["seed_scores"]) > 1:
            extra = " seeds=" + ",".join(_fmt_score(x) for x in h["seed_scores"])
        return f"score={_fmt_score(sc)}{extra} | {cfg_diff(h.get('cfg') or {}, self.knobs)}"

    def build_prompt(self, history: List[Dict[str, Any]], n: int, stage: str = "cts") -> str:
        design = self.design or (history[0].get("design") if history else None)
        # the incumbent (run_search_v2 prior) is already shown as the DEFAULT FLOW reference
        history = [h for h in history if not h.get("is_default")]
        ok = sorted([h for h in history if not h.get("failed") and h.get("score") is not None],
                    key=lambda h: (h["score"], cfg_key(h.get("cfg") or {}, self.knobs)))
        bad = [h for h in history if h.get("failed") or h.get("score") is None]
        best = ok[:self.n_best]
        best_ids = {id(h) for h in best}
        worst = (bad[-self.n_worst:] if bad else []) + \
            [h for h in ok[::-1] if id(h) not in best_ids][:max(0, self.n_worst - len(bad))]
        worst = worst[:self.n_worst]
        ref = self._default_ref(design, stage)
        n_full_best, n_full_worst, ref_full = len(best), len(worst), True
        while True:
            text = self._render(history, n, stage, design, ok, best[:n_full_best], worst[:n_full_worst], ref, ref_full)
            if diag.tokens(_SYSTEM_DIAG) + diag.tokens(text) <= self.max_prompt_tokens:
                return text
            if n_full_worst:
                n_full_worst -= 1
            elif n_full_best > 1:
                n_full_best -= 1
            elif ref_full:
                ref_full = False
            elif n_full_best:
                n_full_best = 0
            else:
                return text

    def _render(self, history, n, stage, design, ok, best, worst, ref, ref_full) -> str:
        L = [f"Design: {design or '?'}; search stage: {stage}.", FLOW_DOC, "", "Knobs (the action space):",
             knob_table(self.knobs), ""]
        if ref:
            L.append(f"DEFAULT FLOW reference (all knobs default), mean over {ref.get('n', '?')} seed(s): "
                     f"wns={ref['wns']:.3f} tns={ref['tns']:.2f} score={-ref['tns'] + 100 * -ref['wns']:.2f}"
                     + (f" seeds={ref['seed_scores']}" if ref.get("seed_scores") else ""))
            if ref_full and (ref.get("result") is not None or ref.get("evidence")):
                L.append(diag.render(self._evidence(ref)))
            L.append("")
        else:
            L += ["DEFAULT FLOW reference: not measured yet.", ""]
        full = [("BEST", h) for h in best] + [("WORST/FAILED", h) for h in worst]
        if full:
            L.append("Evidence for selected configs:")
            for tag, h in full:
                L.append(f"[{tag}] {self._row_head(h)}")
                L.append(diag.render(self._evidence(h)))
            L.append("")
        shown = {id(h) for _, h in full}
        ok_ids = {id(h) for h in ok}
        rest = [h for h in ok if id(h) not in shown] + [h for h in history if id(h) not in shown | ok_ids]
        if rest:
            L.append(f"All other evaluated configs ({len(rest)}, best first; one-line evidence):")
            for h in rest:
                L.append(f"- {self._row_head(h)} :: {diag.render(self._evidence(h), brief=True)}")
            L.append("")
        if self.hypotheses:
            seen = {cfg_key(h.get("cfg") or {}, self.knobs): h for h in history}
            L.append("Your earlier hypotheses and what happened:")
            for hy in self.hypotheses[-self.n_hyp:]:
                h = seen.get(cfg_key(hy["cfg"], self.knobs))
                out = "not evaluated yet" if h is None else ("FAILED" if h.get("failed") else _fmt_score(h.get("score")))
                L.append(f"- {cfg_diff(hy['cfg'], self.knobs)} | hypothesis: {hy['hypothesis'][:200]} | "
                         f"expected: {hy['expected'][:100]} | actual score: {out}")
            L.append("")
        if not history:
            L += ["No configs evaluated yet in this arm-run.", ""]
        L.append(
            f"Propose exactly {n} NEW configs (not already evaluated, all different). For each, state the diagnosis "
            "it tests and the expected effect on WNS/TNS (and runtime if relevant). Reply with JSON only:\n"
            '[{"cfg": {<knob>: <value>, ...}, "hypothesis": "<evidence-based diagnosis being tested>", '
            '"expected": "<predicted effect vs the best/default>"}]\n'
            "Omitted knobs take their defaults.")
        return "\n".join(L)

    # ---------------------------------------------------------------- parse + propose
    def parse(self, text: str) -> List[Dict[str, Any]]:
        """[{cfg (clamped), hypothesis, expected, raw}]; tolerant of fences, prose, a wrapping dict, bare cfgs."""
        data = extract_json(text)
        if isinstance(data, dict):
            for k in ("candidates", "proposals", "configs", "items"):
                if isinstance(data.get(k), list):
                    data = data[k]
                    break
            else:
                data = [data]
        out = []
        for it in data if isinstance(data, list) else []:
            if not isinstance(it, dict):
                continue
            raw = it.get("cfg") if isinstance(it.get("cfg"), dict) else (
                it.get("config") if isinstance(it.get("config"), dict) else
                {k: v for k, v in it.items() if k in self.knobs})
            if not raw:
                continue
            try:
                cfg = clamp_config(raw, self.knobs)
            except Exception:
                continue
            out.append({"cfg": cfg, "hypothesis": str(it.get("hypothesis") or ""),
                        "expected": str(it.get("expected") or ""), "raw": raw})
        return out

    def propose(self, history: List[Dict[str, Any]], n: int, stage: str = "cts") -> List[Dict[str, Any]]:
        self.round += 1
        history = list(history or [])
        prompt = self.build_prompt(history, n, stage)
        resp = self.p.complete(_SYSTEM_DIAG, [Msg("user", prompt)], json_mode=True, max_tokens=8192)
        if self.db is not None:
            try:
                self.db.record_llm(self.arm, self.round, _SYSTEM_DIAG, prompt, resp, phase="propose_diag")
            except Exception:
                pass
        try:
            items = self.parse(resp.text)
        except Exception:
            items = []
            self.invalid += 1
        seen = {cfg_key(h.get("cfg") or {}, self.knobs) for h in history}
        chosen: List[Dict[str, Any]] = []
        n_dupes = 0
        for it in items:
            k = cfg_key(it["cfg"], self.knobs)
            if k in seen:
                n_dupes += 1
                continue
            seen.add(k)
            chosen.append(dict(it, source="llm"))
            if len(chosen) == n:
                break
        if len(chosen) < n:
            if items:
                self.invalid += 1  # short or duplicate-only answer
            tries = 0
            while len(chosen) < n and tries < 50 * n:
                tries += 1
                c = clamp_config(self.fallback.propose(history, 1)[0], self.knobs)
                k = cfg_key(c, self.knobs)
                if k in seen and tries < 50 * n - 1:
                    continue
                seen.add(k)
                chosen.append({"cfg": c, "hypothesis": "", "expected": "", "raw": None, "source": "random_fill"})
        self._log(chosen, stage, history, n_dupes, diag.tokens(_SYSTEM_DIAG) + diag.tokens(prompt), resp)
        self.hypotheses += [c for c in chosen if c["source"] == "llm"]
        return [c["cfg"] for c in chosen[:n]]

    def _log(self, chosen, stage, history, n_dupes, prompt_tokens, resp) -> None:
        if not self.log_path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
            with open(self.log_path, "a") as f:
                for c in chosen:
                    f.write(json.dumps({
                        "ts": round(time.time(), 1), "arm": self.arm, "round": self.round,
                        "design": self.design or (history[0].get("design") if history else None), "stage": stage,
                        "n_history": len(history), "source": c["source"], "cfg": c["cfg"],
                        "hypothesis": c["hypothesis"], "expected": c["expected"], "raw_cfg": c["raw"],
                        "dupes_dropped": n_dupes, "prompt_tokens_est": prompt_tokens,
                        "model": getattr(resp, "model", None)}, sort_keys=True, default=str) + "\n")
        except OSError:
            pass


def make_factory(log_path: Optional[str] = None, work_root: Optional[str] = None,
                 ref: Optional[Dict[str, Any]] = None, design: Optional[str] = None, **kw):
    """experiment.register_arm_v2 factory: (name, rep, db, provider_factory) -> DiagProposer."""

    def factory(name, rep, db, provider_factory):
        if provider_factory is None:
            raise ValueError("the llm_diag arm needs a provider_factory")
        lp = log_path or os.environ.get("PNR_DIAG_LOG")
        return DiagProposer(provider_factory(), FLOW_KNOBS, design=design, ref=ref, db=db, arm=name,
                            fallback=RandomProposer(1000 * rep + 17, FLOW_KNOBS), log_path=lp,
                            work_root=work_root, **kw)

    return factory


def register_llm_diag(log_path: Optional[str] = None, work_root: Optional[str] = None, **kw) -> None:
    """Register the `llm_diag` v2 arm in experiment.PROPOSERS_V2. Hypotheses go to `log_path` (JSONL; default
    $PNR_DIAG_LOG, else not logged); run dirs are looked up under `work_root` (default: from each result's arm_log,
    $NETWT_WORK or ~/work/netwt)."""
    from .experiment import register_arm_v2

    register_arm_v2("llm_diag", make_factory(log_path=log_path, work_root=work_root, **kw))
