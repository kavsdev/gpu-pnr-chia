"""Synthetic evaluator with the same interface as the real node.

Purpose: develop and test the search loop, DB and agent before any tool runs.
It is NOT evidence about placement. The landscape is built so that the cheap
proxy is informative but misleading at high density (routing congestion appears
only after routing), which is the situation the real experiment probes.
"""
from __future__ import annotations

import math
import random

from .contract import EvalRequest, EvalResult

_BASE = {"gcd": 1.0e4, "tinyRocket": 4.0e5, "ariane133": 2.0e7}


class MockEvaluator:
    def __init__(self, noise: float = 0.03):  # ~ measured seed-to-seed sigma on gcd (3%)
        self.noise = noise
        self.tool_id = "mock"
        self.calls = {"proxy": 0, "grt": 0, "route": 0}

    def evaluate(self, req: EvalRequest) -> EvalResult:
        self.calls[req.stage] += 1
        c = req.placer_cfg
        rng = random.Random(f"{req.key()}")
        base = _BASE.get(req.design, 1e5)
        td, dw, g = c["target_density"], math.log10(c["density_weight"]), c["gamma"]
        # proxy HPWL: prefers high density (shorter wires), moderate gamma and weight
        hpwl = base * (1.05 - 0.25 * td + 0.02 * (g - 6) ** 2 / 10 + 0.05 * (dw + 4.2) ** 2 / 4)
        hpwl *= 1 + rng.gauss(0, self.noise)
        if req.stage == "proxy":
            return EvalResult(True, "proxy", proxy_hpwl=hpwl, runtime_s=5.0, provenance={"mock": True})
        # routed: congestion above ~0.85 density costs wirelength, slack and DRCs
        congestion = max(0.0, td - 0.85)
        routed = hpwl * (1.15 + 2.5 * congestion) * (1 + rng.gauss(0, self.noise))
        wns = -0.05 - 6.0 * congestion + rng.gauss(0, 0.005) if congestion > 0.02 else rng.uniform(0.0, 0.05)
        drc = int(max(0, 400 * congestion + rng.gauss(0, 2))) if congestion > 0.02 else 0
        if req.stage == "grt":  # global-route estimate tracks the final wirelength closely
            return EvalResult(True, "grt", proxy_hpwl=hpwl, grt_wl=routed * (1 + rng.gauss(0, 0.005)),
                              runtime_s=60.0, provenance={"mock": True})
        return EvalResult(True, "route", proxy_hpwl=hpwl, routed_wl=routed, wns=wns, tns=min(0.0, wns) * 40,
                          power=1e-3 * routed / base, area=0.9 * base / 1e3, drc_violations=drc,
                          runtime_s=180.0, provenance={"mock": True})


# Synthetic timing landscape for the v2 flow-level arms. NOT evidence about placement: it only lets
# the v2 search loop, DB, proposers and summary run end to end in seconds. TNS is in ns (negative is bad).
_TNS_BASE = {"gcd": -2.0, "tinyRocket": -40.0, "ariane133": -300.0}
_STOP_EFFECT = {"0.6,0.2": 0.0, "0.6,0.4,0.2": -0.04, "0.7,0.3": 0.02, "0.5,0.2": 0.03, "0.6": 0.06, "0.4": 0.10}


class FlowMockEvaluator:
    """Mock for `flow_cfg` requests at stages cts/grt/route. Optimum near target_density 0.52, netwt_pct 15,
    rebuf_stops "0.6,0.4,0.2"; seed noise ~3% (the seed lottery). Deterministic per request identity."""

    def __init__(self, noise: float = 0.03, fail_rate: float = 0.03):
        self.noise, self.fail_rate = noise, fail_rate
        self.tool_id = "mock-flow"
        self.calls: dict = {}

    def evaluate(self, req: EvalRequest) -> EvalResult:
        from .contract import FLOW_KNOBS, clamp_config

        self.calls[req.stage] = self.calls.get(req.stage, 0) + 1
        c = clamp_config(req.flow_cfg, FLOW_KNOBS)
        rng = random.Random(req.key())
        if rng.random() < self.fail_rate or c["target_density"] > 0.75 and c["dp_routability"] is False:
            return EvalResult(False, req.stage, error="mock: DREAMPlace diverged", runtime_s=30.0,
                              provenance={"mock": True, "seed": req.seed})
        rel = (1.0 + 4.0 * (c["target_density"] - 0.52) ** 2 + 0.002 * (c["netwt_pct"] - 15) ** 2
               + 0.004 * (c["netwt_max"] - 6) ** 2 + _STOP_EFFECT[c["rebuf_stops"]]
               + (0.0 if c["rebuf_full"] else 0.05) + (0.02 if c["dp_routability"] else 0.0)
               + 0.03 * (math.log10(c["density_weight"]) + 4.1) ** 2 + 0.003 * (c["gamma"] - 4.5) ** 2
               + 0.03 * (math.log10(c["learning_rate"]) + 2.0) ** 2)
        stage_mult = {"cts": 0.8, "grt": 0.9, "route": 1.0}.get(req.stage, 1.0)
        tns = _TNS_BASE.get(req.design, -50.0) * rel * stage_mult * (1 + rng.gauss(0, self.noise))
        wns = tns / 400.0 * (1 + rng.gauss(0, self.noise))
        wl = 2.0e7 * (1.0 + 0.6 * (0.6 - c["target_density"]) ** 2) * (1 + rng.gauss(0, 0.005))
        r = EvalResult(True, req.stage, wns=wns, tns=tns, runtime_s={"cts": 900.0, "route": 4300.0}.get(req.stage, 1500.0),
                       provenance={"mock": True, "seed": req.seed})
        if req.stage == "route":
            r.routed_wl, r.drc_violations = wl, 0
        elif req.stage == "grt":
            r.grt_wl = wl
        return r
