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
