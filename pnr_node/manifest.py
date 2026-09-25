"""Capability manifest: what the node can do and, just as important, what it cannot.

Scope limits live here, in the node, instead of in prompt text, so an agent (or a reviewer) can read them.
Known-noise numbers are measured values; update them when re-measured.
"""
from __future__ import annotations

from typing import Any, Dict

from .contract import KNOBS


def manifest() -> Dict[str, Any]:
    return {
        "name": "pnr_node",
        "purpose": "Evaluate DREAMPlace global-placement settings by post-route PPA (OpenROAD legalize/CTS/route/STA).",
        "stages": {
            "proxy": "DREAMPlace only (GPU, seconds). Checks feasibility (non-converged placements are rejected) and returns "
                     "proxy_hpwl, which is NOT a reliable ranker: Spearman -0.29 vs routed WL on gcd (dense placements have low "
                     "HPWL and route worse).",
            "grt": "DREAMPlace then stock ORFS through global route; returns grt_wl, an estimate that tracked final routed WL "
                   "at Spearman +0.99 on gcd. Use this to screen candidates.",
            "route": "DREAMPlace then stock OpenROAD/ORFS to detailed route; returns measured routed PPA. Continues from a "
                     "finished grt run of the same identity instead of repeating it.",
        },
        "metrics": ["proxy_hpwl", "grt_wl(um)", "routed_wl(um)", "wns(ns)", "tns(ns)", "power(W)", "area(um^2)", "drc_violations"],
        "knobs": {k: {"lo": v.lo, "hi": v.hi, "default": v.default, "integer": v.integer} for k, v in KNOBS.items()},
        "validated": {
            "designs": ["gcd", "tinyRocket", "ariane133"], "tech": ["nangate45"],
            "evidence": "results/gcd/g1_gcd_handoff.md, results/gcd/noise_floor_gcd.md, "
                        "results/tinyrocket/tinyrocket_arms.md, results/ariane133/ariane133_vs_stock.md",
        },
        "known_noise": {
            "gcd/nangate45": {"routed_wl_sigma_pct": 3.0, "routed_wl_range_pct": 8.0, "n": 5,
                              "note": "seed-to-seed spread; identical input is bit-identical"},
        },
        "known_gaps": [
            "No commercial signoff. NanGate45 has no open DRC/LVS deck: drc_violations is the detailed-router "
            "violation count, not a signoff DRC result. No LVS.",
            "Timing-driven DREAMPlace is not functional in this flow, so placement is wirelength-driven while stock "
            "ORFS places timing-driven; expect a setup-timing gap against stock (Ariane133: WNS -1.10 vs -0.27 ns).",
            "DREAMPlace routability mode (routability_opt=1) crashes in this build/enablement (no routing capacities: "
            "unit_horizontal_capacities is None), so it is not a tunable knob.",
            "The legal knob ranges are the region where DREAMPlace converged on gcd (60/60). Over the earlier wide ranges only "
            "33 of 160 samples (21%) converged; the rest are rejected before any route. Validated on gcd only.",
            "Only the DREAMPlace knobs in `knobs` are tunable; AutoDMP concurrent macro placement is not integrated.",
            "Improvements smaller than about 2 sigma of the per-design noise are not distinguishable from a seed lottery.",
            "Thread-count invariance of OpenROAD results was checked once (gcd, 2 vs 8 threads: bit-identical); untested on larger designs.",
            "FPGA fabrics and designs without an open OpenROAD flow are out of scope.",
        ],
        "trust": {"routed": "measured", "proxy": "estimate", "agent_narrative": "excluded from evidence hash"},
    }
