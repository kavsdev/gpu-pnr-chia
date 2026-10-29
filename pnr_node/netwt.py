"""Net-weight timing-driven DREAMPlace: OpenSTA slack -> per-net weights.

OpenSTA times the real netlist (`net_weights.tcl`, run under ORFS's env) and dumps one worst setup slack per net;
this module turns that into the `net_name weight` file DREAMPlace's `net_weights_file` hook consumes. DREAMPlace
never runs a timer itself.

Weight update = DREAMPlace's own lilith scheme (`ops/timing/src/net_weighting_scheme.h`), applied once per round
instead of every 15 GP iterations:
    nc      = k * max(0, slack / WNS)      for slack < 0 and WNS < 0, else 0    (criticality this round)
    crit    = (1 + crit)^decay * (1 + nc)^(1 - decay) - 1                      (momentum across rounds)
    weight *= (1 + crit), clamped to max_weight
Lilith has k = 1; with only a handful of rounds (vs. dozens of in-GP reweights) a gain k > 1 is exposed so the weights
can reach a useful spread. Clock nets and nets above `ignore_net_degree` pins are never weighted (weight 1): clocks
are CTS's job and DREAMPlace ignores high-degree nets in its wirelength anyway.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

TCL = "net_weights.tcl"  # ORFS-env setup; sources DUMP_TCL from its own directory
DUMP_TCL = "net_slack_dump.tcl"


@dataclass
class SlackDump:
    wns: Optional[float]
    tns: Optional[float]
    viol: Optional[int]
    nets: List[Tuple[str, Optional[float], int, str]] = field(default_factory=list)  # (name, slack|None, degree, flag)


def parse_slacks(path: str) -> SlackDump:
    d = SlackDump(None, None, None)
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            if p[0] == "#":
                kv = dict(zip(p[1::2], p[2::2]))
                d.wns = _num(kv.get("wns"))
                d.tns = _num(kv.get("tns"))
                v = _num(kv.get("viol"))
                d.viol = int(v) if v is not None else None
                continue
            if len(p) < 4:
                continue
            d.nets.append((p[0], _num(p[1]), int(p[2]), p[3]))
    return d


def _num(s: Optional[str]) -> Optional[float]:
    if s is None or s.upper() in ("INF", "-INF", "NAN", ""):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def update_weights(dump: SlackDump, weights: Dict[str, float], crit: Dict[str, float], k: float = 4.0,
                   decay: float = 0.5, max_weight: float = 1024.0,
                   compound: bool = True) -> Tuple[Dict[str, float], Dict[str, float]]:
    """One lilith reweight from one slack dump. Returns new (weights, criticality); inputs are not modified.

    compound=True is lilith (`w *= 1+crit`, weights grow every round a net stays critical). compound=False resets the
    weight each round to `1+crit`: the momentum lives only in the criticality, so a persistently critical net converges
    to 1+k instead of growing geometrically (on tinyRocket compounding reached 30x and hurt the nets it targeted)."""
    w, c = dict(weights), dict(crit)
    wns = dump.wns
    for name, slack, _deg, flag in dump.nets:
        if flag != "sig":  # clock / high-degree: never weighted
            w.pop(name, None)
            c.pop(name, None)
            continue
        if wns is not None and wns < 0:  # lilith leaves criticality untouched when timing is met
            nc = k * max(0.0, slack / wns) if (slack is not None and slack < 0) else 0.0
            c[name] = (1.0 + c.get(name, 0.0)) ** decay * (1.0 + nc) ** (1.0 - decay) - 1.0
        nw = (w.get(name, 1.0) if compound else 1.0) * (1.0 + c.get(name, 0.0))
        w[name] = min(nw, max_weight) if max_weight > 0 else nw
    return w, c


def write_weights(path: str, weights: Dict[str, float]) -> int:
    """Write every non-unit weight (DREAMPlace's default is 1.0). Returns the number of lines written."""
    n = 0
    with open(path, "w") as f:
        for name in sorted(weights):
            if weights[name] != 1.0:
                f.write(f"{name} {weights[name]:.6g}\n")
                n += 1
    return n


def weight_stats(weights: Dict[str, float]) -> Dict:
    ws = [v for v in weights.values() if v != 1.0]
    return {"weighted": len(ws), "max_weight": max(ws) if ws else 1.0,
            "mean_weight": (sum(ws) / len(ws)) if ws else 1.0}


def timing_stats(dump: SlackDump) -> Dict:
    flags: Dict[str, int] = {}
    for _n, _s, _d, fl in dump.nets:
        flags[fl] = flags.get(fl, 0) + 1
    return {"wns": dump.wns, "tns": dump.tns, "viol": dump.viol, "nets": len(dump.nets), "flags": flags}


# gpl's timing-driven defaults (src/gpl/include/gpl/Replace.h: timingNetWeightMax = 5, timingDrivenNetsPercentage = 10;
# the installed ORFS binary was checked with `set_debug_level GPL timing 1`: max weight 5, 10% of nets weighted).
GPL_NETS_PCT = 10.0
GPL_WEIGHT_MAX = 5.0


def gpl_weights(dump: SlackDump, pct: float = GPL_NETS_PCT, max_w: float = GPL_WEIGHT_MAX) -> Dict[str, float]:
    """RePlAce's own timing-driven net weights (gpl TimingBase::executeTimingDriven) from one slack dump.

    Population = every non-clock net with a finite slack (Resizer::findResizeSlacks1 drops clock, special, constant and
    unconstrained nets; high-degree nets stay in). Sort by slack, take the worst ceil(N * pct / 100)
    (Resizer::resizeWorstSlackNets); slack_min/slack_max = the first/last of those. Every net with slack < slack_max
    gets the linear weight 1 + (max_w - 1) * (slack_max - slack) / (slack_max - slack_min) (all 1.0 if
    slack_max == slack_min); everything else is 1. Not cumulative: gpl resets every net to 1.0 before each reweight.
    Slack sign is irrelevant -- the worst pct% are weighted even when timing is met."""
    pop = sorted((s, name) for name, s, _deg, flag in dump.nets if flag != "clk" and s is not None)
    if not pop or pct <= 0:
        return {}
    n = min(len(pop), max(1, math.ceil(len(pop) * pct / 100.0)))
    smin, smax = pop[0][0], pop[n - 1][0]
    w: Dict[str, float] = {}
    for s, name in pop[:n]:
        if s < smax:
            w[name] = 1.0 if smax == smin else 1.0 + (max_w - 1.0) * (smax - s) / (smax - smin)
    return w
