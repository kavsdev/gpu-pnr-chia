"""Turn an experiment (results DB + summary JSON) into the honest table: what each arm found, and whether it is real.

Rules baked in (from results/gcd/noise_floor_gcd.md):
  * the comparison number is the HELD-OUT validated mean, never the selection-seed best (winner's curse);
  * the noise scale sigma is the spread of the default config over seeds (the seed lottery);
  * an arm "clears the noise" only if its validated mean beats the default-config mean by more than 2 sigma,
    where sigma is the per-route seed sigma (a conservative bar for a mean of 4 validation routes).
"""
from __future__ import annotations

import json
import statistics as st
from typing import Any, Dict, List

from .db import ResultsDB


def _default_scores(db: ResultsDB, design: str) -> List[float]:
    """Every routed score of the default config across seeds (search seeds + validation seeds)."""
    out = []
    for arm in [r[0] for r in db.con.execute("SELECT DISTINCT arm FROM evals WHERE arm LIKE 'default_seeds%'").fetchall()]:
        for h in db.evals(arm, design, "route"):
            if h["score"] is not None:
                out.append(h["score"])
    return out


def arms_report(db: ResultsDB, result: Dict[str, Any]) -> str:
    design = result["design"]
    base = _default_scores(db, design)
    if len(base) < 3:
        return "not enough default-seed routes to define the noise floor"
    mu, sd = st.mean(base), st.pstdev(base)
    lines = [f"Design `{design}`. Seed-lottery baseline (default knobs, n={len(base)} seeds): mean {mu:.0f}, sigma {sd:.0f} "
             f"({100 * sd / mu:.1f}%), best {min(base):.0f}, worst {max(base):.0f}.", "",
             "| arm | runs | selection-seed best (mean) | held-out validated mean | vs default mean | clears 2 sigma? | routes/run | grt runs/run |",
             "|---|---|---|---|---|---|---|---|"]
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for s in result["search"]:
        by_arm.setdefault(s["arm"].split("#")[0], []).append(s)
    for arm, runs in by_arm.items():
        sel = [r["best_score"] for r in runs if r.get("best_score") is not None]
        vm = [result["validation"][r["arm"]]["validated_mean"] for r in runs
              if result["validation"].get(r["arm"], {}).get("validated_mean") is not None]
        if not vm:
            lines.append(f"| {arm} | {len(runs)} | - | - | - | - | - | - |")
            continue
        m = st.mean(vm)
        delta = m - mu
        clears = "yes" if delta < -2 * sd else ("no (worse)" if delta > 2 * sd else "no")
        grt = [r.get("grt_runs") for r in runs if r.get("grt_runs") is not None]
        lines.append(f"| {arm} | {len(runs)} | {st.mean(sel):.0f} | {m:.0f} (sd across runs {st.pstdev(vm) if len(vm) > 1 else float('nan'):.0f}) "
                     f"| {delta:+.0f} ({100 * delta / mu:+.1f}%) | {clears} | {runs[0].get('routes_used')} | "
                     f"{(st.mean(grt) if grt else 0):.0f} |")
    n_hits = result.get("cache_hits", 0)
    lines += ["", f"Cache replays during the run: {n_hits} (identical identities are bit-identical, so they cost nothing).",
              "Lower score is better. Selection-seed best is shown only to expose the winner's-curse gap; do not compare arms on it."]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    print(arms_report(ResultsDB(a.db), json.load(open(a.json))))
