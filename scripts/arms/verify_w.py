"""Ariane133 winner (the configuration the routed final round adopted) vs. the default, routed, paired per test seed.
usage: verify_w.py [--runs results/a3/routed_runs.json] [seed ...]   (default: every paired seed)"""
import argparse, json, os, statistics as st
ap = argparse.ArgumentParser()
ap.add_argument("--runs", default=os.path.join(os.path.dirname(__file__), "..", "..", "results", "a3", "routed_runs.json"))
ap.add_argument("seeds", nargs="*", type=int)
a = ap.parse_args()
R = [r for r in json.load(open(a.runs)) if r["design"] == "ariane133" and r["flow"] == "gpu"]
win = {r["seed"]: r for r in R if r["config"] == "winner"}; dflt = {r["seed"]: r for r in R if r["config"] == "default"}
for kind, runs in (("winner", win), ("default", dflt)):  # a failed run is listed, not averaged
    for s, r in sorted(runs.items()):
        if not r["result"].get("ok") and (not a.seeds or s in a.seeds):
            print(f"  FAILED {kind} s{s} ({r['source']}): {r['result'].get('error', '')[-160:]!r}")
ok = lambda r: r["result"].get("ok")
seeds = sorted(x for x in win if x in dflt and ok(win[x]) and ok(dflt[x]) and (not a.seeds or x in a.seeds))
print(f"paired seeds {seeds}")
score = lambda r: -r["tns"] + 100 * -r["wns"]
se2 = lambda xs: 2 * st.stdev(xs) / len(xs) ** .5 if len(xs) > 1 else float("nan")
for x in seeds:
    p, q = win[x]["result"], dflt[x]["result"]
    print(f"  s{x} winner {p['tns']:.1f}/{p['wns']:.3f}/{p['routed_wl']} drc={p['drc_violations']} | "
          f"default {q['tns']:.1f}/{q['wns']:.3f}/{q['routed_wl']} drc={q['drc_violations']} | TNS change {p['tns'] - q['tns']:+.1f}")
if not seeds: raise SystemExit
for name, f in (("score (winner-default, <0 better)", score), ("TNS (winner-default, >0 better)", lambda r: r["tns"]),
                ("WNS (winner-default, >0 better)", lambda r: r["wns"]), ("WL % (winner/default-1)", None)):
    if f is None: d = [(win[x]["result"]["routed_wl"] / dflt[x]["result"]["routed_wl"] - 1) * 100 for x in seeds]
    else: d = [f(win[x]["result"]) - f(dflt[x]["result"]) for x in seeds]
    better = sum((v < 0) if name.startswith(("score", "WL")) else (v > 0) for v in d)
    print(f"  {name}: {st.mean(d):+.3f} ± {se2(d):.3f}  ({better}/{len(d)} seeds winner better)")
