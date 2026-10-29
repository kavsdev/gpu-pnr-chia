"""Table 1: GPU flow (default configuration) vs. stock ORFS, routed, on the test seeds both flows ran.
usage: verify_table_a.py [--runs results/a3/routed_runs.json] [seed ...]   (default: every common seed)"""
import argparse, json, os, statistics as st
ap = argparse.ArgumentParser()
ap.add_argument("--runs", default=os.path.join(os.path.dirname(__file__), "..", "..", "results", "a3", "routed_runs.json"))
ap.add_argument("seeds", nargs="*", type=int)
a = ap.parse_args()
R = [r for r in json.load(open(a.runs)) if r["result"].get("ok")]
def runs(d, flow, cfg): return {r["seed"]: r for r in R if r["design"] == d and r["flow"] == flow and r["config"] == cfg}
# The two flows' seeds are independent (DREAMPlace seed vs RePlAce GPL_RANDOM_SEED): a shared seed number links
# nothing, so the unpaired 2 SE (independent groups) and the all-pairs win count are the right comparison.
def se2u(a, b): return 2 * (st.variance(a) / len(a) + st.variance(b) / len(b)) ** .5
def wins(a, b): return sum(p > q for p in a for q in b)
for d in ["ariane133", "swerv_wrapper", "jpeg", "tinyRocket"]:
    o, s = runs(d, "gpu", "default"), runs(d, "stock", "stock")
    seeds = sorted(x for x in o if x in s and (not a.seeds or x in a.seeds))
    print(f"== {d}: seeds {seeds}")
    for x in seeds:
        p, q = o[x]["result"], s[x]["result"]
        print(f"  s{x} gpu {p['wns']:.3f}/{p['tns']:.1f}/{p['routed_wl']} drc={p['drc_violations']} ({o[x]['source']}) | "
              f"stock {q['wns']:.3f}/{q['tns']:.1f}/{q['routed_wl']} drc={q['drc_violations']}")
    if len(seeds) < 2: continue
    for k in ("tns", "wns"):
        p = [o[x]["result"][k] for x in seeds]; q = [s[x]["result"][k] for x in seeds]
        print(f"  {k}: gpu {st.mean(p):.3f} stock {st.mean(q):.3f} diff {st.mean(p) - st.mean(q):+.3f} ± {se2u(p, q):.3f}"
              f" (2 SE, unpaired); gpu better in {wins(p, q)}/{len(p) * len(q)} run pairs")
    p = [o[x]["result"]["routed_wl"] for x in seeds]; q = [s[x]["result"]["routed_wl"] for x in seeds]
    rel = 2 * (st.variance(p) / len(p) / st.mean(p) ** 2 + st.variance(q) / len(q) / st.mean(q) ** 2) ** .5 * 100
    print(f"  wl: gpu {st.mean(p)/1e6:.3f}M stock {st.mean(q)/1e6:.3f}M ({(st.mean(p)/st.mean(q)-1)*100:+.1f}% ± {rel:.1f}%);"
          f" gpu shorter in {wins(q, p)}/{len(p) * len(q)} run pairs")
