"""Fig. 3a: how well each stage's timing ranks Ariane133 GPU-flow runs like the routed result (Spearman rho).
usage: fidelity_corr.py [--rows results/a3/fidelity_ariane133.json]
Competitive: routed TNS > -700 ns (the timing-driven configurations; the rest are wirelength-driven runs).
Fig. 3a uses the 13 competitive runs that existed when CTS was chosen as the search stage (rows with fig3a = true);
the later runs (search validation, test seeds) are reported as a second set."""
import argparse, json, os
ap = argparse.ArgumentParser()
ap.add_argument("--rows", default=os.path.join(os.path.dirname(__file__), "..", "..", "results", "a3", "fidelity_ariane133.json"))
rows = json.load(open(ap.parse_args().rows))
STAGES = ["globalplace", "placeopt", "detailedplace", "cts", "globalroute"]
def rank(v):
    s = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
    for k, i in enumerate(s): r[i] = k
    return r
def sp(a, b):
    ra, rb = rank(a), rank(b); n = len(a); ma = sum(ra) / n; mb = sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb)); den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** .5
    return num / den if den else float("nan")
comp = [r for r in rows if r["stages"]["finish"][1] > -700]
for name, R in (("Fig. 3a", [r for r in rows if r["fig3a"]]), ("all competitive", comp), ("all", rows)):
    print(f"Spearman vs routed ({name}, n={len(R)})")
    for p in STAGES:
        RR = [r for r in R if p in r["stages"]]
        print(f"  {p:14} WNS {sp([r['stages'][p][0] for r in RR], [r['stages']['finish'][0] for r in RR]):+.2f}"
              f"  TNS {sp([r['stages'][p][1] for r in RR], [r['stages']['finish'][1] for r in RR]):+.2f}")
