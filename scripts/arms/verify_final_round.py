"""Table 2: the routed final round per design. Lead = default - best finalist mean routed score on the selection
seeds; theta = auto_min_gain (2 sd sqrt(2/n) of the default's search-stage score over the search seeds, n = 2);
test delta-score = default - winner, paired by test seed (higher is better), +- 2 SE.
usage: verify_final_round.py [--data results/a3]"""
import argparse, json, os, statistics as st, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from pnr_node.db import ResultsDB  # noqa: E402
from pnr_node.experiment import auto_min_gain  # noqa: E402
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "..", "results", "a3"))
D = ap.parse_args().data
score = lambda r: -r["tns"] + 100 * -r["wns"]
se2 = lambda xs: 2 * st.stdev(xs) / len(xs) ** .5
runs = json.load(open(f"{D}/routed_runs.json"))
for d in ["ariane133", "swerv_wrapper", "jpeg", "tinyRocket"]:
    f = json.load(open(f"{D}/search/{d}_final_round.json")); a = f["comparison"]["arms"]; v = f["validation"]
    lead = a["default"]["selection_mean"] - a["select"]["selection_mean"]
    theta = auto_min_gain(ResultsDB(f"{D}/search/{d}.sqlite"), d, "cts", len(f["settings"]["select_seeds"]))
    if d == "ariane133":  # test seeds beyond the round's own 101/102 were routed separately (verify_w.py)
        R = [r for r in runs if r["design"] == d and r["flow"] == "gpu" and r["result"].get("ok")]
        w = {r["seed"]: score(r["result"]) for r in R if r["config"] == "winner"}
        b = {r["seed"]: score(r["result"]) for r in R if r["config"] == "default"}
    else:
        b = {int(s): x for s, x in v["v2_default#r0"]["seed_scores"].items()}
        w = {int(s): x for s, x in v["v2_select#r0"]["seed_scores"].items()}
    seeds = sorted(set(w) & set(b)); dd = [b[s] - w[s] for s in seeds]
    print(f"{d:14} lead {lead:7.1f}  theta {theta:6.1f}  chosen {'finalist' if lead > theta else 'default':8}"
          f"  test seeds {len(seeds):2}  test delta-score {st.mean(dd):+.1f} ± {se2(dd):.1f}"
          + ("" if lead > theta else "  (best finalist, not adopted)"))
