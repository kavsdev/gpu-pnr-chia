"""Collect the A3 paper's results from a work directory into results/a3/ (the files the verify_* scripts read).

usage: export_results.py [--work DIR] [--out DIR]     (defaults: ~/work, results/a3)

Reads the layout written by scripts/arms/run_arms.py and scripts/gap/run_{arm,stock}.sh:
  <work>/arms_v2/<design>{.sqlite,_v2.json,_final.json,_v2.llm_diag_hypotheses.jsonl}
  <work>/netwt/logs/arm_<design>_*.log            (routed runs outside the search DBs)
  <work>/netwt/stock_<design>_route_s<seed>/      (stock ORFS routes)
  <work>/netwt*/ariane133/<req_key>/result.json   (Ariane133 runs with per-stage timing, for fidelity_corr.py)
  <work>/netwt/{ariane133/<req_key>,stock_ariane133_time3_s<n>}  (runtime runs, for flow_time.py)
  ORFS final report images of the Ariane133 seed-104 routes (Fig. 3b)
Absolute paths inside the records are rewritten to $WORK so the exported files carry no host paths.
"""
import argparse, glob, json, os, re, shutil, sqlite3, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flow_time import run_time  # noqa: E402

DESIGNS = ["ariane133", "swerv_wrapper", "jpeg", "tinyRocket"]
# Ariane133 winner: the configuration the routed final round adopted (ariane133_final.json, v2_select#r0)
WINNER = {"density_weight": 8e-05, "dp_routability": False, "gamma": 2.0, "learning_rate": 0.01, "netwt_max": 8.0,
          "netwt_pct": 10.0, "rebuf_full": True, "rebuf_stops": "0.6,0.4,0.2", "target_density": 0.45}
# Fig. 3b: ORFS's own final report images (6_report) of the seed-104 routes
FIG3B = {"ours": "netwt/ariane133/7bd2e115aace1a45/orfs/reports", "stock": "netwt/stock_ariane133_route_s104/reports"}
RUNTIME = {"gpu_s11": "netwt/ariane133/440263637ea88297", "gpu_s12": "netwt/ariane133/cfdf485c359bcbd1",
           "stock_s1": "netwt/stock_ariane133_time3_s1", "stock_s2": "netwt/stock_ariane133_time3_s2"}
# Fig. 3a: the competitive Ariane133 runs (routed TNS > -700 ns) that existed when CTS was chosen as the search stage
FIG3A = ["06ebfb58626d66e0", "9aee7370f370a058", "bc666e2edbf22637", "af8a8b79ad7d23b9", "fcdaa76a4cb70cbc",
         "f3465f2725ea8445", "a87363714d39126d", "bb8d670dc8fb3c39", "50fa2a4def468e5e", "8031d339967c61c4",
         "f3528ae65674a32e", "985803504b753579", "0a94db4d7e245a61"]
STAGES = [("3_3_place_gp", "globalplace"), ("3_4_place_resized", "placeopt"), ("3_5_place_dp", "detailedplace"),
          ("4_1_cts", "cts"), ("5_1_grt", "globalroute"), ("6_report", "finish")]


def scrub(text, work):
    return text.replace(work, "$WORK").replace(os.path.expanduser("~"), "$HOME")


def lastjson(log):
    lines = [l for l in open(log, errors="replace") if l.startswith("{") and '"stage"' in l]
    return json.loads(lines[-1]) if lines else None


def failed(log):
    return "exit=" in open(log, errors="replace").read()[-200:]


def routed_runs(w):
    """GPU-flow default and Ariane133 winner routes on test seeds, plus stock routes; one record per run."""
    out = {}
    def add(rec):
        out.setdefault((rec["design"], rec["flow"], rec["config"], rec["seed"]), rec)
    for d in DESIGNS:
        db = f"{w}/arms_v2/{d}.sqlite"
        if os.path.exists(db):
            for arm, req, res in sqlite3.connect(db).execute(
                    "select arm, req_json, result_json from evals where stage='route' order by id"):
                q, r = json.loads(req), json.loads(res)
                if q["seed"] < 101:
                    continue
                cfg = "winner" if d == "ariane133" and q.get("flow_cfg") == WINNER else \
                      "default" if arm.startswith("v2_default") else None
                if cfg and r.get("ok"):
                    add(dict(design=d, flow="gpu", config=cfg, seed=q["seed"], source=f"{d}.sqlite", req=q, result=r))
        logs = glob.glob(f"{w}/netwt/logs/arm_{d}_*def_s1[0-9][0-9].log")
        if d == "ariane133":
            logs += glob.glob(f"{w}/netwt/logs/arm_ariane133_confirm_win_s1[0-9][0-9].log")
        for log in sorted(logs):
            s = int(re.search(r"_s(1\d\d)\.log", log).group(1)); r = lastjson(log)
            cfg = "winner" if "_confirm_win_" in log else "default"
            src = os.path.basename(log)
            if r and r.get("ok") and r.get("stage") == "route":
                add(dict(design=d, flow="gpu", config=cfg, seed=s, source=src, result=r))
            elif r is not None or failed(log):
                add(dict(design=d, flow="gpu", config=cfg, seed=s, source=src,
                         result={"ok": False, "error": (r or {}).get("error", "no result JSON")[-300:]}))
        for p in sorted(glob.glob(f"{w}/netwt/stock_{d}_route_s1[0-9][0-9]")):
            b = f"{p}/logs/nangate45/{d}/base"
            try:
                m = json.load(open(f"{b}/6_report.json")); rt = json.load(open(f"{b}/5_2_route.json"))
            except (OSError, ValueError):
                continue
            g = lambda dct, k: dct.get(k)
            add(dict(design=d, flow="stock", config="stock", seed=int(p[-3:]), source=os.path.basename(p),
                     result=dict(ok=True, wns=g(m, "finish__timing__setup__ws"), tns=g(m, "finish__timing__setup__tns"),
                                 routed_wl=g(rt, "detailedroute__route__wirelength"),
                                 drc_violations=g(rt, "detailedroute__route__drc_errors"))))
    return sorted(out.values(), key=lambda r: (DESIGNS.index(r["design"]), r["flow"], r["config"], r["seed"]))


def stage_metrics(logdir):
    out = {}
    for f, p in STAGES:
        fn = os.path.join(logdir, f + ".json")
        if os.path.exists(fn):
            j = json.load(open(fn)); out[p] = [j.get(p + "__timing__setup__ws"), j.get(p + "__timing__setup__tns")]
    return out


def fidelity_rows(w):
    """Ariane133 GPU-flow routes with WNS/TNS at every stage (input of fidelity_corr.py)."""
    rows, seen = [], set()
    for rj in sorted(glob.glob(f"{w}/netwt*/ariane133/[0-9a-f]*/result.json")):
        d = os.path.dirname(rj); j = json.load(open(rj)); key = os.path.basename(d)
        if not j.get("ok") or j.get("stage") != "route" or key in seen:
            continue
        m = stage_metrics(os.path.join(d, "orfs/logs/nangate45/ariane133/base"))
        if "finish" in m:
            seen.add(key); rows.append(dict(req_key=key, fig3a=key in FIG3A, stages=m, routed_wl=j["routed_wl"]))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work", default=os.path.expanduser("~/work"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "results", "a3"))
    a = ap.parse_args(); w = os.path.abspath(a.work); out = os.path.abspath(a.out)
    os.makedirs(f"{out}/search", exist_ok=True)
    def dump(name, obj):
        with open(f"{out}/{name}", "w") as f:
            f.write(scrub(json.dumps(obj, indent=1, sort_keys=True), w) + "\n")
    dump("routed_runs.json", routed_runs(w))
    dump("fidelity_ariane133.json", fidelity_rows(w))
    dump("runtime_ariane133.json", {lb: run_time(f"{w}/{p}") for lb, p in RUNTIME.items()})
    os.makedirs(f"{out}/fig3b", exist_ok=True)
    for who, rep in FIG3B.items():
        for img in ("placement", "congestion"):
            shutil.copyfile(f"{w}/{rep}/nangate45/ariane133/base/final_{img}.webp.png", f"{out}/fig3b/{who}_s104_{img}.png")
    for d in DESIGNS:
        # tinyRocket's search summary is the re-validated one (revalidate.py rewrote it as <d>_v2_final.json)
        search = f"{d}_v2_final.json" if os.path.exists(f"{w}/arms_v2/{d}_v2_final.json") else f"{d}_v2.json"
        for name, dst in ((search, "search.json"), (f"{d}_final.json", "final_round.json"),
                          (f"{d}_v2.llm_diag_hypotheses.jsonl", "llm_diag_hypotheses.jsonl")):
            if os.path.exists(f"{w}/arms_v2/{name}"):
                with open(f"{w}/arms_v2/{name}") as f, open(f"{out}/search/{d}_{dst}", "w") as g:
                    g.write(scrub(f.read(), w))
        src = f"{w}/arms_v2/{d}.sqlite"
        if os.path.exists(src):
            dst = f"{out}/search/{d}.sqlite"; shutil.copyfile(src, dst); c = sqlite3.connect(dst)
            c.create_function("scrub", 1, lambda t: None if t is None else scrub(t, w))
            for tbl, cols in (("evals", ("req_json", "result_json", "note")), ("llm_calls", ("system", "prompt", "response"))):
                c.execute(f"update {tbl} set " + ", ".join(f"{k} = scrub({k})" for k in cols))
            c.commit(); c.execute("vacuum"); c.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
