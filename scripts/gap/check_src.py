"""Compare rebuilt stock --from sources (stock_<d>_src) with the originals: synthesis + floorplan metrics and reports.
usage: python3 scripts/gap/check_src.py   (exit 0 = all match)"""
import json, os, sys, filecmp
N = os.path.expanduser("~/work/netwt")
PAIRS = {"ariane133": "stock_ariane133", "tinyRocket": "stock_tinyRocket", "swerv_wrapper": "stock_swerv_wrapper_route_s101"}
SKIP = ("runtime", "elapsed", "mem", "cpu", "time_units")  # wall time / memory differ run to run
ok = True
for d, old in PAIRS.items():
    o, n = f"{N}/{old}", f"{N}/stock_{d}_src"
    # make's do-copy rule for 2_floorplan.sdc (a plain cp of 2_1_floorplan.sdc) is not run when the target is only
    # 2_floorplan.odb; do the same copy so the source dir is complete.
    r = f"{n}/results/nangate45/{d}/base"
    if os.path.exists(f"{r}/2_1_floorplan.sdc") and not os.path.exists(f"{r}/2_floorplan.sdc"):
        import shutil; shutil.copyfile(f"{r}/2_1_floorplan.sdc", f"{r}/2_floorplan.sdc"); print(f"{d}: copied 2_1_floorplan.sdc -> 2_floorplan.sdc")
    for f in ["1_synth.json", "2_1_floorplan.json", "2_2_floorplan_macro.json", "2_3_floorplan_tapcell.json", "2_4_floorplan_pdn.json"]:
        a = json.load(open(f"{o}/logs/nangate45/{d}/base/{f}")); b = json.load(open(f"{n}/logs/nangate45/{d}/base/{f}"))
        keys = sorted(k for k in set(a) | set(b) if not any(s in k.lower() for s in SKIP))
        diff = [(k, a.get(k), b.get(k)) for k in keys if a.get(k) != b.get(k)]
        print(f"{d} {f}: {len(keys)} keys, {len(diff)} differ {diff[:5]}"); ok &= not diff
    for f in ["results/nangate45/{d}/base/1_synth.sdc", "results/nangate45/{d}/base/2_floorplan.sdc",
              "results/nangate45/{d}/base/2_2_floorplan_macro.tcl", "reports/nangate45/{d}/base/synth_stat.txt",
              "reports/nangate45/{d}/base/2_floorplan_final.rpt"]:
        f = f.format(d=d)
        if not os.path.exists(f"{o}/{f}"): print(f"{d} {f}: (no original to compare)"); continue
        same = filecmp.cmp(f"{o}/{f}", f"{n}/{f}", shallow=False)
        print(f"{d} {f}: {'identical' if same else 'DIFFERENT'}"); ok &= same
print("ALL_MATCH" if ok else "MISMATCH"); sys.exit(0 if ok else 1)
