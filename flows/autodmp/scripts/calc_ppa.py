"""Build AutoDMP's `base_ppa.json` (its reference point for normalizing trial objectives) by scaling a known-good
reference design's metrics by the ratio of real HPWL to that reference's HPWL.

Only HPWL is measured directly here; rsmt/congestion/density are extrapolated from the reference by the same
ratio -- a rough approximation AutoDMP's tuner uses only for normalization, not a claimed real measurement of those
three (flagged, don't cite `rsmt`/`congestion`/`density` from this file as measured results).

Usage: python calc_ppa.py --hpwl-um 385504.27 --out base_ppa.json
"""
import argparse
import json


def calc_ppa(hpwl_um: float, dbu: int, ref_hpwl: float, ref_rsmt: float, ref_cong: float, ref_dens: float):
    hpwl_real = hpwl_um * dbu
    ratio = hpwl_real / ref_hpwl
    return {
        "hpwl": hpwl_real,
        "rsmt": ref_rsmt * ratio,
        "congestion": ref_cong * ratio,
        "density": ref_dens * ratio,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hpwl-um", type=float, required=True, help="real measured HPWL in microns")
    ap.add_argument("--dbu", type=int, default=2000)
    ap.add_argument("--ref-hpwl", type=float, default=1.31e10, help="reference design's HPWL (DBU)")
    ap.add_argument("--ref-rsmt", type=float, default=1.50e10)
    ap.add_argument("--ref-cong", type=float, default=0.65)
    ap.add_argument("--ref-dens", type=float, default=0.51)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = calc_ppa(args.hpwl_um, args.dbu, args.ref_hpwl, args.ref_rsmt, args.ref_cong, args.ref_dens)
    with open(args.out, "w") as f:
        json.dump(res, f)
    print(json.dumps(res, indent=4))
