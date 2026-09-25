"""Convert a DEF+LEF placement snapshot to Bookshelf format for AutoDMP.

AutoDMP's tested input path is Bookshelf (`.aux`/`.nodes`/`.nets`/`.pl`/`.scl`/`.wts`), not DEF+LEF -- feeding it
DEF+LEF directly gave 100% trial failure (Infinity loss) before this conversion was found. Uses DREAMPlace's own
`place_io.write(..., SolutionFileFormat.BOOKSHELFALL)`, since DREAMPlace already has a robust DEF/LEF parser and
AutoDMP is itself DREAMPlace-based.

Must run inside a container with AutoDMP's `dreamplace` package importable (e.g. the `autodmp:latest` image, which
has AutoDMP installed at `/AutoDMP`).

Usage:
    python convert.py --lef TECH.lef --lef MACRO.lef [--lef ...] --def-in placed.def --out-prefix out/design \\
        --macro dcache/data.data_arrays_0.data_arrays_0_ext.mem --macro frontend/icache...
"""
import argparse
import os
import sys

sys.path.append("/AutoDMP/install")
sys.path.append("/AutoDMP")

import dreamplace.Params as Params
import dreamplace.ops.place_io.place_io as place_io


def convert(lef_files, def_input, out_prefix, macros, movable_area=10000):
    params = Params.Params()
    params.lef_input = lef_files
    params.def_input = def_input

    print("Reading DB...")
    raw_db = place_io.PlaceIOFunction.read(params)

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    print("Writing BookshelfALL...")
    sol_format = place_io.SolutionFileFormat.BOOKSHELFALL
    place_io.PlaceIOFunction.write(raw_db, out_prefix, sol_format, None, None)
    print("Success with BOOKSHELFALL")

    if macros:
        with open(out_prefix + ".macros", "w") as f:
            f.write(f"{movable_area}\n")
            f.write(" ".join(macros) + "\n")
        print(f"Wrote {out_prefix}.macros")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lef", action="append", required=True, dest="lef_files", help="repeatable: one per LEF file")
    ap.add_argument("--def-in", required=True, dest="def_input")
    ap.add_argument("--out-prefix", required=True, help="e.g. out/tinyRocket -> out/tinyRocket.{nodes,nets,pl,...}")
    ap.add_argument("--macro", action="append", default=[], dest="macros",
                     help="repeatable: instance name of a movable macro (see AutoDMP's own config for which "
                          "instances need this, it's design-specific)")
    ap.add_argument("--movable-area", type=int, default=10000)
    args = ap.parse_args()
    convert(args.lef_files, args.def_input, args.out_prefix, args.macros, args.movable_area)
