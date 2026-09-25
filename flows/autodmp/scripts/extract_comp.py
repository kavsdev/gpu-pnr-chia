"""Strip a placed DEF down to COMPONENTS-only (drop PINS/NETS/rows/tracks).

`read_def -incremental` re-reads PINS, which appends a second shape to every existing pin and aborts detailed
routing with DRT-0302 ("multiple pins on bterm") -- placement is all that's wanted when importing back into ORFS.
Same function as `pnr_node/raw_handoff.py`'s `components_only_def`; kept standalone here since this runs against
AutoDMP's own output DEF, outside the pnr_node package.

Usage: python extract_comp.py IN.def OUT.def
"""
import argparse


def components_only_def(src: str, dst: str) -> int:
    out, in_comp, n = [], False, 0
    with open(src) as f:
        for line in f:
            tok = line.split(None, 1)[0] if line.strip() else ""
            if tok in ("VERSION", "DIVIDERCHAR", "BUSBITCHARS", "DESIGN", "UNITS", "DIEAREA"):
                out.append(line)
            elif tok == "COMPONENTS":
                in_comp = True
                out.append(line)
            elif in_comp:
                out.append(line)
                if line.lstrip().startswith("- "):
                    n += 1
                if line.startswith("END COMPONENTS"):
                    in_comp = False
            elif tok == "END" and line.strip() == "END DESIGN":
                out.append(line)
    with open(dst, "w") as f:
        f.writelines(out)
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("def_in")
    ap.add_argument("def_out")
    args = ap.parse_args()
    n = components_only_def(args.def_in, args.def_out)
    print(f"Wrote {n} components to {args.def_out}")
