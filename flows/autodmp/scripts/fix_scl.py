"""Generate a Bookshelf .scl (row/site geometry) file.

DREAMPlace's own Bookshelf writer produced an .scl AutoDMP's reader choked on; this regenerates one directly from
the row geometry. Defaults below are tinyRocket/nangate45's real values (from its floorplan) -- pass different
values via flags for any other design/PDK.

Usage: python fix_scl.py --out design.scl [--num-rows 212] [--y-start 5600] [--y-end 596400] ...
"""
import argparse


def write_scl(out_file: str, num_rows: int, y_start: int, y_end: int, row_height: int,
              site_width: int, site_spacing: int, subrow_origin: int, num_sites: int):
    lines = ["UCLA scl 1.0", "", f"NumRows : {num_rows}", ""]
    for y in range(y_start, y_end + row_height, row_height):
        lines += [
            "CoreRow Horizontal",
            f"\tCoordinate : {y}",
            f"\tHeight : {row_height}",
            f"\tSitewidth : {site_width}",
            f"\tSitespacing : {site_spacing}",
            f"\tSiteorient : {1 if (y // row_height) % 2 == 0 else 0}",
            "\tSitesymmetry : 1",
            f"\tSubrowOrigin : {subrow_origin} NumSites : {num_sites}",
            "End",
        ]
    with open(out_file, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-rows", type=int, default=212)
    ap.add_argument("--y-start", type=int, default=5600)
    ap.add_argument("--y-end", type=int, default=596400)
    ap.add_argument("--row-height", type=int, default=2800)
    ap.add_argument("--site-width", type=int, default=380)
    ap.add_argument("--site-spacing", type=int, default=380)
    ap.add_argument("--subrow-origin", type=int, default=4180)
    ap.add_argument("--num-sites", type=int, default=1570)
    args = ap.parse_args()
    write_scl(args.out, args.num_rows, args.y_start, args.y_end, args.row_height,
              args.site_width, args.site_spacing, args.subrow_origin, args.num_sites)
    print(f"Wrote {args.out}")
