"""Candidate figure: tinyRocket placed/routed renders (OpenROAD save_image) for stock RePlAce, our DREAMPlace
handoff and AutoDMP. Inputs: results/figures_data/raw/floorplan_render/ (provenance in results/figures_data/MANIFEST.md)."""
from pathlib import Path

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/figures_data/raw/floorplan_render"
OUT = ROOT / "paper/fig_tinyrocket_floorplans.pdf"
COLS = [("stock", "Stock (RePlAce)"), ("dreamplace", "Ours (DREAMPlace)"), ("autodmp", "AutoDMP")]
ROWS = [("placement", "Placed"), ("routed", "Routed")]

plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                     "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage[tt=false]{libertine}\usepackage[libertine]{newtxmath}", "font.size": 7.5})
fig, axes = plt.subplots(len(ROWS), len(COLS), figsize=(3.4, 2.45))
for i, (stage, rlabel) in enumerate(ROWS):
    for j, (approach, clabel) in enumerate(COLS):
        im = Image.open(SRC / f"tinyrocket_{approach}_{stage}.png").convert("RGB")
        im.thumbnail((700, 700))  # keeps the PDF small; still ~300 dpi at column width
        ax = axes[i][j]
        ax.imshow(im, interpolation="lanczos")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4)
        if i == 0:
            ax.set_title(clabel, fontsize=7, pad=2)
        if j == 0:
            ax.set_ylabel(rlabel, fontsize=7, labelpad=2)
fig.subplots_adjust(left=0.05, right=0.995, top=0.93, bottom=0.005, wspace=0.03, hspace=0.03)
fig.savefig(OUT, dpi=300)
print(OUT)
