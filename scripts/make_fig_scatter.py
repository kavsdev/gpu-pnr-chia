"""Figure 1: DREAMPlace proxy HPWL vs routed WL, 60 gcd configurations, colored by target density."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/figures_data/raw/proxy_vs_routed_scatter/routed_gcd3.json"
OUT = ROOT / "paper/fig_proxy_vs_routed.pdf"

rows = [r for r in json.loads(SRC.read_text())["rows"] if r["ok"] and r["routed_wl"]]
hpwl = [r["proxy_hpwl"] for r in rows]
wl = [r["routed_wl"] for r in rows]
dens = [r["cfg"]["target_density"] for r in rows]
rho, p = spearmanr(hpwl, wl)
print(f"n={len(rows)} rho(hpwl,routed)={rho:+.2f} p={p:.3f}", file=sys.stderr)
print(f"rho(density,hpwl)={spearmanr(dens, hpwl)[0]:+.2f} rho(density,routed)={spearmanr(dens, wl)[0]:+.2f}",
      file=sys.stderr)

# Sequential blue ramp (steps 250 -> 700), light = low density.
cmap = LinearSegmentedColormap.from_list("blue", ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                     "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage[tt=false]{libertine}\usepackage[libertine]{newtxmath}", "font.size": 7.5, "axes.linewidth": 0.6})
fig, ax = plt.subplots(figsize=(3.3, 2.05))
sc = ax.scatter([h / 1e3 for h in hpwl], [w / 1e3 for w in wl], c=dens, cmap=cmap, s=16,
                edgecolors="white", linewidths=0.6, zorder=3)
ax.set_xlabel(r"DREAMPlace HPWL (k$\upmu$m)")
ax.set_ylabel(r"Routed WL (k$\upmu$m)")
ax.text(0.97, 0.95, rf"Spearman $\rho={rho:+.2f}$", transform=ax.transAxes, ha="right", va="top", color="#333333")
ax.grid(True, color="#e6e6e6", linewidth=0.5, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(width=0.6, length=2.5, colors="#333333")
cb = fig.colorbar(sc, ax=ax, pad=0.02, aspect=25)
cb.set_label("Target density")
cb.outline.set_linewidth(0.4)
cb.ax.tick_params(width=0.5, length=2)
fig.tight_layout(pad=0.3)
fig.savefig(OUT)
print(OUT)
