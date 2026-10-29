"""Paper Fig. 3: (a) cost vs. fidelity, (b) Ariane133 seed 104 layouts, GPU flow vs. stock, (c) Ariane133 winner - default TNS per
test seed. Numbers: results/a3 (scripts/arms/verify_*.py, fidelity_corr.py). usage: make_fig_results.py [IMG_DIR [OUT.pdf]]  (defaults: results/a3/fig3b, paper/fig_results.pdf)"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
S = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "a3", "fig3b")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "paper", "fig_results.pdf")
plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42})
fig = plt.figure(figsize=(7.0, 2.7))
gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.55, 0.95], wspace=0.38, left=0.065, right=0.985, bottom=0.14, top=0.88)

# (a) cost vs fidelity: Ariane133 TNS Spearman (13 competitive runs) per stage; minutes to reach it (same-day mean)
ax = fig.add_subplot(gs[0])
st = ["global\nplace", "detail.\nplace", "CTS", "global\nroute", "full\nroute"]
rho = [0.45, 0.55, 0.86, 0.74]; mins = [5.3, 8.3, 20.6, 45.9, 81.3]  # no rho at full route: it is the reference
x = np.arange(len(st))
ax2 = ax.twinx(); ax2.bar(x, mins, width=0.55, color="0.85", zorder=0); ax2.set_ylim(0, 100)
ax2.set_ylabel("min to reach (Ariane133)", fontsize=6.5); ax2.tick_params(labelsize=6)
for i, m in enumerate(mins): ax2.text(i, m + 2, f"{m:.0f}", ha="center", fontsize=5.8, color="0.35")
ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
ax.plot(x[:4], rho, "o-", color="C3", ms=3.5, lw=1, label="Ariane133 TNS")
ax.set_xticks(x); ax.set_xticklabels(st, fontsize=6); ax.set_ylim(0, 1.15); ax.set_ylabel(r"Spearman $\rho$ vs. routed TNS", fontsize=6.5)
ax.legend(fontsize=5.5, loc="upper left", frameon=False, handletextpad=0.3, borderaxespad=0.2)
ax.set_title("(a) cost vs. fidelity", fontsize=7, pad=3)

# (b) layouts
sub = gs[1].subgridspec(2, 2, wspace=0.03, hspace=0.06)
def crop(p):
    im = Image.open(p).convert("RGB"); a = np.asarray(im); m = a.max(axis=2) > 20
    ys, xs = np.where(m); return im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
labs = [("ours_s104_placement", "GPU flow"), ("stock_s104_placement", "stock"),
        ("ours_s104_congestion", ""), ("stock_s104_congestion", "")]
for i, (f, t) in enumerate(labs):
    a = fig.add_subplot(sub[i % 2 * 0 + (0 if i < 2 else 1), i % 2]); a.imshow(crop(f"{S}/{f}.png")); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values(): sp.set_visible(False)
    if t: a.set_title(t, fontsize=6.5, pad=1.5)
    if i % 2 == 0: a.set_ylabel(["placement", "congestion"][i // 2], fontsize=6.5, labelpad=2)

# (c) Ariane133 winner - default per-seed TNS change (positive = winner better), seeds 101-107, 109
ax = fig.add_subplot(gs[2])
seeds = [101, 102, 103, 104, 105, 106, 107, 109]
d = [398.1, 103.3, 22.7, 133.2, 116.6, 133.5, 34.9, -189.6]
ax.axhline(0, color="0.5", lw=0.6)
ax.axhline(np.mean(d), color="C3", lw=0.8, ls="--")
ax.scatter(range(len(d)), d, s=10, color=["C2" if v > 0 else "C3" for v in d], zorder=3)
ax.set_xticks(range(len(d))); ax.set_xticklabels([str(s)[1:] for s in seeds], fontsize=5.5)
ax.set_xlabel("test seed (1xx)", fontsize=6, labelpad=1); ax.set_ylabel(r"$\Delta$TNS (ns)", fontsize=6.5, labelpad=1)
ax.tick_params(labelsize=6); ax.text(7.3, 300, "mean +94", fontsize=5.5, color="C3", ha="right")
ax.set_title(r"(c) Ariane133 winner $-$ default", fontsize=6.3, pad=3)
fig.text(0.375, 0.955, "(b) Ariane133, seed 104 (TNS: GPU flow $-285$, stock $-452$ ns)", fontsize=7)
fig.savefig(OUT, dpi=300)
