"""Figure 3 (two readers): automated vs each reader, Bland-Altman against the reader
mean, and reader-vs-reader Bland-Altman. Same style as build_manuscript_figures.py."""
import csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT = HERE.parents[1] / "manuscript/figures"
rows = list(csv.DictReader(open(HERE / "crest_measurements.csv")))
st = json.load(open(HERE / "interrater_stats.json"))
A = np.array([float(r["ashley_obliquity_mm"]) for r in rows])
J = np.array([float(r["jerick_obliquity_mm"]) for r in rows])
M = np.array([float(r["automated_obliquity_mm"]) for r in rows])

MM = 1 / 25.4
DOUBLE_COL = 174 * MM
BLACK, GREY, LIGHT = "#000000", "#8c8c8c", "#d0d0d0"
plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
})


def panel(ax, letter, x=-0.24):
    ax.text(x, 1.02, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def ba(ax, x, y, s, ylabel, xlabel):
    mean, diff = (x + y) / 2, y - x
    ax.scatter(mean, diff, s=14, facecolor=BLACK, edgecolor="none", zorder=2)
    ax.axhline(0, color=LIGHT, lw=0.6, zorder=0)
    for v, lab, ls in ((s["bias_mm"], f"Mean {s['bias_mm']:+.2f}", "-"),
                       (s["loa_mm"][1], f"+1.96 SD {s['loa_mm'][1]:+.2f}", "--"),
                       (s["loa_mm"][0], f"−1.96 SD {s['loa_mm'][0]:+.2f}", "--")):
        ax.axhline(v, color=BLACK, ls=ls, lw=0.7)
        ax.text(24, v + 0.06, lab.replace("-", "−"), fontsize=6.5, ha="right", va="bottom")
    ax.set_xlim(0, 24); ax.set_ylim(-4.2, 3.6)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)


fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(DOUBLE_COL, DOUBLE_COL * 0.36))
lim = (0, 22)
a1.plot(lim, lim, color=GREY, lw=0.6, ls="--", zorder=0)
a1.scatter(A, M, s=14, facecolor=BLACK, edgecolor="none", zorder=2, label="Reader 1")
a1.scatter(J, M, s=16, facecolor="white", edgecolor=BLACK, linewidth=0.7, zorder=3, label="Reader 2")
a1.text(0.04, 0.97, f"ICC(2,1)\nReader 1: {st['automated_vs_ashley']['icc_2_1']:.3f}\n"
        f"Reader 2: {st['automated_vs_jerick']['icc_2_1']:.3f}", transform=a1.transAxes, va="top", fontsize=6.5)
a1.legend(loc="lower right", frameon=False, handletextpad=0.2, borderaxespad=0.1)
a1.set_xlim(lim); a1.set_ylim(lim); a1.set_aspect("equal")
a1.set_xticks(range(0, 21, 5)); a1.set_yticks(range(0, 21, 5))
a1.set_xlabel("Manual obliquity (mm)"); a1.set_ylabel("Automated obliquity (mm)")
panel(a1, "A")
ba(a2, (A + J) / 2, M, st["automated_vs_reader_mean"], "Automated − reader mean (mm)", "Mean of automated and readers (mm)")
panel(a2, "B")
ba(a3, A, J, st["ashley_vs_jerick"], "Reader 2 − reader 1 (mm)", "Mean of readers (mm)")
panel(a3, "C")
fig.tight_layout(w_pad=2.2)
for ext, kw in (("tif", {"pil_kwargs": {"compression": "tiff_lzw"}}), ("pdf", {}), ("png", {"dpi": 300})):
    fig.savefig(OUT / f"Figure3.{ext}", bbox_inches="tight", pad_inches=0.02, **kw)
with open(OUT / "source_data/Figure3.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["case_id", "reader1_obliquity_mm", "reader2_obliquity_mm", "automated_obliquity_mm"])
    w.writerows([[r["case_id"], a, j, m] for r, a, j, m in zip(rows, A, J, M)])
p = OUT / "provenance.json"; prov = json.load(open(p))
prov["figures"]["Figure3"] = {"n": len(rows), "source": "results/llif/validation/interrater/", "stats": st}
json.dump(prov, open(p, "w"), indent=2)
print("Figure 3 written")
