"""Figure 1: bar chart of L4/L4-L5/L5 classification percentages across the
789 clean cases in the real full-cohort output file. Percentages are
recomputed here from the file, not hardcoded."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FULL_PATH = os.path.join(HERE, "crest_height_full802.jsonl")
OUT_PATH = os.path.join(HERE, "fig1_distribution_bar.png")

recs = [json.loads(l) for l in open(FULL_PATH) if l.strip()]
clean = [r for r in recs if r["qc_flags"] == ["ok"]]

pooled = {}
for r in clean:
    for side in ("right", "left"):
        lv = r[side]["level"]
        pooled[lv] = pooled.get(lv, 0) + 1
n_pooled = sum(pooled.values())
order = ["L4", "L4-L5", "L5"]
pcts = [100 * pooled[k] / n_pooled for k in order]
counts = [pooled[k] for k in order]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 12, "axes.edgecolor": "#333333",
    "axes.linewidth": 1.0, "figure.facecolor": "white", "axes.facecolor": "white",
})
fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=300)
colors = ["#4C72B0", "#DD8452", "#55A868"]
bars = ax.bar(order, pcts, color=colors, width=0.6, edgecolor="black", linewidth=0.8)
for bar, pct, cnt in zip(bars, pcts, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
            f"{pct:.1f}%\n(n={cnt})", ha="center", va="bottom", fontsize=11)

ax.set_ylabel("Percentage of side-observations")
ax.set_xlabel("Iliac crest apex level")
ax.set_title(f"Iliac Crest Height Classification\n(N={len(clean)} cases, {n_pooled} sides)")
ax.set_ylim(0, max(pcts) + 12)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, facecolor="white")
print(f"wrote -> {OUT_PATH}")
print("percentages:", dict(zip(order, [round(p, 1) for p in pcts])))
