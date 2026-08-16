"""Figure 3: Bland-Altman plot, automated vs. manual obliquity, n=10.
Reads the real automated output file; manual values are the same hard-coded
set used in validate_crest_height_10case.py (from the user's ITK-SNAP
measurements)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = os.path.dirname(__file__)
AUTO_PATH = os.path.join(HERE, "crest_height_10case.jsonl")
OUT_PATH = os.path.join(HERE, "fig3_bland_altman.png")

MANUAL = {
    "0441": 18.7, "0049": 17.7, "0231": 6.05, "0680": 9.61, "0368": 17.5,
    "0184": 0.112, "0508": 0.55, "0910": 4.42, "0655": 11.3, "1058": 0.323,
}

auto = {}
with open(AUTO_PATH) as fh:
    for line in fh:
        rec = json.loads(line)
        auto[rec["case_id"]] = rec["obliquity_mm"]

case_ids = list(MANUAL.keys())
a = np.array([auto[c] for c in case_ids])
m = np.array([MANUAL[c] for c in case_ids])
mean_of_pair = (a + m) / 2
diff = a - m
n = len(case_ids)
bias = diff.mean()
sd = diff.std(ddof=1)
loa_lo, loa_hi = bias - 1.96 * sd, bias + 1.96 * sd

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 12, "axes.edgecolor": "#333333",
    "axes.linewidth": 1.0, "figure.facecolor": "white", "axes.facecolor": "white",
})
fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)
ax.scatter(mean_of_pair, diff, s=60, color="#4C72B0", edgecolor="black", linewidth=0.6, zorder=3)
for c, x, y in zip(case_ids, mean_of_pair, diff):
    ax.annotate(c, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8, color="#555555")

ax.axhline(bias, color="black", linewidth=1.3, label=f"Mean bias = {bias:.2f} mm")
ax.axhline(loa_hi, color="#C44E52", linestyle="--", linewidth=1.1,
          label=f"+1.96 SD = {loa_hi:.2f} mm")
ax.axhline(loa_lo, color="#C44E52", linestyle="--", linewidth=1.1,
          label=f"-1.96 SD = {loa_lo:.2f} mm")
ax.axhline(0, color="#999999", linewidth=0.8, zorder=1)

ax.set_xlabel("Mean of automated and manual obliquity (mm)")
ax.set_ylabel("Automated - Manual obliquity (mm)")
ax.set_title(f"Bland-Altman: Automated vs. Manual Crest Obliquity (n={n})")
ax.legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, facecolor="white")
print(f"wrote -> {OUT_PATH}")
print(f"bias={bias:.3f}  sd={sd:.3f}  LoA=[{loa_lo:.3f}, {loa_hi:.3f}]")
