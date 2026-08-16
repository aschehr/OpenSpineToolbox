"""Table 2: summary validation statistics (CSV + rendered PNG). Recomputes
from the real files rather than hard-coding the numbers, so it can't drift
from Table 1 / Figure 3."""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = os.path.dirname(__file__)
AUTO_PATH = os.path.join(HERE, "crest_height_10case.jsonl")
CSV_OUT = os.path.join(HERE, "table2_summary_stats.csv")
PNG_OUT = os.path.join(HERE, "table2_summary_stats.png")

MANUAL = {
    "0441": {"R": "L5", "L": "L4", "obliquity_mm": 18.7},
    "0049": {"R": "L5", "L": "L4", "obliquity_mm": 17.7},
    "0231": {"R": "L5", "L": "L5", "obliquity_mm": 6.05},
    "0680": {"R": "L5", "L": "L5", "obliquity_mm": 9.61},
    "0368": {"R": "L5", "L": "L4", "obliquity_mm": 17.5},
    "0184": {"R": "L4", "L": "L4", "obliquity_mm": 0.112},
    "0508": {"R": "L5", "L": "L5", "obliquity_mm": 0.55},
    "0910": {"R": "L5", "L": "L5", "obliquity_mm": 4.42},
    "0655": {"R": "L4", "L": "L4", "obliquity_mm": 11.3},
    "1058": {"R": "L4", "L": "L4", "obliquity_mm": 0.323},
}
case_ids = list(MANUAL.keys())
auto = {json.loads(l)["case_id"]: json.loads(l) for l in open(AUTO_PATH) if l.strip()}

a = np.array([auto[c]["obliquity_mm"] for c in case_ids])
m = np.array([MANUAL[c]["obliquity_mm"] for c in case_ids])
n = len(case_ids)

r, p = stats.pearsonr(a, m)
diffs = a - m
bias = diffs.mean()
sd_diff = diffs.std(ddof=1)
se_diff = sd_diff / np.sqrt(n)
tcrit = stats.t.ppf(0.975, df=n - 1)
ci_lo, ci_hi = bias - tcrit * se_diff, bias + tcrit * se_diff

X = np.column_stack([a, m])
k = X.shape[1]
grand_mean = X.mean()
row_means = X.mean(axis=1)
col_means = X.mean(axis=0)
SST = np.sum((X - grand_mean) ** 2)
SSR = k * np.sum((row_means - grand_mean) ** 2)
SSC = n * np.sum((col_means - grand_mean) ** 2)
SSE = SST - SSR - SSC
MSR, MSC, MSE = SSR / (n - 1), SSC / (k - 1), SSE / ((n - 1) * (k - 1))
icc21 = (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)

auto_levels, man_levels = [], []
for c in case_ids:
    for side, key in (("right", "R"), ("left", "L")):
        auto_levels.append(auto[c][side]["level"])
        man_levels.append(MANUAL[c][key])
labels = sorted(set(auto_levels) | set(man_levels))
idx = {lab: i for i, lab in enumerate(labels)}
n_pairs = len(auto_levels)
conf = np.zeros((len(labels), len(labels)), dtype=int)
for x, y in zip(auto_levels, man_levels):
    conf[idx[x], idx[y]] += 1
po = np.trace(conf) / n_pairs
row_t, col_t = conf.sum(axis=1), conf.sum(axis=0)
pe = np.sum(row_t * col_t) / (n_pairs ** 2)
kappa = (po - pe) / (1 - pe)

rows = [
    {"Statistic": "Pearson r (obliquity, n=10)", "Value": f"{r:.4f}", "Detail": f"p = {p:.2e}, df = {n-2}"},
    {"Statistic": "ICC(2,1) (obliquity, n=10)", "Value": f"{icc21:.4f}", "Detail": "two-way random, absolute agreement"},
    {"Statistic": "Mean bias, auto - manual (mm)", "Value": f"{bias:.3f}", "Detail": f"95% CI [{ci_lo:.3f}, {ci_hi:.3f}]"},
    {"Statistic": "Cohen's kappa (level, n=20 sides)", "Value": f"{kappa:.4f}", "Detail": f"observed agreement {po:.1%} ({int(np.trace(conf))}/{n_pairs})"},
]

with open(CSV_OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["Statistic", "Value", "Detail"])
    w.writeheader()
    w.writerows(rows)
print(f"wrote -> {CSV_OUT}")
for r_ in rows:
    print(r_)

plt.rcParams.update({"font.family": "sans-serif", "figure.facecolor": "white"})
fig, ax = plt.subplots(figsize=(8, 0.5 * (len(rows) + 1) + 0.6), dpi=300)
ax.axis("off")
cell_text = [[r_["Statistic"], r_["Value"], r_["Detail"]] for r_ in rows]
tbl = ax.table(cellText=cell_text, colLabels=["Statistic", "Value", "Detail"],
               loc="center", cellLoc="left", colLoc="left")
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 1.9)
tbl.auto_set_column_width([0, 1, 2])
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#CCCCCC")
    if row == 0:
        cell.set_facecolor("#4C72B0")
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor("#F7F7F7" if row % 2 == 0 else "white")
        if col == 1:
            cell.set_text_props(fontweight="bold")

ax.set_title("Table 2. Validation Summary Statistics (n=10)", fontsize=13, pad=14)
plt.tight_layout()
plt.savefig(PNG_OUT, dpi=300, facecolor="white", bbox_inches="tight")
print(f"wrote -> {PNG_OUT}")
