"""Table 1: full n=10 validation table (CSV + rendered PNG), reading the real
automated output file. Manual values are the same hard-coded set used
throughout (from the user's ITK-SNAP measurements)."""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
AUTO_PATH = os.path.join(HERE, "crest_height_10case.jsonl")
CSV_OUT = os.path.join(HERE, "table1_validation.csv")
PNG_OUT = os.path.join(HERE, "table1_validation.png")

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
CASE_IDS = list(MANUAL.keys())

auto = {}
with open(AUTO_PATH) as fh:
    for line in fh:
        rec = json.loads(line)
        auto[rec["case_id"]] = rec

rows = []
for c in CASE_IDS:
    a = auto[c]
    m = MANUAL[c]
    diff = round(a["obliquity_mm"] - m["obliquity_mm"], 3)
    rows.append({
        "Case": c,
        "Manual R": m["R"], "Automated R": a["right"]["level"],
        "Manual L": m["L"], "Automated L": a["left"]["level"],
        "Manual Obliquity (mm)": m["obliquity_mm"],
        "Automated Obliquity (mm)": a["obliquity_mm"],
        "Difference (mm)": diff,
    })

with open(CSV_OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"wrote -> {CSV_OUT}")

# --- rendered PNG table ---
plt.rcParams.update({"font.family": "sans-serif", "figure.facecolor": "white"})
col_labels = ["Case", "Manual\nR", "Auto\nR", "Manual\nL", "Auto\nL",
             "Manual\nObliquity (mm)", "Auto\nObliquity (mm)", "Diff\n(mm)"]
cell_text = [[r["Case"], r["Manual R"], r["Automated R"], r["Manual L"], r["Automated L"],
             f'{r["Manual Obliquity (mm)"]:.2f}', f'{r["Automated Obliquity (mm)"]:.2f}',
             f'{r["Difference (mm)"]:+.2f}'] for r in rows]

fig, ax = plt.subplots(figsize=(9.5, 0.45 * (len(rows) + 1) + 0.6), dpi=300)
ax.axis("off")
tbl = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.6)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#CCCCCC")
    if row == 0:
        cell.set_facecolor("#4C72B0")
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor("#F7F7F7" if row % 2 == 0 else "white")
        if col in (2, 4) and cell_text[row - 1][col] != cell_text[row - 1][col - 1]:
            cell.set_text_props(color="#C44E52", fontweight="bold")

ax.set_title("Table 1. Automated vs. Manual Validation (n=10)", fontsize=13, pad=14)
plt.tight_layout()
plt.savefig(PNG_OUT, dpi=300, facecolor="white", bbox_inches="tight")
print(f"wrote -> {PNG_OUT}")
