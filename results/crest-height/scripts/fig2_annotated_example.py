"""Figure 2: single annotated coronal CT example (case 0184 -- one of the
10 manually-validated cases: clean, both sides classified L4, small
obliquity). Coronal bone-window MIP with crest apex points, the L4-L5 disc
reference line, and the classification labels overlaid."""
import json
import os
import sys

sys.path.insert(0, "/Users/ashleyschehr/OpenSpineToolbox")
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ostk.io import load_ct
from figure_utils import coronal_mip

HERE = os.path.dirname(__file__)
SNAP = ("/Users/ashleyschehr/.cache/huggingface/hub/"
        "datasets--anonymous-mlhc--CTSpinoPelvic1K/snapshots/"
        "9f69480a572624029a927bc6661309a068f1f2c3")
CASE_ID = "0184"
OUT_PATH = os.path.join(HERE, "fig2_annotated_example.png")

recs = {json.loads(l)["case_id"]: json.loads(l)
        for l in open(os.path.join(HERE, "crest_height_full802.jsonl")) if l.strip()}
rec = recs[CASE_ID]

ct, aff = load_ct(os.path.join(SNAP, "ct", f"{CASE_ID}_ct.nii.gz"))
img, w2p, left_lbl, right_lbl = coronal_mip(ct, aff)

r_apex = rec["landmarks_world_mm"]["right_crest_apex"]
l_apex = rec["landmarks_world_mm"]["left_crest_apex"]
disc_z = rec["landmarks_world_mm"]["l4_l5_disc_ref_z"][0]

r_col, r_row = w2p(r_apex)
l_col, l_row = w2p(l_apex)
_, disc_row = w2p([0.0, 0.0, disc_z])

# crop to a region around the pelvis/lumbar spine for a legible figure
pad_rows = 140
row_lo = int(max(0, min(r_row, l_row, disc_row) - pad_rows))
row_hi = int(min(img.shape[0], max(r_row, l_row, disc_row) + pad_rows))
col_center = (r_col + l_col) / 2
half_w = 240
col_lo = int(max(0, col_center - half_w))
col_hi = int(min(img.shape[1], col_center + half_w))

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11, "figure.facecolor": "white",
})
fig, ax = plt.subplots(figsize=(7, 8), dpi=300)
ax.imshow(img[row_lo:row_hi, col_lo:col_hi], cmap="gray", vmin=200, vmax=1500,
         extent=[col_lo, col_hi, row_hi, row_lo], aspect="equal")

ax.plot(r_col, r_row, marker="o", markersize=9, markerfacecolor="#DD2C2C",
       markeredgecolor="white", markeredgewidth=1.2, zorder=5)
ax.plot(l_col, l_row, marker="o", markersize=9, markerfacecolor="#DD2C2C",
       markeredgecolor="white", markeredgewidth=1.2, zorder=5)
ax.axhline(disc_row, xmin=0.05, xmax=0.95, color="#FFD54F", linewidth=1.8,
          linestyle="--", zorder=4)

ax.annotate(f"R crest apex\n{rec['right']['level']}, {rec['right']['height_mm']:+.1f} mm",
           (r_col, r_row), xytext=(r_col + 25, r_row - 40), color="white", fontsize=9,
           arrowprops=dict(arrowstyle="->", color="white", lw=1))
ax.annotate(f"L crest apex\n{rec['left']['level']}, {rec['left']['height_mm']:+.1f} mm",
           (l_col, l_row), xytext=(l_col - 75, l_row - 40), color="white", fontsize=9,
           ha="left",
           arrowprops=dict(arrowstyle="->", color="white", lw=1))
ax.text(col_hi - 10, disc_row - 6, "L4-L5 disc reference", color="#FFD54F",
       fontsize=9, ha="right", va="bottom")

ax.text(col_lo + 8, row_hi - 8, left_lbl, color="white", fontsize=13, fontweight="bold",
       ha="left", va="bottom")
ax.text(col_hi - 8, row_hi - 8, right_lbl, color="white", fontsize=13, fontweight="bold",
       ha="right", va="bottom")

ax.set_title(f"Case {CASE_ID}: Iliac Crest Height vs. L4-L5 (coronal bone-window MIP)\n"
            f"Obliquity = {rec['obliquity_mm']:.1f} mm", fontsize=11)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, facecolor="white")
print(f"wrote -> {OUT_PATH}")
