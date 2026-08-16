"""Figure 4: three-panel comparison, one representative example each of L4
body, L4-L5 disc space, and L5 body classification. Cases chosen as the
closest-to-median height within each category (data-driven, not
eyeballed), excluding the known boundary case 0368."""
import json
import os
import sys

sys.path.insert(0, "/Users/ashleyschehr/OpenSpineToolbox")
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ostk.io import load_ct
from figure_utils import coronal_mip

HERE = os.path.dirname(__file__)
SNAP = ("/Users/ashleyschehr/.cache/huggingface/hub/"
        "datasets--anonymous-mlhc--CTSpinoPelvic1K/snapshots/"
        "9f69480a572624029a927bc6661309a068f1f2c3")
OUT_PATH = os.path.join(HERE, "fig4_three_panel.png")

# (case_id, side, panel label) -- picked as closest-to-median height for
# each category from the real full-cohort file (see selection step earlier).
PANELS = [("0056", "right", "L4 body"), ("0107", "right", "L4-L5 disc space"),
         ("0081", "left", "L5 body")]

recs = {json.loads(l)["case_id"]: json.loads(l)
        for l in open(os.path.join(HERE, "crest_height_full802.jsonl")) if l.strip()}

plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "figure.facecolor": "white"})
fig, axes = plt.subplots(1, 3, figsize=(15, 7), dpi=300)

for ax, (case_id, side, panel_label) in zip(axes, PANELS):
    rec = recs[case_id]
    ct, aff = load_ct(os.path.join(SNAP, "ct", f"{case_id}_ct.nii.gz"))
    img, w2p, left_lbl, right_lbl = coronal_mip(ct, aff)

    r_apex = rec["landmarks_world_mm"]["right_crest_apex"]
    l_apex = rec["landmarks_world_mm"]["left_crest_apex"]
    disc_z = rec["landmarks_world_mm"]["l4_l5_disc_ref_z"][0]
    r_col, r_row = w2p(r_apex)
    l_col, l_row = w2p(l_apex)
    _, disc_row = w2p([0.0, 0.0, disc_z])

    focus_col, focus_row = (r_col, r_row) if side == "right" else (l_col, l_row)
    spine_col = (r_col + l_col) / 2
    pad_rows = 110
    row_lo = int(max(0, min(focus_row, disc_row) - pad_rows))
    row_hi = int(min(img.shape[0], max(focus_row, disc_row) + pad_rows))
    # symmetric window around the midline that's guaranteed to include the
    # focus point plus a fixed margin, so all three panels share one scale
    half_span = abs(focus_col - spine_col) + 180
    col_lo = int(max(0, spine_col - half_span))
    col_hi = int(min(img.shape[1], spine_col + half_span))

    ax.imshow(img[row_lo:row_hi, col_lo:col_hi], cmap="gray", vmin=300, vmax=1400,
             extent=[col_lo, col_hi, row_hi, row_lo], aspect="equal")
    ax.plot(focus_col, focus_row, marker="o", markersize=11, markerfacecolor="#DD2C2C",
           markeredgecolor="white", markeredgewidth=1.3, zorder=5)
    ax.axhline(disc_row, xmin=0.05, xmax=0.95, color="#FFD54F", linewidth=1.8,
              linestyle="--", zorder=4)

    height = rec[side]["height_mm"]
    ax.set_title(f"{panel_label}\ncase {case_id} ({side}), {height:+.1f} mm", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

fig.suptitle("Representative Crest-Height Classification Examples", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, facecolor="white", bbox_inches="tight")
print(f"wrote -> {OUT_PATH}")
