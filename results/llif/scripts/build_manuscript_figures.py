"""Build the LLIF manuscript figures in standard journal format, with a provenance trail.

Conventions followed (common to Spine, Eur Spine J, Global Spine J, Radiology author guides):
  - no chart titles inside figures (the legend carries the title); bold panel letters A, B
  - Arial 8 pt text; double-column width 174 mm (6.85 in)
  - 600-dpi TIFF (LZW) for submission + vector PDF; RGB, greyscale-safe palette
  - images in radiological convention (patient right on the viewer's left) with a mm scale bar
  - Bland-Altman with bias and 95% limits of agreement; 3-set Venn for flag overlap;
    forest-style plot with numeric columns for the reference comparison

Transparency: every number is recomputed from repo data files at run time. The script also writes
  figures/source_data/FigureN.csv  - the exact values plotted in each figure
  figures/provenance.json          - SHA-256 of every input file, git commit, package versions, counts
See figures/README.md for what each figure shows and how it is computed.

Run: python results/llif/scripts/build_manuscript_figures.py
"""
import csv
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results/crest-height/scripts"))

LLIF = ROOT / "results/llif"
CREST = ROOT / "results/crest-height"
OUT = LLIF / "manuscript/figures"
SRC = OUT / "source_data"
SRC.mkdir(parents=True, exist_ok=True)
SNAP = Path.home() / (".cache/huggingface/hub/datasets--anonymous-mlhc--CTSpinoPelvic1K/snapshots/"
                      "9f69480a572624029a927bc6661309a068f1f2c3")
EXAMPLE_CASE = "0184"

INPUTS = {
    "llif_report": LLIF / "data/llif_report_full802.jsonl",
    "vertebral_hu": LLIF / "data/vertebral_hu_full802.jsonl",
    "crest_full": CREST / "data/crest_height_full802.jsonl",
    "manual_validation": CREST / "tables/table1_validation.csv",
    "morphometry": LLIF / "validation/cadaveric_validation.csv",
    "example_ct": SNAP / f"ct/{EXAMPLE_CASE}_ct.nii.gz",
}

MM = 1 / 25.4
DOUBLE_COL = 174 * MM
BLACK, GREY, LIGHT, ACCENT = "#000000", "#8c8c8c", "#d0d0d0", "#1f5fa8"
plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
})
provenance = {"figures": {}}


def save(fig, name, notes):
    for ext, kw in (("tif", {"pil_kwargs": {"compression": "tiff_lzw"}}), ("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02, **kw)
    plt.close(fig)
    provenance["figures"][name] = notes


def write_csv(name, header, rows):
    with open(SRC / f"{name}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def panel(ax, letter, x=-0.14):
    ax.text(x, 1.02, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def load_jsonl(p):
    return {(r := json.loads(l))["case_id"]: r for l in open(p) if l.strip()}


ok = lambda d: bool(d) and d.get("qc_flags") == ["ok"]
R = load_jsonl(INPUTS["llif_report"])
H = load_jsonl(INPUTS["vertebral_hu"])

# ---------------------------------------------------------------- Figure 1
from ostk.io import load_ct  # noqa: E402
from figure_utils import axis_mapping, coronal_mip  # noqa: E402

crest = load_jsonl(INPUTS["crest_full"])[EXAMPLE_CASE]
ct, aff = load_ct(str(INPUTS["example_ct"]))
img, w2p, left_lbl, right_lbl = coronal_mip(ct, aff)
ax_for_world, _ = axis_mapping(aff)
sp_lr = float(np.linalg.norm(aff[:3, ax_for_world[0]]))
sp_si = float(np.linalg.norm(aff[:3, ax_for_world[2]]))
flip = right_lbl == "R"  # image currently has patient right on viewer's right -> mirror it
if flip:
    img = img[:, ::-1]
W = img.shape[1]


def to_mm(world):
    c, r = w2p(world)
    if flip:
        c = W - 1 - c
    return c * sp_lr, r * sp_si


lm = crest["landmarks_world_mm"]
rx, ry = to_mm(lm["right_crest_apex"])
lx, ly = to_mm(lm["left_crest_apex"])
_, dy = to_mm([0.0, 0.0, lm["l4_l5_disc_ref_z"][0]])
cx = (rx + lx) / 2
x0, x1 = max(cx - 190, 0.0), min(cx + 190, W * sp_lr)
y0, y1 = min(ry, ly, dy) - 120, max(ry, ly, dy) + 90
fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.62, DOUBLE_COL * 0.62 * (y1 - y0) / (x1 - x0)))
ax.imshow(img, cmap="gray", vmin=200, vmax=1500, extent=[0, W * sp_lr, img.shape[0] * sp_si, 0],
          interpolation="lanczos")
ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
ax.axhline(dy, color="#ffd54f", lw=1.0, ls=(0, (4, 3)))
for (x, y, side) in ((rx, ry, "right"), (lx, ly, "left")):
    ax.plot(x, y, "o", ms=5, mfc="#e53935", mec="white", mew=0.8, zorder=5)
    s = crest[side]
    on_left = x < cx
    tx = x0 + 8 if on_left else x1 - 8
    ax.annotate(f"{side.capitalize()} crest apex\n{s['level']}, {s['height_mm']:+.1f} mm", (x, y),
                xytext=(tx, y - 55), color="white", fontsize=7, ha="left" if on_left else "right", va="bottom",
                arrowprops=dict(arrowstyle="-", color="white", lw=0.6, shrinkA=0, shrinkB=4))
ax.text(x0 + 8, dy + 4, "L4-5 disc level", color="#ffd54f", fontsize=7, ha="left", va="top")
ax.text(x0 + 8, y1 - 8, "R", color="white", fontsize=9, fontweight="bold", ha="left", va="bottom")
ax.text(x1 - 8, y1 - 8, "L", color="white", fontsize=9, fontweight="bold", ha="right", va="bottom")
ax.plot([cx - 25, cx + 25], [y1 - 10, y1 - 10], color="white", lw=1.5)
ax.text(cx, y1 - 14, "50 mm", color="white", fontsize=7, ha="center", va="bottom")
ax.axis("off")
write_csv("Figure1", ["landmark", "level", "height_above_disc_ref_mm", "world_x", "world_y", "world_z"], [
    ["right_crest_apex", crest["right"]["level"], crest["right"]["height_mm"], *lm["right_crest_apex"]],
    ["left_crest_apex", crest["left"]["level"], crest["left"]["height_mm"], *lm["left_crest_apex"]],
    ["l4_l5_disc_ref", "", "", "", "", lm["l4_l5_disc_ref_z"][0]],
])
save(fig, "Figure1", {"case": EXAMPLE_CASE, "view": "coronal bone-window MIP (200-1500 HU), radiological orientation",
                      "pixel_spacing_mm": {"lr": sp_lr, "si": sp_si}, "obliquity_mm": crest["obliquity_mm"]})

# ---------------------------------------------------------------- Figure 2
levels, lvl_rows = Counter(), []
for cid, r in R.items():
    c = r["crest_height"]
    if ok(c):
        for side in ("right", "left"):
            levels[c[side]["level"]] += 1
            lvl_rows.append([cid, side, c[side]["level"]])
yang_rows = [[cid, r["yang_crest_height"]["value"]] for cid, r in R.items() if ok(r["yang_crest_height"])]
yang = np.array([v for _, v in yang_rows])
order, names = ["L4", "L4-L5", "L5"], {"L4": "L4 body", "L4-L5": "L4-5 disc", "L5": "L5 body"}
tot = sum(levels.values())
fig, (a1, a2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.36), gridspec_kw={"width_ratios": [1, 1.7]})
vals = [levels[k] for k in order]
a1.bar(range(3), [100 * v / tot for v in vals], color=GREY, width=0.6)
for i, v in enumerate(vals):
    a1.text(i, 100 * v / tot + 1.2, f"{100 * v / tot:.1f}%\n(n = {v})", ha="center", va="bottom", fontsize=7)
a1.set_xticks(range(3), [names[k] for k in order])
a1.set_ylabel("Sides (%)")
a1.set_ylim(0, 60)
a1.set_xlabel("Level of iliac crest apex")
panel(a1, "A", -0.22)
lo, hi = -40, 44
shown = yang[(yang >= lo) & (yang <= hi)]
bins = np.arange(lo, hi + 1, 2.0)  # 12 mm falls on a bin edge
_, edges, patches = a2.hist(shown, bins=bins, color=GREY, edgecolor="white", linewidth=0.3)
for p, left in zip(patches, edges[:-1]):
    if left >= 12:
        p.set_facecolor(ACCENT)
a2.axvline(12, color=BLACK, ls="--", lw=0.8)
n_above = int((yang > 12).sum())
a2.text(13.2, a2.get_ylim()[1] * 0.92, f"> 12 mm: {n_above}/{len(yang)} ({100 * n_above / len(yang):.1f}%)",
        fontsize=7, va="top", color=ACCENT)
a2.set_xlabel("Crest height above L4-5 ventral disc midpoint (mm)")
a2.set_ylabel("Patients (n)")
panel(a2, "B", -0.12)
fig.tight_layout(w_pad=2.5)
write_csv("Figure2A", ["case_id", "side", "crest_apex_level"], lvl_rows)
write_csv("Figure2B", ["case_id", "yang_crest_height_mm"], yang_rows)
save(fig, "Figure2", {"A_sides": tot, "A_counts": dict(levels), "B_n": len(yang), "B_above_12mm": n_above,
                      "B_outside_axis_not_drawn": int(len(yang) - len(shown)), "cutoff_rule": "value > 12.0 mm"})

# ---------------------------------------------------------------- Figure 3
rows3 = list(csv.DictReader(open(INPUTS["manual_validation"])))
man = np.array([float(r["Manual Obliquity (mm)"]) for r in rows3])
aut = np.array([float(r["Automated Obliquity (mm)"]) for r in rows3])
diff, mean = aut - man, (aut + man) / 2
bias, sd = diff.mean(), diff.std(ddof=1)
loa = (bias - 1.96 * sd, bias + 1.96 * sd)
rr = float(np.corrcoef(man, aut)[0, 1])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.40))
lim = (0, 21)
a1.plot(lim, lim, color=GREY, lw=0.6, ls="--", zorder=0)
a1.scatter(man, aut, s=14, facecolor=BLACK, edgecolor="none", zorder=2)
a1.text(0.04, 0.96, f"r = {rr:.3f}\nICC(2,1) = 0.989", transform=a1.transAxes, va="top", fontsize=7)
a1.set_xlim(lim); a1.set_ylim(lim); a1.set_aspect("equal")
a1.set_xticks(range(0, 21, 5)); a1.set_yticks(range(0, 21, 5))
a1.set_xlabel("Manual obliquity (mm)"); a1.set_ylabel("Automated obliquity (mm)")
panel(a1, "A", -0.2)
a2.scatter(mean, diff, s=14, facecolor=BLACK, edgecolor="none", zorder=2)
xmax = 24
for y, lab, style in ((bias, f"Mean {bias:+.2f}", "-"), (loa[1], f"+1.96 SD {loa[1]:+.2f}", "--"),
                      (loa[0], f"−1.96 SD {loa[0]:+.2f}", "--")):
    a2.axhline(y, color=BLACK, ls=style, lw=0.7)
    a2.text(xmax, y + 0.05, lab, fontsize=7, ha="right", va="bottom")
a2.axhline(0, color=LIGHT, lw=0.6, zorder=0)
a2.set_xlim(0, xmax); a2.set_ylim(-3.2, 2.0)
a2.set_xlabel("Mean of automated and manual (mm)")
a2.set_ylabel("Automated − manual (mm)")
panel(a2, "B", -0.2)
fig.tight_layout(w_pad=3)
write_csv("Figure3", ["case_id", "manual_obliquity_mm", "automated_obliquity_mm", "mean_mm", "difference_mm"],
          [[r["Case"], m, a, (m + a) / 2, a - m] for r, m, a in zip(rows3, man, aut)])
save(fig, "Figure3", {"n": len(man), "pearson_r": rr, "bias_mm": bias, "sd_diff_mm": sd, "loa_mm": loa,
                      "icc_source": "results/crest-height/tables/table2_summary_stats.csv"})

# ---------------------------------------------------------------- Figure 4 (3-set Venn)
combos, per_case = Counter(), []
for cid, r in R.items():
    y, p = r["yang_crest_height"], r["pi_ll"]
    z = (H.get(cid) or {}).get("zhou_l1_l4_bmd") or {}
    if not (ok(y) and ok(p) and ok(z)):
        continue
    k = (y["value"] > 12, p["schwab"]["PI-LL"] in ("+", "++"), z["value"] < 115.7)
    combos[k] += 1
    per_case.append([cid, *[int(f) for f in k]])
n_eval = sum(combos.values())
fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.55, DOUBLE_COL * 0.50))
centres = {0: (-0.55, 0.32), 1: (0.55, 0.32), 2: (0.0, -0.58)}
labels = ["Crest height > 12 mm", "PI-LL ≥ 10°", "L1-L4 HU < 115.7"]
for i, (x, y) in centres.items():
    ax.add_patch(mpatches.Circle((x, y), 1.0, facecolor=GREY, alpha=0.18, edgecolor=BLACK, lw=0.7))
    total = sum(c for k, c in combos.items() if k[i])
    lx, ly = {0: (-1.25, 1.45), 1: (1.25, 1.45), 2: (0, -1.78)}[i]
    ax.text(lx, ly, f"{labels[i]}\n(n = {total})", ha="center", va="center", fontsize=7, fontweight="bold")
pos = {(1, 0, 0): (-1.0, 0.55), (0, 1, 0): (1.0, 0.55), (0, 0, 1): (0, -1.1), (1, 1, 0): (0, 0.82),
       (1, 0, 1): (-0.62, -0.42), (0, 1, 1): (0.62, -0.42), (1, 1, 1): (0, 0.0)}
for k, (x, y) in pos.items():
    c = combos[tuple(bool(v) for v in k)]
    ax.text(x, y, f"{c}\n({100 * c / n_eval:.1f}%)", ha="center", va="center", fontsize=7)
none = combos[(False, False, False)]
ax.text(1.88, -1.88, f"None: {none} ({100 * none / n_eval:.1f}%)\nTotal: {n_eval}", ha="right", va="center", fontsize=7)
ax.set_xlim(-1.9, 1.9); ax.set_ylim(-2.0, 1.75); ax.set_aspect("equal"); ax.axis("off")
write_csv("Figure4", ["case_id", "crest_gt_12mm", "pi_ll_ge_10deg", "l1_l4_hu_lt_115_7"], per_case)
save(fig, "Figure4", {"n_evaluable": n_eval, "regions": {"".join(str(int(v)) for v in k): c for k, c in combos.items()},
                      "region_key": "crest,pi_ll,hu (1 = flagged)"})

# ---------------------------------------------------------------- Figure 5 (forest-style)
mrows = [r for r in csv.DictReader(open(INPUTS["morphometry"])) if r["pct_diff"]]
short = lambda s: s.split(" ")[0] + " " + s.split(" ")[-1]
n = len(mrows)
fig = plt.figure(figsize=(DOUBLE_COL, 0.17 * n + 0.6))
ax = fig.add_axes([0.40, 0.08, 0.30, 0.86])
ax.axvspan(-10, 10, color=LIGHT, alpha=0.35, lw=0, zorder=0)
ax.axvline(0, color=BLACK, lw=0.7)
ax.set_ylim(-0.7, n - 0.3)
fig.canvas.draw()
yfig = lambda y: fig.transFigure.inverted().transform(ax.transData.transform((0, y)))[1]
for i, r in enumerate(mrows):
    y = n - 1 - i
    prim = r["verification_level"] == "primary"
    poor = r["comparability"] != "good"
    col = GREY if poor else BLACK
    ax.plot(float(r["pct_diff"]), y, "s" if prim else "o", ms=4, mfc=col if prim else "white", mec=col, mew=0.8)
    fig.text(0.01, yfig(y), f"{r['parameter']}  ({short(r['source'])})", fontsize=6.5, va="center")
    for xf, txt in ((0.75, r["ref_mean"]), (0.83, f"{float(r['ours_mean']):.1f} ± {float(r['ours_sd']):.1f}"),
                    (0.91, f"{float(r['pct_diff']):+.1f}"), (0.98, r["smd"] or "–")):
        fig.text(xf, yfig(y), txt, fontsize=6.5, va="center", ha="center")
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Difference from reference mean (%)")
for xf, txt in ((0.01, "Parameter (reference)"), (0.75, "Reference\nmean"), (0.83, "Ours\nmean ± SD"),
                (0.91, "Difference\n(%)"), (0.98, "SMD")):
    fig.text(xf, 0.955, txt, fontsize=6.5, fontweight="bold", va="bottom", ha="left" if xf < 0.1 else "center")
write_csv("Figure5", ["parameter", "source", "verification_level", "comparability", "ref_mean", "ours_mean",
                      "ours_sd", "ours_n", "pct_diff", "smd"],
          [[r[k] for k in ("parameter", "source", "verification_level", "comparability", "ref_mean", "ours_mean",
                           "ours_sd", "ours_n", "pct_diff", "smd")] for r in mrows])
save(fig, "Figure5", {"rows": n, "source": str(INPUTS["morphometry"].relative_to(ROOT))})


# ---------------------------------------------------------------- provenance
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def show(p):
    return str(p.relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p).replace(str(Path.home()), "~")


import matplotlib as _m, nibabel as _nb, PIL as _pil, scipy as _sp  # noqa: E402
provenance.update({
    "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "script": show(Path(__file__)),
    "git_commit": git("rev-parse", "HEAD"),
    "git_uncommitted_changes": bool(git("status", "--porcelain")),
    "dataset": "CTSpinoPelvic1K v4, Hugging Face revision 9f69480a572624029a927bc6661309a068f1f2c3",
    "inputs": {k: {"path": show(p), "sha256": sha256(p)} for k, p in INPUTS.items()},
    "software": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": _m.__version__,
                 "scipy": _sp.__version__, "nibabel": _nb.__version__, "pillow": _pil.__version__},
})
json.dump(provenance, open(OUT / "provenance.json", "w"), indent=2, default=float)
print(json.dumps(provenance["figures"], indent=1, default=float))
