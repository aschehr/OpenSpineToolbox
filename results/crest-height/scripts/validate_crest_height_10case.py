"""Validation statistics for the 10-case crest-height run against real manual
ITK-SNAP measurements (provided by the user in-conversation, not fabricated).

Reads the ACTUAL automated output file written by run_crest_height_10case.py
(crest_height_10case.jsonl) -- does not recompute or assume any automated
value here, only loads what the pipeline already wrote to disk.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(__file__)
AUTO_PATH = os.path.join(HERE, "crest_height_10case.jsonl")

# Real manual ITK-SNAP measurements, as given by the user.
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

# --- load the ACTUAL automated output (not recomputed here) -----------------
auto = {}
with open(AUTO_PATH, "r", encoding="utf-8") as fh:
    for line in fh:
        rec = json.loads(line)
        auto[rec["case_id"]] = rec

case_ids = list(MANUAL.keys())
assert set(case_ids) == set(auto.keys()), "case set mismatch between manual and automated files"

# =============================================================================
# 1. Continuous: obliquity (mm) -- ICC(2,1), Pearson r, mean bias + 95% CI
# =============================================================================
auto_obl = np.array([auto[c]["obliquity_mm"] for c in case_ids], dtype=float)
man_obl = np.array([MANUAL[c]["obliquity_mm"] for c in case_ids], dtype=float)
n = len(case_ids)

print("=" * 70)
print("OBLIQUITY (mm) -- paired values actually compared")
print("=" * 70)
for c, a, m in zip(case_ids, auto_obl, man_obl):
    print(f"  {c}: auto={a:6.2f}  manual={m:6.3f}  diff={a - m:+6.2f}")

# --- Pearson r ---------------------------------------------------------------
r, p = stats.pearsonr(auto_obl, man_obl)
print(f"\nPearson r = {r:.4f}  (p = {p:.4g}, df = {n - 2})")

# --- mean bias + 95% CI (paired-difference t-interval) -----------------------
diffs = auto_obl - man_obl
mean_bias = diffs.mean()
sd_diff = diffs.std(ddof=1)
se_diff = sd_diff / np.sqrt(n)
tcrit = stats.t.ppf(0.975, df=n - 1)
ci_lo, ci_hi = mean_bias - tcrit * se_diff, mean_bias + tcrit * se_diff
print(f"Mean bias (auto - manual) = {mean_bias:.3f} mm")
print(f"  sd(diff) = {sd_diff:.3f} mm, se = {se_diff:.3f} mm, t_crit(df={n-1}) = {tcrit:.3f}")
print(f"  95% CI = [{ci_lo:.3f}, {ci_hi:.3f}] mm")

# --- ICC(2,1): two-way random, single measures, absolute agreement -----------
# Shrout & Fleiss (1979) formula via a subjects x raters (n x 2) ANOVA.
X = np.column_stack([auto_obl, man_obl])  # (n subjects, k=2 raters)
k = X.shape[1]
grand_mean = X.mean()
row_means = X.mean(axis=1)
col_means = X.mean(axis=0)

SST = np.sum((X - grand_mean) ** 2)
SSR = k * np.sum((row_means - grand_mean) ** 2)
SSC = n * np.sum((col_means - grand_mean) ** 2)
SSE = SST - SSR - SSC

MSR = SSR / (n - 1)
MSC = SSC / (k - 1)
MSE = SSE / ((n - 1) * (k - 1))

icc21 = (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)
print(f"\nICC(2,1) [two-way random, absolute agreement, single measures]:")
print(f"  MSR={MSR:.4f} MSC={MSC:.4f} MSE={MSE:.4f}")
print(f"  ICC(2,1) = {icc21:.4f}")

# =============================================================================
# 2. Categorical: per-side level (L4 / L4-L5 / L5) -- Cohen's kappa
# =============================================================================
auto_levels, man_levels = [], []
for c in case_ids:
    for side, key in (("right", "R"), ("left", "L")):
        auto_levels.append(auto[c][side]["level"])
        man_levels.append(MANUAL[c][key])

print("\n" + "=" * 70)
print("LEVEL CLASSIFICATION (per side, 20 pairs) -- paired labels compared")
print("=" * 70)
for c in case_ids:
    for side, key in (("right", "R"), ("left", "L")):
        a = auto[c][side]["level"]
        m = MANUAL[c][key]
        flag = "" if a == m else "  <-- MISMATCH"
        print(f"  {c} {side:5s}: auto={a:6s} manual={m:6s}{flag}")

labels = sorted(set(auto_levels) | set(man_levels))
idx = {lab: i for i, lab in enumerate(labels)}
n_pairs = len(auto_levels)
conf = np.zeros((len(labels), len(labels)), dtype=int)
for a, m in zip(auto_levels, man_levels):
    conf[idx[a], idx[m]] += 1

print(f"\nConfusion matrix (rows=automated, cols=manual), labels={labels}:")
print(conf)

po = np.trace(conf) / n_pairs  # observed agreement
row_totals = conf.sum(axis=1)
col_totals = conf.sum(axis=0)
pe = np.sum(row_totals * col_totals) / (n_pairs ** 2)  # expected agreement by chance
kappa = (po - pe) / (1 - pe) if pe != 1 else float("nan")

print(f"\nObserved agreement p_o = {po:.4f}  ({int(np.trace(conf))}/{n_pairs})")
print(f"Expected (chance) agreement p_e = {pe:.4f}")
print(f"Cohen's kappa = {kappa:.4f}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"n cases = {n}, n side-level pairs = {n_pairs}")
print(f"Pearson r (obliquity)        = {r:.4f}  (p={p:.4g})")
print(f"ICC(2,1) (obliquity)         = {icc21:.4f}")
print(f"Mean bias (obliquity)        = {mean_bias:.3f} mm  [95% CI {ci_lo:.3f}, {ci_hi:.3f}]")
print(f"Cohen's kappa (level, n=20)  = {kappa:.4f}")
