"""Two-reader validation of automated iliac crest obliquity (10 cases, 20 sides).

Source: OneDrive "manual iliac crest level and LLIF parameter measurements.xlsx"
(Ashley Schehr and Jerick Kim, independent ITK-SNAP reads; automated values from
the crest-height pipeline). Values are transcribed in crest_measurements.csv.
"""
import csv, json
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).parent
rows = list(csv.DictReader(open(HERE / "crest_measurements.csv")))
A = np.array([float(r["ashley_obliquity_mm"]) for r in rows])
J = np.array([float(r["jerick_obliquity_mm"]) for r in rows])
M = np.array([float(r["automated_obliquity_mm"]) for r in rows])


def icc_2_1(*raters, alpha=0.05):
    """ICC(2,1): two-way random effects, absolute agreement, single rater, with the
    McGraw & Wong (1996) F-based 95% CI."""
    Y = np.column_stack(raters); n, k = Y.shape
    gm = Y.mean()
    msr = k * ((Y.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((Y.mean(0) - gm) ** 2).sum() / (k - 1)
    mse = ((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    icc = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    a, b = k * icc / (n * (1 - icc)), 1 + k * icc * (n - 1) / (n * (1 - icc))
    v = (a * msc + b * mse) ** 2 / ((a * msc) ** 2 / (k - 1) + (b * mse) ** 2 / ((n - 1) * (k - 1)))
    fl, fu = stats.f.ppf(1 - alpha / 2, n - 1, v), stats.f.ppf(1 - alpha / 2, v, n - 1)
    lo = n * (msr - fl * mse) / (fl * (k * msc + (k * n - k - n) * mse) + n * msr)
    hi = n * (fu * msr - mse) / (k * msc + (k * n - k - n) * mse + n * fu * msr)
    return float(icc), (float(lo), float(hi))


def bland_altman(x, y):
    """Bias and 95% limits of agreement for y - x, with a t-based 95% CI on the bias."""
    d = y - x; bias, sd = d.mean(), d.std(ddof=1); n = len(d)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return {"bias_mm": float(bias), "bias_ci_mm": [float(bias - half), float(bias + half)], "sd_mm": float(sd),
            "loa_mm": [float(bias - 1.96 * sd), float(bias + 1.96 * sd)],
            "mean_abs_diff_mm": float(np.abs(d).mean()), "max_abs_diff_mm": float(np.abs(d).max()),
            "pearson_r": float(np.corrcoef(x, y)[0, 1])}


def kappa(a, b):
    cats = sorted(set(a) | set(b)); n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return {"agree": sum(x == y for x, y in zip(a, b)), "n": n, "kappa": float((po - pe) / (1 - pe)) if pe < 1 else 1.0}


lv = lambda who: [r[f"{who}_{s}"] for r in rows for s in ("right_level", "left_level")]
out = {"n_cases": len(rows), "n_sides": 2 * len(rows)}
for name, (x, y) in {"ashley_vs_jerick": (A, J), "automated_vs_ashley": (A, M), "automated_vs_jerick": (J, M),
                     "automated_vs_reader_mean": ((A + J) / 2, M)}.items():
    icc, ci = icc_2_1(x, y)
    out[name] = {"icc_2_1": icc, "icc_ci": ci, **bland_altman(x, y)}
icc3, ci3 = icc_2_1(A, J, M)
out["all_three_icc_2_1"] = {"icc_2_1": icc3, "icc_ci": ci3}
out["level_ashley_vs_jerick"] = kappa(lv("ashley"), lv("jerick"))
out["level_automated_vs_ashley"] = kappa(lv("ashley"), lv("automated"))
out["level_automated_vs_jerick"] = kappa(lv("jerick"), lv("automated"))
json.dump(out, open(HERE / "interrater_stats.json", "w"), indent=2)
print(json.dumps(out, indent=2))
