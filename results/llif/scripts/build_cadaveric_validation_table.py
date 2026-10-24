"""Compare the full-802 LLIF cohort's population distributions against
cadaveric (and one same-modality CT) literature values in
validation/cadaveric_references.json.

This is a POPULATION-LEVEL plausibility check, not per-case validation: the
cadavers are different people from our patients, so agreement means our
cohort's distribution is anatomically plausible and disagreement points at
a systematic bias -- neither says any single case was measured correctly
(that's what the n=10 manual ITK-SNAP comparison in results/crest-height/
is for). Reference numbers are read from the JSON (each with its own
verification trail), never hard-coded here; cohort numbers are recomputed
from llif_report_full802.jsonl every run."""
from __future__ import annotations

import csv
import json
import math
import os

import numpy as np

HERE = os.path.dirname(__file__)
REPORT_PATH = os.path.join(HERE, "..", "data", "llif_report_full802.jsonl")
REFS_PATH = os.path.join(HERE, "..", "validation", "cadaveric_references.json")
OUT_CSV = os.path.join(HERE, "..", "validation", "cadaveric_validation.csv")
OUT_MD = os.path.join(HERE, "..", "validation", "cadaveric_validation.md")

PINNED_REVISION = "9f69480a572624029a927bc6661309a068f1f2c3"
MANIFEST = ("/Users/ashleyschehr/.cache/huggingface/hub/"
            "datasets--anonymous-mlhc--CTSpinoPelvic1K/snapshots/" + PINNED_REVISION
            + "/manifest.json")


def _get(rec, path):
    for k in path:
        rec = rec.get(k) if isinstance(rec, dict) else None
    return rec


def _qc_ok(rec, path) -> bool:
    """Only use values whose owning sub-report passed its own QC -- the
    nearest ancestor carrying qc_flags (e.g. disc_footprint, or
    vertebral_wedging.L4 for per-level sub-reports)."""
    for i in range(len(path) - 1, 0, -1):
        flags = _get(rec, list(path[:i]) + ["qc_flags"])
        if flags is not None:
            return flags == ["ok"]
    return False


def _value(rec, spec):
    if spec == "max(disc_height.anterior_mm, middle_mm, posterior_mm)":
        if not _qc_ok(rec, ["disc_height", "anterior_mm"]):
            return None
        vals = [_get(rec, ["disc_height", k]) for k in ("anterior_mm", "middle_mm", "posterior_mm")]
        return max(vals) if all(isinstance(v, (int, float)) for v in vals) else None
    if not _qc_ok(rec, spec):
        return None
    v = _get(rec, spec)
    return float(v) if isinstance(v, (int, float)) else None


def _ref_sd(src, dispersion_value):
    """SD for a standardized difference -- only where the dispersion type is
    VERIFIED (SEM is converted with the source's per-level n); None when the
    source doesn't say SD vs SEM, rather than guessing."""
    if src["dispersion"] == "SD":
        return dispersion_value
    if src["dispersion"] == "SEM":
        return dispersion_value * math.sqrt(src["n_per_level"])
    return None


def _smd_label(d):
    # Cohen's conventional benchmarks -- a reporting convention, not a
    # validated clinical threshold.
    a = abs(d)
    return "negligible" if a < 0.2 else "small" if a < 0.5 else "medium" if a < 0.8 else "large"


def _stats(v):
    v = np.asarray(v, float)
    return len(v), float(v.mean()), float(v.std(ddof=1))


def main() -> None:
    records = [json.loads(l) for l in open(REPORT_PATH, encoding="utf-8") if l.strip()]
    records = [r for r in records if "error" not in r]
    sex = {}
    if os.path.exists(MANIFEST):
        sex = {c["volume_id"]: c.get("sex") for c in json.load(open(MANIFEST, encoding="utf-8"))}
    refs = json.load(open(REFS_PATH, encoding="utf-8"))
    sources = refs["sources"]

    rows = []
    for ref in refs["references"]:
        vals = [(r["case_id"], _value(r, ref["our_path"])) for r in records]
        vals = [(cid, v) for cid, v in vals if v is not None]
        if len(vals) < 2:
            raise SystemExit(f"no QC-passing values for {ref['parameter']} at {ref['our_path']}")
        n, mean, sd = _stats([v for _, v in vals])
        by_sex = {s: [v for cid, v in vals if sex.get(cid) == s] for s in ("male", "female")}
        for rv in ref["values"]:
            src = sources[rv["source"]]
            rsd = _ref_sd(src, rv["dispersion_value"])
            diff = mean - rv["mean"]
            smd = diff / math.sqrt((sd ** 2 + rsd ** 2) / 2) if rsd is not None else None
            row = {
                "parameter": ref["parameter"], "units": ref["units"],
                "source": src["citation"].split(".")[0].split(",")[0] + " " + rv["source"][-4:],
                "specimen": src["specimen"], "verification_level": rv["verification_level"],
                "verified_against": rv["verified_against"],
                "ref_mean": rv["mean"],
                "ref_dispersion": f"{rv['dispersion_value']} ({src['dispersion']})",
                "ref_sd_used": round(rsd, 2) if rsd is not None else "",
                "ours_n": n, "ours_mean": round(mean, 2), "ours_sd": round(sd, 2),
                "diff": round(diff, 2), "pct_diff": round(100 * diff / rv["mean"], 1),
                "smd": round(smd, 2) if smd is not None else "",
                "smd_label": _smd_label(smd) if smd is not None else "",
                "comparability": ref["comparability"],
            }
            for s in ("male", "female"):
                if s in rv and len(by_sex[s]) > 1:
                    ns, ms, _ = _stats(by_sex[s])
                    row[f"{s}_diff"] = round(ms - rv[s][0], 2)
                    row[f"{s}_ours_vs_ref"] = f"{ms:.1f} (n={ns}) vs {rv[s][0]}"
                else:
                    row[f"{s}_diff"] = ""
                    row[f"{s}_ours_vs_ref"] = ""
            rows.append(row)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    cols = ["parameter", "source", "specimen", "verification_level", "ref_mean", "ref_dispersion",
            "ours_mean", "ours_sd", "ours_n", "diff", "pct_diff", "smd", "smd_label",
            "male_ours_vs_ref", "female_ours_vs_ref", "comparability"]
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("<!-- generated by scripts/build_cadaveric_validation_table.py; do not edit by hand -->\n")
        fh.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            fh.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")
    print(f"DONE -> {OUT_CSV}\n        {OUT_MD}")
    for r in rows:
        print(f"  {r['parameter']:32s} {r['source']:22s} ref={r['ref_mean']:6} ours={r['ours_mean']:6} "
              f"({r['pct_diff']:+5.1f}%) smd={r['smd']!s:6} {r['smd_label']}")


if __name__ == "__main__":
    main()
