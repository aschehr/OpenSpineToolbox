"""Demographics pull straight from the pinned snapshot's manifest.json --
no NIfTI loading, just the dataset's own per-case metadata records."""
from __future__ import annotations

import json
import os
import statistics as stats

PINNED_REVISION = "9f69480a572624029a927bc6661309a068f1f2c3"
SNAPSHOT = (
    "/Users/ashleyschehr/.cache/huggingface/hub/"
    "datasets--anonymous-mlhc--CTSpinoPelvic1K/snapshots/" + PINNED_REVISION
)
MANIFEST = os.path.join(SNAPSHOT, "manifest.json")
OUT_PATH = os.path.join(os.path.dirname(__file__), "demographics_802case.json")


def main() -> None:
    with open(MANIFEST, "r", encoding="utf-8") as fh:
        cases = json.load(fh)

    ages = [c["age"] for c in cases if c.get("age") is not None]
    sexes = [c["sex"] for c in cases if c.get("sex") is not None]
    age_bands = [c["age_band"] for c in cases if c.get("age_band") is not None]

    sex_counts = {}
    for s in sexes:
        sex_counts[s] = sex_counts.get(s, 0) + 1

    band_counts = {}
    for b in age_bands:
        band_counts[b] = band_counts.get(b, 0) + 1

    summary = {
        "n_cases_total": len(cases),
        "n_age_present": len(ages),
        "n_sex_present": len(sexes),
        "age_mean": round(stats.mean(ages), 2) if ages else None,
        "age_sd": round(stats.stdev(ages), 2) if len(ages) > 1 else None,
        "age_min": min(ages) if ages else None,
        "age_max": max(ages) if ages else None,
        "sex_counts": sex_counts,
        "sex_pct": {k: round(100 * v / len(sexes), 1) for k, v in sex_counts.items()} if sexes else {},
        "age_band_counts": dict(sorted(band_counts.items())),
    }

    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nwrote -> {OUT_PATH}")


if __name__ == "__main__":
    main()
