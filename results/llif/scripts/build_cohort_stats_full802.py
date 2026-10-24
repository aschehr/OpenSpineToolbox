"""Build llif_cohort_stats_full802.json from llif_report_full802.jsonl (all
802 cases, not the earlier 100-case subset) via llif_risk.build_cohort_stats.
Run once run_llif_report_full802.py has finished."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, "/Users/ashleyschehr/OpenSpineToolbox")

from ostk.llif_risk import build_cohort_stats  # noqa: E402

IN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "llif_report_full802.jsonl")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "llif_cohort_stats_full802.json")


def main() -> None:
    records = []
    with open(IN_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "error" not in rec:
                records.append(rec)
    print(f"{len(records)} usable records (of the lines in {IN_PATH})", flush=True)

    stats = build_cohort_stats(records)
    with open(OUT_PATH, "w", encoding="utf-8") as out_fh:
        json.dump(stats, out_fh, indent=2)
    print(f"DONE -> {OUT_PATH}", flush=True)
    for name, s in stats.items():
        print(f"  {name}: n={s['n']} p10={s['p10']} p90={s['p90']}", flush=True)


if __name__ == "__main__":
    main()
