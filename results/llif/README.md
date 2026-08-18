# LLIF Preop-Planning Parameters — Results

Data and scripts backing `ostk/llif.py` and `ostk/llif_risk.py`, generated
against CTSpinoPelvic1K v4 (`refs/heads/v4` =
`9f69480a572624029a927bc6661309a068f1f2c3`, same pinned/verified revision
as `results/crest-height/`).

## Contents

- `data/llif_report_10case.jsonl` — full `llif_level_report_from_label`
  output (L4-L5) for the same 10 manually-validated cases used throughout
  this project.
- `data/llif_cohort_calibration_100case.jsonl` — raw `llif_level_report_from_label`
  + `adjacent_disc_height_ratio_from_label` output over the first 100 cases
  (by case id) in the pinned snapshot, used to build real percentile
  reference ranges (`llif_risk.build_cohort_stats`) rather than inventing
  clinical thresholds.
- `data/llif_cohort_stats.json` — the resulting `cohort_stats` dict (p10/p90
  + n per parameter), consumed by `llif_risk.llif_risk_flags_from_label`.
  `crest_obliquity_mm` uses the real 802-case cohort from
  `results/crest-height/` (n=801) instead of this 100-case subset, since
  that full-cohort data already existed and is far more statistically
  robust; every other parameter is calibrated from n=99-100.
- `scripts/` — the runners that produced the above. Same portability caveat
  as `results/crest-height/scripts/`: paths assume this repo checked out at
  `/Users/ashleyschehr/OpenSpineToolbox` with the pinned v4 snapshot cached
  locally via `huggingface_hub`.

## Important caveat

100 cases (99-100 for most parameters) is a real, empirical reference
range — not an invented number — but it is **not** the full 802-case
cohort `crest_height` has. Treat `llif_cohort_stats.json` as a first-pass
calibration; percentile bounds will shift as the reference cohort grows.
`llif_risk_flags_from_label`'s `not_calibrated` status exists specifically
so an uncalibrated parameter is never silently treated as normal.

`llif_risk.py` is explicitly a geometric screening aid, not a treatment
recommendation — see its module docstring.
