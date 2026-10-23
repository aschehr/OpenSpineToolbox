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
- `data/vertebral_hu_10case.jsonl` — `vertebral_hu_from_label` output (L4
  and L5, mean/median/SD HU + voxel count) for the same 10 manually-
  validated cases, via `run_vertebral_hu_10case.py`. Mean HU ranges 87.6-
  233.5 across the 10 cases (real between-patient spread), with L4≈L5
  within each case except 0655 (132.3 vs. 183.1 -- not yet explained,
  worth a manual look). This is the ONE parameter in this toolbox needing
  the CT intensity volume, not just labels -- see `ostk/vertebral_hu.py`'s
  module docstring for why its manual validation (point-sampling, not
  point-landmark ICC/Pearson/kappa) is structurally different from every
  other parameter here.
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

## Cadaveric / literature validation (population-level)

`validation/cadaveric_validation.{csv,md}` (built by
`scripts/build_cadaveric_validation_table.py` from
`validation/cadaveric_references.json`) compares the full-802 cohort's
distributions against cadaveric morphometry (Panjabi 1992, Tan 2002, Berry
1987, Weinberg 2016, Can 2024) plus one same-modality CT study
(Pluemvitayaporn 2025). Only values whose sub-report passed QC are used.

This is a **plausibility / systematic-bias check, not per-case
validation** -- the specimens are different people from our patients.
Every reference number carries a `verification_level`: `primary` (checked
against the paper's own abstract/full text) or `secondary_reproduction`
(the Tan/Berry per-level endplate values, checked only against
PMC11875720 Table 2 because the primaries were unavailable). Panjabi 1992
values were verified against the primary's Table 2 (p. 301) and match the
PMC11875720 reproduction exactly (which rounds the SEMs).

Findings (2026-09-28):

- **Pelvic incidence agrees**: 48.5 vs. 46.0 deg (Weinberg, n=880 dry
  pelves), SMD 0.23.
- **Endplate footprint is systematically undersized** against every
  reference, cadaveric and CT alike, in both sexes separately (so not a
  sex-mix artifact): L4 inferior ML width -17% vs. Panjabi (SMD -1.9),
  L5 superior AP depth -10%. Leading hypothesis, untested:
  `isolate_vertebral_body`'s 10mm erode/dilate is a morphological opening
  that shaves off the endplate rim (ring apophysis), which flares beyond
  the body's waist and is <20mm thick. Separately, `llif.py` doesn't pass
  `lr` to `disc_footprint_from_label`, so ML width is measured on the
  scanner x-axis rather than the patient L-R axis (small effect).
- **Disc height fails**: the "match" with Can 2024's cadaveric anterior
  height (6.78 vs. 6.8mm) is coincidental -- post-mortem discs dehydrate,
  and against same-modality CT maximum disc height our value is 39% low
  (7.6 vs. 12.4mm), alongside 1-voxel (0.8mm) values and 51 cases with
  identical ant/mid/post heights.
- **Not validatable against cadavers**: rotation, listhesis, slip, coronal
  angle, LL, crest height (patient/posture-specific) and HU (post-mortem
  attenuation changes). Anterior body height has no verified cadaveric
  reference (Panjabi reports posterior height only).
- **Posterior vertebral body height reads tall**: L4 26.4 vs. 24.1mm, L5
  24.9 vs. 22.9mm (Panjabi; +9%, SMD 0.54/0.44). Together with disc height
  reading short, this is consistent with our endplate landmarks sitting
  slightly into the disc space -- a hypothesis, not yet tested.
- **Positioning caveat**: the manifest records 377/802 cases as prone and
  3 as decubitus, but every record is marked `supine_ct: true`. Irrelevant
  for bony morphology (endplates, PI), but it matters for posture-dependent
  parameters (LL, PT, slip, crest height).

### Footprint fix attempt (2026-09-28) -- reverted, not yet solved

Axial-slice inspection confirmed the 3D erode/dilate in
`isolate_vertebral_body` shaves the endplate rim (e.g. case 0441 L4
inferior: ~33 vs. ~43mm body width 1.5mm from the endplate). Two fixes were
tried and rejected on real data (not just synthetic phantoms):

- **Slice-wise (2D) opening**: keeps the rim, but at L5 the pedicle +
  transverse-process root is >20mm wide in-plane, so the pedicles survive
  (first 42 cases: L5 superior ML 61 +/- 10mm, max 79mm, case 0051).
- **Growing the 3D-opened body back by k mm (posterior-wall cutoff)**: no
  plateau -- width keeps rising with k (0051 L5: 62/68/73mm at k=4/7/10)
  because it picks up osteophytes and L5 transverse-process roots.

Complication: this cohort is 50-89 years old, and marginal osteophytes are
real bone that widens the segmented "endplate" (0051 L4 inferior is ~60mm
wide at the rim). A correct fix needs an explicit definition (rim vs.
osteophyte; osteophytes are usually resected, not cage-bearing) and a
reference standard -- manual endplate measurements -- rather than tuning
toward cadaveric means, which would make the cadaveric check circular.
Until then the footprint stays on endplate-footprint-v3 with its known
undersizing bias, and should not be used for cage sizing.
