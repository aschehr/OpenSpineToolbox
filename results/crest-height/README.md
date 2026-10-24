# Iliac Crest Height Classification — Results

Data, figures, and tables for the crest-height-vs-L4/L5 abstract, generated
from `ostk/crest_height.py` against CTSpinoPelvic1K v4
(`refs/heads/v4` = `9f69480a572624029a927bc6661309a068f1f2c3`, verified via
`git ls-remote` against the HuggingFace dataset repo).

## Contents

- `data/` — real pipeline output, one JSON record per case
  - `crest_height_10case.jsonl` — the 10 manually-validated cases
  - `crest_height_full802.jsonl` — full-cohort run (802/802 cases)
  - `demographics_802case.json` — age/sex pulled from the dataset manifest
  - `lstv_subanalysis_result.json` — LSTV/transitional-anatomy comparison
- `figures/` — fig1 (level distribution bar chart), fig2 (single annotated
  coronal example), fig3 (Bland-Altman, n=10), fig4 (three-panel L4 /
  L4-L5 / L5 comparison)
- `tables/` — table1 (full n=10 validation table) and table2 (summary
  statistics: Pearson r, ICC(2,1), mean bias + 95% CI, Cohen's kappa), each
  as CSV + a rendered PNG
- `scripts/` — the code that produced everything above. Each script
  recomputes its numbers from the data files rather than hard-coding
  results, so re-running them regenerates the figures/tables byte-for-byte.
  Scripts assume this repo is checked out at `/Users/ashleyschehr/OpenSpineToolbox`
  and that the pinned CTSpinoPelvic1K v4 snapshot is cached locally via
  `huggingface_hub` (paths are not portable as-is).

## Headline numbers

- Level distribution (789 clean cases, 1578 side-observations): L4 47.5%,
  L5 34.5%, L4-L5 disc space 18.0%
- Validation (n=10 manual ITK-SNAP): Pearson r=0.9936 (p=7.44e-9),
  ICC(2,1)=0.9885, mean bias -0.627mm (95% CI [-1.306, 0.053]), Cohen's
  kappa=0.9043 (n=20 side-observations)
- LSTV sub-analysis: no significant difference in obliquity vs. the
  general sample (see `data/lstv_subanalysis_result.json`)
