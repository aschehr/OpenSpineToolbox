"""ostk.llif_risk — geometric suitability/risk-flag SCREENING SUMMARY for a
target LLIF level.

NOT a treatment recommendation, and not designed to become one: it flags
which geometric parameters fall outside the range observed in a real
reference cohort (this toolbox's own validated output on real cases), for
a surgeon to weigh alongside symptoms, neuro exam, comorbidities, bone
quality, and other imaging -- none of which this CT-only pipeline has any
access to. A parameter with no cohort reference reports 'not_calibrated',
never a silently-assumed normal range or an invented clinical cutoff (the
same discipline this project applied to the Yang et al. crest-height
threshold: cite literature as context, don't compute against an unverified
number as if it were our own validated finding).

`cohort_stats` is a plain {parameter_name: {"p10": x, "p90": y, "n": n}}
dict -- build one with `build_cohort_stats` from a batch of real
`llif_risk_flags_from_label` runs (see `scripts` in this toolbox's
results/ for the crest-height precedent). Percentile choice (p10/p90) is a
reporting convention -- "outside the middle 80% of a cohort we've actually
measured" -- not a diagnostic threshold.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from .crest_height_yang import YANG_CITATION
from .disc_height import adjacent_disc_height_ratio_from_label
from .geometry import WORLD_SUPERIOR
from .llif import llif_level_report_from_label

METHOD_VERSION = "llif-risk-v1"

SCHWAB_CITATION = ("Schwab F, Ungar B, Blondel B, et al. Scoliosis Research Society-Schwab "
                   "Adult Spinal Deformity Classification: A Validation Study. "
                   "Neurosurgery. 2012;71(6):1189-1196.")

# Separate from _FLAG_SPECS below on purpose: these are the ONLY flags tied
# to a specific, human-verified citation with a validated cutoff/grade
# (checked via a real literature search + the paper's actual Methods text
# for Yang et al., not an AI-summarized guess -- see crest_height_yang.py's
# module docstring for that verification trail). Every other geometric
# parameter in this toolbox is real and internally consistent, but "outside
# our own cohort's typical range" (below, `_FLAG_SPECS`) is a DIFFERENT,
# weaker kind of claim than "a published study associated this specific
# measurement with this specific outcome" -- conflating the two would
# overstate what most of these parameters actually have behind them.
_LITERATURE_FLAGS = [
    {
        "name": "yang_crest_height_above_12mm",
        "citation": YANG_CITATION,
        "finding": ("Iliac crest height >12mm above the L4-5 disc's ventral midpoint was "
                   "associated with higher subsidence risk after L4-5 OLIF (ROC AUC 0.688, "
                   "p=0.042; 43% vs. 10-19% subsidence across crest-height groups, p=0.01)."),
        "value": lambda r: _get(r, "yang_crest_height", "value"),
        "flagged": lambda r: _get(r, "yang_crest_height", "above_yang_12mm_cutoff"),
    },
    {
        "name": "pi_ll_schwab_elevated",
        "citation": SCHWAB_CITATION,
        "finding": ("PI-LL mismatch >=10 degrees (SRS-Schwab modifier '+' or '++') indicates "
                   "a sagittal-deformity severity grade at which single-level correction may "
                   "not restore alignment on its own."),
        "value": lambda r: _get(r, "pi_ll", "mismatch", "abs_pi_minus_ll"),
        "flagged": lambda r: _get(r, "pi_ll", "schwab", "PI-LL") in ("+", "++"),
    },
]


def _max_abs(values: List[Optional[float]]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return max(values, key=abs) if values else None


def _max_dev_from_one(values: List[Optional[float]]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return max(values, key=lambda v: abs(v - 1.0)) if values else None


def _get(d, *path):
    for k in path:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


# name -> extractor(full_report_dict) -> Optional[float]. One-line addition
# here to flag a new parameter; nothing else needs to change.
_FLAG_SPECS: List[Tuple[str, Callable[[Dict], Optional[float]]]] = [
    ("crest_obliquity_mm", lambda r: _get(r, "crest_height", "obliquity_mm")),
    ("disc_height_middle_mm", lambda r: _get(r, "disc_height", "middle_mm")),
    ("disc_footprint_ap_mismatch_pct", lambda r: _get(r, "disc_footprint", "ap_mismatch_pct")),
    ("disc_footprint_ml_mismatch_pct", lambda r: _get(r, "disc_footprint", "ml_mismatch_pct")),
    ("lateral_listhesis_abs_mm",
     lambda r: abs(v) if (v := _get(r, "lateral_listhesis", "value")) is not None else None),
    ("coronal_disc_angle_deg", lambda r: _get(r, "coronal_disc_angle", "value")),
    ("adjacent_disc_height_ratio", lambda r: _get(r, "adjacent_disc_height_ratio", "value")),
    ("vertebral_rotation_abs_deg",
     lambda r: _max_abs([_get(r, "vertebral_rotation", lv, "value")
                         for lv in (r.get("vertebral_rotation") or {})])),
    ("vertebral_wedge_ratio_dev",
     lambda r: _max_dev_from_one([_get(r, "vertebral_wedging", lv, "wedge_ratio")
                                  for lv in (r.get("vertebral_wedging") or {})])),
    ("pi_ll_mismatch_abs_deg", lambda r: _get(r, "pi_ll", "mismatch", "abs_pi_minus_ll")),
    ("sagittal_slip_abs_mm",
     lambda r: abs(v) if (v := _get(r, "sagittal_slip", "slip_mm")) is not None else None),
]


def literature_flag_values(report: Dict) -> Dict:
    """Evaluate the `_LITERATURE_FLAGS` (Yang et al. crest height, SRS-Schwab
    PI-LL) against one case's report. Each entry carries its own citation
    and the specific finding it's grounded in -- callers should surface
    those directly rather than just a bare True/False, since "flagged" only
    means something alongside the actual claim being cited."""
    out: Dict = {}
    for spec in _LITERATURE_FLAGS:
        value = spec["value"](report)
        flagged = spec["flagged"](report)
        out[spec["name"]] = {
            "value": round(float(value), 4) if value is not None else None,
            "flagged": bool(flagged) if flagged is not None else None,
            "citation": spec["citation"],
            "finding": spec["finding"],
        }
    return out


def flag_value(value: Optional[float], cohort_stats: Optional[Dict], name: str) -> Dict:
    """Compare one value against its cohort reference range. Never raises:
    a missing value or a parameter with no cohort entry yet is reported as
    a status, not silently dropped."""
    if value is None:
        return {"value": None, "status": "unavailable"}
    ref = (cohort_stats or {}).get(name)
    if ref is None:
        return {"value": round(float(value), 4), "status": "not_calibrated"}
    lo, hi = ref["p10"], ref["p90"]
    if value < lo:
        status = "below_typical_range"
    elif value > hi:
        status = "above_typical_range"
    else:
        status = "within_typical_range"
    return {"value": round(float(value), 4), "status": status,
           "cohort_p10": lo, "cohort_p90": hi, "cohort_n": ref.get("n")}


def llif_risk_flags_from_label(label, affine, upper_level: str, lower_level: str,
                               label_ids: Dict[str, int], *, case_id: str = "",
                               sup_axis=WORLD_SUPERIOR,
                               cohort_stats: Optional[Dict] = None) -> Dict:
    """Geometric screening summary for the `upper_level`-`lower_level` disc
    space: every parameter this toolbox computes for that level, each
    compared to a cohort reference range when one is available.

    Deliberately returns a per-parameter STATUS BREAKDOWN and a count, not
    a single yes/no verdict -- this pipeline has no access to the clinical
    information (symptoms, exam, comorbidities, other imaging) a real LLIF
    suitability decision requires, and collapsing multiple independent
    geometric findings into one number would overstate what CT geometry
    alone can support."""
    report = llif_level_report_from_label(label, affine, upper_level, lower_level,
                                          label_ids, case_id=case_id, sup_axis=sup_axis)
    report["adjacent_disc_height_ratio"] = adjacent_disc_height_ratio_from_label(
        label, affine, upper_level, lower_level, label_ids, case_id=case_id, sup_axis=sup_axis)

    flags = {name: flag_value(extractor(report), cohort_stats, name)
             for name, extractor in _FLAG_SPECS}

    statuses = [f["status"] for f in flags.values()]
    lit_flags = literature_flag_values(report)
    return {
        "case_id": case_id, "parameter": "llif_risk_flags",
        "level": f"{upper_level}-{lower_level}", "method_version": METHOD_VERSION,
        "literature_flags": lit_flags,
        "n_literature_flags_positive": sum(bool(f["flagged"]) for f in lit_flags.values()
                                           if f["flagged"] is not None),
        "flags": flags,
        "n_outside_typical_range": sum(s in ("below_typical_range", "above_typical_range")
                                       for s in statuses),
        "n_within_typical_range": sum(s == "within_typical_range" for s in statuses),
        "n_not_calibrated": sum(s == "not_calibrated" for s in statuses),
        "note": ("Geometric screening aid only, NOT a treatment recommendation. "
                "'literature_flags' are the ONLY flags tied to a specific, verified "
                "published citation; 'flags' below are purely descriptive (outside "
                "THIS toolbox's own cohort range), not a clinical risk claim. "
                "'not_calibrated' parameters have no cohort reference range yet."),
        "raw_report": report,
        "supine_ct": True,
    }


def build_cohort_stats(records: List[Dict], lo_pct: float = 10.0, hi_pct: float = 90.0) -> Dict:
    """Build a `cohort_stats` dict from a batch of REAL `llif_risk_flags_from_label`
    (or equivalent raw-report) outputs: for each parameter in `_FLAG_SPECS`,
    the (lo_pct, hi_pct) percentiles of its observed values across `records`
    (values from qc-clean cases only implicitly, since missing values are
    already None and skipped). Requires >= 10 usable values per parameter to
    report a range at all -- fewer than that isn't a cohort, it's noise."""
    import numpy as np

    stats: Dict[str, Dict] = {}
    for name, extractor in _FLAG_SPECS:
        vals = [v for r in records if (v := extractor(r)) is not None]
        if len(vals) < 10:
            continue
        lo, hi = np.percentile(vals, [lo_pct, hi_pct])
        stats[name] = {"p10": round(float(lo), 4), "p90": round(float(hi), 4), "n": len(vals)}
    return stats
