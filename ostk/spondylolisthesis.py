"""ostk.spondylolisthesis — sagittal (anterior-posterior) translational
slip between two adjacent vertebrae, for LLIF vs. ALIF/TLIF approach
planning: significant AP slip changes the trajectory and can favor an
approach other than a true lateral one. Fills in the previously-
unimplemented `projects/spondylolisthesis/` spec.

Reuses `spine.endplate_corner_landmarks` (corners + midpoint, shared with
`disc_height.py`/`vertebral_wedging.py`/`coronal_alignment.py`) -- the
sagittal analog of `coronal_alignment.lateral_listhesis_from_label`,
projected onto the anterior axis instead of the L-R axis.

Label ids are passed explicitly, not `ostk.labels.lid()` -- see
`crest_height.py`'s docstring for why.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .geometry import WORLD_SUPERIOR, unit
from .spine import anterior_axis, endplate_corner_landmarks

METHOD_VERSION = "spondylolisthesis-v1"


def _meyerding_grade(pct: float) -> str:
    """Meyerding grade from percent slip: I (0-25%), II (25-50%),
    III (50-75%), IV (75-100%), V / spondyloptosis (>100%)."""
    if pct < 25.0:
        return "I"
    if pct < 50.0:
        return "II"
    if pct < 75.0:
        return "III"
    if pct < 100.0:
        return "IV"
    return "V"


def sagittal_slip_from_label(label, affine, upper_level: str, lower_level: str,
                             label_ids: Dict[str, int], *, case_id: str = "",
                             sup_axis=WORLD_SUPERIOR, lr=(1.0, 0.0, 0.0)) -> Dict:
    """Sagittal (A-P) translational offset between `upper_level`'s inferior-
    endplate midpoint and `lower_level`'s superior-endplate midpoint -- the
    classic spondylolisthesis measurement. Positive = upper body shifted
    ANTERIORLY relative to lower (anterolisthesis, the common direction at
    L4-L5/L5-S1); negative = retrolisthesis.

    Meyerding grade (%) = |slip| / the lower body's AP corner-to-corner
    width x 100 -- the standard clinical percent-slip convention. Uses the
    cheap corner-to-corner AP distance already computed for the endplate
    (consistent with `disc_height.py`'s approach), not the body-isolated
    footprint width from `endplate_footprint.py`, which needs the more
    expensive erosion step for a marginal accuracy gain here.

    Never silently drops a bad case: missing/unfittable input -> value None
    plus a qc_flags entry (SPEC §4)."""
    flags: list = []
    upper = endplate_corner_landmarks(label, affine, upper_level, "inferior", label_ids,
                                      sup_axis=sup_axis, lr=lr)
    if upper is None:
        flags.append(f"missing_label:{upper_level}")
    lower = endplate_corner_landmarks(label, affine, lower_level, "superior", label_ids,
                                      sup_axis=sup_axis, lr=lr)
    if lower is None:
        flags.append(f"missing_label:{lower_level}")

    result: Dict = {
        "case_id": case_id, "parameter": "sagittal_slip",
        "level": f"{upper_level}-{lower_level}", "units": "mm",
        "slip_mm": None, "meyerding_pct": None, "meyerding_grade": None,
        "qc_flags": flags, "method_version": METHOD_VERSION, "supine_ct": True,
    }
    if upper is not None and lower is not None:
        ap = anterior_axis(unit(sup_axis), lr)
        slip = float((upper["mid"] - lower["mid"]) @ ap)
        lower_ap_width = float(np.linalg.norm(lower["posterior_corner"] - lower["anterior_corner"]))
        result["slip_mm"] = round(slip, 3)
        if lower_ap_width > 1e-6:
            pct = abs(slip) / lower_ap_width * 100.0
            result["meyerding_pct"] = round(pct, 2)
            result["meyerding_grade"] = _meyerding_grade(pct)
    if not flags:
        flags.append("ok")
    return result
