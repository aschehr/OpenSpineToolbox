"""ostk.disc_height — segmental disc height (anterior/middle/posterior) for
LLIF level selection and collapse/degeneration assessment.

Unlike `endplate_footprint` (which needed a new body-isolation step because
transverse processes contaminate a raw width measurement), disc height
needs no new geometry: it's exactly what `spine.endplate_corner_landmarks`
(anterior/posterior cortical corners of the disc-facing endplate surface)
was already built for. Anterior height = vertical gap between the two
surfaces' anterior corners; posterior height = same at the posterior
corners; middle height = same at the corner-chord midpoints.

Label ids are passed explicitly, not `ostk.labels.lid()` — see
`crest_height.py`'s docstring for why (the real v4 dataset renumbers ids).
"""
from __future__ import annotations

from typing import Dict

from .geometry import WORLD_SUPERIOR, unit
from .spine import endplate_corner_landmarks

METHOD_VERSION = "disc-height-v1"


def disc_height_from_label(label, affine, upper_level: str, lower_level: str,
                           label_ids: Dict[str, int], *, case_id: str = "",
                           sup_axis=WORLD_SUPERIOR, lr=(1.0, 0.0, 0.0)) -> Dict:
    """Anterior/middle/posterior height (mm) of the disc space between
    `upper_level`'s inferior endplate and `lower_level`'s superior endplate
    (e.g. 'L4','L5' for the L4-L5 disc). Reporting all three (not just an
    average) matters clinically: anterior-only collapse vs. uniform collapse
    vs. posterior-only collapse are different findings.

    Never silently drops a bad case: missing/unfittable input -> the
    affected field(s) stay None plus a qc_flags entry (SPEC §4)."""
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
        "case_id": case_id, "parameter": "disc_height",
        "level": f"{upper_level}-{lower_level}", "units": "mm",
        "anterior_mm": None, "middle_mm": None, "posterior_mm": None,
        "qc_flags": flags, "method_version": METHOD_VERSION, "supine_ct": True,
    }
    if upper is not None and lower is not None:
        # vertical (cranial-caudal) gap only -- NOT raw 3D distance between
        # the two independently-picked corners. Verified on a real case
        # (0441) that those corners aren't laterally aligned between
        # adjacent vertebrae (30mm apart in L-R for a true 6.4mm vertical
        # gap): a straight-line distance conflates corner-pick jitter with
        # actual disc height. A clinician measuring "anterior disc height"
        # on a sagittal slice measures vertically, not point-to-point.
        a = unit(sup_axis)
        result["anterior_mm"] = round(
            float(abs((upper["anterior_corner"] - lower["anterior_corner"]) @ a)), 3)
        result["posterior_mm"] = round(
            float(abs((upper["posterior_corner"] - lower["posterior_corner"]) @ a)), 3)
        result["middle_mm"] = round(float(abs((upper["mid"] - lower["mid"]) @ a)), 3)
    if not flags:
        flags.append("ok")
    return result
