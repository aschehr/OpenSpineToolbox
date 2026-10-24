"""ostk.vertebral_wedging — anterior vs. posterior height of a SINGLE
vertebral body (compression/wedging assessment) for LLIF planning: a
wedged/collapsed body at the target level changes the cage angle needed
and flags a level that isn't simple degenerative disc disease.

Reuses `spine.endplate_corner_landmarks` -- the SAME corner-fit primitive
`disc_height.py` uses between two vertebrae, applied within a single
vertebra instead (its own superior vs inferior endplate). Fills in the
previously-unimplemented `projects/vertebral-body-wedging-index/` spec.
"""
from __future__ import annotations

from typing import Dict

from .geometry import WORLD_SUPERIOR, unit
from .spine import endplate_corner_landmarks

METHOD_VERSION = "vertebral-wedging-v1"


def vertebral_wedging_from_label(label, affine, level: str, label_ids: Dict[str, int], *,
                                 case_id: str = "", sup_axis=WORLD_SUPERIOR,
                                 lr=(1.0, 0.0, 0.0)) -> Dict:
    """Anterior/posterior height (mm) and wedge ratio of `level`'s own
    vertebral body: the vertical gap between its OWN superior and inferior
    endplate corners at matching A-P position (anterior-to-anterior,
    posterior-to-posterior) -- same vertical-gap-only logic `disc_height.py`
    uses between two vertebrae (not raw 3D corner distance; see that
    module's fix for why that's wrong).

    wedge_ratio = anterior_height / posterior_height. Lumbar bodies are
    NORMALLY somewhat anterior-tall (ratio > 1.0 -- this is physiologic,
    contributing to lumbar lordosis, most pronounced at L5) -- that is NOT
    itself a pathological finding. The clinically actionable pattern is
    ratio < 1.0 (posterior taller than anterior, e.g. an osteoporotic
    anterior-compression fracture) or a ratio far outside the normal range
    for that level, which this function reports but does not itself judge
    (no hard-coded normal-range threshold here). Never silently drops a bad
    case: missing/unfittable input -> value None plus a qc_flags entry
    (SPEC §4)."""
    flags: list = []
    sup = endplate_corner_landmarks(label, affine, level, "superior", label_ids,
                                    sup_axis=sup_axis, lr=lr)
    inf = endplate_corner_landmarks(label, affine, level, "inferior", label_ids,
                                    sup_axis=sup_axis, lr=lr)
    if sup is None or inf is None:
        flags.append(f"missing_label:{level}")

    result: Dict = {
        "case_id": case_id, "parameter": "vertebral_wedging", "level": level, "units": "mm",
        "anterior_height_mm": None, "posterior_height_mm": None, "wedge_ratio": None,
        "qc_flags": flags, "method_version": METHOD_VERSION, "supine_ct": True,
    }
    if sup is not None and inf is not None:
        a = unit(sup_axis)
        ant_h = float(abs((sup["anterior_corner"] - inf["anterior_corner"]) @ a))
        post_h = float(abs((sup["posterior_corner"] - inf["posterior_corner"]) @ a))
        result["anterior_height_mm"] = round(ant_h, 3)
        result["posterior_height_mm"] = round(post_h, 3)
        result["wedge_ratio"] = round(ant_h / post_h, 4) if post_h > 1e-6 else None
    if not flags:
        flags.append("ok")
    return result
