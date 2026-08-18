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

from typing import Dict, List

from .geometry import WORLD_SUPERIOR, unit
from .metrics import LL_ENDPLATE_CHAIN
from .spine import endplate_corner_landmarks

METHOD_VERSION = "disc-height-v1"
RATIO_METHOD_VERSION = "disc-height-ratio-v1"


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


def adjacent_disc_height_ratio_from_label(label, affine, upper_level: str, lower_level: str,
                                          label_ids: Dict[str, int], *, case_id: str = "",
                                          sup_axis=WORLD_SUPERIOR, lr=(1.0, 0.0, 0.0)) -> Dict:
    """How collapsed the target disc space is RELATIVE to its immediate
    neighbors on the L1-S1 chain: ratio = target middle height / mean(the
    one or two neighboring discs' middle heights). Distinguishes a FOCAL
    single-level problem (ratio well below 1.0, a stronger single-level
    LLIF indication) from broad multilevel degeneration (ratio near 1.0,
    where a single-level cage may not address the patient's problem) --
    absolute disc height alone can't tell these apart on its own.

    `upper_level`/`lower_level` must be adjacent entries on
    `metrics.LL_ENDPLATE_CHAIN` (e.g. 'L4','L5'). Uses whichever neighbor(s)
    exist -- one at the top of the chain (L1-L2) or bottom (L5-S1) only has
    one neighbor, not two. Never silently drops a bad case: missing/
    unfittable input -> ratio None plus a qc_flags entry (SPEC §4)."""
    flags: List[str] = []
    try:
        i = LL_ENDPLATE_CHAIN.index(upper_level)
        if LL_ENDPLATE_CHAIN[i + 1] != lower_level:
            raise ValueError
    except (ValueError, IndexError):
        return {
            "case_id": case_id, "parameter": "adjacent_disc_height_ratio",
            "level": f"{upper_level}-{lower_level}", "units": "ratio", "value": None,
            "qc_flags": ["level_not_on_LL_chain"], "method_version": RATIO_METHOD_VERSION,
            "supine_ct": True,
        }

    target = disc_height_from_label(label, affine, upper_level, lower_level, label_ids,
                                    case_id=case_id, sup_axis=sup_axis, lr=lr)
    neighbor_heights: List[float] = []
    if i - 1 >= 0:
        above = disc_height_from_label(label, affine, LL_ENDPLATE_CHAIN[i - 1], upper_level,
                                       label_ids, case_id=case_id, sup_axis=sup_axis, lr=lr)
        if above["middle_mm"] is not None:
            neighbor_heights.append(above["middle_mm"])
        else:
            flags.append(f"missing_neighbor:{LL_ENDPLATE_CHAIN[i - 1]}-{upper_level}")
    if i + 2 < len(LL_ENDPLATE_CHAIN):
        below = disc_height_from_label(label, affine, lower_level, LL_ENDPLATE_CHAIN[i + 2],
                                       label_ids, case_id=case_id, sup_axis=sup_axis, lr=lr)
        if below["middle_mm"] is not None:
            neighbor_heights.append(below["middle_mm"])
        else:
            flags.append(f"missing_neighbor:{lower_level}-{LL_ENDPLATE_CHAIN[i + 2]}")
    if not neighbor_heights:
        flags.append("no_neighbors_available")

    combined_flags = [f for f in target["qc_flags"] if f != "ok"] + flags
    if not combined_flags:
        combined_flags = ["ok"]

    result: Dict = {
        "case_id": case_id, "parameter": "adjacent_disc_height_ratio",
        "level": f"{upper_level}-{lower_level}", "units": "ratio", "value": None,
        "target_middle_mm": target["middle_mm"], "neighbor_middle_mm": neighbor_heights,
        "qc_flags": combined_flags,
        "method_version": RATIO_METHOD_VERSION, "supine_ct": True,
    }
    if target["middle_mm"] is not None and neighbor_heights:
        mean_neighbor = sum(neighbor_heights) / len(neighbor_heights)
        result["value"] = round(target["middle_mm"] / mean_neighbor, 4) if mean_neighbor > 1e-6 else None
    return result
