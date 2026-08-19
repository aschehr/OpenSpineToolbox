"""ostk.llif — LLIF (lateral lumbar interbody fusion) preop-planning
summary: one call, one target level, every geometric parameter this
toolbox currently supports for LLIF -- reusing every primitive already
built and validated rather than re-deriving any of them. Same "one-call
clinical summary" shape as `metrics.spinopelvic_summary_from_label`.

Deliberately NOT included: rib-to-disc corridor clearance -- parked in
`rib_corridor.py` (see its docstring) after the dataset's own rib labels
proved unreliable on real validation cases, independent of a genuine
methodology fix. Also not included: psoas/vessel-based corridor safety
(Moro zone, vessel distance) -- those labels are still unpopulated in the
v4 data as of the pinned revision (checked directly, empty in every case
sampled).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .coronal_alignment import coronal_disc_angle_from_label, lateral_listhesis_from_label
from .crest_height import crest_height_from_label
from .disc_height import disc_height_from_label
from .endplate_footprint import disc_footprint_from_label, isolate_vertebral_body
from .geometry import WORLD_SUPERIOR
from .masks import binary_mask, largest_component
from .metrics import (LL_ENDPLATE_CHAIN, _pi_from_label_core, lumbar_lordosis_from_label,
                      pi_ll_mismatch, schwab_sagittal_modifiers)
from .spondylolisthesis import sagittal_slip_from_label
from .vertebral_rotation import _lr_axis, vertebral_axial_rotation_from_label
from .vertebral_wedging import vertebral_wedging_from_label

METHOD_VERSION = "llif-summary-v1"

_LL_SEGMENTS = {f"{a}-{b}" for a, b in zip(LL_ENDPLATE_CHAIN, LL_ENDPLATE_CHAIN[1:])}


def _segmental_lordosis(ll_measurement, upper_level: str, lower_level: str
                        ) -> Tuple[Optional[float], List[str]]:
    """Pull ONE segment's lordosis out of an ALREADY-COMPUTED
    `lumbar_lordosis_from_label` Measurement's per-segment chain
    (L1-L2 ... L5-S1) -- no new geometry, just surfacing an existing,
    fully-tested result (and not recomputing LL a second time; the caller
    needs the full LL value anyway for PI-LL mismatch). Only defined for
    consecutive levels on that chain; None (with a flag) otherwise, e.g.
    T12-L1 -- above the chain's top."""
    seg_key = f"{upper_level}-{lower_level}"
    if seg_key not in _LL_SEGMENTS:
        return None, ["level_outside_LL_chain"]
    per_seg = (ll_measurement.landmarks_world_mm.get("per_segment_lordosis_deg")
              if ll_measurement.value is not None else None)
    if not per_seg or seg_key not in per_seg:
        return None, [f"segment_unavailable:{seg_key}"]
    return round(float(per_seg[seg_key]), 3), []


def llif_level_report_from_label(label, affine, upper_level: str, lower_level: str,
                                 label_ids: Dict[str, int], *, case_id: str = "",
                                 sup_axis=WORLD_SUPERIOR) -> Dict:
    """One-call LLIF preop-planning summary for the disc space between
    `upper_level` and `lower_level` (e.g. 'L4','L5'): disc height (ant/mid/
    post), cage footprint (AP/ML at both bounding endplates), vertebral
    axial rotation (both levels), vertebral wedging (both levels), lateral
    listhesis, coronal disc angle, sagittal slip (Meyerding grade),
    segmental lordosis + PI-LL mismatch/Schwab modifiers (when on the
    L1-S1 chain), and -- only when the target IS L4-L5, since the crest
    only constrains access at that specific level -- iliac crest height,
    the parameter this toolbox validated first.

    PI-LL is composed manually from `metrics._pi_from_label_core` +
    `pi_ll_mismatch`/`schwab_sagittal_modifiers` here -- deliberately NOT
    via `metrics.spinopelvic_summary_from_label`, which also runs
    `surgical_recommendation` (an existing, pre-this-project function that
    outputs an actual prescriptive surgical plan). That's a real,
    literature-cited feature already in this codebase, but including it
    here was a decision left to the user rather than one this report makes
    silently by reusing the convenience wrapper that happens to bundle it.

    Every sub-parameter keeps ITS OWN qc_flags (nothing here silently drops
    a bad case); this function does not invent a combined flag."""
    # Both computed ONCE and threaded through every sub-call that needs them
    # below, rather than each sub-function recomputing its own -- these are
    # the two expensive steps (femoral-head sphere fit for `lr`; erosion-
    # based body isolation for the two masks), and naive composition without
    # this sharing was measured redundantly re-running the fit up to 4x and
    # the erosion up to 2x per level per case.
    lr = _lr_axis(label, affine, label_ids, sup_axis, 0.35, 50)
    body_masks = {}
    for lv in {upper_level, lower_level}:
        m = largest_component(binary_mask(label, label_ids[lv]))
        body_masks[lv] = isolate_vertebral_body(m, affine, erosion_mm=10.0) if m.any() else None

    report: Dict = {
        "case_id": case_id, "parameter": "llif_level_report",
        "level": f"{upper_level}-{lower_level}", "method_version": METHOD_VERSION,
        "supine_ct": True,
    }
    report["disc_height"] = disc_height_from_label(
        label, affine, upper_level, lower_level, label_ids, case_id=case_id, sup_axis=sup_axis)
    report["disc_footprint"] = disc_footprint_from_label(
        label, affine, upper_level, lower_level, label_ids, case_id=case_id, sup_axis=sup_axis,
        upper_body_mask=body_masks[upper_level], lower_body_mask=body_masks[lower_level])
    report["vertebral_rotation"] = {
        upper_level: vertebral_axial_rotation_from_label(
            label, affine, upper_level, label_ids, case_id=case_id, sup_axis=sup_axis, lr=lr,
            body_mask=body_masks[upper_level]),
        lower_level: vertebral_axial_rotation_from_label(
            label, affine, lower_level, label_ids, case_id=case_id, sup_axis=sup_axis, lr=lr,
            body_mask=body_masks[lower_level]),
    }
    report["vertebral_wedging"] = {
        upper_level: vertebral_wedging_from_label(
            label, affine, upper_level, label_ids, case_id=case_id, sup_axis=sup_axis),
        lower_level: vertebral_wedging_from_label(
            label, affine, lower_level, label_ids, case_id=case_id, sup_axis=sup_axis),
    }
    report["lateral_listhesis"] = lateral_listhesis_from_label(
        label, affine, upper_level, lower_level, label_ids, case_id=case_id, sup_axis=sup_axis, lr=lr)
    report["coronal_disc_angle"] = coronal_disc_angle_from_label(
        label, affine, upper_level, lower_level, label_ids, case_id=case_id, sup_axis=sup_axis, lr=lr)
    report["sagittal_slip"] = sagittal_slip_from_label(
        label, affine, upper_level, lower_level, label_ids, case_id=case_id, sup_axis=sup_axis, lr=lr)

    # PI-LL: PI's femoral-head fit needs full sphere objects (center+radius),
    # not just the direction `lr` already gives us, so this is the one
    # remaining un-cached fit in this report -- a deliberate stopping point
    # given the earlier 4x/2x redundancy fixes already cut per-case time by
    # a third; unifying this too would need a deeper refactor of PI's
    # internals for a comparatively small further gain.
    pi_r, pi_flags = _pi_from_label_core(label, affine, sup_axis, 0.15, 0.35, 50, label_ids)
    ll_measurement = lumbar_lordosis_from_label(label, affine, case_id=case_id, sup_axis=sup_axis,
                                                label_ids=label_ids, lr=lr)
    PI = round(pi_r["PI"], 3) if pi_r else None
    PT = round(pi_r["PT"], 3) if pi_r else None
    LL = ll_measurement.value
    pi_ll_flags = [f"PI:{f}" for f in pi_flags] if pi_r is None else []
    if LL is None:
        pi_ll_flags += [f"LL:{f}" for f in ll_measurement.qc_flags]
    report["pi_ll"] = {
        "PI": PI, "LL": LL, "PT": PT,
        "mismatch": pi_ll_mismatch(PI, LL) if PI is not None and LL is not None else None,
        "schwab": (schwab_sagittal_modifiers(PI, LL, PT)
                  if PI is not None and LL is not None and PT is not None else None),
        "qc_flags": pi_ll_flags or ["ok"],
    }

    seg_ll, seg_flags = _segmental_lordosis(ll_measurement, upper_level, lower_level)
    report["segmental_lordosis"] = {
        "value": seg_ll, "units": "degrees", "qc_flags": seg_flags or ["ok"],
    }

    if {upper_level, lower_level} == {"L4", "L5"}:
        report["crest_height"] = crest_height_from_label(
            label, affine, label_ids, case_id=case_id, sup_axis=sup_axis)

    return report
