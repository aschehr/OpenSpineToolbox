"""ostk.crest_height — iliac crest height classified against L4/L5.

The iliac crest apex (highest point of the ilium) commonly sits over the L4
body, the L4-L5 disc space, or the L5 body — the classic intercristal-line
landmark used for lumbar-puncture level and pedicle-screw entry planning.
This module finds that apex per side straight from the hip label, then
measures/classifies it against the SAME L4/L5 corner-fit endplates
`ostk.spine.fit_endplate` uses for PI/LL elsewhere, so the level boundary is
consistent with the rest of the toolbox rather than a re-derived landmark.

Label ids are NOT taken from `ostk.labels` here: that module encodes the v3
scheme (and a v3-shaped guess at where v4 would extend it), but the real v4
dataset on the Hub renumbers everything because it adds C1-C7 ahead of the
thoracolumbar levels (verified against
snapshot 9f69480a572624029a927bc6661309a068f1f2c3, refs/heads/v4 of
anonymous-mlhc/CTSpinoPelvic1K — L4=23/L5=24/hips=30-31 there, not 4/5/9-10).
So every function here takes an explicit `label_ids` mapping (structure name
-> int id); load it from the dataset's own `dataset_labels.json` with
`load_label_ids` rather than hard-coding a guess.
"""
from __future__ import annotations

import json
from typing import Dict, Optional, Tuple

import numpy as np

from .geometry import WORLD_SUPERIOR, unit
from .masks import binary_mask, largest_component, mask_world
from .spine import fit_endplate

METHOD_VERSION = "crest-height-v1"


def load_label_ids(dataset_labels_json: str) -> Dict[str, int]:
    """Load a {structure_name: id} map from a dataset's own label-scheme file
    (e.g. `<snapshot>/dataset_labels.json`) — the ground truth for whichever
    revision the caller pinned, so callers never hard-code ids that only hold
    for one dataset version."""
    with open(dataset_labels_json, "r", encoding="utf-8") as fh:
        return json.load(fh)


def crest_apex_from_label(label, affine, side: str, label_ids: Dict[str, int], *,
                          sup_axis=WORLD_SUPERIOR, min_voxels: int = 50
                          ) -> Tuple[Optional[np.ndarray], int]:
    """Highest point of the `side` ("left"/"right") hip-bone mask, world mm.

    Keeps only the LARGEST connected component of the hip label before taking
    the extreme, so a fragmented/artifactual island (a real segmentation QC
    issue on hip masks) can't pull a spurious high point in. Returns
    (apex_point, n_voxels_kept); apex_point is None if too small/absent.
    """
    m = largest_component(binary_mask(label, label_ids[f"{side}_hip"]))
    n_voxels = int(m.sum())
    if n_voxels < min_voxels:
        return None, n_voxels
    pts = mask_world(m, affine)
    a = unit(sup_axis)
    apex = pts[int(np.argmax(pts @ a))]
    return apex, n_voxels


def _level_span(label, affine, level: str, label_ids: Dict[str, int], sup_axis
                ) -> Optional[Tuple[float, float]]:
    """(z_inferior, z_superior) of a vertebral body along sup_axis, from the
    same corner-fit endplates PI/LL use. None if the body is absent/unfittable
    (e.g. too few voxels for `fit_endplate`'s min_points)."""
    m = largest_component(binary_mask(label, label_ids[level]))
    if not m.any():
        return None
    pts = mask_world(m, affine)
    sup = fit_endplate(pts, sup_axis, "superior")
    inf = fit_endplate(pts, sup_axis, "inferior")
    if sup is None or inf is None:
        return None
    a = unit(sup_axis)
    return float(inf[0] @ a), float(sup[0] @ a)


def classify_crest_level(apex_z: float, l4_span: Tuple[float, float],
                         l5_span: Tuple[float, float]) -> str:
    """Classify a crest-apex height (already projected onto sup_axis) against
    the L4/L5 body spans: 'L4', 'L4-L5' (disc space), or 'L5'.

    An apex above the L4 superior endplate or below the L5 inferior endplate
    is folded into the adjacent body level ('L4' / 'L5') rather than reported
    as a separate 'above'/'below' class: manual review (case 0655 validation)
    showed these are near-boundary calls within the endplate fit's own rms,
    not a distinct anatomical level, and the manual scheme being validated
    against only ever picks L4 or L5."""
    l4_inf, l4_sup = l4_span
    l5_inf, l5_sup = l5_span
    if apex_z >= l4_inf:
        return "L4"
    if apex_z >= l5_sup:
        return "L4-L5"
    return "L5"


def crest_height_from_label(label, affine, label_ids: Dict[str, int], *,
                            case_id: str = "", sup_axis=WORLD_SUPERIOR,
                            min_voxels: int = 50) -> Dict:
    """Per-case crest-height measurement: for each side, the iliac-crest apex
    classified against the L4/L5 body spans, plus a continuous height (mm,
    signed +cranial) measured from the L4-L5 disc-space midpoint, and the
    left-right obliquity (mm, |right height - left height|).

    Never silently drops a bad case: missing/small inputs -> that side's
    level/height stay None plus a qc_flags entry, mirroring the rest of ostk
    (SPEC §4) — not an exception.
    """
    a = unit(sup_axis)
    flags: list = []
    landmarks: Dict[str, list] = {}

    l4_span = _level_span(label, affine, "L4", label_ids, sup_axis)
    l5_span = _level_span(label, affine, "L5", label_ids, sup_axis)
    if l4_span is None:
        flags.append("missing_label:L4")
    if l5_span is None:
        flags.append("missing_label:L5")

    result: Dict = {
        "case_id": case_id, "parameter": "crest_height", "units": "mm",
        "right": {"level": None, "height_mm": None},
        "left": {"level": None, "height_mm": None},
        "obliquity_mm": None,
        "landmarks_world_mm": landmarks,
        "qc_flags": flags,
        "method_version": METHOD_VERSION,
        "supine_ct": True,
    }

    if l4_span is None or l5_span is None:
        flags.append("insufficient_input")
        return result

    disc_z = 0.5 * (l4_span[0] + l5_span[1])  # midpoint of the L4-L5 disc space
    landmarks["l4_l5_disc_ref_z"] = [float(disc_z)]

    heights: Dict[str, float] = {}
    for side in ("right", "left"):
        apex, n_vox = crest_apex_from_label(label, affine, side, label_ids,
                                            sup_axis=sup_axis, min_voxels=min_voxels)
        if apex is None:
            flags.append(f"low_voxels:{side}_hip")
            continue
        apex_z = float(apex @ a)
        height_mm = apex_z - disc_z
        result[side] = {
            "level": classify_crest_level(apex_z, l4_span, l5_span),
            "height_mm": round(height_mm, 3),
        }
        landmarks[f"{side}_crest_apex"] = apex.tolist()
        heights[side] = height_mm

    if "right" in heights and "left" in heights:
        result["obliquity_mm"] = round(abs(heights["right"] - heights["left"]), 3)

    if not flags:
        flags.append("ok")
    return result
