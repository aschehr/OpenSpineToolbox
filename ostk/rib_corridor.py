"""ostk.rib_corridor — rib-to-disc vertical clearance for upper-lumbar LLIF
corridor accessibility (T12-L1, L1-L2).

PARKED / EXPERIMENTAL -- do not trust or validate this against manual
measurements yet. On the real 10-case validation cohort, values were
anatomically implausible (median clearance ~-45mm; rib 11, which should
essentially never reach L1, showed -35 to -89mm in every case) even though
none of those 10 cases appear on the dataset's own rib-QC worklists. A
posterior-segment-restriction fix (v2, below) produced a BYTE-IDENTICAL
result for rib 12 -- ruling out "wrong point along the rib's arc" as the
cause. Combined with the dataset's own acknowledged, active rib-labeling
QC work (`rib_worklist.json`: 152/802 cases with a single rib bone split
across two adjacent numeric labels; ongoing "rib-spine detachment" checks),
the conclusion is that rib labels in this dataset are not yet reliable
enough to build on, not a fixable bug in this module. Revisit once that
label set matures.

Only buildable now because ribs are populated in the pinned v4 snapshot
(verified empirically -- rib_left/right_4or5 through _12 all non-empty
across a real sample of cases; SPEC.md still documents ribs as "RESERVED,
empty in v3"). A low-riding 11th/12th rib is a cited constraint on true
lateral access at the thoracolumbar junction, same clinical shape as the
iliac-crest-height check at L4-L5 (`crest_height.py`) -- this is
deliberately the same kind of measurement: a single vertical clearance
between a bony landmark and a target disc space.

v1 checked vertical clearance using the single most-inferior point of the
WHOLE rib -- wrong, verified on real data: a rib slopes downward as it
curves from its posterior costovertebral attachment out to its anterior/
lateral arc, so the whole-bone minimum is often way out at the anterior
segment, nowhere near a lateral surgical corridor. v2 restricts to the
POSTERIOR segment (within `posterior_window_mm` of the rib's own most
posterior point, by A-P position along the same data-derived patient L-R
axis PI/SS/PT/crest-height use) before taking the inferior point -- the
segment actually adjacent to a lateral retroperitoneal approach.

Also worth stating plainly: even restricted this way, this checks vertical
(cranial-caudal) clearance only, not whether the rib sits over the true
lateral trajectory in A-P/L-R -- that needs the retroperitoneal corridor
(psoas) to define, and psoas is still unpopulated (checked directly,
verified empirically empty in every case sampled). And the dataset ships
its own QC worklists (`rib_worklist.json`, etc.) documenting active,
unresolved rib-labeling issues (e.g. a single rib bone split across two
adjacent numeric labels) on 152 cases -- our 10-case cohort isn't on any
of those worklists, but that's evidence of absence from THOSE specific
checks, not proof of clean rib labels generally.

Label ids are passed explicitly, not `ostk.labels.lid()` -- see
`crest_height.py`'s docstring for why.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .geometry import WORLD_SUPERIOR, unit
from .masks import binary_mask, largest_component, mask_world
from .spine import anterior_axis, fit_endplate
from .vertebral_rotation import _lr_axis

METHOD_VERSION = "rib-corridor-v2"


def rib_tip_from_label(label, affine, rib_name: str, label_ids: Dict[str, int], *,
                       sup_axis=WORLD_SUPERIOR, lr=(1.0, 0.0, 0.0),
                       posterior_window_mm: float = 35.0, min_voxels: int = 20):
    """Most INFERIOR point within the POSTERIOR segment of a rib mask (world
    mm) -- the segment near the costovertebral joint, adjacent to a lateral
    corridor, not the rib's whole curved arc. Returns (None, n_voxels) if
    too small/absent -- e.g. `rib_name='rib_left_12'`."""
    m = largest_component(binary_mask(label, label_ids[rib_name]))
    n_voxels = int(m.sum())
    if n_voxels < min_voxels:
        return None, n_voxels
    pts = mask_world(m, affine)
    a = unit(sup_axis)
    ap = anterior_axis(a, lr)
    ap_proj = pts @ ap
    posterior_pts = pts[ap_proj <= ap_proj.min() + posterior_window_mm]
    if len(posterior_pts) < min_voxels:
        posterior_pts = pts  # rib too short/fragment to restrict; use whole mask
    tip = posterior_pts[int(np.argmin(posterior_pts @ a))]
    return tip, n_voxels


def rib_corridor_clearance_from_label(label, affine, rib_name: str, upper_level: str,
                                      lower_level: str, label_ids: Dict[str, int], *,
                                      case_id: str = "", sup_axis=WORLD_SUPERIOR,
                                      head_frac: float = 0.35, min_voxels: int = 20,
                                      posterior_window_mm: float = 35.0) -> Dict:
    """Vertical clearance (mm) between `rib_name`'s posterior-segment
    inferior tip and the disc space's superior boundary (`upper_level`'s
    inferior endplate) -- e.g. rib_name='rib_left_12', upper_level='T12',
    lower_level='L1' for the T12-L1 disc. Positive = the rib tip sits above
    (cranial to) the disc space, i.e. clear; near-zero/negative = the rib
    extends into or past the disc level, a corridor-accessibility flag.

    Never silently drops a bad case: missing/small input -> value None plus
    a qc_flags entry (SPEC §4)."""
    flags: list = []
    a = unit(sup_axis)

    lr = _lr_axis(label, affine, label_ids, sup_axis, head_frac, min_voxels)
    if lr is None:
        flags.append("sagittal_ref_fallback")
        lr = unit(np.array([1.0, 0.0, 0.0]))

    tip, n_vox = rib_tip_from_label(label, affine, rib_name, label_ids, sup_axis=sup_axis,
                                    lr=lr, posterior_window_mm=posterior_window_mm,
                                    min_voxels=min_voxels)
    if tip is None:
        flags.append(f"missing_label:{rib_name}")

    m = largest_component(binary_mask(label, label_ids[upper_level]))
    disc_top = None
    if m.any():
        pts = mask_world(m, affine)
        ep = fit_endplate(pts, sup_axis, "inferior")
        if ep is not None:
            disc_top = ep[0]
    if disc_top is None:
        flags.append(f"missing_label:{upper_level}")

    result: Dict = {
        "case_id": case_id, "parameter": "rib_corridor_clearance",
        "level": f"{upper_level}-{lower_level}", "rib": rib_name, "units": "mm",
        "value": None, "qc_flags": flags,
        "method_version": METHOD_VERSION, "supine_ct": True,
    }
    if tip is not None and disc_top is not None:
        clearance = float((tip - disc_top) @ a)
        result["value"] = round(clearance, 3)
        result["landmarks_world_mm"] = {
            "rib_tip": tip.tolist(), "disc_superior_boundary": disc_top.tolist(),
        }
    if not flags:
        flags.append("ok")
    return result
