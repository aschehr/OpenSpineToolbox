"""ostk.vertebral_hu — mean trabecular Hounsfield-unit (HU) sampling
within a vertebral body: a bone-density (BMD) proxy relevant to LLIF cage
subsidence and screw-pullout risk (SPEC.md §6 #11, "Osteoporosis /
trabecular HU").

Needs the CT intensity volume, not just labels -- the only parameter in
this toolbox that does.

Reuses `endplate_footprint.isolate_vertebral_body` to separate the body
from posterior elements first (SPEC's "avoid... posterior elements"),
then applies a FURTHER, smaller inward erosion -- from the isolated
body's own surface, a fresh shrink, not a repeat of the topological
pedicle-severing erosion isolate_vertebral_body already did -- to exclude
the dense cortical shell and sample only the central trabecular core
(SPEC's "avoid cortex").

Validation note: unlike every other parameter in this toolbox, this one
has no crisp anatomical landmark a human can click to validate against --
it's a REGION mean, not a point; two reasonable-but-different ROI
placements in the same trabecular area will naturally give somewhat
different (but overlapping) mean HU. Manual validation therefore uses
POINT-SAMPLING (a human clicks several points inside the trabecular
region and we compare the average of those points' CT intensities to
this function's ROI mean) rather than the ICC/Pearson/kappa-against-a-
landmark protocol used elsewhere -- a "same ballpark" check, not a
precise-agreement one. See results/llif/ for that comparison once run.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .endplate_footprint import isolate_vertebral_body
from .masks import binary_mask, largest_component

METHOD_VERSION = "vertebral-hu-v1"


def trabecular_roi(body_mask: np.ndarray, affine: np.ndarray,
                   cortical_margin_mm: float = 4.0) -> np.ndarray:
    """Erode `body_mask` inward by `cortical_margin_mm` from ITS OWN
    surface (a fresh shrink, not `isolate_vertebral_body`'s topological
    pedicle-severing erosion, which restores full size via a dilate-back
    step -- here the shrinkage is the point) to exclude the cortical
    shell, leaving the central trabecular core. Crops to a local bounding
    box first for the same reason `isolate_vertebral_body` does --
    full-CT-volume distance transforms are enormously wasteful."""
    from scipy import ndimage
    empty = np.zeros_like(body_mask)
    spacing = np.array([np.linalg.norm(affine[:3, i]) for i in range(3)])
    idx = np.argwhere(body_mask)
    if len(idx) == 0:
        return empty
    margin_vox = np.ceil(cortical_margin_mm / spacing).astype(int) + 2
    lo = np.maximum(idx.min(axis=0) - margin_vox, 0)
    hi = np.minimum(idx.max(axis=0) + margin_vox + 1, np.array(body_mask.shape))
    sl = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
    sub = body_mask[sl]
    dist = ndimage.distance_transform_edt(sub, sampling=spacing)
    core_sub = dist >= cortical_margin_mm
    core = np.zeros_like(body_mask)
    core[sl] = core_sub
    return core


def vertebral_hu_from_label(label, ct, affine, level: str, label_ids: Dict[str, int], *,
                            case_id: str = "", erosion_mm: float = 10.0,
                            cortical_margin_mm: float = 4.0, min_voxels: int = 100,
                            body_mask=None) -> Dict:
    """Mean/median/SD Hounsfield units within `level`'s central trabecular
    core -- a bone-density (BMD) proxy. `ct` must be on the SAME grid as
    `label` (i.e. `ostk.io.load_ct` on the same case's CT file).

    `body_mask`, if given, is an ALREADY-ISOLATED body mask (from
    `endplate_footprint.isolate_vertebral_body`) for this level -- lets
    callers that already paid for that erosion (e.g. `llif.py`'s report)
    skip repeating it here.

    Never silently drops a bad case: missing/unfittable input -> value
    None plus a qc_flags entry (SPEC §4)."""
    flags: list = []
    body = body_mask
    if body is None:
        m = largest_component(binary_mask(label, label_ids[level]))
        if not m.any():
            flags.append(f"missing_label:{level}")
        else:
            body = isolate_vertebral_body(m, affine, erosion_mm=erosion_mm)
            if not body.any():
                flags.append(f"body_isolation_failed:{level}")
                body = None
    elif not body.any():
        flags.append(f"body_isolation_failed:{level}")
        body = None

    result: Dict = {
        "case_id": case_id, "parameter": "vertebral_hu", "level": level, "units": "HU",
        "mean_hu": None, "median_hu": None, "sd_hu": None, "n_voxels": 0,
        "qc_flags": flags, "method_version": METHOD_VERSION, "supine_ct": True,
    }
    if body is not None:
        core = trabecular_roi(body, affine, cortical_margin_mm=cortical_margin_mm)
        n_voxels = int(core.sum())
        if n_voxels < min_voxels:
            flags.append("low_voxels:trabecular_roi")
        else:
            hu = np.asarray(ct)[core]
            result["mean_hu"] = round(float(np.mean(hu)), 2)
            result["median_hu"] = round(float(np.median(hu)), 2)
            result["sd_hu"] = round(float(np.std(hu)), 2)
            result["n_voxels"] = n_voxels
    if not flags:
        flags.append("ok")
    return result
