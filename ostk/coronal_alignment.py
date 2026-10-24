"""ostk.coronal_alignment — coronal-plane alignment between two adjacent
vertebrae at a target level, for LLIF safety planning.

Two related parameters, both between `upper_level` and `lower_level`
(e.g. 'L4','L5' for the L4-L5 disc):

- lateral listhesis: left-right offset between the two endplate-corner
  midpoints -- degenerative CORONAL subluxation risks cage mis-centering
  and (at more lateral offsets) vascular injury on a lateral trajectory.
  Reuses `spine.endplate_corner_landmarks` (fine for this: listhesis only
  needs the corner-chord MIDPOINT, not the normal).
- coronal disc angle: the Cobb angle between the two endplate normals,
  viewed in the coronal plane -- a severely angulated/collapsed disc space
  complicates cage insertion and correction planning.

coronal disc angle does NOT reuse `endplate_corner_landmarks` -- a real bug
was found and fixed here: that helper's normal is `cross(lr, Pc-A))`, which
is mathematically perpendicular to `lr` BY CONSTRUCTION, for any input.
Projected into the coronal view (which keeps only the {lr, sup_axis}
plane), that normal can only ever land exactly on sup_axis -- Cobb angle
between any two such normals is IDENTICALLY 0 degrees, regardless of real
anatomy. Confirmed on real data: all 10 validation cases returned exactly
0.0 deg, which is what exposed it (real coronal tilt varying to exactly
zero in every one of 10 different patients is not plausible). That corner
formula is correct for SAGITTAL tilt (what `disc_height.py`/
`vertebral_wedging.py`/PI/LL need) but structurally cannot carry coronal
information. `cobb.py`'s existing, working `coronal_cobb_from_label` avoids
this -- not by using a different formula on purpose, but because it passes
`fit_endplate(..., "superior", ant_frac, ...)` with `ant_frac` landing
positionally in the `method` parameter slot; since `0.6 != "corner"`,
`fit_endplate` falls through to its `method="surface"` branch (a true
best-fit 3-D plane, which DOES carry coronal orientation). Below, the same
`method="surface"` is requested explicitly rather than relying on that
accidental fallthrough.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .geometry import WORLD_SUPERIOR, cobb_angle, unit
from .masks import binary_mask, largest_component, mask_world
from .spine import anterior_axis, corner_params_for_level, endplate_corner_landmarks, fit_endplate
from .vertebral_rotation import _lr_axis

METHOD_VERSION = "coronal-alignment-v2"


def lateral_listhesis_from_label(label, affine, upper_level: str, lower_level: str,
                                 label_ids: Dict[str, int], *, case_id: str = "",
                                 sup_axis=WORLD_SUPERIOR, head_frac: float = 0.35,
                                 min_voxels: int = 30, lr=None) -> Dict:
    """Left-right offset (mm, signed: positive = lower body shifted toward
    patient right relative to upper) between `upper_level`'s inferior-
    endplate midpoint and `lower_level`'s superior-endplate midpoint,
    projected onto the patient's true L-R axis. `lr`: optional precomputed
    axis (see `vertebral_rotation.vertebral_axial_rotation_from_label`'s
    docstring -- same caching purpose for multi-parameter callers). Never
    silently drops a bad case: missing/unfittable input -> value None plus
    a qc_flags entry."""
    flags: list = []
    if lr is None:
        lr = _lr_axis(label, affine, label_ids, sup_axis, head_frac, min_voxels)
    if lr is None:
        flags.append("sagittal_ref_fallback")
        lr = unit(np.array([1.0, 0.0, 0.0]))

    upper = endplate_corner_landmarks(label, affine, upper_level, "inferior", label_ids,
                                      sup_axis=sup_axis, lr=lr)
    if upper is None:
        flags.append(f"missing_label:{upper_level}")
    lower = endplate_corner_landmarks(label, affine, lower_level, "superior", label_ids,
                                      sup_axis=sup_axis, lr=lr)
    if lower is None:
        flags.append(f"missing_label:{lower_level}")

    result: Dict = {
        "case_id": case_id, "parameter": "lateral_listhesis",
        "level": f"{upper_level}-{lower_level}", "units": "mm", "value": None,
        "qc_flags": flags, "method_version": METHOD_VERSION, "supine_ct": True,
    }
    if upper is not None and lower is not None:
        result["value"] = round(float((lower["mid"] - upper["mid"]) @ lr), 3)
    if not flags:
        flags.append("ok")
    return result


def coronal_disc_angle_from_label(label, affine, upper_level: str, lower_level: str,
                                  label_ids: Dict[str, int], *, case_id: str = "",
                                  sup_axis=WORLD_SUPERIOR, head_frac: float = 0.35,
                                  min_voxels: int = 30, lr=None) -> Dict:
    """Coronal-plane Cobb angle (degrees, unsigned) between `upper_level`'s
    inferior endplate and `lower_level`'s superior endplate -- flags a
    severely angulated/collapsed disc space. `lr`: optional precomputed
    axis, same caching purpose as `lateral_listhesis_from_label`. Never
    silently drops a bad case: missing/unfittable input -> value None plus
    a qc_flags entry."""
    flags: list = []
    if lr is None:
        lr = _lr_axis(label, affine, label_ids, sup_axis, head_frac, min_voxels)
    if lr is None:
        flags.append("sagittal_ref_fallback")
        lr = unit(np.array([1.0, 0.0, 0.0]))

    def _surface_normal(level, which):
        m = largest_component(binary_mask(label, label_ids[level]))
        if not m.any():
            return None
        pts = mask_world(m, affine)
        fit = fit_endplate(pts, sup_axis, which, method="surface", lr=lr,
                           **corner_params_for_level(level))
        return None if fit is None else fit[1]  # (centroid, normal, rms)

    upper_n = _surface_normal(upper_level, "inferior")
    if upper_n is None:
        flags.append(f"missing_label:{upper_level}")
    lower_n = _surface_normal(lower_level, "superior")
    if lower_n is None:
        flags.append(f"missing_label:{lower_level}")

    result: Dict = {
        "case_id": case_id, "parameter": "coronal_disc_angle",
        "level": f"{upper_level}-{lower_level}", "units": "degrees", "value": None,
        "qc_flags": flags, "method_version": METHOD_VERSION, "supine_ct": True,
    }
    if upper_n is not None and lower_n is not None:
        view_normal = anterior_axis(unit(sup_axis), lr)  # coronal view = project out A-P
        result["value"] = round(cobb_angle(upper_n, lower_n, view_normal), 3)
    if not flags:
        flags.append("ok")
    return result
