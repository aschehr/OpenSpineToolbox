"""ostk.vertebral_rotation — axial rotation of a vertebral body relative to
the patient L-R axis, for LLIF corridor-safety assessment (rotation changes
where the disc space and lumbar plexus sit relative to a lateral trajectory,
especially in degenerative/scoliotic curves).

Reuses `endplate_footprint.isolate_vertebral_body` first: a naive PCA over
the WHOLE vertebra (body + posterior elements) would have its long-axis
estimate biased by the pedicles/transverse-processes/facets, the same
contamination problem `endplate_footprint.py` already found and fixed for
width measurement. And reuses the same data-derived, femoral-head-based
patient L-R axis PI/SS/PT/crest-height all use (SPEC §3: "don't derive a
second, inconsistent L-R estimate") -- via `metrics.femoral_head_center`
directly rather than `metrics._lr_axis_from_label`, which hard-codes v3
structure NAMES; this module needs explicit ids for the real v4 data (same
reason as `crest_height.py`). `metrics.femoral_head_center` itself was
extended (backward-compatibly) to accept an int id as well as a name, for
exactly this.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .geometry import WORLD_SUPERIOR, project_out, unit
from .masks import binary_mask, largest_component, mask_world
from .endplate_footprint import isolate_vertebral_body
from .metrics import femoral_head_center

METHOD_VERSION = "vertebral-rotation-v1"


def _lr_axis(label, affine, label_ids: Dict[str, int], sup_axis, head_frac, min_voxels
            ) -> Optional[np.ndarray]:
    L = femoral_head_center(label, affine, label_ids["femur_left"], label_ids["left_hip"],
                            sup_axis=sup_axis, slab_frac=head_frac, min_voxels=min_voxels)
    R = femoral_head_center(label, affine, label_ids["femur_right"], label_ids["right_hip"],
                            sup_axis=sup_axis, slab_frac=head_frac, min_voxels=min_voxels)
    if L is None or R is None:
        return None
    return unit(R[0] - L[0])


def vertebral_axial_rotation_from_label(label, affine, level: str, label_ids: Dict[str, int], *,
                                        case_id: str = "", sup_axis=WORLD_SUPERIOR,
                                        head_frac: float = 0.35, min_voxels: int = 50,
                                        erosion_mm: float = 10.0, lr=None,
                                        body_mask=None) -> Dict:
    """Axial rotation (degrees) of `level`'s vertebral body: the angle
    between the body's own transverse (long) axis, found by 2-D PCA of the
    body-isolated point cloud projected into the axial plane, and the
    patient's true L-R axis. Signed; positive = body's right side rotated
    anteriorly (right-hand rule about the cranial axis). Folded to (-90, 90]
    since an AXIS (not a directed vector) has 180-degree ambiguity.

    `lr`: optional precomputed patient L-R axis (from `_lr_axis`). `body_mask`:
    optional precomputed, already-isolated body mask (from
    `isolate_vertebral_body`). Pass either when calling for multiple levels/
    parameters on the SAME case (e.g. `llif.llif_level_report_from_label`,
    which also needs the body mask for `endplate_footprint`) so the
    expensive femoral-head fit / erosion isn't redundantly recomputed each
    time. Both None (default) computes them.

    Never silently drops a bad case: missing/unfittable input -> value None
    plus a qc_flags entry (SPEC §4)."""
    flags: list = []
    if lr is None:
        lr = _lr_axis(label, affine, label_ids, sup_axis, head_frac, min_voxels)
    if lr is None:
        flags.append("sagittal_ref_fallback")
        lr = unit(np.array([1.0, 0.0, 0.0]))

    result: Dict = {
        "case_id": case_id, "parameter": "vertebral_axial_rotation", "level": level,
        "units": "degrees", "value": None, "qc_flags": flags,
        "method_version": METHOD_VERSION, "supine_ct": True,
    }

    body = body_mask
    if body is None:
        m = largest_component(binary_mask(label, label_ids[level]))
        if not m.any():
            flags.append(f"missing_label:{level}")
            return result
        body = isolate_vertebral_body(m, affine, erosion_mm=erosion_mm)
    if not body.any():
        flags.append(f"body_isolation_failed:{level}")
        return result

    pts = mask_world(body, affine)
    a = unit(sup_axis)
    in_plane = project_out(pts - pts.mean(0), a)  # drop the cranial-caudal component

    e1 = unit(project_out(lr, a))
    e2 = unit(np.cross(a, e1))
    uv = np.stack([in_plane @ e1, in_plane @ e2], axis=-1)
    cov = np.cov(uv.T)
    w, V = np.linalg.eigh(cov)
    long_axis = V[:, int(np.argmax(w))]  # (u,v) of the body's own transverse axis

    angle = float(np.degrees(np.arctan2(long_axis[1], long_axis[0])))
    if angle <= -90.0:
        angle += 180.0
    elif angle > 90.0:
        angle -= 180.0

    result["value"] = round(angle, 3)
    result["n_body_voxels"] = int(body.sum())
    if not flags:
        flags.append("ok")
    return result
