"""ostk.endplate_footprint — vertebral endplate AP/ML dimensions for LLIF
cage-sizing planning.

AP/ML width needs the vertebral BODY alone (not the whole vertebra): the
transverse processes are wider than the true body, and -- verified
empirically -- their tips sit at roughly the SAME A-P position as the
body's own posterior half, so no positional (A-P or L-R fractional) crop
can separate them from the body; only `isolate_vertebral_body`'s
morphological erosion/largest-component/dilate approach does. Width is
then measured as the bounding-box extent of that isolated body's top
(disc-facing) surface -- the caliper measurement a surgeon would take, not
an endplate-line/plane fit (that's what `spine.fit_endplate` is for).

Label ids are passed explicitly (not `ostk.labels.lid()`) for the same
reason as `crest_height.py`: the real v4 dataset renumbers ids relative to
what `ostk.labels` encodes. Load them from the dataset's own
`dataset_labels.json` via `crest_height.load_label_ids`.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .geometry import WORLD_SUPERIOR, unit
from .masks import binary_mask, largest_component, mask_world
from .spine import anterior_axis

METHOD_VERSION = "endplate-footprint-v3"


def isolate_vertebral_body(mask: np.ndarray, affine: np.ndarray,
                           erosion_mm: float = 10.0) -> np.ndarray:
    """Isolate the vertebral BODY from a full-vertebra binary mask by
    severing its connection to the posterior arch through the pedicles.

    The pedicles are a thin bony "neck" (diameter roughly 8-18mm) joining
    the body to the lamina/transverse-processes/spinous-process; a plain
    AP-position or L-R-position crop CANNOT separate body from processes,
    because the transverse-process tips sit at roughly the SAME A-P
    position as the body's own posterior half -- they're offset laterally,
    not posteriorly (verified empirically: scanning ML extent by A-P bin on
    a real case showed the widest point, the transverse-process tips, in
    the MIDDLE of the A-P range, not at the posterior extreme).

    So this isolates by MORPHOLOGY instead: erode the mask by `erosion_mm`
    in real space (Euclidean distance transform, correct even with
    anisotropic voxel spacing), keep the largest surviving connected
    component (the body, since the thinner pedicle vanishes first), then
    dilate that component back by the same radius -- a plain dilation, not
    geodesic reconstruction, so it cannot re-bridge to the arch even though
    the pedicle voxels are still technically present in the input mask.

    Crops to the mask's own bounding box (+ margin) first: a vertebra is a
    tiny fraction of a full CT volume, and `distance_transform_edt` on the
    full array (e.g. 512x667x512) is enormously wasteful -- this dropped a
    single-case runtime from >12 CPU-minutes (killed, still not done) to
    under a second."""
    from scipy import ndimage
    empty = np.zeros_like(mask)
    spacing = np.array([np.linalg.norm(affine[:3, i]) for i in range(3)])
    margin_vox = np.ceil(erosion_mm / spacing).astype(int) + 2
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return empty
    lo = np.maximum(idx.min(axis=0) - margin_vox, 0)
    hi = np.minimum(idx.max(axis=0) + margin_vox + 1, np.array(mask.shape))
    sl = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
    sub = mask[sl]

    dist_in = ndimage.distance_transform_edt(sub, sampling=spacing)
    eroded = dist_in >= erosion_mm
    if not eroded.any():
        # erosion_mm too large for this structure -- FAILS LOUDLY (empty
        # mask) rather than silently falling back to the un-eroded input,
        # which would mask the failure behind a plausible-looking number.
        return empty
    lab, n = ndimage.label(eroded)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    seed = lab == int(counts.argmax())
    dist_out = ndimage.distance_transform_edt(~seed, sampling=spacing)
    body_sub = (dist_out <= erosion_mm) & sub

    body = np.zeros_like(mask)
    body[sl] = body_sub
    return body


def _top_surface(points, normal_axis, which: str, lr=(1.0, 0.0, 0.0), nbins: int = 30):
    """Per-(A-P, L-R)-cell extreme voxel along `normal_axis` (the disc-facing
    cortical surface), no positional cropping -- the caller is expected to
    have already isolated the body (e.g. via `isolate_vertebral_body`)."""
    P = np.asarray(points, dtype=np.float64)
    a = unit(normal_axis)
    ap = anterior_axis(a, lr)
    lrv = unit(lr)
    u, v, w = P @ lrv, P @ ap, P @ a
    ui = np.floor((u - u.min()) / (np.ptp(u) + 1e-9) * nbins).astype(int)
    vi = np.floor((v - v.min()) / (np.ptp(v) + 1e-9) * nbins).astype(int)
    key = ui * (nbins + 1) + vi
    sgn = -1.0 if which == "superior" else 1.0
    order = np.lexsort((sgn * w, key))
    sk = key[order]
    first = np.ones(len(order), bool)
    first[1:] = sk[1:] != sk[:-1]
    return P[order[first]]


def endplate_footprint(body_points, normal_axis=WORLD_SUPERIOR, which: str = "superior",
                       lr=(1.0, 0.0, 0.0), min_points: int = 20) -> Optional[Dict]:
    """AP width and ML width (bounding-box extent, mm) of an ALREADY BODY-
    ISOLATED vertebra's disc-facing surface -- the caliper measurement a
    surgeon would take for cage sizing. `body_points` must come from a mask
    already run through `isolate_vertebral_body` (this function does not
    isolate the body itself -- see `endplate_footprint_from_label`)."""
    surf = _top_surface(body_points, normal_axis, which, lr=lr)
    if len(surf) < min_points:
        return None
    ap = anterior_axis(unit(normal_axis), lr)
    lrv = unit(lr)
    ap_proj = surf @ ap
    lat_proj = surf @ lrv
    return {
        "ap_width_mm": round(float(ap_proj.max() - ap_proj.min()), 3),
        "ml_width_mm": round(float(lat_proj.max() - lat_proj.min()), 3),
        "n_surface_points": int(len(surf)),
    }


def endplate_footprint_from_label(label, affine, level: str, which: str,
                                  label_ids: Dict[str, int], *,
                                  sup_axis=WORLD_SUPERIOR, lr=(1.0, 0.0, 0.0),
                                  erosion_mm: float = 10.0) -> Optional[Dict]:
    """Convenience: fit one endplate's footprint straight from a label volume
    + structure name, isolating the vertebral body first."""
    m = largest_component(binary_mask(label, label_ids[level]))
    if not m.any():
        return None
    body_mask = isolate_vertebral_body(m, affine, erosion_mm=erosion_mm)
    if not body_mask.any():
        return None
    pts = mask_world(body_mask, affine)
    return endplate_footprint(pts, sup_axis, which, lr=lr)


def disc_footprint_from_label(label, affine, upper_level: str, lower_level: str,
                              label_ids: Dict[str, int], *, case_id: str = "",
                              sup_axis=WORLD_SUPERIOR, lr=(1.0, 0.0, 0.0)) -> Dict:
    """Cage-sizing footprint for the disc space BETWEEN `upper_level` and
    `lower_level` (e.g. 'L4','L5' for the L4-L5 disc): the inferior endplate
    of the upper body and the superior endplate of the lower body -- the two
    surfaces that actually bound the disc space, reported separately (not
    averaged) since surgeons compare them, e.g. to undersize to the smaller.

    Never silently drops a bad case: a missing/unfittable surface stays None
    with a qc_flags entry, mirroring the rest of ostk (SPEC §4)."""
    flags: list = []
    upper_fp = endplate_footprint_from_label(label, affine, upper_level, "inferior",
                                             label_ids, sup_axis=sup_axis, lr=lr)
    if upper_fp is None:
        flags.append(f"missing_label:{upper_level}")
    lower_fp = endplate_footprint_from_label(label, affine, lower_level, "superior",
                                             label_ids, sup_axis=sup_axis, lr=lr)
    if lower_fp is None:
        flags.append(f"missing_label:{lower_level}")
    if not flags:
        flags.append("ok")
    return {
        "case_id": case_id, "parameter": "disc_footprint",
        "level": f"{upper_level}-{lower_level}", "units": "mm",
        f"{upper_level}_inferior": upper_fp,
        f"{lower_level}_superior": lower_fp,
        "qc_flags": flags,
        "method_version": METHOD_VERSION,
        "supine_ct": True,
    }
