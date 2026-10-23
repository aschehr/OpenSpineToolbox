"""ostk.vertebral_hu_zhou — L1-L4 average trabecular HU ("vertebral body
BMD"), reconstructed to replicate Zhou et al. 2021's validated OLIF
cage-subsidence protocol closely enough to apply their cutoff honestly --
same discipline as `crest_height_yang.py`: a SEPARATE, protocol-matched
implementation, not `vertebral_hu.py`'s general-purpose single-level
metric with someone else's cutoff bolted onto it. `vertebral_hu.py`'s own
docstring flags exactly this risk; this module is how we earned the right
to actually cite a number.

Verified against the paper's own Methods text (PMC8600787, fetched
directly -- not from training-data recall):

  Zhou J, Yuan C, Liu C, et al. Hounsfield unit value on CT as a
  predictor of cage subsidence following stand-alone oblique lumbar
  interbody fusion. BMC Musculoskelet Disord. 2021;22:960.

  "Three measurements were obtained: just superior to the inferior
  endplate; the mid-vertebral body; and just inferior to the superior
  endplate." Per vertebra, "an elliptical area of interest (AOI) was
  drawn on an axial image of the lumbar vertebral body, including the
  largest possible cancellous bone and excluding cortical edges,
  osteophytes, and osseous abnormalities" and "vertebral body BMD was
  defined as the average HU value of the three AOIs" (intra-/inter-
  observer ICC 0.989/0.972). The "L1-L4 horizontal plane" average of
  those four per-vertebra BMDs had AUC 0.909 (95% CI 0.834-0.984,
  P<0.001; 93.8% sensitivity, 81.7% specificity) at a 115.7 HU cutoff
  for predicting cage subsidence after stand-alone OLIF (n=76 patients,
  84 segments).

We have no PACS software and no human drawing ellipses, so the "exclude
cortex, maximize cancellous bone" AOI at each of the three described
heights is reconstructed with `vertebral_hu.trabecular_roi` (an inward
erosion from an already-isolated vertebral body -- same exclusion intent,
morphological instead of manual) intersected with a thin axial slab at
each height, rather than one whole-body 3D core -- the one deliberate
methodology difference from `vertebral_hu.py`, made specifically to track
this paper's own three-slice protocol.

L5 is NOT included in the average: the paper's own Methods text never
explains why it limits "L1-L4 horizontal plane" to four vertebrae, so we
don't extend the average to L5 on our own guess -- if any of L1-L4 is
missing/unfittable, the whole case reports `value=None` with a qc_flags
entry rather than silently averaging over fewer vertebrae than the
validated protocol used.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .endplate_footprint import isolate_vertebral_body
from .geometry import WORLD_SUPERIOR, unit
from .io import voxels_to_world
from .masks import binary_mask, largest_component, mask_indices
from .vertebral_hu import trabecular_roi

METHOD_VERSION = "vertebral-hu-zhou-v1"

ZHOU_SUBSIDENCE_CUTOFF_HU = 115.7
ZHOU_CITATION = ("Zhou J, Yuan C, Liu C, et al. Hounsfield unit value on CT as a predictor of "
                 "cage subsidence following stand-alone oblique lumbar interbody fusion. "
                 "BMC Musculoskelet Disord. 2021;22:960.")
ZHOU_LEVELS = ("L1", "L2", "L3", "L4")
_SLICE_FRACTIONS = {"near_inferior_endplate": 0.15, "mid_body": 0.5, "near_superior_endplate": 0.85}


def _vertebra_bmd_zhou(label, ct, affine, level: str, label_ids: Dict[str, int], *,
                       sup_axis, erosion_mm: float, cortical_margin_mm: float,
                       slab_halfwidth_mm: float, min_voxels_per_slice: int,
                       body_mask=None) -> Dict:
    """One vertebra's "BMD" (mean HU of three axial-slab cancellous-bone
    samples) per Zhou et al.'s definition. Never raises: returns
    value=None + qc_flags on any failure."""
    flags: List[str] = []
    body = body_mask
    if body is None:
        m = largest_component(binary_mask(label, label_ids[level]))
        if not m.any():
            return {"value": None, "qc_flags": [f"missing_label:{level}"], "slices": {}}
        body = isolate_vertebral_body(m, affine, erosion_mm=erosion_mm)
    if body is None or not body.any():
        return {"value": None, "qc_flags": [f"body_isolation_failed:{level}"], "slices": {}}

    core = trabecular_roi(body, affine, cortical_margin_mm=cortical_margin_mm)
    idx = mask_indices(core)
    if len(idx) == 0:
        return {"value": None, "qc_flags": [f"trabecular_roi_empty:{level}"], "slices": {}}

    a = unit(sup_axis)
    proj = voxels_to_world(idx, affine) @ a
    z_lo, z_hi = float(proj.min()), float(proj.max())

    slice_means: Dict[str, Optional[float]] = {}
    for name, frac in _SLICE_FRACTIONS.items():
        z_target = z_lo + frac * (z_hi - z_lo)
        in_slab = np.abs(proj - z_target) <= slab_halfwidth_mm
        n = int(in_slab.sum())
        if n < min_voxels_per_slice:
            slice_means[name] = None
            flags.append(f"low_voxels:{level}:{name}")
            continue
        vox = idx[in_slab]
        hu = np.asarray(ct)[vox[:, 0], vox[:, 1], vox[:, 2]]
        slice_means[name] = float(np.mean(hu))

    available = [v for v in slice_means.values() if v is not None]
    value = round(float(np.mean(available)), 2) if available else None
    if value is None:
        flags.append(f"no_slices_available:{level}")
    elif len(available) < 3:
        flags.append(f"partial_slices:{level}")
    return {"value": value, "qc_flags": flags,
           "slices": {k: (round(v, 2) if v is not None else None) for k, v in slice_means.items()}}


def zhou_l1_l4_bmd_from_label(label, ct, affine, label_ids: Dict[str, int], *,
                              case_id: str = "", sup_axis=WORLD_SUPERIOR,
                              erosion_mm: float = 10.0, cortical_margin_mm: float = 4.0,
                              slab_halfwidth_mm: float = 3.0, min_voxels_per_slice: int = 30,
                              body_masks: Optional[Dict[str, np.ndarray]] = None) -> Dict:
    """L1-L4 average "vertebral body BMD" (mean HU), replicating Zhou et al.
    2021's OLIF cage-subsidence protocol, plus the `above/below` call
    against their validated 115.7 HU cutoff. `ct` must be on the SAME grid
    as `label` (`ostk.io.load_ct` on the same case's CT file).

    `body_masks`: optional {level: already-isolated body mask} to reuse
    masks a caller (e.g. `llif.py`) already paid to compute for L1-L4.

    Never silently drops a bad case: any of L1-L4 missing/unfittable ->
    value=None (we don't average over fewer vertebrae than the validated
    protocol used) plus a qc_flags entry per affected level (SPEC §4)."""
    per_level: Dict[str, Dict] = {}
    flags: List[str] = []
    for lv in ZHOU_LEVELS:
        bm = (body_masks or {}).get(lv)
        r = _vertebra_bmd_zhou(label, ct, affine, lv, label_ids, sup_axis=sup_axis,
                               erosion_mm=erosion_mm, cortical_margin_mm=cortical_margin_mm,
                               slab_halfwidth_mm=slab_halfwidth_mm,
                               min_voxels_per_slice=min_voxels_per_slice, body_mask=bm)
        per_level[lv] = r
        flags.extend(r["qc_flags"])

    values = [per_level[lv]["value"] for lv in ZHOU_LEVELS]
    value = round(float(np.mean(values)), 2) if all(v is not None for v in values) else None
    if value is None:
        flags.append("l1_l4_average_unavailable")

    result: Dict = {
        "case_id": case_id, "parameter": "zhou_l1_l4_bmd", "level": "L1-L4", "units": "HU",
        "value": value, "cutoff_hu": ZHOU_SUBSIDENCE_CUTOFF_HU,
        "below_zhou_115_7_cutoff": (bool(value < ZHOU_SUBSIDENCE_CUTOFF_HU)
                                    if value is not None else None),
        "per_level": per_level, "citation": ZHOU_CITATION,
        "qc_flags": flags or ["ok"], "method_version": METHOD_VERSION, "supine_ct": True,
    }
    return result
