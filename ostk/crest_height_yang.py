"""ostk.crest_height_yang — iliac crest height relative to the L4-L5 disc
space, replicating Yang et al. 2024's exact CT measurement protocol, so
their validated 12mm subsidence-risk cutoff can be applied to our own
output directly rather than just cited as general context.

Citation: Yang JH, Lee KJ, Lee SY, Lee HR. "Relationship of the Iliac
Crest Height with Subsidence After Oblique Lateral Interbody Fusion at
L4-5: A Quantitative and Categorical Analysis." J Clin Med.
2024;13(20):6223. PMC11508602. Verified against the paper's actual
Methods text and Figure 2/3 captions (not a lossy AI summary of a fetch,
which gave three mutually-inconsistent descriptions before the real text
was obtained).

Their protocol (Methods §2.2, Figures 2-3):
  1. Coronal plane: the highest point of EACH ilium (left and right) --
     the crest apex -- connected by a line; its MIDPOINT is the bilateral
     crest reference point ("thick red line represents the connection
     between the highest points of both iliac crests... the midpoint of
     this line serving as a reference").
  2. Sagittal plane: that midpoint's height is projected as a horizontal
     reference line (done to reduce rotational error from oblique patient
     positioning on the CT table -- the reason they use a data-derived
     construction rather than a raw scanner-axis one, the same reasoning
     this toolbox's `lr` axis convention already follows).
  3. The L4-L5 disc space's ventral (anterior) midpoint: the point on the
     disc's anterior margin, cranio-caudally centered between L4's
     inferior endplate and L5's superior endplate.
  4. Height = vertical (cranial-caudal) distance from (1) down to (3).

Their result: CT-measured height >12mm was associated with higher L4-5
OLIF subsidence risk (ROC AUC 0.688, p=0.042; 43% vs. 10-19% subsidence
across their three crest-level groups, p=0.01).

This is deliberately a SEPARATE function from `crest_height.py`'s own
obliquity/per-side metric, which measures something different (each
side's height independently, plus left-right asymmetry) and has its own
independent manual validation (ICC 0.9885 against real ITK-SNAP
measurements) that has nothing to do with Yang et al.'s cutoff. Applying
Yang's 12mm to that OTHER metric would have been exactly the kind of
measurement mismatch this module exists to avoid.

Reuses `crest_height.crest_apex_from_label` (the per-side apex point) and
`spine.endplate_corner_landmarks` (the anterior corner) -- no new
geometry primitives, just this specific combination of existing ones.
"""
from __future__ import annotations

from typing import Dict

from .crest_height import crest_apex_from_label
from .geometry import WORLD_SUPERIOR, unit
from .spine import endplate_corner_landmarks
from .vertebral_rotation import _lr_axis

METHOD_VERSION = "crest-height-yang-v1"
YANG_SUBSIDENCE_CUTOFF_MM = 12.0  # Yang et al. 2024, ROC AUC 0.688, p=0.042
YANG_CITATION = ("Yang JH, Lee KJ, Lee SY, Lee HR. Relationship of the Iliac Crest "
                 "Height with Subsidence After Oblique Lateral Interbody Fusion at "
                 "L4-5. J Clin Med. 2024;13(20):6223. PMC11508602.")


def yang_crest_height_from_label(label, affine, label_ids: Dict[str, int], *,
                                 case_id: str = "", sup_axis=WORLD_SUPERIOR,
                                 head_frac: float = 0.35, min_voxels: int = 50) -> Dict:
    """Iliac crest height relative to the L4-L5 disc space's ventral
    midpoint, replicating Yang et al. 2024's CT protocol (see module
    docstring). Positive = crest above (cranial to) the disc's ventral
    midpoint. `above_yang_12mm_cutoff`: their validated subsidence-risk
    flag, since THIS function (unlike `crest_height.py`'s own metric) is
    built to match the measurement they actually validated it against.

    Uses the same data-derived patient L-R axis (bicoxofemoral vector) the
    rest of this toolbox uses for the anterior-corner lookups -- Yang et
    al. specifically designed their sagittal-plane construction to reduce
    error from oblique/rotated patient positioning, so a scanner-axis
    fallback would work against their own stated rationale.

    Never silently drops a bad case: missing/unfittable input -> value
    None plus a qc_flags entry (SPEC §4)."""
    flags: list = []
    a = unit(sup_axis)

    lr = _lr_axis(label, affine, label_ids, sup_axis, head_frac, min_voxels)
    if lr is None:
        flags.append("sagittal_ref_fallback")
        lr = unit((1.0, 0.0, 0.0))

    apex_pts = {}
    for side in ("right", "left"):
        apex, n_vox = crest_apex_from_label(label, affine, side, label_ids,
                                            sup_axis=sup_axis, min_voxels=min_voxels)
        if apex is None:
            flags.append(f"low_voxels:{side}_hip")
        else:
            apex_pts[side] = apex

    l4 = endplate_corner_landmarks(label, affine, "L4", "inferior", label_ids,
                                   sup_axis=sup_axis, lr=lr)
    if l4 is None:
        flags.append("missing_label:L4")
    l5 = endplate_corner_landmarks(label, affine, "L5", "superior", label_ids,
                                   sup_axis=sup_axis, lr=lr)
    if l5 is None:
        flags.append("missing_label:L5")

    result: Dict = {
        "case_id": case_id, "parameter": "yang_crest_height", "units": "mm",
        "value": None, "above_yang_12mm_cutoff": None,
        "qc_flags": flags, "method_version": METHOD_VERSION, "supine_ct": True,
        "citation": YANG_CITATION,
    }
    if "right" in apex_pts and "left" in apex_pts and l4 is not None and l5 is not None:
        crest_ref = 0.5 * (apex_pts["right"] + apex_pts["left"])
        disc_ventral_mid = 0.5 * (l4["anterior_corner"] + l5["anterior_corner"])
        height = float((crest_ref - disc_ventral_mid) @ a)
        result["value"] = round(height, 3)
        result["above_yang_12mm_cutoff"] = bool(height > YANG_SUBSIDENCE_CUTOFF_MM)
        result["landmarks_world_mm"] = {
            "bilateral_crest_midpoint": crest_ref.tolist(),
            "disc_ventral_midpoint": disc_ventral_mid.tolist(),
        }
    if not flags:
        flags.append("ok")
    return result
