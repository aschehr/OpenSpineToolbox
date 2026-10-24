"""Shared geometry for coronal CT figures: builds a coronal bone-window MIP
(maximum intensity projection collapsed along the anterior-posterior axis)
and a world-mm -> image-pixel mapper, both derived directly from the case's
own NIfTI affine (no assumption about voxel axis order beyond what the
affine actually encodes -- CTSpinoPelvic1K affines are axis-aligned
permutations, no shear, which is verified below).
"""
from __future__ import annotations

import numpy as np


def axis_mapping(affine: np.ndarray):
    """For an axis-aligned (permutation + scale + sign) affine, return which
    ARRAY axis is dominant for each WORLD axis (X=L-R, Y=A-P, Z=S-I), and the
    sign of that relationship (positive = increasing array index increases
    the world coordinate). Asserts the affine really is axis-aligned (no
    shear) so this mapping is exact, not approximate."""
    M = affine[:3, :3]
    absM = np.abs(M)
    array_axis_for_world = np.argmax(absM, axis=1)  # row=world axis -> col=array axis
    assert len(set(array_axis_for_world.tolist())) == 3, "affine is not a clean permutation"
    for w in range(3):
        row = absM[w].copy()
        dom = row[array_axis_for_world[w]]
        row[array_axis_for_world[w]] = 0
        assert row.max() < 1e-6 * max(dom, 1e-9), "affine has off-axis shear; mapping unsafe"
    sign_for_world = np.sign(M[np.arange(3), array_axis_for_world])
    return array_axis_for_world, sign_for_world  # [X,Y,Z] order


def coronal_mip(ct: np.ndarray, affine: np.ndarray):
    """Coronal (frontal) MIP: collapses the array axis dominant for world Y
    (anterior-posterior). Returns (image, pixel_mapper) where pixel_mapper
    takes a world-mm (x,y,z) point and returns (col, row) in `image`, with
    row 0 = most superior and increasing column = increasing world X."""
    array_axis_for_world, sign_for_world = axis_mapping(affine)
    lr_axis, ap_axis, si_axis = array_axis_for_world
    sign_lr, _, sign_si = sign_for_world

    mip = ct.max(axis=int(ap_axis))
    remaining = [ax for ax in range(3) if ax != ap_axis]
    lr_pos = remaining.index(lr_axis)
    si_pos = remaining.index(si_axis)
    img = np.transpose(mip, axes=(si_pos, lr_pos))  # (dim_si, dim_lr)
    flip_rows = sign_si > 0  # larger si-axis index = more superior -> must end up at row 0
    if flip_rows:
        img = img[::-1, :]
    n_si = img.shape[0]

    inv_affine = np.linalg.inv(affine)

    def world_to_pixel(xyz):
        vox = inv_affine @ np.array([xyz[0], xyz[1], xyz[2], 1.0])
        i_si = vox[si_axis]
        i_lr = vox[lr_axis]
        row = (n_si - 1 - i_si) if flip_rows else i_si
        col = i_lr
        return float(col), float(row)

    right_edge_label = "R" if sign_lr > 0 else "L"
    left_edge_label = "L" if sign_lr > 0 else "R"
    return img, world_to_pixel, left_edge_label, right_edge_label
