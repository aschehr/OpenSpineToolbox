import numpy as np

from ostk import geometry as g
from ostk import spine


def _vertebra(center, tilt_deg, ap_extent=30.0, lr_extent=45.0, height=24.0,
              seed=0):
    """A solid box vertebral BODY (DENSE 1 mm voxel grid, like a real mask) whose
    superior endplate is tilted `tilt_deg` about the L–R axis, plus a posterior
    'spinous process' spike that must NOT influence the endplate fit."""
    gx = np.arange(-lr_extent / 2, lr_extent / 2)
    gy = np.arange(-ap_extent / 2, ap_extent / 2)
    gz = np.arange(-height / 2, height / 2)
    b = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), -1).reshape(-1, 3)
    a = np.deg2rad(tilt_deg)
    R = np.array([[1, 0, 0],
                  [0, np.cos(a), -np.sin(a)],
                  [0, np.sin(a), np.cos(a)]])
    body = b @ R.T + np.asarray(center)
    # posterior spike (−y), below the endplate
    sx = np.arange(-6, 6); sy = np.arange(-12, 2); sz = np.arange(-28, 2)
    s = np.stack(np.meshgrid(sx, sy, sz, indexing="ij"), -1).reshape(-1, 3)
    s += np.array([0, -int(ap_extent * 0.75), -int(height * 0.3)])
    spike = s @ R.T + np.asarray(center)
    return np.vstack([body, spike])


def test_fit_endplate_recovers_tilt_and_ignores_posterior():
    pts = _vertebra(center=[0, 0, 100.0], tilt_deg=20.0)
    res = spine.fit_endplate(pts, which="superior")
    assert res is not None
    c, n, rms = res
    # superior endplate normal tilted 20° from vertical, cranial (+z)
    assert n[2] > 0
    tilt = np.degrees(np.arccos(np.clip(abs(n @ np.array([0, 0, 1.0])), 0, 1)))
    assert abs(tilt - 20.0) < 6.0          # recovered tilt despite posterior spike


def test_fit_endplate_flat_is_horizontal():
    pts = _vertebra(center=[0, 0, 0.0], tilt_deg=0.0, seed=3)
    c, n, rms = spine.fit_endplate(pts, which="superior")
    tilt = np.degrees(np.arccos(np.clip(abs(n @ np.array([0, 0, 1.0])), 0, 1)))
    assert tilt < 5.0


def test_fit_endplate_too_few_points():
    assert spine.fit_endplate(np.zeros((5, 3))) is None


def test_anterior_axis_orientation():
    ap = spine.anterior_axis()
    assert ap[1] > 0                       # points to +Y (anterior)
    assert abs(ap @ np.array([0, 0, 1.0])) < 1e-9
