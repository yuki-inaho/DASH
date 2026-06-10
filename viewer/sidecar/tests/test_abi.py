"""ABI tests: no torch/CUDA needed (pure numpy)."""

import numpy as np

from dash_viewer_sidecar.abi import (
    GAUSSIAN3D_DTYPE,
    GAUSSIAN3D_STRIDE,
    inverse_sigmoid_np,
    normalize_quat_np,
    pack_gaussian3d,
)


def test_stride_is_240():
    assert GAUSSIAN3D_STRIDE == 240
    assert GAUSSIAN3D_DTYPE.itemsize == 240


def test_pack_roundtrip():
    n = 5
    pos = np.random.rand(n, 3).astype(np.float32)
    op = np.random.rand(n).astype(np.float32)
    sc = np.random.rand(n, 3).astype(np.float32)
    rot = np.random.rand(n, 4).astype(np.float32)
    sh = np.random.rand(n, 48).astype(np.float32)

    blob = pack_gaussian3d(pos, op, sc, rot, sh)
    assert len(blob) == n * 240

    arr = np.frombuffer(blob, dtype=GAUSSIAN3D_DTYPE)
    assert np.allclose(arr["position"], pos)
    assert np.allclose(arr["opacity"], op)
    assert np.allclose(arr["scale"], sc)
    assert np.allclose(arr["rotation"], rot)
    assert np.allclose(arr["sh"], sh)
    assert (arr["_pad0"] == 0).all()


def test_inverse_sigmoid_is_inverse():
    x = np.array([0.05, 0.1, 0.5, 0.9], dtype=np.float32)
    back = 1.0 / (1.0 + np.exp(-inverse_sigmoid_np(x)))
    assert np.allclose(back, x, atol=1e-4)


def test_normalize_quat_unit():
    q = np.array([[0.0, 0.0, 0.0, 2.0], [1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    n = normalize_quat_np(q)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-5)
