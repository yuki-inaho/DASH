"""Gaussian3d ABI (single source of truth on the Python side).

Byte layout (240 bytes, matches dash-runtime's GAUSSIAN3D_STRIDE and the viewer's
`#[repr(C)] struct Gaussian3d`):

    position [3]f32 | opacity f32 | scale [3]f32 | _pad0 u32 | rotation [4]f32 | sh [48]f32
"""

from __future__ import annotations

import numpy as np

GAUSSIAN3D_STRIDE = 240

GAUSSIAN3D_DTYPE = np.dtype(
    [
        ("position", "<f4", (3,)),
        ("opacity", "<f4"),
        ("scale", "<f4", (3,)),
        ("_pad0", "<u4"),
        ("rotation", "<f4", (4,)),
        ("sh", "<f4", (48,)),
    ],
    align=False,
)

if GAUSSIAN3D_DTYPE.itemsize != GAUSSIAN3D_STRIDE:
    raise RuntimeError(
        f"Gaussian3d ABI stride mismatch: {GAUSSIAN3D_DTYPE.itemsize} != {GAUSSIAN3D_STRIDE}"
    )


def inverse_sigmoid_np(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x)).astype(np.float32)


def normalize_quat_np(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    return (q / np.maximum(norm, eps)).astype(np.float32)


def pack_gaussian3d(
    position: np.ndarray,
    opacity_raw: np.ndarray,
    scale_log: np.ndarray,
    rotation: np.ndarray,
    sh: np.ndarray,
) -> bytes:
    """Pack per-Gaussian attribute arrays into the 240-byte ABI byte string."""
    n = position.shape[0]
    out = np.empty(n, dtype=GAUSSIAN3D_DTYPE)
    out["position"] = position
    out["opacity"] = opacity_raw.reshape(n)
    out["scale"] = scale_log
    out["_pad0"] = 0
    out["rotation"] = rotation
    out["sh"] = sh
    return out.tobytes(order="C")
