"""Contract test against a real trained DASH model (needs CUDA + model)."""

import os
from pathlib import Path

import numpy as np
import pytest

from dash_viewer_sidecar.abi import GAUSSIAN3D_DTYPE

DASH_ROOT = Path(os.environ.get("DASH_ROOT", Path(__file__).resolve().parents[2]))
MODEL = Path(
    os.environ.get("DASH_MODEL_DIR", DASH_ROOT / "output" / "tva_nyx650_400_smoke")
)


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


requires_model_cuda = pytest.mark.skipif(
    not (_cuda_available() and MODEL.exists()),
    reason="needs CUDA and a trained DASH model directory",
)


@requires_model_cuda
def test_info_and_frame_contract():
    from dash_viewer_sidecar.runtime import DashModelRuntime

    rt = DashModelRuntime(DASH_ROOT, MODEL, -2, "cuda", "all")

    info = rt.info()
    assert info["stride"] == 240
    assert info["gaussian_count"] > 0
    n = info["gaussian_count"]

    p0 = rt.frame(0.0)
    p5 = rt.frame(0.5)
    assert len(p0) == n * 240
    assert len(p5) == n * 240

    a = np.frombuffer(p0, dtype=GAUSSIAN3D_DTYPE)
    assert np.isfinite(a["position"]).all()
    assert np.isfinite(a["opacity"]).all()
    assert np.isfinite(a["scale"]).all()
    assert (a["_pad0"] == 0).all()
    # rotations are normalized quaternions
    rot_norm = np.linalg.norm(a["rotation"], axis=1)
    assert np.allclose(rot_norm, 1.0, atol=1e-3)
