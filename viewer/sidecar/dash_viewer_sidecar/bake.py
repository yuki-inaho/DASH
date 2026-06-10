"""Export a sequence of raw Gaussian3d frames (for offline / wasm consumers).

The native PNG bake (web fallback) is produced by the Rust ``dash_bake`` binary,
which renders with the real GPU pipeline. This raw export is provided for reuse
(e.g. a future WebGPU player that rasterizes the gaussians itself)."""

from __future__ import annotations

import json
from pathlib import Path

from .abi import GAUSSIAN3D_STRIDE
from .protocol import log
from .runtime import DashModelRuntime


def bake_frames(runtime: DashModelRuntime, out_dir: Path, frames: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = max(1, frames)
    for i in range(frames):
        t = i / frames
        payload = runtime.frame(t)
        (out_dir / f"frame_{i:04d}.bin").write_bytes(payload)
    manifest = {
        "frames": frames,
        "gaussian_count": runtime.n,
        "stride": GAUSSIAN3D_STRIDE,
        "format": "gaussian3d-raw",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    log(f"baked {frames} raw frames to {out_dir}")
