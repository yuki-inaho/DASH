"""DASH viewer sidecar.

A small, SOLID-split service that loads a trained DASH model once and, for a
normalized time ``t in [0, 1)``, returns deformed Gaussians packed in the
wgpu-gs-viewer ``Gaussian3d`` 240-byte ABI. Split into focused modules:

  abi               240-byte Gaussian3d numpy dtype + packing helpers
  protocol          length-prefixed TCP framing
  model_repository  iteration resolution + cfg_args / ply / deform.pth loading
  runtime           DASH hash-grid + DeformNetwork evaluation -> frame bytes
  server            threading TCP server speaking the protocol
  bake              export a frame sequence (raw payloads) for offline use
  cli               argument parsing + serve/info/bake entry points
"""

from .abi import GAUSSIAN3D_STRIDE

__all__ = ["GAUSSIAN3D_STRIDE"]
