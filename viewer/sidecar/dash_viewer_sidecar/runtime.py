"""Load a trained DASH model and evaluate deformed Gaussian frames.

Requires CUDA: the uploaded DASH implementation builds ``DeformModel`` with
``.cuda()`` and uses a CUDA hash encoder.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from . import abi
from .model_repository import (
    filter_kwargs,
    load_cfg_args,
    resolve_model_files,
)
from .protocol import log


class DashModelRuntime:
    def __init__(
        self,
        dash_root: Path,
        model_dir: Path,
        iteration: int,
        device: str,
        mask_mode: str,
    ) -> None:
        if str(dash_root) not in sys.path:
            sys.path.insert(0, str(dash_root))

        import torch  # noqa: WPS433
        from argparse import ArgumentParser  # noqa: WPS433
        from arguments import ModelParams  # noqa: WPS433
        from scene.deform_model import DeformModel  # noqa: WPS433
        from scene.gaussian_model import GaussianModel  # noqa: WPS433
        from scene.network import DeformNetwork  # noqa: WPS433

        if device != "cuda":
            raise RuntimeError(
                "This DASH sidecar requires --device cuda (DeformModel uses .cuda() "
                "and a CUDA hash encoder)."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, but DASH evaluation requires CUDA.")

        self.torch = torch
        self.device = device
        self.mask_mode = mask_mode

        ply_path, deform_path, cfg_path, resolved_iteration = resolve_model_files(
            model_dir, iteration
        )
        cfg = load_cfg_args(cfg_path)

        defaults = ModelParams(ArgumentParser(), sentinel=True).extract(
            Namespace(
                source_path=str(getattr(cfg, "source_path", model_dir)),
                model_path=str(model_dir),
                images=getattr(cfg, "images", "images"),
                resolution=getattr(cfg, "resolution", -1),
                white_background=getattr(cfg, "white_background", False),
                data_device=getattr(cfg, "data_device", "cuda"),
                eval=getattr(cfg, "eval", True),
                load2gpu_on_the_fly=getattr(cfg, "load2gpu_on_the_fly", False),
                sh_degree=getattr(cfg, "sh_degree", 3),
            )
        )

        # dict-valued fields stay on the class instance; prefer cfg_args overrides.
        model_defaults = ModelParams(ArgumentParser(), sentinel=True)
        grid_args = dict(getattr(model_defaults, "grid_args"))
        grid_args.update(dict(getattr(cfg, "grid_args", {})))

        network_args = dict(getattr(model_defaults, "network_args"))
        network_args.update(dict(getattr(cfg, "network_args", {})))
        network_args = filter_kwargs(
            DeformNetwork.__init__,
            network_args,
            excluded={"self", "d3_in_dim", "d4_in_dim"},
        )

        sh_degree = int(getattr(cfg, "sh_degree", getattr(defaults, "sh_degree", 3)))
        scale_xyz = getattr(cfg, "scale_xyz", getattr(model_defaults, "scale_xyz", 1.0))

        self.gaussians = GaussianModel(sh_degree)
        self.gaussians.load_ply(str(ply_path))

        self.deform = DeformModel(
            grid_args=grid_args,
            net_args=network_args,
            scale_xyz=scale_xyz,
            reg_temporal_able=False,
        )

        log(f"loading DASH deformation weights: {deform_path}")
        # Load directly to avoid DeformModel.load_weights re-resolving iterations.
        grid_weight, network_weight = torch.load(str(deform_path), map_location="cuda")
        self.deform.dash.load_state_dict(grid_weight)
        self.deform.deform.load_state_dict(network_weight)
        self.deform.dash.eval()
        self.deform.deform.eval()

        self.xyz = self.gaussians.get_xyz.detach()
        self.n = int(self.xyz.shape[0])
        dynamic_mask_raw = self.gaussians.get_dynamic.detach().bool()
        if dynamic_mask_raw.ndim == 1:
            self.dynamic_mask_for_step = dynamic_mask_raw[:, None]
            self.dynamic_mask_flat = dynamic_mask_raw
        else:
            self.dynamic_mask_for_step = dynamic_mask_raw
            self.dynamic_mask_flat = dynamic_mask_raw.squeeze(1)

        self._last_time: Optional[float] = None
        self._last_payload: Optional[bytes] = None

        log(
            f"loaded model: iteration={resolved_iteration}, gaussians={self.n}, "
            f"dynamic={int(self.dynamic_mask_flat.sum().item())}, ply={ply_path}"
        )

    def info(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "kind": "dash-runtime",
            "gaussian_count": self.n,
            "stride": abi.GAUSSIAN3D_STRIDE,
            "mask_mode": self.mask_mode,
            "device": self.device,
        }

    def frame(self, t: float) -> bytes:
        # The viewer drives t in [0, 1); DASH render.py uses the same range.
        t = float(t % 1.0)
        if (
            self._last_time is not None
            and abs(t - self._last_time) < 1e-7
            and self._last_payload is not None
        ):
            return self._last_payload

        torch = self.torch
        with torch.no_grad():
            time_input = torch.full((self.n, 1), t, device="cuda", dtype=torch.float32)

            if self.mask_mode == "ply_dynamic":
                gs_mask = self.dynamic_mask_for_step
            elif self.mask_mode == "all":
                gs_mask = torch.ones((self.n, 1), device="cuda", dtype=torch.bool)
            elif self.mask_mode == "dash_spatial":
                gs_mask = None
            else:
                raise ValueError(f"unknown mask_mode: {self.mask_mode}")

            if gs_mask is None:
                viewpoint_loc = torch.zeros((1, 3), device="cuda", dtype=torch.float32)
                deform_pkgs = self.deform.step(
                    self.xyz,
                    time_input,
                    viewpoint_loc=viewpoint_loc,
                    vis_filter=None,
                    extent=1.0,
                    stage="fine",
                    gs_mask=None,
                    test=True,
                )
            else:
                deform_pkgs = self.deform.step(
                    self.xyz,
                    time_input,
                    viewpoint_loc=None,
                    vis_filter=None,
                    extent=None,
                    stage="fine",
                    gs_mask=gs_mask,
                    test=True,
                )

            d_xyz = deform_pkgs["d_xyz"]
            d_rotation = deform_pkgs["d_rotation"]
            d_scaling = deform_pkgs["d_scaling"]
            d_opacity = deform_pkgs["d_opacity"]
            d_shs = deform_pkgs["d_shs"]

            position = (self.gaussians.get_xyz + d_xyz).detach().float().cpu().numpy()

            opacity_activated = (
                (self.gaussians.get_opacity + d_opacity).detach().float().cpu().numpy()
            )
            opacity_raw = abi.inverse_sigmoid_np(opacity_activated).reshape(self.n)

            scale_activated = (
                (self.gaussians.get_scaling + d_scaling).detach().float().cpu().numpy()
            )
            scale_log = np.log(np.clip(scale_activated, 1e-8, None)).astype(np.float32)

            rotation = (self.gaussians.get_rotation + d_rotation).detach().float().cpu().numpy()
            rotation = abi.normalize_quat_np(rotation)

            features = (self.gaussians.get_features + d_shs).detach().float().cpu().numpy()
            sh = np.empty((self.n, 48), dtype=np.float32)
            sh[:, 0:3] = features[:, 0, :]
            sh[:, 3:48] = np.transpose(features[:, 1:16, :], (0, 2, 1)).reshape(self.n, 45)

        payload = abi.pack_gaussian3d(position, opacity_raw, scale_log, rotation, sh)
        self._last_time = t
        self._last_payload = payload
        return payload
