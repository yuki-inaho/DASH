"""splaTV ``.splatv`` export for Gaussian PLY files.

The container format is the one used by ``antimatter15/splaTV``: an 8-byte
little-endian header, a JSON chunk manifest, then an ``RGBA32UI`` texture payload
with 16 ``uint32`` words per Gaussian.  The texture layout is 4DGS/STG-Lite
compatible: base position/rotation/scale/color plus cubic motion, angular
velocity, and temporal RBF parameters.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from beartype import beartype
from plyfile import PlyData
from tqdm import tqdm

SPLATV_MAGIC = 0x674B
SPLATV_TEXWIDTH = 4096
WORDS_PER_GAUSSIAN = 16
GAUSSIANS_PER_TEXTURE_ROW = SPLATV_TEXWIDTH // 4
SH_C0 = 0.28209479177387814

ColorMode = Literal["auto", "raw", "sh"]

REQUIRED_3DGS_FIELDS = (
    "x",
    "y",
    "z",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
)

REQUIRED_4DGS_FIELDS = (
    "trbf_center",
    "trbf_scale",
    "motion_0",
    "motion_1",
    "motion_2",
    "motion_3",
    "motion_4",
    "motion_5",
    "motion_6",
    "motion_7",
    "motion_8",
    "omega_0",
    "omega_1",
    "omega_2",
    "omega_3",
)


@dataclass(frozen=True)
class PlyGaussianData:
    """Numpy view of the Gaussian fields needed by the splaTV texture layout."""

    fields: dict[str, np.ndarray]
    count: int
    is_4d: bool

    def require(self, name: str) -> np.ndarray:
        try:
            return self.fields[name]
        except KeyError as exc:
            raise ValueError(f"missing PLY vertex property: {name}") from exc


@dataclass(frozen=True)
class SplatvSummary:
    input: str
    output: str
    gaussian_count: int
    source_kind: str
    color_mode: str
    texwidth: int
    texheight: int
    byte_size: int


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def _pack_half2x16(x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    y_arr = np.broadcast_to(np.asarray(y, dtype=np.float32), x_arr.shape)
    lo = x_arr.astype("<f2").view("<u2").astype("<u4")
    hi = y_arr.astype("<f2").view("<u2").astype("<u4")
    return (lo | (hi << np.uint32(16))).astype("<u4")


def _read_field(vertex: np.ndarray, name: str) -> np.ndarray:
    return np.asarray(vertex[name], dtype=np.float32)


def _validate_fields(names: set[str], required: tuple[str, ...], label: str) -> None:
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"{label} PLY is missing required fields: {', '.join(missing)}")


@beartype
def load_gaussian_ply(path: Path, require_4d: bool = False) -> PlyGaussianData:
    """Load a 3DGS or 4DGS/STG-Lite Gaussian PLY into typed arrays."""

    ply = PlyData.read(path)
    try:
        vertex = ply["vertex"].data
    except KeyError as exc:
        raise ValueError(f"PLY has no vertex element: {path}") from exc

    names = set(vertex.dtype.names or ())
    _validate_fields(names, REQUIRED_3DGS_FIELDS, "3DGS")

    has_any_4d = any(name in names for name in REQUIRED_4DGS_FIELDS)
    has_all_4d = all(name in names for name in REQUIRED_4DGS_FIELDS)
    if require_4d and not has_all_4d:
        _validate_fields(names, REQUIRED_4DGS_FIELDS, "4DGS")
    if has_any_4d and not has_all_4d:
        _validate_fields(names, REQUIRED_4DGS_FIELDS, "partial 4DGS")

    required = list(REQUIRED_3DGS_FIELDS)
    if has_all_4d:
        required.extend(REQUIRED_4DGS_FIELDS)

    fields = {name: _read_field(vertex, name) for name in required}
    count = len(vertex)
    return PlyGaussianData(fields=fields, count=count, is_4d=has_all_4d)


def _importance(data: PlyGaussianData) -> np.ndarray:
    size = (
        np.exp(data.require("scale_0"))
        * np.exp(data.require("scale_1"))
        * np.exp(data.require("scale_2"))
    )
    return (size * _sigmoid(data.require("opacity"))).astype(np.float32)


def _resolve_color_mode(data: PlyGaussianData, color_mode: ColorMode) -> Literal["raw", "sh"]:
    if color_mode == "auto":
        return "raw" if data.is_4d else "sh"
    return color_mode


def _rgb01(data: PlyGaussianData, order: np.ndarray, color_mode: Literal["raw", "sh"]) -> np.ndarray:
    rgb = np.column_stack(
        (
            data.require("f_dc_0")[order],
            data.require("f_dc_1")[order],
            data.require("f_dc_2")[order],
        )
    ).astype(np.float32)
    if color_mode == "sh":
        rgb = 0.5 + SH_C0 * rgb
    return np.clip(rgb, 0.0, 1.0)


def _default_cameras(position: np.ndarray) -> list[dict[str, Any]]:
    center = np.median(position, axis=0) if len(position) else np.zeros(3, dtype=np.float32)
    radius = 1.0
    if len(position):
        dist = np.linalg.norm(position - center[None, :], axis=1)
        radius = float(np.quantile(dist, 0.95))
        radius = max(radius, 1e-3)

    width = 1280
    height = 720
    camera_distance = max(radius * 2.5, 1.0)
    camera_position = center + np.array([0.0, 0.0, -camera_distance], dtype=np.float32)
    focal = float(max(width, height) * 0.9)

    return [
        {
            "id": 0,
            "img_name": "auto",
            "width": width,
            "height": height,
            "position": [float(x) for x in camera_position],
            "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "fx": focal,
            "fy": focal,
        }
    ]


@beartype
def load_cameras(camera_json: Path | None, position: np.ndarray) -> list[dict[str, Any]]:
    if camera_json is None:
        return _default_cameras(position)

    raw = json.loads(camera_json.read_text(encoding="utf-8"))
    cameras = raw if isinstance(raw, list) else [raw]
    if not cameras:
        raise ValueError(f"camera JSON is empty: {camera_json}")
    return cameras


def _write_static_temporal_fields(
    records: np.ndarray,
    start: int,
    end: int,
    static_trbf_center: float,
    static_trbf_scale: float,
) -> None:
    records[start:end, 15] = _pack_half2x16(
        np.full(end - start, static_trbf_center, dtype=np.float32),
        np.full(end - start, static_trbf_scale, dtype=np.float32),
    )


def _pack_records(
    data: PlyGaussianData,
    *,
    color_mode: Literal["raw", "sh"],
    chunk_size: int,
    progress: bool,
    static_trbf_center: float,
    static_trbf_scale: float,
) -> tuple[np.ndarray, int]:
    count = data.count
    texheight = max(1, math.ceil(count / GAUSSIANS_PER_TEXTURE_ROW))
    texdata = np.zeros(SPLATV_TEXWIDTH * texheight * 4, dtype="<u4")
    records = texdata.reshape((-1, WORDS_PER_GAUSSIAN))
    records_f32 = texdata.view("<f4").reshape((-1, WORDS_PER_GAUSSIAN))

    order = np.argsort(-_importance(data), kind="stable")
    total = count
    with tqdm(
        total=total,
        desc="packing .splatv",
        unit="gaussians",
        disable=not progress,
    ) as bar:
        for start in range(0, count, chunk_size):
            end = min(start + chunk_size, count)
            idx = order[start:end]

            records_f32[start:end, 0] = data.require("x")[idx]
            records_f32[start:end, 1] = data.require("y")[idx]
            records_f32[start:end, 2] = data.require("z")[idx]

            records[start:end, 3] = _pack_half2x16(
                data.require("rot_0")[idx], data.require("rot_1")[idx]
            )
            records[start:end, 4] = _pack_half2x16(
                data.require("rot_2")[idx], data.require("rot_3")[idx]
            )

            records[start:end, 5] = _pack_half2x16(
                np.exp(data.require("scale_0")[idx]),
                np.exp(data.require("scale_1")[idx]),
            )
            records[start:end, 6] = _pack_half2x16(np.exp(data.require("scale_2")[idx]), 0.0)

            rgb8 = (_rgb01(data, idx, color_mode) * 255.0).astype(np.uint8)
            alpha8 = (_sigmoid(data.require("opacity")[idx]) * 255.0).astype(np.uint8)
            records[start:end, 7] = (
                rgb8[:, 0].astype("<u4")
                | (rgb8[:, 1].astype("<u4") << np.uint32(8))
                | (rgb8[:, 2].astype("<u4") << np.uint32(16))
                | (alpha8.astype("<u4") << np.uint32(24))
            )

            if data.is_4d:
                records[start:end, 8] = _pack_half2x16(
                    data.require("motion_0")[idx], data.require("motion_1")[idx]
                )
                records[start:end, 9] = _pack_half2x16(
                    data.require("motion_2")[idx], data.require("motion_3")[idx]
                )
                records[start:end, 10] = _pack_half2x16(
                    data.require("motion_4")[idx], data.require("motion_5")[idx]
                )
                records[start:end, 11] = _pack_half2x16(
                    data.require("motion_6")[idx], data.require("motion_7")[idx]
                )
                records[start:end, 12] = _pack_half2x16(data.require("motion_8")[idx], 0.0)
                records[start:end, 13] = _pack_half2x16(
                    data.require("omega_0")[idx], data.require("omega_1")[idx]
                )
                records[start:end, 14] = _pack_half2x16(
                    data.require("omega_2")[idx], data.require("omega_3")[idx]
                )
                records[start:end, 15] = _pack_half2x16(
                    data.require("trbf_center")[idx],
                    np.exp(data.require("trbf_scale")[idx]),
                )
            else:
                _write_static_temporal_fields(
                    records,
                    start,
                    end,
                    static_trbf_center,
                    static_trbf_scale,
                )

            bar.update(end - start)

    return texdata, texheight


@beartype
def write_splatv(
    output: Path,
    *,
    texdata: np.ndarray,
    texheight: int,
    cameras: list[dict[str, Any]],
    gaussian_count: int,
    source_kind: str,
    color_mode: str,
) -> None:
    manifest = [
        {
            "type": "splat",
            "size": int(texdata.nbytes),
            "texwidth": SPLATV_TEXWIDTH,
            "texheight": texheight,
            "cameras": cameras,
            "gaussian_count": gaussian_count,
            "source_kind": source_kind,
            "color_mode": color_mode,
        }
    ]
    manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        f.write(struct.pack("<II", SPLATV_MAGIC, len(manifest_bytes)))
        f.write(manifest_bytes)
        f.write(texdata.tobytes(order="C"))


@beartype
def convert_ply_to_splatv(
    input_path: Path,
    output_path: Path,
    *,
    camera_json: Path | None = None,
    color_mode: ColorMode = "auto",
    require_4d: bool = False,
    chunk_size: int = 65536,
    progress: bool = True,
    static_trbf_center: float = 0.5,
    static_trbf_scale: float = 100.0,
) -> SplatvSummary:
    """Convert a Gaussian PLY to a splaTV-compatible ``.splatv`` file."""

    if chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if static_trbf_scale <= 0:
        raise ValueError("--static-trbf-scale must be positive")

    data = load_gaussian_ply(input_path, require_4d=require_4d)
    resolved_color_mode = _resolve_color_mode(data, color_mode)

    position = np.column_stack(
        (data.require("x"), data.require("y"), data.require("z"))
    ).astype(np.float32)
    cameras = load_cameras(camera_json, position)

    texdata, texheight = _pack_records(
        data,
        color_mode=resolved_color_mode,
        chunk_size=chunk_size,
        progress=progress,
        static_trbf_center=static_trbf_center,
        static_trbf_scale=static_trbf_scale,
    )
    source_kind = "4dgs" if data.is_4d else "static-3dgs"
    write_splatv(
        output_path,
        texdata=texdata,
        texheight=texheight,
        cameras=cameras,
        gaussian_count=data.count,
        source_kind=source_kind,
        color_mode=resolved_color_mode,
    )

    return SplatvSummary(
        input=str(input_path),
        output=str(output_path),
        gaussian_count=data.count,
        source_kind=source_kind,
        color_mode=resolved_color_mode,
        texwidth=SPLATV_TEXWIDTH,
        texheight=texheight,
        byte_size=output_path.stat().st_size,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dash_viewer_sidecar splatv",
        description="Convert a Gaussian 4DGS/STG-Lite or 3DGS PLY to splaTV .splatv.",
    )
    parser.add_argument("--input", required=True, type=Path, help="input point_cloud.ply")
    parser.add_argument("--output", required=True, type=Path, help="output .splatv path")
    parser.add_argument("--camera-json", type=Path, help="optional splaTV camera JSON/list")
    parser.add_argument(
        "--color-mode",
        choices=["auto", "raw", "sh"],
        default="auto",
        help="auto=4DGS raw color, 3DGS SH DC conversion",
    )
    parser.add_argument(
        "--require-4d",
        action="store_true",
        help="fail unless motion_*/omega_*/trbf_* fields are present",
    )
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--static-trbf-center", type=float, default=0.5)
    parser.add_argument(
        "--static-trbf-scale",
        type=float,
        default=100.0,
        help="activated temporal RBF scale used when exporting static 3DGS PLY",
    )
    parser.add_argument("--no-progress", action="store_true", help="disable tqdm progress bar")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = convert_ply_to_splatv(
        args.input,
        args.output,
        camera_json=args.camera_json,
        color_mode=args.color_mode,
        require_4d=args.require_4d,
        chunk_size=args.chunk_size,
        progress=not args.no_progress,
        static_trbf_center=args.static_trbf_center,
        static_trbf_scale=args.static_trbf_scale,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False))
    return 0
