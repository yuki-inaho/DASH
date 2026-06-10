import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from dash_viewer_sidecar.splatv import (
    REQUIRED_3DGS_FIELDS,
    REQUIRED_4DGS_FIELDS,
    SPLATV_MAGIC,
    SPLATV_TEXWIDTH,
    convert_ply_to_splatv,
)


def _write_float_ply(path: Path, fields: list[str], rows: list[list[float]]) -> None:
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(rows)}",
        *[f"property float {name}" for name in fields],
        "end_header",
    ]
    body = b"".join(struct.pack("<" + "f" * len(fields), *row) for row in rows)
    path.write_bytes(("\n".join(header) + "\n").encode("ascii") + body)


def _read_splatv(path: Path):
    raw = path.read_bytes()
    magic, json_len = struct.unpack_from("<II", raw, 0)
    manifest = json.loads(raw[8 : 8 + json_len].decode("utf-8"))
    tex = np.frombuffer(raw[8 + json_len :], dtype="<u4")
    return magic, manifest, tex.reshape((-1, 16)), tex.view("<f4").reshape((-1, 16))


def _unpack_half2x16(word: np.uint32) -> tuple[float, float]:
    value = int(word)
    pair = np.array([value & 0xFFFF, value >> 16], dtype="<u2").view("<f2")
    return float(pair[0]), float(pair[1])


def _row(values: dict[str, float], fields: list[str]) -> list[float]:
    return [values.get(name, 0.0) for name in fields]


def _base_values(**overrides: float) -> dict[str, float]:
    values = {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "opacity": 0.0,
        "scale_0": 0.0,
        "scale_1": 0.0,
        "scale_2": 0.0,
        "rot_0": 1.0,
        "rot_1": 0.0,
        "rot_2": 0.0,
        "rot_3": 0.0,
        "f_dc_0": 0.0,
        "f_dc_1": 0.0,
        "f_dc_2": 0.0,
    }
    values.update(overrides)
    return values


def test_convert_true_4dgs_ply_to_splatv(tmp_path: Path) -> None:
    fields = list(REQUIRED_3DGS_FIELDS) + list(REQUIRED_4DGS_FIELDS)
    low = _base_values(
        x=1.0,
        opacity=0.0,
        f_dc_0=0.1,
        f_dc_1=0.2,
        f_dc_2=0.3,
        motion_0=0.01,
        motion_1=0.02,
        trbf_center=0.25,
        trbf_scale=math.log(2.0),
    )
    high = _base_values(
        x=2.0,
        opacity=2.0,
        scale_0=1.0,
        scale_1=1.0,
        scale_2=1.0,
        f_dc_0=0.4,
        f_dc_1=0.5,
        f_dc_2=0.6,
        motion_0=0.11,
        motion_1=0.12,
        omega_0=0.21,
        omega_1=0.22,
        trbf_center=0.75,
        trbf_scale=math.log(3.0),
    )
    ply = tmp_path / "input_4d.ply"
    out = tmp_path / "model.splatv"
    _write_float_ply(ply, fields, [_row(low, fields), _row(high, fields)])

    summary = convert_ply_to_splatv(
        ply,
        out,
        require_4d=True,
        chunk_size=1,
        progress=False,
    )

    assert summary.source_kind == "4dgs"
    assert summary.color_mode == "raw"
    magic, manifest, records, records_f32 = _read_splatv(out)
    assert magic == SPLATV_MAGIC
    assert manifest[0]["type"] == "splat"
    assert manifest[0]["texwidth"] == SPLATV_TEXWIDTH
    assert manifest[0]["gaussian_count"] == 2
    assert records_f32[0, 0] == pytest.approx(2.0)

    rgba = int(records[0, 7])
    assert rgba & 0xFF == int(0.4 * 255)
    assert (rgba >> 8) & 0xFF == int(0.5 * 255)
    assert (rgba >> 16) & 0xFF == int(0.6 * 255)
    assert (rgba >> 24) & 0xFF == int((1.0 / (1.0 + math.exp(-2.0))) * 255)

    motion0, motion1 = _unpack_half2x16(records[0, 8])
    omega0, omega1 = _unpack_half2x16(records[0, 13])
    center, trbf_scale = _unpack_half2x16(records[0, 15])
    assert motion0 == pytest.approx(0.11, abs=1e-3)
    assert motion1 == pytest.approx(0.12, abs=1e-3)
    assert omega0 == pytest.approx(0.21, abs=1e-3)
    assert omega1 == pytest.approx(0.22, abs=1e-3)
    assert center == pytest.approx(0.75, abs=1e-3)
    assert trbf_scale == pytest.approx(3.0, abs=1e-3)


def test_static_3dgs_export_is_explicit_compatibility_mode(tmp_path: Path) -> None:
    fields = list(REQUIRED_3DGS_FIELDS)
    values = _base_values(x=1.0, opacity=0.0, f_dc_0=1.0)
    ply = tmp_path / "input_3d.ply"
    out = tmp_path / "static.splatv"
    _write_float_ply(ply, fields, [_row(values, fields)])

    summary = convert_ply_to_splatv(ply, out, progress=False)

    assert summary.source_kind == "static-3dgs"
    assert summary.color_mode == "sh"
    _, manifest, records, _ = _read_splatv(out)
    assert manifest[0]["source_kind"] == "static-3dgs"
    assert (records[0, 8:15] == 0).all()
    center, trbf_scale = _unpack_half2x16(records[0, 15])
    assert center == pytest.approx(0.5, abs=1e-3)
    assert trbf_scale == pytest.approx(100.0, abs=1e-2)

    rgba = int(records[0, 7])
    assert rgba & 0xFF == int((0.5 + 0.28209479177387814) * 255)


def test_require_4d_rejects_plain_3dgs(tmp_path: Path) -> None:
    fields = list(REQUIRED_3DGS_FIELDS)
    ply = tmp_path / "input_3d.ply"
    out = tmp_path / "model.splatv"
    _write_float_ply(ply, fields, [_row(_base_values(), fields)])

    with pytest.raises(ValueError, match="4DGS PLY is missing"):
        convert_ply_to_splatv(ply, out, require_4d=True, progress=False)
