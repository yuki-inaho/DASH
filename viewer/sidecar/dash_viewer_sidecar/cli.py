"""Argument parsing and entry points (serve / info / bake)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .bake import bake_frames
from .runtime import DashModelRuntime
from .server import serve as serve_runtime
from .splatv import main as splatv_main


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dash-root", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument(
        "--mask-mode",
        default="ply_dynamic",
        choices=["ply_dynamic", "all", "dash_spatial"],
    )


def _build_runtime(a: argparse.Namespace) -> DashModelRuntime:
    return DashModelRuntime(
        dash_root=a.dash_root.resolve(),
        model_dir=a.model_dir.resolve(),
        iteration=a.iteration,
        device=a.device,
        mask_mode=a.mask_mode,
    )


def serve_main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="dash_viewer_sidecar serve")
    _add_common(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", default=0, type=int)
    a = p.parse_args(argv)
    return serve_runtime(_build_runtime(a), a.host, a.port)


def info_main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="dash_viewer_sidecar info")
    _add_common(p)
    a = p.parse_args(argv)
    print(json.dumps(_build_runtime(a).info()))
    return 0


def bake_main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="dash_viewer_sidecar bake")
    _add_common(p)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--frames", default=24, type=int)
    a = p.parse_args(argv)
    bake_frames(_build_runtime(a), a.out, a.frames)
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: dash_viewer_sidecar {serve|info|bake|splatv} ...", file=sys.stderr)
        return 2
    cmd, rest = argv[0], argv[1:]
    table = {
        "serve": serve_main,
        "info": info_main,
        "bake": bake_main,
        "splatv": splatv_main,
    }
    fn = table.get(cmd)
    if fn is None:
        print(f"unknown command: {cmd}", file=sys.stderr)
        return 2
    return fn(rest)
