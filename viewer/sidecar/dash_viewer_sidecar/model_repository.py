"""Locate and load a trained DASH model's artifacts.

Expected model directory layout::

    <model_dir>/cfg_args
    <model_dir>/point_cloud/iteration_<N>/point_cloud.ply
    <model_dir>/deform/iteration_<N>/deform.pth   (or iteration_best)
"""

from __future__ import annotations

import inspect
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def find_max_iteration(root: Path) -> int:
    if not root.exists():
        raise FileNotFoundError(f"iteration directory does not exist: {root}")
    values = []
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith("iteration_"):
            suffix = child.name[len("iteration_") :]
            if suffix.isdigit():
                values.append(int(suffix))
    if not values:
        raise FileNotFoundError(f"no iteration_* directories found in: {root}")
    return max(values)


def resolve_iteration(model_dir: Path, iteration: int) -> Tuple[int, bool]:
    """Return (resolved_iteration, is_best). -2 = best, -1 = latest."""
    if iteration == -2:
        return iteration, True
    if iteration == -1:
        return find_max_iteration(model_dir / "point_cloud"), False
    return iteration, False


def resolve_model_files(
    model_dir: Path, iteration: int
) -> Tuple[Path, Path, Optional[Path], int]:
    resolved_iteration, is_best = resolve_iteration(model_dir, iteration)

    if is_best:
        deform_path = model_dir / "deform" / "iteration_best" / "deform.pth"
        # DASH does not normally store point_cloud/iteration_best; use the latest.
        point_iter = find_max_iteration(model_dir / "point_cloud")
    else:
        point_iter = resolved_iteration
        deform_path = model_dir / "deform" / f"iteration_{resolved_iteration}" / "deform.pth"

    ply_path = model_dir / "point_cloud" / f"iteration_{point_iter}" / "point_cloud.ply"
    cfg_path = model_dir / "cfg_args"

    if not ply_path.exists():
        raise FileNotFoundError(f"point_cloud.ply not found: {ply_path}")
    if not deform_path.exists():
        raise FileNotFoundError(f"deform.pth not found: {deform_path}")
    if not cfg_path.exists():
        cfg_path = None

    return ply_path, deform_path, cfg_path, point_iter


def load_cfg_args(cfg_path: Optional[Path]) -> Namespace:
    if cfg_path is None:
        return Namespace()
    text = cfg_path.read_text(encoding="utf-8")
    # DASH writes `argparse.Namespace(...)` directly. This is a trusted local
    # training artifact; builtins are disabled to keep eval scope narrow.
    return eval(text, {"Namespace": Namespace, "__builtins__": {}}, {})


def filter_kwargs(fn: Any, values: Dict[str, Any], excluded: set) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys()) - excluded
    return {k: v for k, v in values.items() if k in allowed}
