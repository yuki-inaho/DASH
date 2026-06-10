from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def _load_validator():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scene" / "colmap_processed.py"
    spec = importlib.util.spec_from_file_location("_dash_colmap_processed", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load validator module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a processed COLMAP dataset")
    parser.add_argument("--source-path", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--expect-images", type=int, default=None)
    parser.add_argument("--expect-registered", type=int, default=None)
    parser.add_argument("--validate-source-paths", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validator = _load_validator()
    try:
        summary = validator.validate_processed_colmap_dataset(
            args.source_path,
            args.mapping,
            expect_images=args.expect_images,
            expect_registered=args.expect_registered,
            validate_source_paths=args.validate_source_paths,
        )
    except validator.ColmapProcessedError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"source_path: {summary.source_path}")
    print(f"mapping_path: {summary.mapping_path}")
    print(f"images: {summary.image_count}")
    print(f"mapping_rows: {summary.mapping_rows}")
    print(f"registered: {summary.registered_count}")
    print(f"points: {summary.point_count}")
    print(f"camera_models: {','.join(summary.camera_models)}")
    print(f"source_paths_validated: {args.validate_source_paths}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
