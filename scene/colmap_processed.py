from __future__ import annotations

import csv
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


class ColmapProcessedError(ValueError):
    """Raised when a processed COLMAP/GLOMAP dataset is inconsistent."""


@dataclass(frozen=True)
class MappingEntry:
    subset_index: int
    subset_filename: str
    source_index_0based: int
    source_index_1based: int
    source_filename: str
    source_path: str
    fid: float


@dataclass(frozen=True)
class ProcessedMapping:
    mapping_path: Path
    entries: tuple[MappingEntry, ...]
    fid_by_filename: dict[str, float]
    source_index_by_filename: dict[str, int]


@dataclass(frozen=True)
class ProcessedColmapSummary:
    source_path: Path
    mapping_path: Path
    image_count: int
    mapping_rows: int
    registered_count: int
    point_count: int
    camera_models: tuple[str, ...]


REQUIRED_MAPPING_COLUMNS = (
    "subset_index",
    "subset_filename",
    "source_index_0based",
    "source_index_1based",
    "source_filename",
    "source_path",
)


def _as_int(value: str, field: str, row_number: int) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ColmapProcessedError(
            f"invalid integer for {field} at mapping row {row_number}: {value!r}"
        ) from exc


def load_mapping_csv(
    mapping_path: str | Path,
    *,
    images_dir: str | Path | None = None,
    expect_images: int | None = None,
    validate_source_paths: bool = False,
) -> ProcessedMapping:
    path = Path(mapping_path)
    if not path.exists():
        raise ColmapProcessedError(f"mapping.csv not found: {path}")

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ColmapProcessedError(f"mapping.csv has no header: {path}")
        missing = [name for name in REQUIRED_MAPPING_COLUMNS if name not in reader.fieldnames]
        if missing:
            raise ColmapProcessedError(
                f"mapping.csv missing required columns {missing}: {path}"
            )
        rows = list(reader)

    if expect_images is not None and len(rows) != expect_images:
        raise ColmapProcessedError(
            f"mapping row count mismatch: expected {expect_images}, got {len(rows)}"
        )
    if not rows:
        raise ColmapProcessedError(f"mapping.csv has no rows: {path}")

    seen_indices: set[int] = set()
    seen_filenames: set[str] = set()
    seen_source_indices: set[int] = set()
    parsed: list[dict[str, int | str]] = []

    for offset, row in enumerate(rows, start=2):
        subset_index = _as_int(row["subset_index"], "subset_index", offset)
        source_index_0based = _as_int(
            row["source_index_0based"], "source_index_0based", offset
        )
        source_index_1based = _as_int(
            row["source_index_1based"], "source_index_1based", offset
        )
        subset_filename = row["subset_filename"]
        source_filename = row["source_filename"]

        if subset_index in seen_indices:
            raise ColmapProcessedError(f"duplicate subset_index: {subset_index}")
        if subset_filename in seen_filenames:
            raise ColmapProcessedError(f"duplicate subset_filename: {subset_filename}")
        if source_index_0based in seen_source_indices:
            raise ColmapProcessedError(
                f"duplicate source_index_0based: {source_index_0based}"
            )
        if source_index_1based != source_index_0based + 1:
            raise ColmapProcessedError(
                "source_index_1based must equal source_index_0based + 1 "
                f"for {subset_filename}"
            )
        expected_subset_filename = f"{subset_index:05d}.jpg"
        if subset_filename != expected_subset_filename:
            raise ColmapProcessedError(
                f"non-contiguous subset filename: expected {expected_subset_filename}, "
                f"got {subset_filename}"
            )
        expected_source_filename = f"frame_{source_index_1based:05d}.jpg"
        if source_filename != expected_source_filename:
            raise ColmapProcessedError(
                f"source filename mismatch for {subset_filename}: expected "
                f"{expected_source_filename}, got {source_filename}"
            )

        seen_indices.add(subset_index)
        seen_filenames.add(subset_filename)
        seen_source_indices.add(source_index_0based)
        parsed.append(
            {
                "subset_index": subset_index,
                "subset_filename": subset_filename,
                "source_index_0based": source_index_0based,
                "source_index_1based": source_index_1based,
                "source_filename": source_filename,
                "source_path": row["source_path"],
            }
        )

    expected_indices = set(range(len(parsed)))
    if seen_indices != expected_indices:
        missing = sorted(expected_indices - seen_indices)
        extra = sorted(seen_indices - expected_indices)
        raise ColmapProcessedError(
            f"non-contiguous subset_index values: missing={missing}, extra={extra}"
        )

    image_dir_path = Path(images_dir) if images_dir is not None else None
    if image_dir_path is not None:
        for row in parsed:
            image_path = image_dir_path / str(row["subset_filename"])
            if not image_path.exists():
                raise ColmapProcessedError(f"missing image file: {image_path}")

    if validate_source_paths:
        for row in parsed:
            source_path = str(row["source_path"])
            if source_path and not Path(source_path).exists():
                raise ColmapProcessedError(f"missing source image file: {source_path}")

    max_source_index = max(int(row["source_index_0based"]) for row in parsed)
    if max_source_index <= 0:
        raise ColmapProcessedError("maximum source_index_0based must be positive")

    entries = tuple(
        MappingEntry(
            subset_index=int(row["subset_index"]),
            subset_filename=str(row["subset_filename"]),
            source_index_0based=int(row["source_index_0based"]),
            source_index_1based=int(row["source_index_1based"]),
            source_filename=str(row["source_filename"]),
            source_path=str(row["source_path"]),
            fid=float(row["source_index_0based"]) / max_source_index,
        )
        for row in sorted(parsed, key=lambda item: int(item["subset_index"]))
    )
    return ProcessedMapping(
        mapping_path=path,
        entries=entries,
        fid_by_filename={entry.subset_filename: entry.fid for entry in entries},
        source_index_by_filename={
            entry.subset_filename: entry.source_index_0based for entry in entries
        },
    )


def _load_colmap_loader() -> ModuleType:
    module_path = Path(__file__).with_name("colmap_loader.py")
    spec = importlib.util.spec_from_file_location("_dash_colmap_loader", module_path)
    if spec is None or spec.loader is None:
        raise ColmapProcessedError(f"could not load COLMAP loader: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_sparse_counts(source_path: str | Path) -> tuple[int, int, tuple[str, ...]]:
    sparse_dir = Path(source_path) / "sparse" / "0"
    if not sparse_dir.exists():
        raise ColmapProcessedError(f"sparse/0 not found: {sparse_dir}")

    loader = _load_colmap_loader()
    try:
        images = loader.read_extrinsics_binary(sparse_dir / "images.bin")
        cameras = loader.read_intrinsics_binary(sparse_dir / "cameras.bin")
        xyz, _, _ = loader.read_points3D_binary(sparse_dir / "points3D.bin")
    except FileNotFoundError:
        images = loader.read_extrinsics_text(sparse_dir / "images.txt")
        cameras = loader.read_intrinsics_text(sparse_dir / "cameras.txt")
        xyz, _, _ = loader.read_points3D_text(sparse_dir / "points3D.txt")

    camera_models = tuple(sorted({camera.model for camera in cameras.values()}))
    unsupported = [model for model in camera_models if model not in {"PINHOLE", "SIMPLE_PINHOLE"}]
    if unsupported:
        raise ColmapProcessedError(
            "unsupported COLMAP camera model(s): " + ", ".join(unsupported)
        )
    point_count = int(len(xyz)) if xyz is not None else 0
    return len(images), point_count, camera_models


def validate_processed_colmap_dataset(
    source_path: str | Path,
    mapping_path: str | Path,
    *,
    expect_images: int | None = None,
    expect_registered: int | None = None,
    validate_source_paths: bool = False,
) -> ProcessedColmapSummary:
    root = Path(source_path)
    images_dir = root / "images"
    if not images_dir.exists():
        raise ColmapProcessedError(f"images directory not found: {images_dir}")

    mapping = load_mapping_csv(
        mapping_path,
        images_dir=images_dir,
        expect_images=expect_images,
        validate_source_paths=validate_source_paths,
    )
    image_count = len(
        [path for path in images_dir.iterdir() if path.is_file() or path.is_symlink()]
    )
    if expect_images is not None and image_count != expect_images:
        raise ColmapProcessedError(
            f"image count mismatch: expected {expect_images}, got {image_count}"
        )
    if image_count != len(mapping.entries):
        raise ColmapProcessedError(
            f"image count and mapping rows differ: images={image_count}, "
            f"mapping={len(mapping.entries)}"
        )

    registered_count, point_count, camera_models = read_sparse_counts(root)
    if expect_registered is not None and registered_count != expect_registered:
        raise ColmapProcessedError(
            "registered image count mismatch: expected "
            f"{expect_registered}, got {registered_count}"
        )

    return ProcessedColmapSummary(
        source_path=root,
        mapping_path=Path(mapping_path),
        image_count=image_count,
        mapping_rows=len(mapping.entries),
        registered_count=registered_count,
        point_count=point_count,
        camera_models=camera_models,
    )
