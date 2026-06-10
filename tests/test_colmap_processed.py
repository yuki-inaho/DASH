import shutil
from pathlib import Path

import pytest

from scene.colmap_processed import (
    ColmapProcessedError,
    load_mapping_csv,
    validate_processed_colmap_dataset,
)


DATASET = Path("data/tva_nyx650_400_aliked_lg_glomap")
MAPPING = DATASET / "mapping.csv"


def test_mapping_csv_has_400_rows_and_numeric_names():
    if not MAPPING.exists():
        pytest.skip("processed COLMAP mapping.csv is not available")

    mapping = load_mapping_csv(MAPPING, images_dir=DATASET / "images", expect_images=400)

    assert len(mapping.entries) == 400
    assert mapping.entries[0].subset_filename == "00000.jpg"
    assert mapping.entries[0].source_index_0based == 0
    assert mapping.entries[-1].subset_filename == "00399.jpg"
    assert mapping.entries[-1].source_index_0based == 1995
    assert mapping.fid_by_filename["00000.jpg"] == pytest.approx(0.0)
    assert mapping.fid_by_filename["00399.jpg"] == pytest.approx(1.0)


def test_real_processed_dataset_summary_counts():
    if not DATASET.exists():
        pytest.skip("processed COLMAP dataset is not available")

    summary = validate_processed_colmap_dataset(
        DATASET,
        MAPPING,
        expect_images=400,
        expect_registered=400,
    )

    assert summary.image_count == 400
    assert summary.mapping_rows == 400
    assert summary.registered_count == 400
    assert summary.point_count == 41333


def test_dataset_reader_uses_source_index_mapping_for_fid():
    if not DATASET.exists():
        pytest.skip("processed COLMAP dataset is not available")

    from scene.dataset_readers import readColmapSceneInfo

    scene_info = readColmapSceneInfo(str(DATASET), "images", eval=False)
    cams = sorted(scene_info.train_cameras, key=lambda cam: cam.image_name)

    assert len(cams) == 400
    assert cams[0].image_name == "00000"
    assert cams[0].fid == pytest.approx(0.0)
    assert cams[1].image_name == "00001"
    assert cams[1].fid == pytest.approx(5 / 1995)
    assert cams[-1].image_name == "00399"
    assert cams[-1].fid == pytest.approx(1.0)


def test_mapping_rejects_duplicate_subset_index(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "00000.jpg").touch()
    (images_dir / "00001.jpg").touch()
    mapping = tmp_path / "mapping.csv"
    mapping.write_text(
        "subset_index,subset_filename,source_index_0based,source_index_1based,source_filename,source_path\n"
        "0,00000.jpg,0,1,frame_00001.jpg,/tmp/frame_00001.jpg\n"
        "0,00001.jpg,5,6,frame_00006.jpg,/tmp/frame_00006.jpg\n"
    )

    with pytest.raises(ColmapProcessedError, match="duplicate subset_index"):
        load_mapping_csv(mapping, images_dir=images_dir)


def test_mapping_rejects_missing_image(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "00000.jpg").touch()
    mapping = tmp_path / "mapping.csv"
    mapping.write_text(
        "subset_index,subset_filename,source_index_0based,source_index_1based,source_filename,source_path\n"
        "0,00000.jpg,0,1,frame_00001.jpg,/tmp/frame_00001.jpg\n"
        "1,00001.jpg,5,6,frame_00006.jpg,/tmp/frame_00006.jpg\n"
    )

    with pytest.raises(ColmapProcessedError, match="missing image file"):
        load_mapping_csv(mapping, images_dir=images_dir)


def test_source_path_validation_is_optional(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "00000.jpg").touch()
    (images_dir / "00001.jpg").touch()
    mapping = tmp_path / "mapping.csv"
    missing_source = tmp_path / "missing" / "frame_00001.jpg"
    missing_source_2 = tmp_path / "missing" / "frame_00006.jpg"
    mapping.write_text(
        "subset_index,subset_filename,source_index_0based,source_index_1based,source_filename,source_path\n"
        f"0,00000.jpg,0,1,frame_00001.jpg,{missing_source}\n"
        f"1,00001.jpg,5,6,frame_00006.jpg,{missing_source_2}\n"
    )

    loaded = load_mapping_csv(mapping, images_dir=images_dir)

    assert len(loaded.entries) == 2
    with pytest.raises(ColmapProcessedError, match="missing source image file"):
        load_mapping_csv(
            mapping,
            images_dir=images_dir,
            validate_source_paths=True,
        )


def test_validate_rejects_registered_mismatch(tmp_path):
    if not DATASET.exists():
        pytest.skip("processed COLMAP dataset is not available")

    copied = tmp_path / "dataset"
    shutil.copytree(DATASET / "sparse", copied / "sparse", symlinks=True)
    shutil.copytree(DATASET / "images", copied / "images", symlinks=True)
    shutil.copy2(MAPPING, copied / "mapping.csv")

    with pytest.raises(ColmapProcessedError, match="registered image count"):
        validate_processed_colmap_dataset(
            copied,
            copied / "mapping.csv",
            expect_images=400,
            expect_registered=399,
        )
