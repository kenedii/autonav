import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetracer.train.data_frontend.augment_data import get_pixel_columns as get_aug_pixel_columns
from jetracer.train.data_frontend.dataset_csv_creator import create_combined_csv
from jetracer.train.model_training import dataset_loader
from jetracer.train.model_training.dataset_loader import load_dataset


LEGACY_COLUMNS = [
    "timestamp",
    "steer_us",
    "throttle_us",
    "steer_norm",
    "throttle_norm",
    "depth_front",
]
RAW_METADATA_COLUMNS = [
    "rgb_source",
    "depth_source",
    "imu_source",
    "rear_rgb_source",
    "preprocess_profile",
    "run_id",
    "session_id",
    "cam0_path",
    "cam1_path",
    "ir_path",
    "depth_path",
]
FLATTENED_METADATA_COLUMNS = [
    "rgb_source",
    "depth_source",
    "imu_source",
    "rear_rgb_source",
    "preprocess_profile",
    "run_id",
    "session_id",
]
RAW_RUN_COLUMNS = LEGACY_COLUMNS + ["rgb_path"] + RAW_METADATA_COLUMNS


def test_dataset_loader_exports_lightweight_helpers():
    assert callable(load_dataset)
    assert callable(dataset_loader.get_pixel_columns)
    assert dataset_loader.EXPECTED_PIXEL_COLUMNS == 160 * 120 * 3


def make_image(path, color):
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[:, :] = np.array(color, dtype=np.uint8)
    Image.fromarray(frame, mode="RGB").save(path)


def write_run(root_dir, run_name, *, rgb_source, preprocess_profile):
    run_dir = root_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    image_path = run_dir / "rgb_00000.png"
    make_image(image_path, (10, 20, 30))

    row = {
        "timestamp": "1.0",
        "steer_us": "1500",
        "throttle_us": "1500",
        "steer_norm": "0.0",
        "throttle_norm": "0.0",
        "depth_front": "0.0",
        "rgb_path": image_path.name,
        "rgb_source": rgb_source,
        "depth_source": "realsense_d435i",
        "imu_source": "realsense_d435i",
        "rear_rgb_source": "none",
        "preprocess_profile": preprocess_profile,
        "run_id": run_name,
        "session_id": run_name,
        "cam0_path": str(image_path),
        "cam1_path": "",
        "ir_path": "",
        "depth_path": "",
    }

    with open(run_dir / "dataset.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_RUN_COLUMNS)
        writer.writeheader()
        writer.writerow(row)

    return run_dir


def test_combined_csv_rejects_mixed_rgb_sources(tmp_path):
    write_run(tmp_path, "run_a", rgb_source="cam0", preprocess_profile="cam0_fisheye_v1")
    write_run(tmp_path, "run_b", rgb_source="realsense", preprocess_profile="legacy_resize_v0")

    output_csv = tmp_path / "combined.csv"
    with pytest.raises(RuntimeError, match="Mixed rgb_source"):
        create_combined_csv(str(tmp_path), str(output_csv), allow_mixed=False)


def test_combined_csv_rejects_mixed_profiles(tmp_path):
    write_run(tmp_path, "run_a", rgb_source="cam0", preprocess_profile="cam0_fisheye_v1")
    write_run(tmp_path, "run_b", rgb_source="cam0", preprocess_profile="legacy_resize_v0")

    output_csv = tmp_path / "combined.csv"
    with pytest.raises(RuntimeError, match="Mixed preprocess_profile"):
        create_combined_csv(str(tmp_path), str(output_csv), allow_mixed=False)


def test_combined_csv_filter_preserves_metadata_tail(tmp_path):
    write_run(tmp_path, "run_a", rgb_source="cam0", preprocess_profile="cam0_fisheye_v1")
    write_run(tmp_path, "run_b", rgb_source="realsense", preprocess_profile="legacy_resize_v0")

    output_csv = tmp_path / "combined_cam0.csv"
    create_combined_csv(
        str(tmp_path),
        str(output_csv),
        allow_mixed=False,
        rgb_source_filter="cam0",
    )

    with open(output_csv, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    expected_tail = FLATTENED_METADATA_COLUMNS
    assert header[: len(LEGACY_COLUMNS)] == LEGACY_COLUMNS
    assert header[-len(expected_tail) :] == expected_tail
    assert len(rows) == 1
    assert rows[0][len(LEGACY_COLUMNS) + 57600 :] == [
        "cam0",
        "realsense_d435i",
        "realsense_d435i",
        "none",
        "cam0_fisheye_v1",
        "run_a",
        "run_a",
    ]


def test_load_dataset_defaults_legacy_metadata(tmp_path):
    write_run(tmp_path, "run_a", rgb_source="cam0", preprocess_profile="cam0_fisheye_v1")
    output_csv = tmp_path / "combined.csv"
    create_combined_csv(str(tmp_path), str(output_csv), allow_mixed=True)

    df = pd.read_csv(output_csv)
    stripped = df[LEGACY_COLUMNS + [c for c in df.columns if c.startswith(("R", "G", "B"))]].copy()
    stripped_path = tmp_path / "combined_stripped.csv"
    stripped.to_csv(stripped_path, index=False)

    loaded_df, pixel_columns = load_dataset(str(stripped_path))

    assert len(pixel_columns) == 160 * 120 * 3
    assert loaded_df["rgb_source"].eq("realsense").all()
    assert loaded_df["preprocess_profile"].eq("legacy_resize_v0").all()


def test_augment_pixel_column_detection_ignores_metadata():
    df = pd.DataFrame(
        columns=[
            "timestamp",
            "rgb_source",
            "R2",
            "depth_front",
            "G1",
            "preprocess_profile",
            "B1",
            "R1",
        ]
    )

    assert get_aug_pixel_columns(df) == ["R1", "G1", "B1", "R2"]
