import os
import re
import sys
from collections import Counter

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from preprocess_utils import canonicalize_preprocess_profile


LEGACY_METADATA_DEFAULTS = {
    "rgb_source": "realsense",
    "depth_source": "realsense_d435i",
    "imu_source": "realsense_d435i",
    "rear_rgb_source": "none",
    "preprocess_profile": "legacy_resize_v0",
    "run_id": "",
    "session_id": "",
}
EXPECTED_PIXEL_COLUMNS = 160 * 120 * 3

RGB_CHANNELS = ("R", "G", "B")
CHANNEL_ORDER = {channel: index for index, channel in enumerate(RGB_CHANNELS)}
PIXEL_COLUMN_PATTERN = re.compile(r"^([RGB])(\d+)$")


def get_pixel_columns(df):
    pixel_columns = []
    for column in df.columns:
        match = PIXEL_COLUMN_PATTERN.match(str(column))
        if match:
            pixel_columns.append(column)

    pixel_columns.sort(key=lambda column: (
        int(PIXEL_COLUMN_PATTERN.match(str(column)).group(2)),
        CHANNEL_ORDER[PIXEL_COLUMN_PATTERN.match(str(column)).group(1)],
    ))
    return pixel_columns


def load_dataset(dataset_path, rgb_source_filter=None):
    import pandas as pd

    df = pd.read_csv(dataset_path)
    if "depth_front" not in df.columns:
        df["depth_front"] = 0.0

    for key, default in LEGACY_METADATA_DEFAULTS.items():
        if key not in df.columns:
            df[key] = default

    df["rgb_source"] = df["rgb_source"].fillna(LEGACY_METADATA_DEFAULTS["rgb_source"]).astype(str)
    df["preprocess_profile"] = df["preprocess_profile"].fillna(LEGACY_METADATA_DEFAULTS["preprocess_profile"]).astype(str)
    df["depth_source"] = df["depth_source"].fillna(LEGACY_METADATA_DEFAULTS["depth_source"]).astype(str)
    df["imu_source"] = df["imu_source"].fillna(LEGACY_METADATA_DEFAULTS["imu_source"]).astype(str)
    df["rear_rgb_source"] = df["rear_rgb_source"].fillna(LEGACY_METADATA_DEFAULTS["rear_rgb_source"]).astype(str)
    df["run_id"] = df["run_id"].fillna(LEGACY_METADATA_DEFAULTS["run_id"]).astype(str)
    df["session_id"] = df["session_id"].fillna(df["run_id"]).astype(str)

    if rgb_source_filter:
        df = df[df["rgb_source"] == rgb_source_filter].copy()

    if df.empty:
        raise RuntimeError("No rows available after applying filters.")

    df["preprocess_profile"] = df["preprocess_profile"].map(canonicalize_preprocess_profile)
    if "rgb_source" not in df.columns:
        df["rgb_source"] = LEGACY_METADATA_DEFAULTS["rgb_source"]

    pixel_columns = get_pixel_columns(df)
    if len(pixel_columns) != EXPECTED_PIXEL_COLUMNS:
        raise RuntimeError(
            f"Expected {EXPECTED_PIXEL_COLUMNS} flattened RGB columns for 160x120 images, found {len(pixel_columns)}."
        )

    return df, pixel_columns


def log_dataset_metadata(df):
    source_counts = Counter(df["rgb_source"].tolist())
    profile_counts = Counter(df["preprocess_profile"].tolist())
    print(f"Source counts: {dict(source_counts)}")
    print(f"Preprocess profile counts: {dict(profile_counts)}")
    return source_counts, profile_counts


def validate_dataset_metadata(source_counts, profile_counts, allow_mixed=False):
    if allow_mixed:
        return
    if len(source_counts) > 1:
        raise RuntimeError("Mixed rgb_source values detected. Re-run with --allow-mixed or --rgb-source <value>.")
    if len(profile_counts) > 1:
        raise RuntimeError(
            "Mixed preprocess_profile values detected. Re-run with --allow-mixed or filter to a single rgb_source."
        )
