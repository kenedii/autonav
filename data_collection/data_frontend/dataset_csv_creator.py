import argparse
import csv
import os
import re
from collections import Counter
from pathlib import Path

from PIL import Image

import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from preprocess_utils import canonicalize_preprocess_profile


LEGACY_COLUMNS = [
    "timestamp",
    "steer_us",
    "throttle_us",
    "steer_norm",
    "throttle_norm",
    "depth_front",
]
METADATA_COLUMNS = [
    "rgb_source",
    "depth_source",
    "imu_source",
    "rear_rgb_source",
    "preprocess_profile",
    "run_id",
    "session_id",
]
PIXEL_PATTERN = re.compile(r"^[RGB](\d+)$")


def resolve_image_path(run_dir, raw_path):
    if not raw_path:
        return ""

    raw = str(raw_path).strip().replace("\\", "/")
    if os.path.isabs(raw) and os.path.exists(raw):
        return raw

    candidate_names = [raw, os.path.basename(raw)]
    search_roots = [
        run_dir,
        os.path.dirname(run_dir),
        os.getcwd(),
    ]
    for root in search_roots:
        for name in candidate_names:
            candidate = os.path.normpath(os.path.join(root, name))
            if os.path.exists(candidate):
                return candidate
    return os.path.normpath(os.path.join(run_dir, os.path.basename(raw)))


def infer_metadata(row, run_dir):
    rgb_source = (row.get("rgb_source") or "realsense").strip() or "realsense"
    depth_source = (row.get("depth_source") or ("realsense_d435i" if rgb_source != "none" else "none")).strip() or "none"
    imu_source = (row.get("imu_source") or depth_source).strip() or "none"
    rear_rgb_source = (row.get("rear_rgb_source") or "none").strip() or "none"
    preprocess_profile = (
        row.get("preprocess_profile")
        or ("cam0_fisheye_v1" if rgb_source == "cam0" else "legacy_resize_v0")
    ).strip() or "legacy_resize_v0"
    preprocess_profile = canonicalize_preprocess_profile(preprocess_profile)
    run_id = (row.get("run_id") or os.path.basename(run_dir)).strip()
    session_id = (row.get("session_id") or run_id).strip()
    return {
        "rgb_source": rgb_source,
        "depth_source": depth_source,
        "imu_source": imu_source,
        "rear_rgb_source": rear_rgb_source,
        "preprocess_profile": preprocess_profile,
        "run_id": run_id,
        "session_id": session_id,
    }


def get_run_rows(run_dir):
    csv_path = os.path.join(run_dir, "dataset.csv")
    if not os.path.exists(csv_path):
        return []

    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rgb_path = row.get("rgb_path")
            if not rgb_path:
                continue

            image_path = resolve_image_path(run_dir, rgb_path)
            if not os.path.exists(image_path):
                continue

            metadata = infer_metadata(row, run_dir)
            rows.append({
                "row": row,
                "image_path": image_path,
                "metadata": metadata,
            })
    return rows


def collect_runs(root_dir, rgb_source_filter=None):
    run_dirs = [
        os.path.join(root_dir, d)
        for d in sorted(os.listdir(root_dir))
        if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("run_")
    ]

    records = []
    for run_dir in run_dirs:
        for record in get_run_rows(run_dir):
            if rgb_source_filter and record["metadata"]["rgb_source"] != rgb_source_filter:
                continue
            record["run_dir"] = run_dir
            records.append(record)
    return records


def find_sample_image(records):
    for record in records:
        if os.path.exists(record["image_path"]):
            return record["image_path"]
    return None


def get_pixel_columns(width, height):
    return [f"{channel}{index}" for index in range(1, width * height + 1) for channel in ("R", "G", "B")]


def build_combined_rows(records, width, height):
    pixel_columns = get_pixel_columns(width, height)
    output_rows = []
    source_counts = Counter()
    profile_counts = Counter()

    for record in records:
        image = Image.open(record["image_path"]).convert("RGB")
        if image.size != (width, height):
            continue

        row = record["row"]
        metadata = record["metadata"]
        source_counts[metadata["rgb_source"]] += 1
        profile_counts[metadata["preprocess_profile"]] += 1

        pixels = [str(value) for pixel in image.getdata() for value in pixel]
        base_row = [
            str(row.get("timestamp", "")),
            str(row.get("steer_us", "")),
            str(row.get("throttle_us", "")),
            str(row.get("steer_norm", "")),
            str(row.get("throttle_norm", "")),
            str(row.get("depth_front", "")),
        ]
        tail = [metadata[column] for column in METADATA_COLUMNS]
        output_rows.append(base_row + pixels + tail)

    return output_rows, pixel_columns, source_counts, profile_counts


def create_combined_csv(root_dir, output_csv, allow_mixed=False, rgb_source_filter=None):
    records = collect_runs(root_dir, rgb_source_filter=rgb_source_filter)
    sample_image_path = find_sample_image(records)
    if not sample_image_path:
        raise FileNotFoundError("No valid image found to determine size.")

    img = Image.open(sample_image_path).convert("RGB")
    width, height = img.size

    rows, pixel_columns, source_counts, profile_counts = build_combined_rows(records, width, height)
    if not rows:
        raise RuntimeError("No rows were eligible for combination.")

    filtered_sources = [source for source, count in source_counts.items() if count > 0]
    filtered_profiles = [profile for profile, count in profile_counts.items() if count > 0]
    print(f"Source counts: {dict(source_counts)}")
    print(f"Preprocess profile counts: {dict(profile_counts)}")

    if not allow_mixed:
        if len(filtered_sources) > 1:
            raise RuntimeError(
                "Mixed rgb_source values detected. Re-run with --allow-mixed or --rgb-source <value>."
            )
        if len(filtered_profiles) > 1:
            raise RuntimeError(
                "Mixed preprocess_profile values detected. Re-run with --allow-mixed or filter to a single rgb_source."
            )

    header = LEGACY_COLUMNS + pixel_columns + METADATA_COLUMNS
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Combined dataset saved to {output_csv}")
    return output_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Create a pixel-flattened combined dataset CSV.")
    parser.add_argument("--root-dir", default="data", help="Folder containing run_* subfolders")
    parser.add_argument("--output-csv", default="combined_dataset.csv", help="Output CSV path")
    parser.add_argument("--allow-mixed", action="store_true", help="Allow mixed rgb_source and preprocess_profile values")
    parser.add_argument("--rgb-source", default=None, help="Filter to a single rgb_source before combining")
    return parser.parse_args()


def main():
    args = parse_args()
    create_combined_csv(
        args.root_dir,
        args.output_csv,
        allow_mixed=args.allow_mixed,
        rgb_source_filter=args.rgb_source,
    )


if __name__ == "__main__":
    main()
