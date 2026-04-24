#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import statistics
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "autonav-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "docs" / "final_artifacts" / "data_eda"
RUNS_ROOT = REPO_ROOT / "jetracer" / "train" / "runs_rgb_depth"
TRAIN_ROOT = REPO_ROOT / "jetracer" / "train"

sys.path.insert(0, str(REPO_ROOT / "data_collection"))
from preprocess_utils import (  # noqa: E402
    CAM0_FISHEYE_PREPROCESS_PROFILE,
    apply_preprocess_profile,
)


PATH_SENSOR_FIELDS = [
    "rgb_path",
    "cam0_path",
    "cam1_path",
    "ir_path",
    "depth_path",
    "realsense_rgb_path",
]
SCALAR_SENSOR_FIELDS = ["depth_front"]
IMU_FIELDS = [
    "accel_x",
    "accel_y",
    "accel_z",
    "accel_ts_ms",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "gyro_ts_ms",
]
ALL_SENSOR_FIELDS = PATH_SENSOR_FIELDS + SCALAR_SENSOR_FIELDS + IMU_FIELDS

STEERING_LEFT_THRESHOLD = -0.15
STEERING_RIGHT_THRESHOLD = 0.15


def is_blank(value) -> bool:
    if value is None:
        return True
    return str(value).strip() == ""


def parse_float(value):
    if is_blank(value):
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def resolve_recorded_path(relative_value: str | None, run_dir: Path) -> Path | None:
    if is_blank(relative_value):
        return None

    value = str(relative_value).strip()
    candidate = Path(value)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    possible_paths = [
        REPO_ROOT / value,
        TRAIN_ROOT / value,
        run_dir / value,
        run_dir.parent / value,
        run_dir.parent.parent / value,
    ]
    for path in possible_paths:
        if path.exists():
            return path
    return None


def normalize_grayscale(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros(arr.shape[:2], dtype=np.uint8)

    lower = np.percentile(valid, 1)
    upper = np.percentile(valid, 99)
    if upper <= lower:
        lower = float(valid.min())
        upper = float(valid.max())
    if upper <= lower:
        return np.zeros(arr.shape[:2], dtype=np.uint8)

    scaled = np.clip((arr - lower) / (upper - lower), 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def load_rgb_image(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_ir_image(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized = normalize_grayscale(image)
    return cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)


def load_depth_image(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized = normalize_grayscale(image)
    colorized = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)


def format_pct(count: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{(count / total) * 100:.1f}%"


def stats_dict(values: list[float]) -> dict[str, str]:
    if not values:
        return {
            "count": "0",
            "min": "not measured",
            "max": "not measured",
            "mean": "not measured",
            "median": "not measured",
            "stdev": "not measured",
        }
    return {
        "count": str(len(values)),
        "min": f"{min(values):.4f}",
        "max": f"{max(values):.4f}",
        "mean": f"{statistics.mean(values):.4f}",
        "median": f"{statistics.median(values):.4f}",
        "stdev": f"{statistics.stdev(values):.4f}" if len(values) > 1 else "0.0000",
    }


def write_csv_summary(rows: list[dict[str, str]]) -> None:
    csv_path = OUTPUT_DIR / "data_summary.csv"
    fieldnames = ["category", "metric", "value", "notes"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_histogram(values: list[float], title: str, xlabel: str, output_path: Path, color: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.hist(values, bins=21, color=color, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Rows")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def create_sensor_coverage_plot(total_rows: int, counts: dict[str, int]) -> None:
    labels = [
        "CAM0 / rgb_path",
        "CAM1 / cam1_path",
        "IR",
        "depth_path",
        "realsense_rgb_path",
        "IMU accel",
        "IMU gyro",
    ]
    values = [
        counts["rgb_path"],
        counts["cam1_path"],
        counts["ir_path"],
        counts["depth_path"],
        counts["realsense_rgb_path"],
        counts["imu_accel_rows"],
        counts["imu_gyro_rows"],
    ]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    bars = ax.bar(labels, values, color=["#2E86AB", "#4F9D69", "#A23B72", "#F18F01", "#7B6D8D", "#5D737E", "#8F6593"])
    ax.set_title("Sensor Coverage Across Archived Training Runs")
    ax.set_ylabel("Rows with non-empty data")
    ax.set_ylim(0, max(total_rows, max(values) if values else 0) * 1.08 if total_rows else 1)
    ax.grid(axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(total_rows * 0.01, 5),
            f"{value}\n({format_pct(value, total_rows)})",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sensor_coverage.png", bbox_inches="tight")
    plt.close(fig)


def create_sample_grid(sample: dict | None) -> tuple[list[str], str]:
    output_path = OUTPUT_DIR / "sample_sensor_grid.png"
    if sample is None:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        ax.text(0.5, 0.5, "No representative sample row with loadable images was found.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return ["No representative sample row with loadable images was found."], "not available"

    raw_cam0_path = sample["cam0_path"] or sample["rgb_path"]
    cam1_path = sample["cam1_path"]
    ir_path = sample["ir_path"]
    depth_path = sample["depth_path"]
    profile = sample.get("preprocess_profile") or CAM0_FISHEYE_PREPROCESS_PROFILE

    raw_cam0 = load_rgb_image(raw_cam0_path) if raw_cam0_path else None
    cam1 = load_rgb_image(cam1_path) if cam1_path else None
    ir_image = load_ir_image(ir_path) if ir_path else None
    depth_image = load_depth_image(depth_path) if depth_path else None
    preprocessed = apply_preprocess_profile(raw_cam0, profile) if raw_cam0 is not None else None

    failed = []
    if raw_cam0 is None:
        failed.append("CAM0 / rgb_path")
    if cam1_path and cam1 is None:
        failed.append("cam1_path")
    if ir_path and ir_image is None:
        failed.append("ir_path")
    if depth_path and depth_image is None:
        failed.append("depth_path")

    tiles = [
        ("CAM0 / rgb_path", raw_cam0),
        ("CAM1 / cam1_path", cam1),
        ("IR", ir_image),
        ("depth_path", depth_image),
        ("Model input (160x120)", preprocessed),
        ("Sample metadata", None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=150)
    axes = axes.flatten()

    for ax, (title, image) in zip(axes, tiles):
        ax.set_title(title)
        ax.axis("off")
        if image is not None:
            ax.imshow(image)
        elif title == "Sample metadata":
            ax.text(
                0.05,
                0.95,
                "\n".join(
                    [
                        f"run_id: {sample.get('run_id') or sample['run_dir'].name}",
                        f"profile: {profile}",
                        f"rgb: {sample['rgb_path'].name if sample['rgb_path'] else 'missing'}",
                        f"cam1: {sample['cam1_path'].name if sample['cam1_path'] else 'missing'}",
                        f"ir: {sample['ir_path'].name if sample['ir_path'] else 'missing'}",
                        f"depth: {sample['depth_path'].name if sample['depth_path'] else 'missing'}",
                    ]
                ),
                ha="left",
                va="top",
                fontsize=9,
                family="monospace",
                transform=ax.transAxes,
            )
        else:
            ax.text(0.5, 0.5, "Not available", ha="center", va="center")

    fig.suptitle("Representative Multi-Sensor Sample", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    preprocessing_output = OUTPUT_DIR / "preprocessing_example.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=150)
    for ax in axes:
        ax.axis("off")

    if raw_cam0 is not None:
        axes[0].imshow(raw_cam0)
        axes[0].set_title(f"Raw CAM0 input\n{raw_cam0.shape[1]}x{raw_cam0.shape[0]}")
    else:
        axes[0].text(0.5, 0.5, "Raw CAM0 input unavailable", ha="center", va="center")
        axes[0].set_title("Raw CAM0 input")

    if preprocessed is not None:
        axes[1].imshow(preprocessed)
        axes[1].set_title(f"Model-preprocessed input\n{preprocessed.shape[1]}x{preprocessed.shape[0]}")
    else:
        axes[1].text(0.5, 0.5, "Preprocessing unavailable", ha="center", va="center")
        axes[1].set_title("Model-preprocessed input")

    fig.suptitle(f"CAM0 preprocessing example ({profile})", fontsize=13)
    fig.tight_layout()
    fig.savefig(preprocessing_output, bbox_inches="tight")
    plt.close(fig)

    return failed, profile


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(path for path in RUNS_ROOT.iterdir() if path.is_dir() and path.name.startswith("run_"))
    csv_paths = [path / "dataset.csv" for path in run_dirs if (path / "dataset.csv").exists()]

    total_rows = 0
    non_empty_runs = 0
    empty_runs = []
    union_columns: set[str] = set()
    run_row_counts: list[tuple[str, int]] = []
    present_counts = Counter()
    missing_counts = Counter()
    steering_values: list[float] = []
    throttle_values: list[float] = []
    imu_accel_rows = 0
    imu_gyro_rows = 0
    best_sample = None
    best_sample_score = -1

    for csv_path in csv_paths:
        run_dir = csv_path.parent
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            union_columns.update(fieldnames)

            run_rows = 0
            for row in reader:
                run_rows += 1
                total_rows += 1

                steer = parse_float(row.get("steer_norm"))
                throttle = parse_float(row.get("throttle_norm"))
                if steer is not None:
                    steering_values.append(steer)
                if throttle is not None:
                    throttle_values.append(throttle)

                for field in ALL_SENSOR_FIELDS:
                    value = row.get(field, "")
                    if is_blank(value):
                        missing_counts[field] += 1
                    else:
                        present_counts[field] += 1

                if any(not is_blank(row.get(field, "")) for field in ("accel_x", "accel_y", "accel_z", "accel_ts_ms")):
                    imu_accel_rows += 1
                if any(not is_blank(row.get(field, "")) for field in ("gyro_x", "gyro_y", "gyro_z", "gyro_ts_ms")):
                    imu_gyro_rows += 1

                raw_cam0_path = resolve_recorded_path(row.get("cam0_path"), run_dir) or resolve_recorded_path(row.get("rgb_path"), run_dir)
                cam1_path = resolve_recorded_path(row.get("cam1_path"), run_dir)
                ir_path = resolve_recorded_path(row.get("ir_path"), run_dir)
                depth_path = resolve_recorded_path(row.get("depth_path"), run_dir)
                score = sum(path is not None for path in (raw_cam0_path, cam1_path, ir_path, depth_path))
                if score > best_sample_score:
                    best_sample_score = score
                    best_sample = {
                        "run_dir": run_dir,
                        "run_id": row.get("run_id") or run_dir.name,
                        "rgb_path": resolve_recorded_path(row.get("rgb_path"), run_dir),
                        "cam0_path": raw_cam0_path,
                        "cam1_path": cam1_path,
                        "ir_path": ir_path,
                        "depth_path": depth_path,
                        "preprocess_profile": row.get("preprocess_profile"),
                    }

        run_row_counts.append((run_dir.name, run_rows))
        if run_rows > 0:
            non_empty_runs += 1
        else:
            empty_runs.append(run_dir.name)

    combined_csvs = sorted(
        path
        for path in REPO_ROOT.rglob("combined*.csv")
        if OUTPUT_DIR not in path.parents and ".git" not in path.parts and "__pycache__" not in path.parts
    )

    steering_stats = stats_dict(steering_values)
    throttle_stats = stats_dict(throttle_values)

    steering_left = sum(value < STEERING_LEFT_THRESHOLD for value in steering_values)
    steering_center = sum(STEERING_LEFT_THRESHOLD <= value <= STEERING_RIGHT_THRESHOLD for value in steering_values)
    steering_right = sum(value > STEERING_RIGHT_THRESHOLD for value in steering_values)

    throttle_reverse = sum(value < 0 for value in throttle_values)
    throttle_zero = sum(abs(value) < 1e-9 for value in throttle_values)
    throttle_forward = sum(value > 0 for value in throttle_values)

    coverage_counts = {
        "rgb_path": present_counts["rgb_path"],
        "cam1_path": present_counts["cam1_path"],
        "ir_path": present_counts["ir_path"],
        "depth_path": present_counts["depth_path"],
        "realsense_rgb_path": present_counts["realsense_rgb_path"],
        "imu_accel_rows": imu_accel_rows,
        "imu_gyro_rows": imu_gyro_rows,
    }

    create_histogram(
        steering_values,
        "Steering Distribution (steer_norm)",
        "Normalized steering",
        OUTPUT_DIR / "steering_distribution.png",
        "#2E86AB",
    )
    create_histogram(
        throttle_values,
        "Throttle Distribution (throttle_norm)",
        "Normalized throttle",
        OUTPUT_DIR / "throttle_distribution.png",
        "#F18F01",
    )
    create_sensor_coverage_plot(total_rows, coverage_counts)
    failed_loads, sample_profile = create_sample_grid(best_sample)

    sensors_present = []
    if present_counts["rgb_path"] > 0:
        sensors_present.append("CAM0 / rgb_path")
    if present_counts["cam1_path"] > 0:
        sensors_present.append("CAM1 / cam1_path")
    if present_counts["ir_path"] > 0:
        sensors_present.append("IR")
    if present_counts["depth_path"] > 0:
        sensors_present.append("depth_path")
    if present_counts["depth_front"] > 0:
        sensors_present.append("depth_front scalar")
    if present_counts["realsense_rgb_path"] > 0:
        sensors_present.append("realsense_rgb_path")
    if imu_accel_rows > 0:
        sensors_present.append("IMU accel")
    if imu_gyro_rows > 0:
        sensors_present.append("IMU gyro")

    limitations = [
        f"{len(empty_runs)} run folder(s) contain an empty dataset.csv and contribute no labeled rows.",
        "The archived training runs are center-heavy, so recovery and edge-case behavior are underrepresented.",
        "CAM1 / IR / depth are only present on a subset of rows, while the primary CAM0 / rgb_path is available almost everywhere.",
        "No checked-in combined dataset CSVs were found, so train/validation/test split information comes from training code rather than a committed final dataset artifact.",
    ]
    if present_counts["realsense_rgb_path"] == 0:
        limitations.append("The current archived training snapshot does not include realsense_rgb_path images, even though the newer recorder supports them.")
    if imu_accel_rows == 0 and imu_gyro_rows == 0:
        limitations.append("The current archived training snapshot does not include IMU fields, even though the newer recorder supports accel/gyro logging.")
    if failed_loads:
        limitations.append("Some representative sample images could not be loaded for visualization: " + ", ".join(failed_loads) + ".")

    markdown_lines = [
        "# AutoNav Data Preparation and EDA Summary",
        "",
        "## Dataset scope",
        f"- Run folders found: `{len(run_dirs)}`",
        f"- Dataset CSV files found: `{len(csv_paths)}`",
        f"- Total raw rows across all `dataset.csv` files: `{total_rows}`",
        f"- Non-empty runs: `{non_empty_runs}`",
        f"- Empty runs: `{len(empty_runs)}`" + (f" ({', '.join(empty_runs)})" if empty_runs else ""),
        "",
        "## Sensors present in the archived training snapshot",
        "- " + (", ".join(sensors_present) if sensors_present else "No sensor columns with non-empty values were found."),
        "",
        "## Missing-data counts per sensor column",
        "",
        "| Column | Present rows | Missing / empty rows | Coverage |",
        "|---|---:|---:|---:|",
    ]
    for field in PATH_SENSOR_FIELDS + SCALAR_SENSOR_FIELDS + IMU_FIELDS:
        present = present_counts[field]
        missing = missing_counts[field]
        markdown_lines.append(f"| `{field}` | {present} | {missing} | {format_pct(present, total_rows)} |")

    markdown_lines.extend(
        [
            "",
            "## Steering distribution",
            f"- Count: `{steering_stats['count']}`",
            f"- Min / max: `{steering_stats['min']}` / `{steering_stats['max']}`",
            f"- Mean / median / stdev: `{steering_stats['mean']}` / `{steering_stats['median']}` / `{steering_stats['stdev']}`",
            f"- Left / center / right bins using thresholds `< {STEERING_LEFT_THRESHOLD}`, `{STEERING_LEFT_THRESHOLD} to {STEERING_RIGHT_THRESHOLD}`, `> {STEERING_RIGHT_THRESHOLD}`: `{steering_left}` / `{steering_center}` / `{steering_right}`",
            "",
            "## Throttle distribution",
            f"- Count: `{throttle_stats['count']}`",
            f"- Min / max: `{throttle_stats['min']}` / `{throttle_stats['max']}`",
            f"- Mean / median / stdev: `{throttle_stats['mean']}` / `{throttle_stats['median']}` / `{throttle_stats['stdev']}`",
            f"- Reverse / zero / forward rows: `{throttle_reverse}` / `{throttle_zero}` / `{throttle_forward}`",
            "",
            "## Train / validation / test split",
            "- `model_training/train_model_experiments.py` uses a `70 / 15 / 15` split.",
            "- `model_training/train_model_resnet.py` uses a legacy `80 / 20` train/test split.",
            "",
            "## Known dataset limitations",
        ]
    )
    markdown_lines.extend(f"- {item}" for item in limitations)
    markdown_lines.extend(
        [
            "",
            "## Additional notes",
            f"- Representative preprocessing profile used for the sample figures: `{sample_profile}`.",
            f"- Combined dataset CSVs found in repo: `{len(combined_csvs)}`.",
        ]
    )
    if combined_csvs:
        markdown_lines.extend(f"  - `{path.relative_to(REPO_ROOT)}`" for path in combined_csvs[:10])
    else:
        markdown_lines.append("- No `combined*.csv` artifacts are checked into this repo snapshot.")

    (OUTPUT_DIR / "data_summary.md").write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    csv_rows = [
        {"category": "summary", "metric": "run_folders", "value": str(len(run_dirs)), "notes": "run_* folders under jetracer/train/runs_rgb_depth"},
        {"category": "summary", "metric": "dataset_csv_files", "value": str(len(csv_paths)), "notes": "dataset.csv files discovered"},
        {"category": "summary", "metric": "total_raw_rows", "value": str(total_rows), "notes": "non-header rows across all dataset.csv files"},
        {"category": "summary", "metric": "non_empty_runs", "value": str(non_empty_runs), "notes": "runs with at least one labeled row"},
        {"category": "summary", "metric": "empty_runs", "value": str(len(empty_runs)), "notes": ",".join(empty_runs)},
        {"category": "split", "metric": "experiment_train_val_test", "value": "70/15/15", "notes": "model_training/train_model_experiments.py"},
        {"category": "split", "metric": "legacy_train_test", "value": "80/20", "notes": "model_training/train_model_resnet.py"},
        {"category": "steering", "metric": "left_rows", "value": str(steering_left), "notes": f"value < {STEERING_LEFT_THRESHOLD}"},
        {"category": "steering", "metric": "center_rows", "value": str(steering_center), "notes": f"{STEERING_LEFT_THRESHOLD} <= value <= {STEERING_RIGHT_THRESHOLD}"},
        {"category": "steering", "metric": "right_rows", "value": str(steering_right), "notes": f"value > {STEERING_RIGHT_THRESHOLD}"},
        {"category": "throttle", "metric": "reverse_rows", "value": str(throttle_reverse), "notes": "value < 0"},
        {"category": "throttle", "metric": "zero_rows", "value": str(throttle_zero), "notes": "value == 0"},
        {"category": "throttle", "metric": "forward_rows", "value": str(throttle_forward), "notes": "value > 0"},
    ]
    for field in PATH_SENSOR_FIELDS + SCALAR_SENSOR_FIELDS + IMU_FIELDS:
        csv_rows.append({"category": "sensor", "metric": f"{field}_present_rows", "value": str(present_counts[field]), "notes": ""})
        csv_rows.append({"category": "sensor", "metric": f"{field}_missing_rows", "value": str(missing_counts[field]), "notes": ""})
    csv_rows.extend(
        [
            {"category": "sensor", "metric": "imu_accel_rows", "value": str(imu_accel_rows), "notes": "rows with any accel field populated"},
            {"category": "sensor", "metric": "imu_gyro_rows", "value": str(imu_gyro_rows), "notes": "rows with any gyro field populated"},
        ]
    )
    write_csv_summary(csv_rows)

    coverage_lines = [
        "- AutoNav training data was recorded from real RC-car runs stored under `jetracer/train/runs_rgb_depth/run_*`.",
        f"- The archived training snapshot contains `{total_rows}` labeled rows across `{len(run_dirs)}` run folders, with `{non_empty_runs}` non-empty runs.",
        f"- Primary front-camera coverage is strong: `rgb_path` is present on `{present_counts['rgb_path']}` rows ({format_pct(present_counts['rgb_path'], total_rows)}).",
        f"- Secondary modalities are partial: `cam1_path` `{present_counts['cam1_path']}`, `ir_path` `{present_counts['ir_path']}`, `depth_path` `{present_counts['depth_path']}`.",
        f"- Steering labels are center-heavy: `{steering_center}` center rows vs `{steering_left}` left and `{steering_right}` right.",
        f"- The main experiment trainer uses a `70 / 15 / 15` split; the legacy trainer still contains an `80 / 20` split path.",
        "- The archived training snapshot does not include the newer `realsense_rgb_path` or IMU fields, so those newer recorder capabilities should be framed as forward-looking rather than core training inputs for this dataset.",
    ]
    (OUTPUT_DIR / "slide_ready_data_bullets.md").write_text("# Slide-ready data bullets\n\n" + "\n".join(coverage_lines) + "\n", encoding="utf-8")

    speaker_notes = [
        "# Speaker notes: data preparation and EDA",
        "",
        "We collected this dataset by manually driving the RC car and saving synchronized steering, throttle, and sensor outputs into per-run folders.",
        "",
        f"In the archived training snapshot we inspected, there are {len(run_dirs)} run folders and {total_rows} labeled rows. The front driving camera is the most complete modality, while the back camera, IR, and depth streams appear on only part of the dataset.",
        "",
        f"The steering labels are center-heavy, with {steering_center} center rows compared to {steering_left} left and {steering_right} right. That matters because it means straight driving is better represented than aggressive recovery behavior at the tape edges.",
        "",
        "For preprocessing, the live CAM0 path uses the same `cam0_fisheye_v1` crop-and-resize profile that was applied to the training inputs. That consistency is important because the model is learning from a specific image geometry, not from arbitrary raw frames.",
        "",
        "One honest limitation is that this archived training snapshot does not include the newer RealSense RGB sidecar path or IMU columns, even though the recorder now supports them. For the final presentation, we should describe those as newer capabilities rather than implying they were part of the original training set used for the validated live demo.",
    ]
    (OUTPUT_DIR / "speaker_notes_data.md").write_text("\n".join(speaker_notes) + "\n", encoding="utf-8")

    output_files = sorted(path.name for path in OUTPUT_DIR.iterdir() if path.is_file())
    print(f"Run folders: {len(run_dirs)}")
    print(f"Total raw rows: {total_rows}")
    print(f"Non-empty runs: {non_empty_runs}")
    print(f"Sensors present: {', '.join(sensors_present) if sensors_present else 'none'}")
    print(f"Failed sample loads: {', '.join(failed_loads) if failed_loads else 'none'}")
    print("Generated files:")
    for name in output_files:
        print(f"- {OUTPUT_DIR / name}")


if __name__ == "__main__":
    main()
