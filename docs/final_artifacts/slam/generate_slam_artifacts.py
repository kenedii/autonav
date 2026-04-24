#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "docs" / "final_artifacts" / "slam"
RUNS_ROOT = REPO_ROOT / "jetracer" / "train" / "runs_rgb_depth"
PREFERRED_RUN = "run_20260421_040824"

TESTS_DIR = REPO_ROOT / "tests"
CLIENT_API_DIR = REPO_ROOT / "fleet" / "fleet_management_app" / "client_api"
for path in (TESTS_DIR, CLIENT_API_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from slam import VisualSlamSystem  # noqa: E402
from test_slam import draw_pose_overlay, replay_run  # noqa: E402


def write_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def depth_sample_info(run_dir: Path) -> str | None:
    for depth_path in sorted(run_dir.glob("depth_*.png")):
        image = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        return f"{depth_path.name}: shape={tuple(image.shape)} dtype={image.dtype}"
    return None


def pick_run() -> tuple[Path | None, str]:
    preferred = RUNS_ROOT / PREFERRED_RUN
    if preferred.exists():
        return preferred, "preferred run found"

    best_run = None
    best_depth_rows = -1
    best_total_rows = -1

    for csv_path in sorted(RUNS_ROOT.glob("run_*/dataset.csv")):
        run_dir = csv_path.parent
        total = 0
        depth_rows = 0
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                total += 1
                if (row.get("depth_path") or "").strip():
                    depth_rows += 1
        if depth_rows > best_depth_rows or (depth_rows == best_depth_rows and total > best_total_rows):
            best_run = run_dir
            best_depth_rows = depth_rows
            best_total_rows = total

    if best_run is None:
        return None, "no replayable run found"
    return best_run, "preferred run missing; selected best available depth-backed run"


def save_metrics_csv(rows: list[dict[str, object]]) -> None:
    path = OUTPUT_DIR / "slam_replay_metrics.csv"
    fieldnames = [
        "frame_index",
        "motion_source",
        "tracking_points",
        "rgbd_points",
        "x",
        "y",
        "theta",
        "processing_time_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_trajectory_plot(rows: list[dict[str, object]]) -> None:
    xs = [row["x"] for row in rows]
    ys = [row["y"] for row in rows]
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    ax.plot(xs, ys, color="#2E86AB", linewidth=1.5, label="trajectory")
    if xs and ys:
        ax.scatter([xs[0]], [ys[0]], color="#4F9D69", s=40, label="start")
        ax.scatter([xs[-1]], [ys[-1]], color="#A23B72", s=40, label="end")
    ax.set_title("SLAM replay trajectory")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.25)
    ax.axis("equal")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "slam_replay_summary.png", bbox_inches="tight")
    plt.close(fig)


def save_overlay(frame_bgr, pose, frame_idx) -> None:
    overlay = draw_pose_overlay(frame_bgr, pose, frame_idx)
    cv2.imwrite(str(OUTPUT_DIR / "slam_pose_overlay.png"), overlay)


def create_reports(
    run_dir: Path,
    selection_note: str,
    rows: list[dict[str, object]],
    motion_counts: Counter,
    rgbd_points_values: list[int],
    elapsed_seconds: float,
    final_pose: dict,
    depth_info: str | None,
) -> None:
    frames_processed = len(rows)
    update_rate = frames_processed / elapsed_seconds if elapsed_seconds > 0 else None
    median_rgbd = int(np.median(rgbd_points_values)) if rgbd_points_values else None
    max_rgbd = int(max(rgbd_points_values)) if rgbd_points_values else None

    limitations = [
        "This module is experimental RGB-D odometry, not full production SLAM.",
        "There is no loop closure or global map optimization, so drift is expected.",
        "This replay run does not include IMU fields, so gyro fusion was not exercised here.",
        "SLAM state can feed navigation hooks in the fleet runtime, but it is not the validated live lane-follow control path.",
        "Motion source counts can include `reseed` / `bootstrap`, so trajectory continuity depends on feature tracking quality.",
    ]
    if not rgbd_points_values and depth_info is not None:
        limitations.append(
            f"The archived depth files in this run are preview-style images (`{depth_info}`), so the metric RGB-D path rejected them and replay fell back to RGB-only motion."
        )

    report_lines = [
        "# SLAM / RGB-D odometry status report",
        "",
        "## Algorithm summary",
        "- Sparse optical flow between consecutive RGB frames using `cv2.calcOpticalFlowPyrLK`.",
        "- Metric RGB-D correspondences built from tracked 2D points plus depth sampling.",
        "- Rigid transform fit via SVD to estimate camera motion when metric depth is available.",
        "- Fallback to RGB-only visual motion when metric RGB-D motion cannot be recovered.",
        "- Optional gyro fusion exists in code, but was not exercised in this replay because IMU fields were absent.",
        "",
        "## Sensor inputs",
        "- Required: RGB frame",
        "- Optional: depth map",
        "- Optional: IMU gyro / accel",
        "",
        "## Output state",
        "- `x`, `y`, `theta`",
        "- short `trajectory` history",
        "- `tracking_points`",
        "- `rgbd_points`",
        "- `motion_source`",
        "- `last_motion`",
        "",
        "## Replay run used",
        f"- Run directory: `{run_dir.relative_to(REPO_ROOT)}`",
        f"- Selection note: {selection_note}",
        "",
        "## Replay result",
        f"- Frames processed: `{frames_processed}`",
        f"- Final pose: `x={final_pose['x']:.3f} y={final_pose['y']:.3f} theta={final_pose['theta']:.3f}`",
        f"- Motion source counts: `{dict(motion_counts)}`",
        f"- RGB-D correspondences: `median={median_rgbd if median_rgbd is not None else 'not measured'} max={max_rgbd if max_rgbd is not None else 'not measured'} frames={len(rgbd_points_values)}`",
        f"- Update rate: `{update_rate:.2f} FPS`" if update_rate is not None else "- Update rate: `not measured`",
        f"- Depth sample inspected: `{depth_info}`" if depth_info else "- Depth sample inspected: `not available`",
        "",
        "## Drift / limitations",
    ]
    report_lines.extend(f"- {item}" for item in limitations)
    report_lines.extend(
        [
            "",
            "## Live-control status",
            "- Replay and code-review feature: yes",
            "- Validated live lane-follow dependency: no",
            "- Full production SLAM claim: no",
        ]
    )
    write_text(OUTPUT_DIR / "slam_status_report.md", "\n".join(report_lines))

    update_lines = [
        "# SLAM replay update-rate measurement",
        "",
        f"- Command: `python3 docs/final_artifacts/slam/generate_slam_artifacts.py`",
        f"- Replay run: `{run_dir.relative_to(REPO_ROOT)}`",
        f"- Frames processed: `{frames_processed}`",
        f"- Elapsed seconds: `{elapsed_seconds:.4f}`",
        f"- Estimated replay FPS / update rate: `{update_rate:.2f}`" if update_rate is not None else "- Estimated replay FPS / update rate: `not measured`",
        "- Note: this is replay processing throughput on the current host, not a live Jetson timing claim.",
    ]
    write_text(OUTPUT_DIR / "slam_update_rate.md", "\n".join(update_lines))

    slide_lines = [
        "# Slide-ready SLAM bullets",
        "",
        "- AutoNav includes an experimental RGB-D odometry / SLAM helper in `fleet/fleet_management_app/client_api/slam.py`.",
        f"- Preferred replay run was unavailable in this workspace, so the artifacts use `{run_dir.name}`, the strongest available depth-backed archived run.",
        f"- Replay produced a final pose of `x={final_pose['x']:.3f}, y={final_pose['y']:.3f}, theta={final_pose['theta']:.3f}` across `{frames_processed}` processed frames.",
        f"- Motion sources observed: `{dict(motion_counts)}`; RGB-D correspondence median/max: `{median_rgbd if median_rgbd is not None else 'not measured'}` / `{max_rgbd if max_rgbd is not None else 'not measured'}`.",
        f"- Archived depth sample in this run was `{depth_info}`." if depth_info else "- Archived depth sample could not be inspected.",
        f"- Replay throughput on this host was `{update_rate:.2f} FPS`." if update_rate is not None else "- Replay throughput on this host was not measured.",
        "- Final presentation wording should be: experimental RGB-D odometry with replay evidence. In this archived run, depth files are preview-style, so motion fell back to RGB-only updates.",
    ]
    write_text(OUTPUT_DIR / "slide_ready_slam_bullets.md", "\n".join(slide_lines))

    speaker_lines = [
        "# Speaker notes: SLAM / RGB-D odometry",
        "",
        "This feature is best described as an RGB-D visual odometry prototype rather than full production SLAM. The code tracks image features, samples metric depth when available, fits frame-to-frame rigid motion, and integrates that into a 2D pose estimate.",
        "",
        f"For the final evidence bundle, I replayed `{run_dir.name}` because the originally preferred run was not available in this workspace. That archived run has strong depth coverage, so it is the best available replay candidate here.",
        "",
        f"One important detail is that the archived depth sample we inspected was `{depth_info}`. That means the replayed depth files are preview-style images, not raw metric depth, so the RGB-D motion path did not activate and the run mostly used RGB visual motion instead.",
        "",
        f"On that replay, the system processed {frames_processed} frames and ended at x={final_pose['x']:.3f}, y={final_pose['y']:.3f}, theta={final_pose['theta']:.3f}. The motion-source counts and RGB-D correspondence counts are included in the report and CSV artifacts.",
        "",
        "The honest limitation is that there is no loop closure or global map optimization, so drift is expected. Also, this run does not contain IMU fields, so the optional gyro fusion path was not exercised in this replay.",
        "",
        "For presentation, this should be framed as a replay-validated localization prototype and code-review feature, not as a live production navigation dependency.",
    ]
    write_text(OUTPUT_DIR / "speaker_notes_slam.md", "\n".join(speaker_lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_dir, selection_note = pick_run()
    if run_dir is None:
        write_text(
            OUTPUT_DIR / "slam_status_report.md",
            "# SLAM / RGB-D odometry status report\n\n- Status: could not generate artifacts\n- Reason: no replayable run directory was found.\n",
        )
        print("No replayable run directory found.")
        return

    slam = None
    rows = []
    motion_counts = Counter()
    rgbd_points_values = []
    final_pose = None
    best_overlay = None
    best_overlay_score = (-1, -1, -1)
    start = time.perf_counter()
    depth_info = depth_sample_info(run_dir)

    for frame_idx, (frame_bgr, depth_map, imu_data, throttle) in enumerate(replay_run(str(run_dir))):
        if slam is None:
            h, w = frame_bgr.shape[:2]
            slam = VisualSlamSystem(width=w, height=h)

        frame_start = time.perf_counter()
        pose = slam.update(frame_bgr, depth_map=depth_map, throttle_val=throttle, imu_data=imu_data)
        proc_ms = (time.perf_counter() - frame_start) * 1000.0

        motion_source = pose.get("motion_source", "unknown")
        tracking_points = int(pose.get("tracking_points", 0))
        rgbd_points = int(pose.get("rgbd_points", 0))
        rows.append(
            {
                "frame_index": frame_idx,
                "motion_source": motion_source,
                "tracking_points": tracking_points,
                "rgbd_points": rgbd_points,
                "x": float(pose["x"]),
                "y": float(pose["y"]),
                "theta": float(pose["theta"]),
                "processing_time_ms": round(proc_ms, 6),
            }
        )
        final_pose = pose
        motion_counts[motion_source] += 1
        if rgbd_points > 0:
            rgbd_points_values.append(rgbd_points)

        overlay_score = (1 if motion_source == "rgbd" else 0, rgbd_points, tracking_points)
        if overlay_score > best_overlay_score:
            best_overlay_score = overlay_score
            best_overlay = (frame_bgr.copy(), dict(pose), frame_idx)

    elapsed = time.perf_counter() - start

    if not rows or final_pose is None:
        write_text(
            OUTPUT_DIR / "slam_status_report.md",
            f"# SLAM / RGB-D odometry status report\n\n- Status: could not generate artifacts\n- Reason: replay run `{run_dir}` produced no frames.\n",
        )
        print(f"Replay run produced no frames: {run_dir}")
        return

    save_metrics_csv(rows)
    save_trajectory_plot(rows)
    if best_overlay is not None:
        save_overlay(*best_overlay)

    create_reports(run_dir, selection_note, rows, motion_counts, rgbd_points_values, elapsed, final_pose, depth_info)

    print(f"Replay command: python3 docs/final_artifacts/slam/generate_slam_artifacts.py")
    print(f"Replay run used: {run_dir.relative_to(REPO_ROOT)}")
    print(f"Frames processed: {len(rows)}")
    print(f"Elapsed seconds: {elapsed:.6f}")
    print(f"Estimated FPS: {len(rows) / elapsed:.6f}")
    print(f"Final pose: x={final_pose['x']:.6f} y={final_pose['y']:.6f} theta={final_pose['theta']:.6f}")
    print(f"Motion source counts: {dict(motion_counts)}")
    if rgbd_points_values:
        print(
            f"RGB-D correspondences: median={int(np.median(rgbd_points_values))} max={max(rgbd_points_values)} frames={len(rgbd_points_values)}"
        )
    else:
        print("RGB-D correspondences: not measured")


if __name__ == "__main__":
    main()
