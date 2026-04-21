import csv
import os
import sys

import cv2
import numpy as np


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if TESTS_DIR not in sys.path:
    sys.path.append(TESTS_DIR)

from test_slam import replay_run


def test_replay_prefers_aligned_realsense_rgb_when_depth_exists(tmp_path):
    run_dir = tmp_path / "run_demo"
    run_dir.mkdir()

    rgb_path = run_dir / "rgb_00000.png"
    rs_rgb_path = run_dir / "rs_rgb_00000.png"
    depth_path = run_dir / "depth_00000.png"

    canonical = np.full((120, 160, 3), (5, 15, 25), dtype=np.uint8)
    aligned = np.full((240, 424, 3), (55, 105, 155), dtype=np.uint8)
    depth = np.full((240, 424), 1234, dtype=np.uint16)
    cv2.imwrite(str(rgb_path), canonical)
    cv2.imwrite(str(rs_rgb_path), aligned)
    cv2.imwrite(str(depth_path), depth)

    with open(run_dir / "dataset.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rgb_path",
                "realsense_rgb_path",
                "depth_path",
                "throttle_norm",
                "accel_x",
                "accel_y",
                "accel_z",
                "gyro_x",
                "gyro_y",
                "gyro_z",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "rgb_path": str(rgb_path),
                "realsense_rgb_path": str(rs_rgb_path),
                "depth_path": str(depth_path),
                "throttle_norm": "0.25",
                "accel_x": "0.1",
                "accel_y": "0.2",
                "accel_z": "0.3",
                "gyro_x": "1.1",
                "gyro_y": "1.2",
                "gyro_z": "1.3",
            }
        )

    frame_bgr, depth_map, imu_data, throttle = next(replay_run(str(run_dir)))

    assert frame_bgr.shape == aligned.shape
    assert np.array_equal(frame_bgr, aligned)
    assert depth_map.shape == depth.shape
    assert depth_map.dtype == np.uint16
    assert imu_data == {"accel": [0.1, 0.2, 0.3], "gyro": [1.1, 1.2, 1.3]}
    assert throttle == 0.25

