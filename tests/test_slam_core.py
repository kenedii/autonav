import math
import os
import sys

import numpy as np
import pytest


CLIENT_API_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "fleet",
    "fleet_management_app",
    "client_api",
)
if CLIENT_API_DIR not in sys.path:
    sys.path.append(CLIENT_API_DIR)

from slam import VisualSlamSystem


def test_metric_motion_recovers_camera_yaw_and_translation():
    slam = VisualSlamSystem(width=424, height=240, focal_length=320.0)
    rng = np.random.default_rng(7)
    prev_xyz = np.column_stack([
        rng.uniform(-0.5, 0.5, size=48),
        rng.uniform(-0.2, 0.2, size=48),
        rng.uniform(1.0, 3.0, size=48),
    ]).astype(np.float32)

    camera_yaw = math.radians(6.0)
    camera_translation = np.array([0.04, 0.0, 0.18], dtype=np.float32)
    point_rotation = slam._rotation_y(camera_yaw).T
    point_translation = -(point_rotation @ camera_translation)
    curr_xyz = prev_xyz @ point_rotation.T + point_translation

    motion = slam._estimate_metric_motion_from_point_clouds(prev_xyz, curr_xyz)

    assert motion is not None
    assert motion["source"] == "rgbd"
    assert motion["point_count"] == len(prev_xyz)
    assert motion["lateral_m"] == pytest.approx(float(camera_translation[0]), abs=1e-3)
    assert motion["forward_m"] == pytest.approx(float(camera_translation[2]), abs=1e-3)
    assert motion["yaw_rad"] == pytest.approx(camera_yaw, abs=1e-3)


def test_rgbd_correspondences_scale_resized_rgb_points_to_depth_map():
    slam = VisualSlamSystem(width=160, height=120, focal_length=100.0)
    prev_depth = np.full((240, 424), 1.0, dtype=np.float32)
    curr_depth = np.full((240, 424), 1.0, dtype=np.float32)
    points = np.array([
        [40.0, 30.0],
        [80.0, 60.0],
        [120.0, 90.0],
        [32.0, 60.0],
        [96.0, 60.0],
        [80.0, 24.0],
    ], dtype=np.float32)

    prev_xyz, curr_xyz = slam._build_rgbd_correspondences(
        points,
        points.copy(),
        prev_depth,
        curr_depth,
        (120, 160),
    )

    assert prev_xyz.shape == (6, 3)
    assert np.allclose(prev_xyz, curr_xyz)
    assert np.allclose(prev_xyz[:, 2], 1.0)
    assert prev_xyz[1, 0] == pytest.approx(0.0, abs=1e-6)
    assert prev_xyz[1, 1] == pytest.approx(0.0, abs=1e-6)
    assert prev_xyz[0, 0] == pytest.approx(-0.4, abs=1e-2)
    assert prev_xyz[0, 1] == pytest.approx(-0.3, abs=1e-2)


def test_colorized_preview_depth_is_not_treated_as_metric_depth():
    depth_preview = np.zeros((240, 424, 3), dtype=np.uint8)
    assert VisualSlamSystem._normalize_depth_map(depth_preview) is None

