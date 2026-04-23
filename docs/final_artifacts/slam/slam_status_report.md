# SLAM / RGB-D odometry status report

## Algorithm summary
- Sparse optical flow between consecutive RGB frames using `cv2.calcOpticalFlowPyrLK`.
- Metric RGB-D correspondences built from tracked 2D points plus depth sampling.
- Rigid transform fit via SVD to estimate camera motion when metric depth is available.
- Fallback to RGB-only visual motion when metric RGB-D motion cannot be recovered.
- Optional gyro fusion exists in code, but was not exercised in this replay because IMU fields were absent.

## Sensor inputs
- Required: RGB frame
- Optional: depth map
- Optional: IMU gyro / accel

## Output state
- `x`, `y`, `theta`
- short `trajectory` history
- `tracking_points`
- `rgbd_points`
- `motion_source`
- `last_motion`

## Replay run used
- Run directory: `jetracer/train/runs_rgb_depth/run_20260401_222322`
- Selection note: preferred run missing; selected best available depth-backed run

## Replay result
- Frames processed: `3120`
- Final pose: `x=-0.291 y=4.007 theta=2.087`
- Motion source counts: `{'bootstrap': 12, 'rgb': 3097, 'reseed': 11}`
- RGB-D correspondences: `median=not measured max=not measured frames=0`
- Update rate: `391.27 FPS`
- Depth sample inspected: `depth_00000.png: shape=(240, 424, 3) dtype=uint8`

## Drift / limitations
- This module is experimental RGB-D odometry, not full production SLAM.
- There is no loop closure or global map optimization, so drift is expected.
- This replay run does not include IMU fields, so gyro fusion was not exercised here.
- SLAM state can feed navigation hooks in the fleet runtime, but it is not the validated live lane-follow control path.
- Motion source counts can include `reseed` / `bootstrap`, so trajectory continuity depends on feature tracking quality.
- The archived depth files in this run are preview-style images (`depth_00000.png: shape=(240, 424, 3) dtype=uint8`), so the metric RGB-D path rejected them and replay fell back to RGB-only motion.

## Live-control status
- Replay and code-review feature: yes
- Validated live lane-follow dependency: no
- Full production SLAM claim: no
