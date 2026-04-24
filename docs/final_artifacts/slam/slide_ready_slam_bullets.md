# Slide-ready SLAM bullets

- AutoNav includes an experimental RGB-D odometry / SLAM helper in `fleet/fleet_management_app/client_api/slam.py`.
- Preferred replay run was unavailable in this workspace, so the artifacts use `run_20260401_222322`, the strongest available depth-backed archived run.
- Replay produced a final pose of `x=-0.291, y=4.007, theta=2.087` across `3120` processed frames.
- Motion sources observed: `{'bootstrap': 12, 'rgb': 3097, 'reseed': 11}`; RGB-D correspondence median/max: `not measured` / `not measured`.
- Archived depth sample in this run was `depth_00000.png: shape=(240, 424, 3) dtype=uint8`.
- Replay throughput on this host was `391.27 FPS`.
- Final presentation wording should be: experimental RGB-D odometry with replay evidence. In this archived run, depth files are preview-style, so motion fell back to RGB-only updates.
