# Slide-ready data bullets

- AutoNav training data was recorded from real RC-car runs stored under `jetracer/train/runs_rgb_depth/run_*`.
- The archived training snapshot contains `9536` labeled rows across `17` run folders, with `16` non-empty runs.
- Primary front-camera coverage is strong: `rgb_path` is present on `9535` rows (100.0%).
- Secondary modalities are partial: `cam1_path` `9035`, `ir_path` `9035`, `depth_path` `9035`.
- Steering labels are center-heavy: `6280` center rows vs `1033` left and `2222` right.
- The main experiment trainer uses a `70 / 15 / 15` split; the legacy trainer still contains an `80 / 20` split path.
- The archived training snapshot does not include the newer `realsense_rgb_path` or IMU fields, so those newer recorder capabilities should be framed as forward-looking rather than core training inputs for this dataset.
