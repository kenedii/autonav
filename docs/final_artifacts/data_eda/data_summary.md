# AutoNav Data Preparation and EDA Summary

## Dataset scope
- Run folders found: `17`
- Dataset CSV files found: `17`
- Total raw rows across all `dataset.csv` files: `9536`
- Non-empty runs: `16`
- Empty runs: `1` (run_20260401_234117)

## Sensors present in the archived training snapshot
- CAM0 / rgb_path, CAM1 / cam1_path, IR, depth_path, depth_front scalar

## Missing-data counts per sensor column

| Column | Present rows | Missing / empty rows | Coverage |
|---|---:|---:|---:|
| `rgb_path` | 9535 | 1 | 100.0% |
| `cam0_path` | 9535 | 1 | 100.0% |
| `cam1_path` | 9035 | 501 | 94.7% |
| `ir_path` | 9035 | 501 | 94.7% |
| `depth_path` | 9035 | 501 | 94.7% |
| `realsense_rgb_path` | 0 | 9536 | 0.0% |
| `depth_front` | 9535 | 1 | 100.0% |
| `accel_x` | 0 | 9536 | 0.0% |
| `accel_y` | 0 | 9536 | 0.0% |
| `accel_z` | 0 | 9536 | 0.0% |
| `accel_ts_ms` | 0 | 9536 | 0.0% |
| `gyro_x` | 0 | 9536 | 0.0% |
| `gyro_y` | 0 | 9536 | 0.0% |
| `gyro_z` | 0 | 9536 | 0.0% |
| `gyro_ts_ms` | 0 | 9536 | 0.0% |

## Steering distribution
- Count: `9535`
- Min / max: `-1.0000` / `1.0000`
- Mean / median / stdev: `0.1251` / `0.0000` / `0.5646`
- Left / center / right bins using thresholds `< -0.15`, `-0.15 to 0.15`, `> 0.15`: `1033` / `6280` / `2222`

## Throttle distribution
- Count: `9535`
- Min / max: `-0.3000` / `0.3000`
- Mean / median / stdev: `0.1170` / `0.2720` / `0.2118`
- Reverse / zero / forward rows: `1290` / `3152` / `5093`

## Train / validation / test split
- `model_training/train_model_experiments.py` uses a `70 / 15 / 15` split.
- `model_training/train_model_resnet.py` uses a legacy `80 / 20` train/test split.

## Known dataset limitations
- 1 run folder(s) contain an empty dataset.csv and contribute no labeled rows.
- The archived training runs are center-heavy, so recovery and edge-case behavior are underrepresented.
- CAM1 / IR / depth are only present on a subset of rows, while the primary CAM0 / rgb_path is available almost everywhere.
- No checked-in combined dataset CSVs were found, so train/validation/test split information comes from training code rather than a committed final dataset artifact.
- The current archived training snapshot does not include realsense_rgb_path images, even though the newer recorder supports them.
- The current archived training snapshot does not include IMU fields, even though the newer recorder supports accel/gyro logging.

## Additional notes
- Representative preprocessing profile used for the sample figures: `cam0_fisheye_v1`.
- Combined dataset CSVs found in repo: `0`.
- No `combined*.csv` artifacts are checked into this repo snapshot.
