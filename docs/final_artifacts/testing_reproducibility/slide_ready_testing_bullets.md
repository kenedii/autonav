# Slide-ready testing bullets

- We re-ran the targeted final validation set in the current repo state: AutoNav v2, SLAM core/replay, API, mission, and preprocessing tests all passed.
- Total passing targeted tests in this pass: `25`.
- A syntax-only `py_compile` sweep passed for key data collection, training, inference, client API, SLAM, and host-server modules.
- `tests/test_runtime_split.py` is not green: it reflects an older hardware/runtime split and should be presented as stale, not as a live-demo blocker.
- `tests/test_data_pipeline_metadata.py` is also not green in the current host environment because of import-path and pandas/pyarrow/NumPy compatibility issues.
- Reproducibility is improved by the updated READMEs and exact live-demo command, but external weights and Jetson-specific setup are still required.
- Safe final claim: the core lane-follow path and targeted regression coverage are verified; broader or older suites are not all green and should be described honestly.
