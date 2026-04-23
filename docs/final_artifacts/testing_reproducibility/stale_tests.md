# Stale / failing tests

## 1) `tests/test_runtime_split.py`

- Command:
  - `PYTHONPATH=fleet/fleet_management_app/client_api:data_collection pytest tests/test_runtime_split.py -q`
- Result:
  - `11 failed, 2 passed in 0.24s`
- Why it fails:
  - The suite expects older or unimplemented `hardware.py` APIs such as:
    - `resolve_sensor_role_configs`
    - `empty_sensor_snapshot`
    - `CompositeSensorRig`
    - `build_sensor_rig`
    - `_opencv_cuda_device_count`
    - `_opencv_has_gstreamer_build`
- Interpretation:
  - This is a **stale test suite** against the current runtime layout.
  - It is **not** evidence that the validated live lane-following path is broken.
- Safer presentation wording:
  - “`tests/test_runtime_split.py` reflects an older runtime-split architecture and is not part of the green final validation set.”

## 2) `tests/test_data_pipeline_metadata.py`

- Commands tried:
  - `pytest tests/test_data_pipeline_metadata.py -q`
  - `PYTHONPATH=. pytest tests/test_data_pipeline_metadata.py -q`
- Result:
  - both runs failed during collection before assertions executed
- Why it fails in this host environment:
  - import-path issue:
    - `ModuleNotFoundError: No module named 'jetracer.train.data_frontend.augment_data'`
  - dependency compatibility issue:
    - local pandas / pyarrow import path emits `_ARRAY_API not found` against the current NumPy 2.3.5 host environment
- Interpretation:
  - This suite is currently **environment/path-dependent** and non-green in the reviewed host setup.
  - It should not be counted in the passing final validation set.
- Safer presentation wording:
  - “The data-pipeline metadata suite exists, but it is not green in the current host environment due to path/dependency issues and was not part of the validated final pass set.”

## Presentation-safe summary

- Green final validation set:
  - AutoNav v2 tests
  - SLAM core/replay tests
  - API tests
  - mission tests
  - preprocessing tests
- Non-green / not counted:
  - stale runtime-split suite
  - metadata suite with host-environment import/dependency issues
