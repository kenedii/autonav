# Testing summary

This summary captures the exact commands run during the final testing / reproducibility evidence pass.

| Command | Result | Pass / fail | What it proves |
|---|---|---|---|
| `pytest fleet/fleet_management_app/client_api/test_autonav_v2.py -q` | `5 passed in 2.00s` | pass | AutoNav v2 checkpoint-shape inference, default throttle/steering behavior, and control-output helpers are stable. |
| `pytest tests/test_slam_core.py tests/test_slam_replay.py -q` | `4 passed in 0.24s` | pass | SLAM core RGB-D math and replay-path selection logic are stable. |
| `PYTHONPATH=fleet/fleet_management_app/client_api:fleet/fleet_management_app/host_app pytest tests/test_server.py tests/test_client.py -q` | `8 passed in 0.64s` | pass | Fleet client/host API routes and proxy behavior are stable in the reviewed repo state. |
| `PYTHONPATH=fleet/fleet_management_app/client_api pytest tests/test_mission.py -q` | `7 passed in 0.01s` | pass | Mission/depth-stop related logic helpers still pass targeted unit coverage. |
| `PYTHONPATH=data_collection pytest tests/test_preprocess_utils.py -q` | `1 passed in 0.19s` | pass | Preprocess-profile logic used by CAM0 data/runtime flow is stable. |
| `pytest tests/test_data_pipeline_metadata.py -q` | collection error | fail | This suite is **not** green in the current host environment. It fails during import/collection before assertions run. |
| `PYTHONPATH=. pytest tests/test_data_pipeline_metadata.py -q` | same collection error | fail | Confirms the metadata-suite problem is not solved by a simple `PYTHONPATH=.` adjustment. |
| `PYTHONPATH=fleet/fleet_management_app/client_api:data_collection pytest tests/test_runtime_split.py -q` | `11 failed, 2 passed in 0.24s` | fail | This suite is stale against the current `hardware.py` / runtime layout and should not be presented as part of the green validation set. |
| `python3 -m py_compile data_collection/record_data3.py data_collection/realsense_full.py data_collection/preprocess_utils.py model_training/train_model_experiments.py model_training/train_model_resnet.py model_training/dataset_loader.py inference/run_autonomous_resnet.py inference/trt_optimize.py fleet/fleet_management_app/client_api/car.py fleet/fleet_management_app/client_api/hardware.py fleet/fleet_management_app/client_api/main.py fleet/fleet_management_app/client_api/mission.py fleet/fleet_management_app/client_api/models.py fleet/fleet_management_app/client_api/slam.py fleet/fleet_management_app/host_app/server.py` | no output | pass | Key data-collection, training, inference, client API, SLAM, and host-server modules are syntactically valid in the current Python environment. |

## Net result

- Passing targeted validation set: `25` tests
- Stale runtime-split suite: `11 failed, 2 passed`
- Metadata-suite status: collection/import failure in this host environment

## Honest presentation framing

- Present the five passing targeted suites as the current validated regression set.
- Present `tests/test_runtime_split.py` as stale against the current runtime architecture, not as a live-demo blocker.
- Present `tests/test_data_pipeline_metadata.py` as environment/path-dependent and currently non-green in this Python 3.11 host environment.
