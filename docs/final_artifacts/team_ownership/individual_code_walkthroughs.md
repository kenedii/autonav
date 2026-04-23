# Individual Code Walkthroughs

## Nicolas Maitland

### Files to open

- `inference/run_autonomous_resnet.py`
- `inference/trt_optimize.py`
- `data_collection/preprocess_utils.py`
- `fleet/fleet_management_app/client_api/models.py`
- `fleet/fleet_management_app/client_api/slam.py`

### Two-minute script

"I’ll start with the live Jetson path, because that is the most validated part of the project. `run_autonomous_resnet.py` is the final runtime that opens CAM0, applies the same preprocessing profile used in training, runs the AutoNav-v2-34 model, and sends steering and throttle commands through PCA9685. `trt_optimize.py` is the TensorRT step that makes this path usable on the Jetson Nano. `preprocess_utils.py` matters because it keeps the CAM0 fisheye crop and resize consistent between training and deployment. On the prototype side, `models.py` contains the YOLO wrapper, which only produces advisory detections, and `slam.py` contains the experimental visual odometry / RGB-D localization logic used in replay and code review, not in the final live control loop."

### Likely questions and answers

- Why is CAM0 the main live camera?
  - The final model was trained and tuned around the CAM0 domain and preprocessing path, so CAM0 is the most reliable live input.

- What exactly did TensorRT change?
  - TensorRT is the optimized inference path used on the Jetson for the validated live demo command. It improves deployment efficiency without changing model behavior.

- Does YOLO control the car?
  - No. It is advisory only and does not steer, brake, or override the lane-follow model.

- Is the SLAM module production-ready?
  - No. It is best described as experimental visual odometry / RGB-D odometry with replay evidence and known drift limitations.

## Chris Kenedi / `kenedii`

### Files to open

- `model_training/train_model_experiments.py`
- `model_training/train_model_resnet.py`
- `model_training/dataset_loader.py`
- `fleet/fleet_management_app/client_api/main.py`
- `fleet/fleet_management_app/host_app/server.py`
- `fleet/fleet_management_app/host_app/static/app.js`

### Two-minute script

"I’ll cover how the model is trained and how the software around it is exposed as a product. `train_model_experiments.py` is the main experiment-based training path and shows how we compare sensor combinations and split the dataset into train, validation, and test sets. `train_model_resnet.py` is the earlier training path, and `dataset_loader.py` handles how the data is fed into the model. On the product side, `client_api/main.py` exposes the car runtime through API endpoints, while `host_app/server.py` and `host_app/static/app.js` provide the host-side dashboard and deployment support. This is where we can explain model selection, API behavior, and how the dashboard helps operate the system without claiming it is the proof of autonomous driving by itself."

### Likely questions and answers

- Why did you choose ResNet34?
  - It gave a good balance between representation power and Jetson deployment cost, and it matches the final checkpoint selected for the live demo.

- How were train / validation / test splits handled?
  - The experiment trainer uses an explicit train / validation / test split, while the older trainer uses a simpler train / test split.

- What does the dashboard actually do?
  - It supports deployment, status, configuration, and monitoring. It is useful operational tooling, not the main evidence that YOLO or SLAM is production-ready.

- What should we say about the mixed Git history?
  - We should describe ownership by subsystem familiarity rather than claiming that one person wrote every line in a shared module.
