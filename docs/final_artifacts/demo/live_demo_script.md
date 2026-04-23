# Live Demo Script

Today we’re demonstrating AutoNav as a deployed technical system, not just a trained model. The live path we trust for the presentation is the lane-following pipeline running on the Jetson Nano with CAM0 as the primary camera, AutoNav-v2-34 as the selected model, TensorRT for optimized inference, and PCA9685 for steering and throttle control.

Before starting the car, we confirm the practical pieces that matter for reproducibility: the model checkpoint is in the expected location, the TensorRT engine exists on the Jetson, CAM0 opens correctly, and the manual override path is ready. That matters because the presentation is about technical readiness and honest deployment, not about pretending the hardware can never fail.

The command we run is `inference/run_autonomous_resnet.py` with the CAM0, PCA9685, and TensorRT options enabled. What you should notice during the live run is that the Jetson is doing real inference from the forward camera and sending live control outputs to the car. If the run is stable, the car should complete at least one controlled lane-following segment without manual correction.

The repo also includes two additional AI features beyond the main lane-follow model. YOLO exists as a prototype advisory detection path in the fleet runtime, but it does not control steering, braking, or throttle in the validated live demo. SLAM exists as an experimental RGB-D odometry and replay feature, but it is not the validated live navigation path for the final presentation either.

If the car behavior is unreliable, we will switch immediately to the fallback evidence package instead of forcing a bad live run. In that case, we will still show the exact deployment command, the code path that runs on Jetson, the testing and reproducibility artifacts, the model evaluation artifacts, the YOLO prototype evidence, and the SLAM replay outputs. That still satisfies the technical review because it shows the working architecture, measured evidence, and the limits of the system honestly.
