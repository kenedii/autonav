# Speaker notes: deployment and inference benchmark

This is the validated live path we used for the Jetson presentation setup: CAM0 as the driving camera, the AutoNav-v2-34 checkpoint, TensorRT acceleration, and PCA9685 motor control.

In this documentation pass, I did not rerun the live benchmark because the current environment is not the Jetson Nano itself. That means I can document the exact command and the required benchmark procedure, but I should not claim fresh FPS or inference timing numbers from this host.

What I can say honestly is that the checkpoint is present, the validated command is documented, and the remaining benchmark evidence should be captured directly on the Jetson using `--debug-timings` and `tegrastats` during a 30 to 60 second run.

When presenting this slide, keep the emphasis on the exact deployment path and the safety procedure: manual override stays active, one person watches the vehicle, and one person watches the terminal or dashboard.
