# Slide-ready deployment bullets

- Validated live deployment path: CAM0 + AutoNav-v2-34 + TensorRT + PCA9685 on Jetson Nano.
- This repo snapshot contains the main PyTorch checkpoint, but the local TensorRT engine file is not present here.
- Deployment benchmark numbers were not re-measured in this pass because this environment is not the Jetson target.
- The exact Jetson benchmark command is already documented and should be run with `--debug-timings` plus `tegrastats` capture.
- Safety requirement: keep manual override active and assign one team member to rescue control during the live run.
- For the final deck, present measured Jetson numbers only after they are captured on the actual hardware.
