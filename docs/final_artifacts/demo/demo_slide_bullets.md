# Demo Slide Bullets

- Plan A is a live lane-following demo on the validated Jetson path: CAM0 + AutoNav-v2-34 + TensorRT + PCA9685.
- The exact live command is documented and should be run only after camera, model, TensorRT, and manual-override checks pass.
- Plan B is a hybrid fallback: keep the repo and deployment path live, but show recorded lane-follow evidence plus YOLO and SLAM artifacts.
- Plan C is a code-first fallback using the final artifact package, test evidence, and subsystem walkthroughs.
- YOLO is prototype advisory detection only and is not part of the trusted live control loop.
- SLAM is experimental replay-oriented odometry / localization evidence and is not the validated live navigation path.
- The final presentation should prioritize safe, honest demonstration over forcing a failing hardware run.
