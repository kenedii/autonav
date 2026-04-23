# Demo Risk Table

| Symptom | Likely cause | Mitigation | Fallback |
|---|---|---|---|
| CAM0 does not open | camera path busy, cable issue, wrong backend, Jetson camera stack problem | restart the process, verify no other process owns the camera, re-seat or recheck camera before retrying once | switch to Plan B with recorded evidence and code walkthrough |
| TensorRT engine missing or load fails | `inference/best_model_trt.pth` absent or stale on the Jetson | verify file exists before the demo; do not attempt long recovery on stage | switch to Plan B or Plan C and explain that the validated command expects a prepared engine |
| PCA9685/control path does not respond | controller board not initialized, I2C issue, power issue | confirm control path before autonomy and keep manual override ready | stop live run and move to Plan B |
| Car does not follow lane reliably | lighting drift, low battery, poor start position, track condition, model generalization limit | reset starting position once, keep throttle conservative, stop quickly if behavior is unstable | switch to Plan B with rehearsal clip or current artifact bundle |
| Dashboard does not connect cleanly | host/client mismatch, network issue, API not started | do not burn demo time debugging the dashboard | continue with direct CLI lane-follow demo or Plan B |
| YOLO questions come up during demo | audience assumes it is part of live control | state clearly that YOLO is advisory only and prototype-only | show `docs/final_artifacts/yolo/yolo_status_report.md` instead of attempting a live YOLO demo |
| SLAM questions come up during demo | audience assumes live navigation is being used | state clearly that SLAM is replay/code-review evidence, not the live lane-follow path | show `docs/final_artifacts/slam/slam_replay_summary.png` and `slam_pose_overlay.png` |
| Manual override is needed | unexpected control behavior or unsafe motion | hand control to the assigned team member immediately and stop the autonomous run | transition to Plan B without framing the interruption as a success |
