# Speaker notes: SLAM / RGB-D odometry

This feature is best described as an RGB-D visual odometry prototype rather than full production SLAM. The code tracks image features, samples metric depth when available, fits frame-to-frame rigid motion, and integrates that into a 2D pose estimate.

For the final evidence bundle, I replayed `run_20260401_222322` because the originally preferred run was not available in this workspace. That archived run has strong depth coverage, so it is the best available replay candidate here.

One important detail is that the archived depth sample we inspected was `depth_00000.png: shape=(240, 424, 3) dtype=uint8`. That means the replayed depth files are preview-style images, not raw metric depth, so the RGB-D motion path did not activate and the run mostly used RGB visual motion instead.

On that replay, the system processed 3120 frames and ended at x=-0.291, y=4.007, theta=2.087. The motion-source counts and RGB-D correspondence counts are included in the report and CSV artifacts.

The honest limitation is that there is no loop closure or global map optimization, so drift is expected. Also, this run does not contain IMU fields, so the optional gyro fusion path was not exercised in this replay.

For presentation, this should be framed as a replay-validated localization prototype and code-review feature, not as a live production navigation dependency.
