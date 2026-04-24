# Speaker notes: data preparation and EDA

We collected this dataset by manually driving the RC car and saving synchronized steering, throttle, and sensor outputs into per-run folders.

In the archived training snapshot we inspected, there are 17 run folders and 9536 labeled rows. The front driving camera is the most complete modality, while the back camera, IR, and depth streams appear on only part of the dataset.

The steering labels are center-heavy, with 6280 center rows compared to 1033 left and 2222 right. That matters because it means straight driving is better represented than aggressive recovery behavior at the tape edges.

For preprocessing, the live CAM0 path uses the same `cam0_fisheye_v1` crop-and-resize profile that was applied to the training inputs. That consistency is important because the model is learning from a specific image geometry, not from arbitrary raw frames.

One honest limitation is that this archived training snapshot does not include the newer RealSense RGB sidecar path or IMU columns, even though the recorder now supports them. For the final presentation, we should describe those as newer capabilities rather than implying they were part of the original training set used for the validated live demo.
