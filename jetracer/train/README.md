record_data.py:
After xbox controller config is complete and configuration is complete in this script, This script allows the user to drive the car around the track using the XBOX controller.
It will save the flattened RGB, IR (2 channels), and depth data (entire depth pixel map) as independent variables, and the steering and acceleration values are the target variables of the dataset. This data can be used to predict steering and acceleration values when given an image of what the car sees.

record_data2.py:
The first script was causing the XBOX Controller's inputs to be sent to the car with a 4-5 second delay due to the large amount of data being processed, even with only 2 FPS. This script only collects the RGB image along with the depth in front of camera float value in centimetres to reduce the amount of data being processed by the Jetson every second. Using this script will considerably reduce the input delay between the steering/acceleration controls from the XBOX Controller and the RC Car while collecting training data driving the car.

record_data3.py: Utilizes the net_controller_client.py to read the controller via pygame when plugged into another computer and sends normalized axes over UDP to the Jetson Nano. This allows wireless recording of training data using the RC car.

net_controller_client.py: Install Pygame `pip install pygame` and run this script on the computer which you want to have the XBOX Controller plugged into to control the RC Car. Reads the controller via pygame and sends normalized axes over UDP to the Jetson Nano. Jetson IP and port must be configured in script or environment variables.

realsense_full.py: Library with functions that can be used by the record scripts to interact with the Intel Realsense camera to obtain RGB image/IR image/depth data.

## Camera Support

The data recording scripts (`record_data.py`, `record_data2.py`, `record_data3.py`) now support both Intel RealSense cameras and standard USB webcams (OpenCV).

**Arguments:**

- `--camera`: Choose the camera type. Options: `realsense` (default), `opencv`, `other`.
- `--device`: Specify the device ID for the camera when using OpenCV mode (default is `0`).
- `--view_360`: Also record two Jetson CSI cameras (CAM0/CAM1) and append `cam0_path` / `cam1_path` to each `dataset.csv` row.
- `--view_360_cam0_sensor_id`, `--view_360_cam1_sensor_id`: Select the Jetson CSI `sensor-id` values for the 360 cameras.
- `--view_360_width`, `--view_360_height`, `--view_360_fps`: Configure the CSI capture resolution and FPS.
- `--view_360_save_width`, `--view_360_save_height`: Configure the saved PNG size for the 360 cameras. Defaults are `320x240`.
- `--view_360_cam0_flip_method`, `--view_360_cam1_flip_method`: Set Jetson `nvvidconv` flip methods per camera when mounting orientation needs adjustment. Defaults are `2` so the 360 cameras are rotated 180 degrees.

**Examples:**

```bash
# Default (RealSense)
python record_data.py

# Use a specific USB webcam (OpenCV)
python record_data.py --camera opencv --device 0

# Record CAM0/CAM1 alongside the main recorder stream
python record_data3.py --view_360
```

Note: When using `opencv` mode with `record_data.py`, dummy data (pixels with value 0) is generated for depth and infrared channels to maintain dataset compatibility. In `record_data2.py` and `record_data3.py`, depth will be recorded as 0.0.
