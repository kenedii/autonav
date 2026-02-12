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

**Examples:**

```bash
# Default (RealSense)
python record_data.py

# Use a specific USB webcam (OpenCV)
python record_data.py --camera opencv --device 0
```

Note: When using `opencv` mode with `record_data.py`, dummy data (pixels with value 0) is generated for depth and infrared channels to maintain dataset compatibility. In `record_data2.py` and `record_data3.py`, depth will be recorded as 0.0.
