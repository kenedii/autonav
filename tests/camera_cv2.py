# Functions to easily use the camera in OpenCV. Pass the OpenCV integer camera index as input for the 'camera' parameter.
# These functions only support basic camera features and do not make use of the depth sensors.

import cv2
import os

def capture_and_save_image(save_path, camera):
    """
    Captures an image from the default camera and saves it to the specified path.
    
    Args:
        save_path (str): Full path where the image should be saved (e.g., 'images/photo.jpg')
    	camera (int): Index for openCV camera.
    Returns:
        bool: True if image was saved successfully, False otherwise
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
    
    # Initialize camera (0 is usually the default camera)
    cap = cv2.VideoCapture(camera)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return False
    
    # Let camera warm up
    import time
    time.sleep(1)
    
    # Capture frame
    ret, frame = cap.read()
    
    # Release the camera
    cap.release()
    
    if ret:
        success = cv2.imwrite(save_path, frame)
        if success:
            print(f"Image saved successfully to {save_path}")
            return True
        else:
            print("Error: Failed to save image.")
            return False
    else:
        print("Error: Failed to capture image.")
        return False

def capture_image(camera):
    """
    Captures an image from the default camera and returns it.
    Args:
	camera: Index for OpenCV camera
    Returns:
        numpy.ndarray or None: The captured image as a NumPy array, or None if failed
    """
    cap = cv2.VideoCapture(camera)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    
    time.sleep(1)  # Warm up camera
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return frame
    else:
        print("Error: Failed to capture image.")
        return None
