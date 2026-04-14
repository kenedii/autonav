# camera_utils.py
# Pure OpenCV + V4L2 - RGB + IR + Depth on RealSense
import cv2
import numpy as np

# ==============================================================
# Basic RGB / IR capture
# ==============================================================
def capture_and_save(camera_index, filename, width=1280, height=720, fps=None):
    img = _capture_raw(camera_index, width, height, fps)
    if img is None:
        return False
    success = cv2.imwrite(filename, img)
    if success:
        print("[SAVED] " + filename + " (%dx%d)" % (img.shape[1], img.shape[0]))
    else:
        print("[ERROR] Could not save " + filename)
    return success


def capture_image(camera_index, width=1280, height=720, fps=None):
    return _capture_raw(camera_index, width, height, fps)


def _capture_raw(camera_index, width=1280, height=720, fps=None):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera index " + str(camera_index))
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps is not None:
        cap.set(cv2.CAP_PROP_FPS, fps)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("[ERROR] Failed to grab frame from index " + str(camera_index))
        return None
    return frame


# ==============================================================
# Depth-specific functions (index usually 6 or 8)
# ==============================================================
def capture_and_save_depth(camera_index, filename_raw="depth_raw.png", filename_color="depth_color.jpg"):
    depth = _capture_raw(camera_index, 1280, 720)
    if depth is None:
        return False
    cv2.imwrite(filename_raw, depth)
    print("[SAVED] Raw depth (mm) -> " + filename_raw)
    depth_8bit = cv2.convertScaleAbs(depth, alpha=(255.0/5000.0))
    depth_color = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
    cv2.imwrite(filename_color, depth_color)
    print("[SAVED] Depth visualization -> " + filename_color)
    return True


def get_depth_image(camera_index, width=1280, height=720):
    return _capture_raw(camera_index, width, height)


def get_distance_mm(camera_index, x, y):
    depth = get_depth_image(camera_index)
    if depth is None or y >= depth.shape[0] or x >= depth.shape[1]:
        return 0
    return int(depth[y, x])


def get_distance_meters(camera_index, x=None, y=None):
    depth = get_depth_image(camera_index)
    if depth is None:
        return 0.0
    h, w = depth.shape
    if x is None:
        x = w // 2
    if y is None:
        y = h // 2
    mm = depth[y, x]
    if mm == 0:
        return 0.0
    return round(mm / 1000.0, 3)


# ==============================================================
# Quick test
# ==============================================================
if __name__ == "__main__":
    RGB_INDEX   = 2      # colour camera
    IR_INDEX    = 1      # left IR with dots
    DEPTH_INDEX = 6      # usually 6 or 8 - check with: ls /dev/video*

    print("=== Testing RGB ===")
    capture_and_save(RGB_INDEX, "test_rgb.jpg")

    print("=== Testing IR ===")
    capture_and_save(IR_INDEX, "test_ir.jpg")

    print("=== Testing Depth ===")
    capture_and_save_depth(DEPTH_INDEX)

    dist = get_distance_meters(DEPTH_INDEX)
    print("Distance in front of camera: " + str(dist) + " meters")

