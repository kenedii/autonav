# realsense_full.py
import pyrealsense2 as rs
import numpy as np
import cv2

# Global objects - created once
pipeline = None
align = None

def start_pipeline():
    global pipeline, align
    if pipeline is None:
        pipeline = rs.pipeline()
        config = rs.config()

        # Enable the three streams we need
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
        # Note: '1' is the index for the left IR camera on most D400 series
        config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

        pipeline.start(config)

        # FIX: Align depth and IR to the color image by passing rs.stream.color
        align = rs.align(rs.stream.color)
        print("[RealSense] Pipeline started - RGB + IR + Depth ready")


# --------------------- RGB ---------------------
def get_rgb_image():
    start_pipeline()
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    color_frame = aligned.get_color_frame()
    if not color_frame:
        return None
    return np.asanyarray(color_frame.get_data())  # RGB, shape (720,1280,3)


def save_rgb_image(filename):
    img = get_rgb_image()
    if img is not None:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, bgr)
        print("[SAVED] RGB -> " + filename)
        return True
    return False


# --------------------- IR (dots) ---------------------
def get_ir_image():
    start_pipeline()
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    # The left IR stream index is 1, as enabled in config.
    ir_frame = aligned.get_infrared_frame(1)
    if not ir_frame:
        return None
    ir = np.asanyarray(ir_frame.get_data())
    # Convert single-channel IR (grayscale) to 3-channel for consistent model input/saving
    return cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)


def save_ir_image(filename):
    img = get_ir_image()
    if img is not None:
        cv2.imwrite(filename, img)
        print("[SAVED] IR dots -> " + filename)
        return True
    return False


# --------------------- DEPTH ---------------------
def get_depth_image():
    start_pipeline()
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    depth_frame = aligned.get_depth_frame()
    if not depth_frame:
        return None
    return np.asanyarray(depth_frame.get_data())  # uint16 in millimeters


def save_depth_image(filename, colored=True):
    depth = get_depth_image()
    if depth is None:
        return False

    # Save raw 16-bit depth (perfect for later use)
    raw_name = filename.replace(".png", "_raw.png")
    cv2.imwrite(raw_name, depth)
    print("[SAVED] Raw depth -> " + raw_name)

    if colored:
        # Scale depth for visualization (e.g., scale 5 meters (5000mm) to 255)
        vis = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=255/5000), cv2.COLORMAP_JET)
        color_name = filename.replace(".png", "_color.jpg")
        cv2.imwrite(color_name, vis)
        print("[SAVED] Depth color -> " + color_name)

    return True


# --------------------- Distance helper ---------------------
def get_center_distance():
    depth = get_depth_image()
    if depth is None:
        return 0
    h, w = depth.shape
    dist_mm = depth[h//2, w//2]
    
    # Depth value of 0 means no valid depth detected (e.g., outside range or reflective)
    if dist_mm == 0:
        return 0
    return dist_mm / 1000.0  # return meters


# --------------------- Quick test ---------------------
if __name__ == "__main__":
    print("--- Starting Camera Test ---")
    save_rgb_image("test_rgb.jpg")
    save_ir_image("test_ir.jpg")
    save_depth_image("test_depth.png", colored=True)
    
    dist = get_center_distance()
    if dist > 0:
        print("\nDistance in front of camera: **%.3f meters**" % dist)
    else:
        print("\nNo valid depth at center (Is the object too far, too close, or reflective?)")
    
    # Stop the pipeline explicitly
    if pipeline:
        pipeline.stop()
        print("[RealSense] Pipeline stopped.")
