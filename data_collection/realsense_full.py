# realsense_full.py
try:
    import pyrealsense2 as rs
except ImportError:
    rs = None
import numpy as np
import cv2
import time
import threading

try:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
    Gst.init(None)
    GST_AVAILABLE = True
except Exception:
    Gst = None
    GST_AVAILABLE = False

# Configuration
CAMERA_TYPE = "realsense"  # or "opencv" / "csi"
OPENCV_DEVICE_ID = 0
CSI_SENSOR_ID = 0
CSI_WIDTH = 640
CSI_HEIGHT = 480
CSI_FPS = 15
CSI_FLIP_METHOD = 2
CSI_BACKEND = "auto"
REALSENSE_ENABLE_DEPTH = True
REALSENSE_ENABLE_IR = True
REALSENSE_ENABLE_DEPTH_PREVIEW = True

# Global objects - created once
pipeline = None
align = None
cap = None # for opencv
gst_pipeline = None
gst_sink = None
csi_backend = None
# Store RGB, center depth, IR image and a small colorized depth map.
# `rgb_seq` increments on each new frame so consumers can skip duplicate work.
latest_frames = {
    "rgb": None,
    "rgb_seq": 0,
    "depth_center": 0.0,
    "ir": None,
    "depth_map": None,
}

frame_lock = threading.Lock()
stop_event = threading.Event()

def set_camera_type(
    type_name,
    device_id=0,
    sensor_id=0,
    width=640,
    height=480,
    fps=15,
    flip_method=2,
    backend="auto",
    enable_depth=True,
    enable_ir=True,
    enable_depth_preview=True,
):
    global CAMERA_TYPE, OPENCV_DEVICE_ID, CSI_SENSOR_ID, CSI_WIDTH, CSI_HEIGHT, CSI_FPS, CSI_FLIP_METHOD, CSI_BACKEND
    global REALSENSE_ENABLE_DEPTH, REALSENSE_ENABLE_IR, REALSENSE_ENABLE_DEPTH_PREVIEW
    CAMERA_TYPE = type_name
    OPENCV_DEVICE_ID = device_id
    CSI_SENSOR_ID = sensor_id
    CSI_WIDTH = width
    CSI_HEIGHT = height
    CSI_FPS = fps
    CSI_FLIP_METHOD = flip_method
    CSI_BACKEND = backend
    REALSENSE_ENABLE_DEPTH = bool(enable_depth)
    REALSENSE_ENABLE_IR = bool(enable_ir)
    REALSENSE_ENABLE_DEPTH_PREVIEW = bool(enable_depth_preview) and REALSENSE_ENABLE_DEPTH
    with frame_lock:
        latest_frames["rgb"] = None
        latest_frames["rgb_seq"] = 0
        latest_frames["depth_center"] = 0.0
        latest_frames["ir"] = None
        latest_frames["depth_map"] = None


def build_csi_pipeline(sensor_id, width, height, fps, flip_method, appsink_name=None):
    appsink = "appsink drop=true max-buffers=1 sync=false"
    if appsink_name is not None:
        appsink = (
            f"appsink name={appsink_name} emit-signals=false "
            "drop=true max-buffers=1 sync=false"
        )

    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        "video/x-raw(memory:NVMM), "
        f"width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, format=(string)BGRx ! "
        "videoconvert ! "
        f"video/x-raw, format=(string)RGB ! "
        f"{appsink}"
    )


def _sample_to_array(sample):
    caps = sample.get_caps()
    if caps is None or caps.get_size() == 0:
        return None

    structure = caps.get_structure(0)
    width = structure.get_value("width")
    height = structure.get_value("height")
    pixel_format = structure.get_value("format")

    buffer = sample.get_buffer()
    if buffer is None:
        return None

    ok, map_info = buffer.map(Gst.MapFlags.READ)
    if not ok:
        return None

    try:
        if pixel_format in ("BGR", "RGB"):
            channels = 3
        elif pixel_format in ("BGRx", "RGBx"):
            channels = 4
        else:
            channels = None
        if channels is None:
            return None

        frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape((height, width, channels))
        if channels == 4:
            frame = frame[:, :, :3]
        return frame.copy()
    finally:
        buffer.unmap(map_info)


def _pull_gstreamer_sample():
    if gst_sink is None:
        return None

    sample = gst_sink.emit("try-pull-sample", 200000000)
    if sample is None:
        return None

    return _sample_to_array(sample)


def _start_native_csi_pipeline():
    global gst_pipeline, gst_sink, csi_backend
    pipeline_desc = build_csi_pipeline(
        CSI_SENSOR_ID,
        CSI_WIDTH,
        CSI_HEIGHT,
        CSI_FPS,
        CSI_FLIP_METHOD,
        appsink_name="capture_sink",
    )
    gst_pipeline = Gst.parse_launch(pipeline_desc)
    gst_sink = gst_pipeline.get_by_name("capture_sink")
    if gst_sink is None:
        raise RuntimeError("appsink not found in GStreamer pipeline")

    state_ret = gst_pipeline.set_state(Gst.State.PLAYING)
    if state_ret == Gst.StateChangeReturn.FAILURE:
        raise RuntimeError("pipeline failed to enter PLAYING state")

    csi_backend = "python-gstreamer"

def camera_worker():
    """Background fetcher: copies RGB, reads center depth only (float)."""
    global latest_frames
    
    if CAMERA_TYPE in ("opencv", "csi"):
        while not stop_event.is_set():
            frame = None
            if CAMERA_TYPE == "csi" and gst_sink is not None:
                frame = _pull_gstreamer_sample()
                if frame is None:
                    time.sleep(0.01)
                    continue
                rgb = frame
            elif cap and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                # OpenCV / V4L2 return BGR, convert to RGB.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                time.sleep(0.1)
                continue

            with frame_lock:
                latest_frames["rgb"] = rgb
                latest_frames["rgb_seq"] += 1
                latest_frames["depth_center"] = 0.0
        return

    while not stop_event.is_set():
        try:
            frames = pipeline.wait_for_frames(timeout_ms=2000)
            aligned_frames = align.process(frames) if align is not None else frames

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame() if REALSENSE_ENABLE_DEPTH else None
            ir_frame = aligned_frames.get_infrared_frame(1) if REALSENSE_ENABLE_IR else None

            if color_frame and (depth_frame is not None or not REALSENSE_ENABLE_DEPTH):
                rgb = np.asanyarray(color_frame.get_data()).copy()

                depth_center = 0.0
                if depth_frame is not None:
                    # Read only center depth to avoid copying the whole depth map (~600KB/frame)
                    w = depth_frame.get_width()
                    h = depth_frame.get_height()
                    depth_center = float(depth_frame.get_distance(w // 2, h // 2))

                depth_colormap = None
                if REALSENSE_ENABLE_DEPTH_PREVIEW and depth_frame is not None:
                    try:
                        depth_data = np.asanyarray(depth_frame.get_data()).copy()
                        depth_mm = depth_data.astype(np.float32)
                        vmax = np.percentile(depth_mm[depth_mm > 0], 95) if np.any(depth_mm > 0) else 1.0
                        vmin = np.percentile(depth_mm[depth_mm > 0], 5) if np.any(depth_mm > 0) else 0.0
                        if vmax <= vmin:
                            vmax = vmin + 1.0
                        norm = np.clip((depth_mm - vmin) / (vmax - vmin), 0.0, 1.0)
                        depth_colormap = (255 * cv2.applyColorMap((255 * (1.0 - norm)).astype(np.uint8), cv2.COLORMAP_JET)).astype(np.uint8)
                    except Exception:
                        depth_colormap = None

                ir_image = None
                if REALSENSE_ENABLE_IR and ir_frame:
                    try:
                        ir_image = np.asanyarray(ir_frame.get_data()).copy()
                        # IR is single-channel; keep as-is (uint8)
                    except Exception:
                        ir_image = None

                with frame_lock:
                    latest_frames["rgb"] = rgb
                    latest_frames["rgb_seq"] += 1
                    latest_frames["depth_center"] = depth_center
                    latest_frames["ir"] = ir_image
                    latest_frames["depth_map"] = depth_colormap
        except Exception as e:
            print(f"[RealSense Thread Error] {e}")

def start_pipeline():
    global pipeline, align, cap, gst_pipeline, gst_sink, csi_backend
    
    if CAMERA_TYPE == "opencv":
        if cap is None:
            cap = cv2.VideoCapture(OPENCV_DEVICE_ID)
            # Set resolution to match realsense config roughly if possible, or just default
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 424)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            print(f"[OpenCV] Camera started on device {OPENCV_DEVICE_ID}")

            stop_event.clear()
            t = threading.Thread(target=camera_worker, daemon=True)
            t.start()
        return

    if CAMERA_TYPE == "csi":
        if cap is None and gst_pipeline is None:
            if CSI_BACKEND != "v4l2" and GST_AVAILABLE:
                try:
                    _start_native_csi_pipeline()
                except Exception as e:
                    print(f"[CSI] Native GStreamer open failed: {e}")
                    if gst_pipeline is not None:
                        try:
                            gst_pipeline.set_state(Gst.State.NULL)
                        except Exception:
                            pass
                    gst_pipeline = None
                    gst_sink = None
                    csi_backend = None

            if gst_pipeline is None:
                if CSI_BACKEND == "argus":
                    raise RuntimeError(
                        f"Could not open Jetson CSI camera sensor-id={CSI_SENSOR_ID} "
                        f"with backend={CSI_BACKEND}"
                    )

                cap = cv2.VideoCapture(CSI_SENSOR_ID)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CSI_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CSI_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, CSI_FPS)
                if not cap.isOpened():
                    cap.release()
                    cap = None
                    raise RuntimeError(
                        f"Could not open Jetson CSI camera sensor-id={CSI_SENSOR_ID} "
                        f"({CSI_WIDTH}x{CSI_HEIGHT}@{CSI_FPS}, flip={CSI_FLIP_METHOD})"
                    )
                csi_backend = f"v4l2-/dev/video{CSI_SENSOR_ID}"

            print(
                f"[CSI] Camera started on sensor-id={CSI_SENSOR_ID} using {csi_backend} "
                f"({CSI_WIDTH}x{CSI_HEIGHT}@{CSI_FPS}, flip={CSI_FLIP_METHOD})"
            )

            stop_event.clear()
            t = threading.Thread(target=camera_worker, daemon=True)
            t.start()
        return

    if rs is None:
        raise RuntimeError(
            "pyrealsense2 is required when CAMERA_TYPE is 'realsense'. "
            "Use --camera cam0/opencv or install the RealSense Python bindings."
        )

    if pipeline is None:
        pipeline = rs.pipeline()
        config = rs.config()

        # Reduced resolution + 15 FPS to lower CPU load on Jetson Nano
        config.enable_stream(rs.stream.color, 424, 240, rs.format.rgb8, 15)
        if REALSENSE_ENABLE_DEPTH:
            config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 15)
        if REALSENSE_ENABLE_IR:
            config.enable_stream(rs.stream.infrared, 1, 424, 240, rs.format.y8, 15)

        pipeline.start(config)

        align = rs.align(rs.stream.color) if REALSENSE_ENABLE_DEPTH else None
        enabled_streams = ["RGB"]
        if REALSENSE_ENABLE_DEPTH:
            enabled_streams.append("Depth")
        if REALSENSE_ENABLE_IR:
            enabled_streams.append("IR")
        print("[RealSense] Pipeline started - %s ready" % " + ".join(enabled_streams))
        
        # Start background thread
        t = threading.Thread(target=camera_worker, daemon=True)
        t.start()

def get_all_frames(copy_frames=True):
    """Return a tuple (rgb, depth_center, ir_image, depth_map).
    Any of the images may be None if they are not available yet.
    """
    start_pipeline()
    with frame_lock:
        if latest_frames["rgb"] is None:
            return None, None, None, None
        rgb = latest_frames["rgb"] if latest_frames["rgb"] is not None else None
        depth_center = float(latest_frames.get("depth_center", 0.0) or 0.0)
        ir = latest_frames.get("ir")
        if copy_frames and rgb is not None:
            rgb = rgb.copy()
        if copy_frames and ir is not None:
            ir = ir.copy()
        depth_map = latest_frames.get("depth_map")
        if copy_frames and depth_map is not None:
            depth_map = depth_map.copy()
        return rgb, depth_center, ir, depth_map


def get_rgb_frame_and_seq(copy_frame=True):
    start_pipeline()
    with frame_lock:
        rgb = latest_frames.get("rgb")
        if rgb is None:
            return None, 0
        if copy_frame:
            rgb = rgb.copy()
        return rgb, int(latest_frames.get("rgb_seq", 0) or 0)

def stop_pipeline():
    global pipeline, align, cap, gst_pipeline, gst_sink, csi_backend
    stop_event.set()
    # Give the thread a moment to exit the loop
    time.sleep(0.5)
    
    if CAMERA_TYPE in ("opencv", "csi"):
        if cap:
            cap.release()
            cap = None
        if gst_pipeline is not None:
            try:
                gst_pipeline.set_state(Gst.State.NULL)
            except Exception as e:
                print(f"[CSI] Error stopping GStreamer pipeline: {e}")
            gst_pipeline = None
            gst_sink = None
        csi_backend = None
        label = "CSI" if CAMERA_TYPE == "csi" else "OpenCV"
        print(f"[{label}] Camera stopped.")
        return

    if pipeline:
        try:
            pipeline.stop()
        except Exception as e:
            print(f"[RealSense] Error stopping pipeline: {e}")
        pipeline = None
        align = None
    print("[RealSense] Pipeline stopped.")

def get_aligned_frames():
    """Returns (rgb, depth_center_float) with single lock acquisition"""
    start_pipeline()
    with frame_lock:
        if latest_frames["rgb"] is None:
            return None, None
        return latest_frames["rgb"], latest_frames["depth_center"]

# --------------------- RGB ---------------------
def get_rgb_image():
    start_pipeline()
    with frame_lock:
        if latest_frames["rgb"] is None:
            return None
        return latest_frames["rgb"].copy()


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
    with frame_lock:
        if latest_frames["ir"] is None:
            return None
        return latest_frames["ir"].copy()


def save_ir_image(filename):
    img = get_ir_image()
    if img is not None:
        cv2.imwrite(filename, img)
        print("[SAVED] IR dots -> " + filename)
        return True
    return False


# --------------------- DEPTH ---------------------
def get_depth_image():
    # Depth map is no longer stored to reduce CPU. Return None to indicate unavailable.
    return None


def save_depth_image(filename, colored=True):
    depth = get_depth_image()
    if depth is None:
        print("[Depth] Full depth map not stored (optimized mode).")
        return False
    # If enabled in the future, the code below can run.
    return False


# --------------------- Distance helper ---------------------
def get_center_distance():
    _, depth_center = get_aligned_frames()
    if depth_center is None:
        return 0
    if depth_center == 0:
        return 0
    return depth_center


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
