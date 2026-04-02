import cv2
import numpy as np
import time
import logging
from smbus2 import SMBus
import platform
import subprocess

# Try importing GStreamer Python bindings for CSI fallback when OpenCV
# lacks GStreamer support on Jetson.
try:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
    Gst.init(None)
    HAS_GI_GST = True
except Exception:
    Gst = None
    HAS_GI_GST = False

# Try importing pyrealsense2, handle if missing
try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False

class PCA9685:
    def __init__(self, bus=1, address=0x40):
        self.bus = SMBus(bus)
        self.address = address
        self.set_pwm_freq(50)

    def set_pwm_freq(self, freq_hz=50):
        prescaleval = 25000000.0
        prescaleval /= 4096.0
        prescaleval /= float(freq_hz)
        prescaleval -= 1.0
        prescale = int(prescaleval + 0.5)

        oldmode = self.bus.read_byte_data(self.address, 0x00)
        newmode = (oldmode & 0x7F) | 0x10                    # sleep
        self.bus.write_byte_data(self.address, 0x00, newmode)  # go to sleep
        self.bus.write_byte_data(self.address, 0xFE, prescale)  # set prescale
        self.bus.write_byte_data(self.address, 0x00, oldmode)   # restore
        time.sleep(0.005)
        self.bus.write_byte_data(self.address, 0x00, oldmode | 0x80)  # restart

    def set_pwm(self, channel, on, off):
        self.bus.write_byte_data(self.address, 0x06 + 4*channel, on & 0xFF)
        self.bus.write_byte_data(self.address, 0x07 + 4*channel, on >> 8)
        self.bus.write_byte_data(self.address, 0x08 + 4*channel, off & 0xFF)
        self.bus.write_byte_data(self.address, 0x09 + 4*channel, off >> 8)

    def set_us(self, channel, microseconds):
        """Convert microseconds -> correct 12-bit off value at 50Hz"""
        pulse = int(microseconds * 4096 * 50 / 1000000 + 0.5)
        self.set_pwm(channel, 0, pulse)

def get_cpu_ram_info():
    try:
        # Simple linux commands
        cpu_info = subprocess.check_output("cat /proc/cpuinfo | grep 'model name' | head -1", shell=True).decode().strip().split(': ')[1]
    except:
        cpu_info = platform.machine() # fallback
    
    try:
        mem_info = subprocess.check_output("free -h | grep Mem | awk '{print $2}'", shell=True).decode().strip()
    except:
        mem_info = "Unknown"
        
    return f"{cpu_info} / {mem_info}"

ROLE_PRIMARY_RGB = "primary_rgb"
ROLE_SIDECAR_DEPTH_IMU = "sidecar_depth_imu"
ROLE_REAR_PREVIEW = "rear_preview"
ROLE_ORDER = [ROLE_PRIMARY_RGB, ROLE_SIDECAR_DEPTH_IMU, ROLE_REAR_PREVIEW]
ROLE_USED_FOR = {
    ROLE_PRIMARY_RGB: ["lane_following", "apriltag", "forward_preview"],
    ROLE_SIDECAR_DEPTH_IMU: ["obstacle_stop", "state_context"],
    ROLE_REAR_PREVIEW: ["rear_preview_only"],
}
logger = logging.getLogger("Hardware")
_CV2_GSTREAMER_ENABLED = None


def _opencv_cuda_device_count():
    cuda_module = getattr(cv2, "cuda", None)
    if cuda_module is None:
        return 0
    try:
        return int(cuda_module.getCudaEnabledDeviceCount())
    except Exception:
        return 0


def _opencv_has_gstreamer_build():
    global _CV2_GSTREAMER_ENABLED
    if _CV2_GSTREAMER_ENABLED is not None:
        return _CV2_GSTREAMER_ENABLED
    if not hasattr(cv2, "CAP_GSTREAMER"):
        _CV2_GSTREAMER_ENABLED = False
        return _CV2_GSTREAMER_ENABLED
    try:
        build_info = cv2.getBuildInformation()
    except Exception:
        _CV2_GSTREAMER_ENABLED = False
        return _CV2_GSTREAMER_ENABLED

    enabled = False
    for line in build_info.splitlines():
        if "GStreamer" in line:
            enabled = "YES" in line.upper()
            break
    _CV2_GSTREAMER_ENABLED = enabled
    return _CV2_GSTREAMER_ENABLED


def _coerce_bool(value, default=True):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off", "")
    return bool(value)


def normalize_camera_config(config):
    cfg = dict(config or {})
    cfg["type"] = str(cfg.get("type") or "opencv").strip().lower()
    role = str(cfg.get("role") or "").strip().lower()
    cfg["role"] = role or None
    cfg["enabled"] = _coerce_bool(cfg.get("enabled"), True)

    defaults = {
        "index": 0,
        "sensor_id": 0,
        "width": 640,
        "height": 480,
        "fps": 30,
        "flip_method": 0,
    }
    for key, default in defaults.items():
        try:
            cfg[key] = int(cfg.get(key, default))
        except Exception:
            cfg[key] = default
    return cfg


def normalize_camera_configs(camera_configs):
    return [normalize_camera_config(config) for config in camera_configs or []]


def resolve_sensor_role_configs(camera_configs):
    enabled_configs = [cfg for cfg in normalize_camera_configs(camera_configs) if cfg.get("enabled", True)]
    resolved = {
        ROLE_PRIMARY_RGB: None,
        ROLE_SIDECAR_DEPTH_IMU: None,
        ROLE_REAR_PREVIEW: None,
        "camera_configs": enabled_configs,
        "role_based": any(cfg.get("role") for cfg in enabled_configs),
    }
    if not enabled_configs:
        return resolved

    if resolved["role_based"]:
        for role in ROLE_ORDER:
            for cfg in enabled_configs:
                if cfg.get("role") == role:
                    resolved[role] = cfg
                    break
        if resolved[ROLE_SIDECAR_DEPTH_IMU] is None:
            primary = resolved[ROLE_PRIMARY_RGB]
            if primary and primary.get("type") == "realsense":
                resolved[ROLE_SIDECAR_DEPTH_IMU] = primary
        return resolved

    primary_cfg = enabled_configs[0]
    resolved[ROLE_PRIMARY_RGB] = primary_cfg
    if len(enabled_configs) == 1 and primary_cfg.get("type") == "realsense":
        resolved[ROLE_SIDECAR_DEPTH_IMU] = primary_cfg
    return resolved


def get_system_specs(cameras=None):
    specs = {
        "device": f"{platform.system()} {platform.release()}",
        "cpu_ram": get_cpu_ram_info(),
        "cameras": [],
        "inference": "CUDA (GPU)" if _opencv_cuda_device_count() > 0 else "CPU",
        "resnet_version": "ResNet-101 (Default)",
        "yolo_version": "YOLOv8n (Default)"
    }
    role_map = resolve_sensor_role_configs(cameras or [])
    role_labels_by_id = {}
    for role in ROLE_ORDER:
        cfg = role_map.get(role)
        if cfg is None:
            continue
        role_labels_by_id.setdefault(id(cfg), []).append(role)

    for cfg in role_map.get("camera_configs", []):
        item = {
            "type": cfg.get("type"),
            "role": "+".join(role_labels_by_id.get(id(cfg), [cfg.get("role") or "camera"])),
            "width": cfg.get("width"),
            "height": cfg.get("height"),
            "fps": cfg.get("fps"),
        }
        if cfg.get("type") == "csi":
            item["sensor_id"] = cfg.get("sensor_id")
            item["flip_method"] = cfg.get("flip_method")
        else:
            item["index"] = cfg.get("index")
        specs["cameras"].append(item)
    return specs


class CameraInterface:
    def read(self):
        raise NotImplementedError

    def release(self):
        raise NotImplementedError


def _build_csi_pipeline(sensor_id, width, height, fps, flip_method):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        "video/x-raw(memory:NVMM), "
        f"width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        "video/x-raw, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )


def _build_csi_gst_appsink_pipeline(sensor_id, width, height, fps, flip_method, appsink_name="sink"):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        "video/x-raw(memory:NVMM), "
        f"width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        "video/x-raw, format=(string)BGRx ! "
        "videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink name={appsink_name} "
        "drop=true max-buffers=1 sync=false"
    )


class OpenCVCamera(CameraInterface):
    def __init__(self, index=0, width=640, height=480, fps=30):
        self.width = width
        self.height = height

        if index == 0 and _opencv_has_gstreamer_build():
            pipeline = _build_csi_pipeline(0, width, height, fps, 0)
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if not self.cap.isOpened():
                print("[Camera] GStreamer failed, falling back to V4L2...")
                self.cap = cv2.VideoCapture(index)
        else:
            self.cap = cv2.VideoCapture(index)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open OpenCV camera index {index}")

    def read(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None, None, None
        
        # Ensure correct size (sometimes GStreamer or V4L2 returns different sizes)
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))

        # Convert BGR to RGB, return None for depth and imu
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), None, None

    def release(self):
        self.cap.release()


class GstAppSinkCSIInterface(CameraInterface):
    def __init__(self, sensor_id=0, width=640, height=480, fps=15, flip_method=0):
        if not HAS_GI_GST:
            raise RuntimeError("Python GStreamer bindings not installed")

        self.width = width
        self.height = height
        self.sensor_id = sensor_id
        self.flip_method = flip_method
        self.pipeline = Gst.parse_launch(
            _build_csi_gst_appsink_pipeline(sensor_id, width, height, fps, flip_method)
        )
        self.sink = self.pipeline.get_by_name("sink")
        if self.sink is None:
            self.pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("Failed to create CSI GStreamer appsink")
        result = self.pipeline.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            self.pipeline.set_state(Gst.State.NULL)
            raise RuntimeError(
                f"Could not start CSI GStreamer pipeline sensor-id {sensor_id} "
                f"({width}x{height}@{fps}, flip={flip_method})"
            )

    def read(self):
        sample = self.sink.emit("try-pull-sample", 500000000)
        if sample is None:
            return None, None, None

        buffer = sample.get_buffer()
        caps = sample.get_caps()
        if buffer is None or caps is None:
            return None, None, None

        structure = caps.get_structure(0)
        width = int(structure.get_value("width"))
        height = int(structure.get_value("height"))
        ok, map_info = buffer.map(Gst.MapFlags.READ)
        if not ok:
            return None, None, None
        try:
            frame = np.frombuffer(map_info.data, dtype=np.uint8)
            expected_size = width * height * 3
            if frame.size < expected_size:
                return None, None, None
            frame = frame[:expected_size].reshape((height, width, 3)).copy()
        finally:
            buffer.unmap(map_info)

        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), None, None

    def release(self):
        if getattr(self, "pipeline", None) is not None:
            self.pipeline.set_state(Gst.State.NULL)


class CSICamera(CameraInterface):
    def __init__(self, sensor_id=0, width=640, height=480, fps=15, flip_method=0):
        self.width = width
        self.height = height
        self.sensor_id = sensor_id
        self.flip_method = flip_method
        self.cap = None
        self.gst_camera = None
        self.backend = None

        if _opencv_has_gstreamer_build():
            pipeline = _build_csi_pipeline(sensor_id, width, height, fps, flip_method)
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if self.cap.isOpened():
                self.backend = "opencv_gstreamer"
            else:
                logger.warning(
                    "OpenCV GStreamer CSI open failed for sensor-id %s; attempting GI Gst fallback.",
                    sensor_id,
                )
                self.cap.release()
                self.cap = None
        elif HAS_GI_GST:
            logger.info(
                "OpenCV lacks GStreamer support; using GI Gst appsink fallback for CSI sensor-id %s.",
                sensor_id,
            )

        if self.backend is None and HAS_GI_GST:
            self.gst_camera = GstAppSinkCSIInterface(
                sensor_id=sensor_id,
                width=width,
                height=height,
                fps=fps,
                flip_method=flip_method,
            )
            self.backend = "gi_gstreamer"

        if self.backend is None:
            raise RuntimeError(
                f"Could not open CSI camera sensor-id {sensor_id} "
                f"({width}x{height}@{fps}, flip={flip_method})"
            )

    def read(self):
        if self.backend == "gi_gstreamer" and self.gst_camera is not None:
            return self.gst_camera.read()

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None, None, None
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), None, None

    def release(self):
        if self.gst_camera is not None:
            self.gst_camera.release()
        if self.cap is not None:
            self.cap.release()


class RealSenseCamera(CameraInterface):
    def __init__(self, width=640, height=480, fps=15, enable_depth=True, enable_color=True, enable_imu=True):
        if not HAS_REALSENSE:
            raise RuntimeError("pyrealsense2 not installed")
        
        self.pipeline = rs.pipeline()
        config = rs.config()
        self.color_enabled = bool(enable_color)
        self.depth_enabled = bool(enable_depth)
        self.imu_enabled = bool(enable_imu)
        self.motion_sensor = None
        self.motion_queue = None
        self._last_imu_data = {}
        self._last_imu_ts = None
        
        # RealSense supports specific FPS: 6, 15, 30, 60.
        # Mapping to closest supported.
        if fps > 45: r_fps = 60
        elif fps > 22: r_fps = 30
        elif fps > 10: r_fps = 15
        else: r_fps = 6

        print(
            f"[Camera] RealSense requesting {width}x{height}@{r_fps} FPS "
            f"(Color: {self.color_enabled}, Depth: {self.depth_enabled}, IMU: {self.imu_enabled})"
        )
        
        try:
            if self.color_enabled:
                config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, r_fps)
            if self.depth_enabled:
                config.enable_stream(rs.stream.depth, width, height, rs.format.z16, r_fps)

            self.profile = self.pipeline.start(config)
            
            # Optimization: Turn off laser if we don't need depth (saves power/heat)
            if not self.depth_enabled:
                try:
                    dev = self.profile.get_device()
                    depth_sensor = dev.first_depth_sensor()
                    if depth_sensor.supports(rs.option.emitter_enabled):
                        depth_sensor.set_option(rs.option.emitter_enabled, 0)
                except: pass

        except Exception as e:
            print(f"[Camera] Profile fail: {e}. Trying fallback...")
            config = rs.config()
            if self.color_enabled:
                config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 15)
            if self.depth_enabled:
                try:
                    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
                except: pass
            self.profile = self.pipeline.start(config)

        self.align = rs.align(rs.stream.color) if self.depth_enabled and self.color_enabled else None
        
        # Power management: Set auto-exposure for stable FPS
        try:
            sensors = self.profile.get_device().query_sensors()
            for s in sensors:
                if s.supports(rs.option.enable_auto_exposure):
                    s.set_option(rs.option.enable_auto_exposure, 1)
        except: pass

        self._start_motion_sensor()

    def _select_motion_profile(self, profiles, stream_type, preferred_fps):
        candidates = []
        for profile in profiles:
            try:
                if profile.stream_type() != stream_type:
                    continue
                if profile.format() != rs.format.motion_xyz32f:
                    continue
                candidates.append(profile)
            except Exception:
                continue
        if not candidates:
            return None
        for fps in preferred_fps:
            for profile in candidates:
                try:
                    if profile.fps() == fps:
                        return profile
                except Exception:
                    continue
        return candidates[0]

    def _start_motion_sensor(self):
        if not self.imu_enabled:
            return
        try:
            device = self.profile.get_device()
            motion_sensor = None
            for sensor in device.query_sensors():
                try:
                    name = sensor.get_info(rs.camera_info.name)
                except Exception:
                    continue
                if "Motion" in name:
                    motion_sensor = sensor
                    break
            if motion_sensor is None:
                print("[Camera] Motion sensor not available on RealSense device")
                return

            profiles = list(motion_sensor.get_stream_profiles())
            accel_profile = self._select_motion_profile(profiles, rs.stream.accel, preferred_fps=(63, 250))
            gyro_profile = self._select_motion_profile(profiles, rs.stream.gyro, preferred_fps=(200, 400))
            selected_profiles = [profile for profile in (accel_profile, gyro_profile) if profile is not None]
            if not selected_profiles:
                print("[Camera] No compatible IMU motion profiles found")
                return

            self.motion_queue = rs.frame_queue(100)
            motion_sensor.open(selected_profiles)
            motion_sensor.start(self.motion_queue)
            self.motion_sensor = motion_sensor
            print("[Camera] Enabling IMU streams via motion sensor")
        except Exception as imu_e:
            print(f"[Camera] IMU motion sensor start failed: {imu_e}")
            self.motion_sensor = None
            self.motion_queue = None

    def _read_motion_frames(self):
        if self.motion_queue is None:
            return dict(self._last_imu_data or {})

        imu_data = dict(self._last_imu_data or {})
        updated = False
        while True:
            try:
                ok, frame = self.motion_queue.try_wait_for_frame(0)
            except Exception:
                break
            if not ok or frame is None:
                break
            try:
                if not frame.is_motion_frame():
                    continue
                stream_type = frame.profile.stream_type()
                motion = frame.as_motion_frame().get_motion_data()
                values = [motion.x, motion.y, motion.z]
                if stream_type == rs.stream.accel:
                    imu_data["accel"] = values
                    updated = True
                elif stream_type == rs.stream.gyro:
                    imu_data["gyro"] = values
                    updated = True
            except Exception:
                continue

        if updated:
            self._last_imu_data = dict(imu_data)
            self._last_imu_ts = time.time()
            return imu_data

        if self._last_imu_ts is not None and (time.time() - self._last_imu_ts) <= 0.5:
            return dict(self._last_imu_data)
        return {}
            
    def read(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=500)
            
            color_frame = frames.get_color_frame() if self.color_enabled else None
            depth_data = None
            imu_data = self._read_motion_frames()

            if self.depth_enabled and self.align is not None:
                aligned = self.align.process(frames)
                depth_frame = aligned.get_depth_frame()
                if depth_frame:
                    depth_data = np.asanyarray(depth_frame.get_data())
            elif self.depth_enabled:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_data = np.asanyarray(depth_frame.get_data())
            
            if self.color_enabled and not color_frame:
                return None, None, None

            color_data = np.asanyarray(color_frame.get_data()) if color_frame else None
            return color_data, depth_data, imu_data
        except Exception as e:
            # Don't print every frame to avoid log spam, just yield
            return None, None, None

    def release(self):
        try:
            if self.motion_sensor is not None:
                self.motion_sensor.stop()
                self.motion_sensor.close()
        except:
            pass
        try:
            self.pipeline.stop()
            print("[Camera] RealSense pipeline stopped.")
            time.sleep(0.5)
        except:
            pass


class CompositeSensorRig:
    def __init__(
        self,
        primary_rgb_camera=None,
        primary_rgb_config=None,
        primary_rgb_error=None,
        sidecar_depth_imu_sensor=None,
        sidecar_depth_imu_config=None,
        sidecar_depth_imu_error=None,
        rear_preview_camera=None,
        rear_preview_config=None,
        rear_preview_error=None,
        depth_aligned_to_primary=False,
    ):
        self.primary_rgb_camera = primary_rgb_camera
        self.primary_rgb_config = primary_rgb_config
        self.sidecar_depth_imu_sensor = sidecar_depth_imu_sensor
        self.sidecar_depth_imu_config = sidecar_depth_imu_config
        self.rear_preview_camera = rear_preview_camera
        self.rear_preview_config = rear_preview_config
        self.depth_aligned_to_primary = bool(depth_aligned_to_primary)
        self.forward_preview_role = ROLE_PRIMARY_RGB
        self._last_status = {
            ROLE_PRIMARY_RGB: self._initial_role_status(
                primary_rgb_camera,
                primary_rgb_config,
                error=primary_rgb_error,
            ),
            ROLE_SIDECAR_DEPTH_IMU: self._initial_role_status(
                sidecar_depth_imu_sensor,
                sidecar_depth_imu_config,
                aliased_to=ROLE_PRIMARY_RGB if primary_rgb_camera is sidecar_depth_imu_sensor and primary_rgb_camera is not None else None,
                error=sidecar_depth_imu_error,
            ),
            ROLE_REAR_PREVIEW: self._initial_role_status(
                rear_preview_camera,
                rear_preview_config,
                error=rear_preview_error,
            ),
        }

    def _initial_role_status(self, camera, config, aliased_to=None, error=None):
        if config is None:
            status_name = "disabled"
        elif camera is None:
            status_name = "unavailable"
        else:
            status_name = "configured"
        return {
            "status": status_name,
            "frame_available": False,
            "depth_available": False,
            "imu_available": False,
            "last_frame_ts": None,
            "last_depth_ts": None,
            "last_imu_ts": None,
            "aliased_to": aliased_to,
            "present": camera is not None,
            "error": str(error) if error else None,
        }

    def _age_ms(self, ts, now_ts=None):
        if ts is None:
            return None
        if now_ts is None:
            now_ts = time.time()
        return int(max(0.0, (now_ts - ts) * 1000.0))

    def _is_recent(self, ts, now_ts=None, max_age_ms=2000):
        age_ms = self._age_ms(ts, now_ts=now_ts)
        return age_ms is not None and age_ms <= max_age_ms

    def _modality_status(self, configured, available, role_status):
        if not configured:
            return "disabled"
        if available:
            return "available"
        if role_status == "configured":
            return "configured"
        return "unavailable"

    def _update_status(self, role, frame_available=False, depth_available=False, imu_available=False):
        status = self._last_status[role]
        status["frame_available"] = bool(frame_available)
        status["depth_available"] = bool(depth_available)
        status["imu_available"] = bool(imu_available)
        if frame_available or depth_available or imu_available:
            status["status"] = "available"
            now_ts = time.time()
            if frame_available:
                status["last_frame_ts"] = now_ts
            if depth_available:
                status["last_depth_ts"] = now_ts
            if imu_available:
                status["last_imu_ts"] = now_ts
        elif status["status"] != "disabled":
            status["status"] = "unavailable"

    def snapshot(self):
        now_ts = time.time()
        return {
            "forward_preview_role": self.forward_preview_role,
            "depth_aligned_to_primary": bool(self.depth_aligned_to_primary),
            ROLE_PRIMARY_RGB: self._role_snapshot(ROLE_PRIMARY_RGB, self.primary_rgb_config, now_ts=now_ts),
            ROLE_SIDECAR_DEPTH_IMU: self._role_snapshot(ROLE_SIDECAR_DEPTH_IMU, self.sidecar_depth_imu_config, now_ts=now_ts),
            ROLE_REAR_PREVIEW: self._role_snapshot(ROLE_REAR_PREVIEW, self.rear_preview_config, now_ts=now_ts),
            "depth": self._alias_snapshot(ROLE_SIDECAR_DEPTH_IMU, "depth", now_ts=now_ts, used_for=["obstacle_stop"]),
            "imu": self._alias_snapshot(ROLE_SIDECAR_DEPTH_IMU, "imu", now_ts=now_ts, used_for=["state_context"]),
            "rear": self._alias_snapshot(ROLE_REAR_PREVIEW, "rear", now_ts=now_ts, used_for=["rear_preview_only"]),
        }

    def _role_snapshot(self, role, config, now_ts=None):
        status = dict(self._last_status[role])
        status_name = status.get("status", "disabled")
        frame_recent = self._is_recent(status.get("last_frame_ts"), now_ts=now_ts)
        depth_recent = self._is_recent(status.get("last_depth_ts"), now_ts=now_ts)
        imu_recent = self._is_recent(status.get("last_imu_ts"), now_ts=now_ts)
        depth_state = self._modality_status(
            config is not None,
            status.get("depth_available", False),
            status_name,
        )
        imu_state = self._modality_status(
            config is not None,
            status.get("imu_available", False),
            status_name,
        )
        depth_frame_age_ms = None
        imu_frame_age_ms = None
        if role == ROLE_SIDECAR_DEPTH_IMU and config is not None:
            depth_frame_age_ms = self._age_ms(status.get("last_depth_ts"), now_ts=now_ts)
            imu_frame_age_ms = self._age_ms(status.get("last_imu_ts"), now_ts=now_ts)
        frame_age_ms = self._age_ms(status.get("last_frame_ts"), now_ts=now_ts)
        if role != ROLE_SIDECAR_DEPTH_IMU:
            depth_frame_age_ms = None
            imu_frame_age_ms = None
        healthy = False
        if status_name == "available":
            if role == ROLE_SIDECAR_DEPTH_IMU:
                healthy = bool(frame_recent or depth_recent or imu_recent)
            else:
                healthy = bool(frame_recent)
        snapshot = {
            "role": role,
            "configured": config is not None,
            "enabled": bool((config or {}).get("enabled", False)),
            "present": bool(status.get("present", False)),
            "status": status_name,
            "healthy": healthy,
            "frame_available": status.get("frame_available", False),
            "depth_available": status.get("depth_available", False),
            "imu_available": status.get("imu_available", False),
            "last_frame_ts": status.get("last_frame_ts"),
            "frame_age_ms": frame_age_ms,
            "depth_status": depth_state if role == ROLE_SIDECAR_DEPTH_IMU else None,
            "depth_frame_age_ms": depth_frame_age_ms,
            "imu_status": imu_state if role == ROLE_SIDECAR_DEPTH_IMU else None,
            "imu_frame_age_ms": imu_frame_age_ms,
            "aliased_to": status.get("aliased_to"),
            "depth": depth_state,
            "imu": imu_state,
            "used_for": list(ROLE_USED_FOR.get(role, [])),
            "error": status.get("error"),
        }
        if config:
            label = "Sensor"
            source = config.get("type")
            if role == ROLE_PRIMARY_RGB:
                label = "CAM0 Primary RGB" if config.get("type") == "csi" else "Primary RGB"
                if config.get("type") == "csi":
                    source = f"cam{config.get('sensor_id', 0)}"
            elif role == ROLE_SIDECAR_DEPTH_IMU:
                label = "RealSense Sidecar Depth/IMU"
                source = "realsense"
            elif role == ROLE_REAR_PREVIEW:
                label = "CAM1 Rear Preview" if config.get("type") == "csi" else "Rear Preview"
                if config.get("type") == "csi":
                    source = f"cam{config.get('sensor_id', 1)}"
            snapshot.update({
                "label": label,
                "source": source,
                "type": config.get("type"),
                "width": config.get("width"),
                "height": config.get("height"),
                "fps": config.get("fps"),
                "sensor_id": config.get("sensor_id"),
                "index": config.get("index"),
                "flip_method": config.get("flip_method"),
            })
        return snapshot

    def _alias_snapshot(self, source_role, alias_role, now_ts=None, used_for=None):
        snapshot = dict(self._role_snapshot(source_role, getattr(self, f"{source_role}_config", None), now_ts=now_ts))
        snapshot.update({
            "role": alias_role,
            "used_for": list(used_for or ROLE_USED_FOR.get(alias_role, [])),
        })
        if alias_role == "depth":
            depth_status = self._modality_status(
                snapshot.get("configured"),
                snapshot.get("depth_available"),
                snapshot.get("status"),
            )
            snapshot["status"] = depth_status
            snapshot["healthy"] = depth_status == "available" and self._is_recent(
                self._last_status[source_role].get("last_depth_ts"),
                now_ts=now_ts,
            )
            snapshot["frame_available"] = bool(snapshot.get("depth_available"))
            snapshot["frame_age_ms"] = snapshot.get("depth_frame_age_ms")
            snapshot["depth"] = depth_status
            snapshot["depth_status"] = depth_status
            snapshot["imu"] = "disabled"
            snapshot["imu_status"] = "disabled"
            snapshot["source"] = "realsense" if snapshot.get("configured") else snapshot.get("source")
            snapshot["label"] = "Depth Sidecar"
        elif alias_role == "imu":
            imu_status = self._modality_status(
                snapshot.get("configured"),
                snapshot.get("imu_available"),
                snapshot.get("status"),
            )
            snapshot["status"] = imu_status
            snapshot["healthy"] = imu_status == "available" and self._is_recent(
                self._last_status[source_role].get("last_imu_ts"),
                now_ts=now_ts,
            )
            snapshot["frame_available"] = bool(snapshot.get("imu_available"))
            snapshot["frame_age_ms"] = snapshot.get("imu_frame_age_ms")
            snapshot["depth"] = "disabled"
            snapshot["depth_status"] = "disabled"
            snapshot["imu"] = imu_status
            snapshot["imu_status"] = imu_status
            snapshot["source"] = "realsense" if snapshot.get("configured") else snapshot.get("source")
            snapshot["label"] = "IMU Sidecar"
        elif alias_role == "rear":
            snapshot["label"] = "Rear Preview"
        return snapshot

    def read(self, include_rear=False):
        packet = {
            "primary_rgb": None,
            "depth": None,
            "imu": {},
            "rear_rgb": None,
            "depth_aligned_to_primary": bool(self.depth_aligned_to_primary),
        }

        if self.primary_rgb_camera is not None and self.primary_rgb_camera is self.sidecar_depth_imu_sensor:
            primary_rgb, depth, imu = self.primary_rgb_camera.read()
            packet["primary_rgb"] = primary_rgb
            packet["depth"] = depth
            packet["imu"] = imu or {}
            self._update_status(
                ROLE_PRIMARY_RGB,
                frame_available=primary_rgb is not None,
                depth_available=depth is not None,
                imu_available=bool(packet["imu"]),
            )
            self._update_status(
                ROLE_SIDECAR_DEPTH_IMU,
                frame_available=primary_rgb is not None,
                depth_available=depth is not None,
                imu_available=bool(packet["imu"]),
            )
        else:
            if self.primary_rgb_camera is not None:
                primary_rgb, _, _ = self.primary_rgb_camera.read()
                packet["primary_rgb"] = primary_rgb
                self._update_status(
                    ROLE_PRIMARY_RGB,
                    frame_available=primary_rgb is not None,
                )
            if self.sidecar_depth_imu_sensor is not None:
                _, depth, imu = self.sidecar_depth_imu_sensor.read()
                packet["depth"] = depth
                packet["imu"] = imu or {}
                self._update_status(
                    ROLE_SIDECAR_DEPTH_IMU,
                    depth_available=depth is not None,
                    imu_available=bool(packet["imu"]),
                )

        if include_rear and self.rear_preview_camera is not None:
            rear_rgb, _, _ = self.rear_preview_camera.read()
            packet["rear_rgb"] = rear_rgb
            self._update_status(
                ROLE_REAR_PREVIEW,
                frame_available=rear_rgb is not None,
            )

        packet["sensor_snapshot"] = self.snapshot()
        return packet

    def release(self):
        released = set()
        for camera in (
            self.primary_rgb_camera,
            self.sidecar_depth_imu_sensor,
            self.rear_preview_camera,
        ):
            if camera is None or id(camera) in released:
                continue
            try:
                camera.release()
            finally:
                released.add(id(camera))


def empty_sensor_snapshot():
    return CompositeSensorRig().snapshot()


def get_camera(config, enable_depth=True, enable_color=True, enable_imu=True):
    cfg = normalize_camera_config(config)
    c_type = cfg.get("type", "opencv")
    print(f"[Camera] Initializing {c_type} (Need Depth: {enable_depth})...")
    if c_type == "realsense":
        return RealSenseCamera(
            width=cfg.get("width", 640),
            height=cfg.get("height", 480),
            fps=cfg.get("fps", 15),
            enable_depth=enable_depth,
            enable_color=enable_color,
            enable_imu=enable_imu,
        )
    if c_type == "csi":
        return CSICamera(
            sensor_id=cfg.get("sensor_id", 0),
            width=cfg.get("width", 640),
            height=cfg.get("height", 480),
            fps=cfg.get("fps", 15),
            flip_method=cfg.get("flip_method", 0),
        )
    else:
        return OpenCVCamera(
            index=cfg.get("index", 0),
            width=cfg.get("width", 640),
            height=cfg.get("height", 480),
            fps=cfg.get("fps", 30)
        )


def build_sensor_rig(camera_configs, enable_depth=True):
    resolved = resolve_sensor_role_configs(camera_configs)
    primary_cfg = resolved.get(ROLE_PRIMARY_RGB)
    sidecar_cfg = resolved.get(ROLE_SIDECAR_DEPTH_IMU)
    rear_cfg = resolved.get(ROLE_REAR_PREVIEW)

    primary_camera = None
    sidecar_camera = None
    rear_camera = None
    primary_error = None
    sidecar_error = None
    rear_error = None

    if primary_cfg is not None:
        primary_enable_depth = bool(enable_depth and primary_cfg is sidecar_cfg and primary_cfg.get("type") == "realsense")
        try:
            primary_camera = get_camera(
                primary_cfg,
                enable_depth=primary_enable_depth,
                enable_color=True,
                enable_imu=bool(primary_cfg.get("type") == "realsense"),
            )
        except Exception as exc:
            primary_error = str(exc)
            logger.warning("Primary RGB camera init failed: %s", exc)

    if sidecar_cfg is not None:
        if sidecar_cfg is primary_cfg:
            if primary_camera is not None:
                sidecar_camera = primary_camera
            else:
                try:
                    sidecar_camera = get_camera(
                        sidecar_cfg,
                        enable_depth=enable_depth,
                        enable_color=True,
                        enable_imu=bool(sidecar_cfg.get("type") == "realsense"),
                    )
                    primary_camera = sidecar_camera
                    primary_error = None
                except Exception as exc:
                    sidecar_error = str(exc)
                    logger.warning("Sidecar depth/IMU camera init failed: %s", exc)
        else:
            try:
                sidecar_camera = get_camera(
                    sidecar_cfg,
                    enable_depth=enable_depth,
                    enable_color=False if sidecar_cfg.get("type") == "realsense" else True,
                    enable_imu=bool(sidecar_cfg.get("type") == "realsense"),
                )
            except Exception as exc:
                sidecar_error = str(exc)
                logger.warning("Sidecar depth/IMU camera init failed: %s", exc)

    if rear_cfg is not None:
        try:
            rear_camera = get_camera(rear_cfg, enable_depth=False)
        except Exception as exc:
            rear_error = str(exc)
            logger.warning("Rear preview camera init failed: %s", exc)

    depth_aligned_to_primary = bool(
        primary_camera is not None
        and primary_camera is sidecar_camera
        and primary_cfg is not None
        and primary_cfg.get("type") == "realsense"
        and enable_depth
    )

    return CompositeSensorRig(
        primary_rgb_camera=primary_camera,
        primary_rgb_config=primary_cfg,
        primary_rgb_error=primary_error,
        sidecar_depth_imu_sensor=sidecar_camera,
        sidecar_depth_imu_config=sidecar_cfg,
        sidecar_depth_imu_error=sidecar_error,
        rear_preview_camera=rear_camera,
        rear_preview_config=rear_cfg,
        rear_preview_error=rear_error,
        depth_aligned_to_primary=depth_aligned_to_primary,
    )
