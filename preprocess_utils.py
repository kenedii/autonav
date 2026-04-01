import cv2
import numpy as np


LEGACY_PREPROCESS_PROFILE = "legacy_resize_v0"
CAM0_FISHEYE_PREPROCESS_PROFILE = "cam0_fisheye_v1"
PREPROCESS_OUTPUT_WIDTH = 160
PREPROCESS_OUTPUT_HEIGHT = 120
SUPPORTED_PREPROCESS_PROFILES = {
    LEGACY_PREPROCESS_PROFILE,
    CAM0_FISHEYE_PREPROCESS_PROFILE,
}


def canonicalize_preprocess_profile(profile):
    if profile in SUPPORTED_PREPROCESS_PROFILES:
        return profile
    return LEGACY_PREPROCESS_PROFILE


def infer_preprocess_profile(camera_configs=None, explicit_profile=None):
    if explicit_profile:
        return canonicalize_preprocess_profile(explicit_profile)

    for config in camera_configs or []:
        role = str((config or {}).get("role") or "").strip().lower()
        camera_type = str((config or {}).get("type") or "").strip().lower()
        if role == "primary_rgb" and camera_type == "csi":
            return CAM0_FISHEYE_PREPROCESS_PROFILE
    return LEGACY_PREPROCESS_PROFILE


def _ensure_uint8_rgb(frame_rgb):
    if frame_rgb is None:
        return None
    if frame_rgb.dtype == np.uint8:
        return frame_rgb
    return np.clip(frame_rgb, 0, 255).astype(np.uint8)


def _apply_legacy_resize(frame_rgb):
    frame_rgb = _ensure_uint8_rgb(frame_rgb)
    if frame_rgb is None:
        return None
    if frame_rgb.shape[1] == PREPROCESS_OUTPUT_WIDTH and frame_rgb.shape[0] == PREPROCESS_OUTPUT_HEIGHT:
        return frame_rgb.copy()
    return cv2.resize(frame_rgb, (PREPROCESS_OUTPUT_WIDTH, PREPROCESS_OUTPUT_HEIGHT))


def _apply_cam0_fisheye_v1(frame_rgb):
    frame_rgb = _ensure_uint8_rgb(frame_rgb)
    if frame_rgb is None:
        return None

    height = frame_rgb.shape[0]
    crop_start = max(0, int(height * 0.30))
    cropped = frame_rgb[crop_start:, :, :]
    if cropped.size == 0:
        cropped = frame_rgb
    return cv2.resize(cropped, (PREPROCESS_OUTPUT_WIDTH, PREPROCESS_OUTPUT_HEIGHT))


def apply_preprocess_profile(frame_rgb, profile):
    profile = canonicalize_preprocess_profile(profile)
    if profile == CAM0_FISHEYE_PREPROCESS_PROFILE:
        return _apply_cam0_fisheye_v1(frame_rgb)
    return _apply_legacy_resize(frame_rgb)
