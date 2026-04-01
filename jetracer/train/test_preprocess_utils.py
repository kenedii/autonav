import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocess_utils import (
    CAM0_FISHEYE_PREPROCESS_PROFILE,
    LEGACY_PREPROCESS_PROFILE,
    apply_preprocess_profile,
    canonicalize_preprocess_profile,
    infer_preprocess_profile,
    PREPROCESS_OUTPUT_HEIGHT,
    PREPROCESS_OUTPUT_WIDTH,
)


def test_preprocess_profiles_are_versioned_and_fixed_size():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    frame[3:, :, :] = 255

    legacy = apply_preprocess_profile(frame, LEGACY_PREPROCESS_PROFILE)
    cam0 = apply_preprocess_profile(frame, CAM0_FISHEYE_PREPROCESS_PROFILE)

    assert legacy.shape == (PREPROCESS_OUTPUT_HEIGHT, PREPROCESS_OUTPUT_WIDTH, 3)
    assert cam0.shape == (PREPROCESS_OUTPUT_HEIGHT, PREPROCESS_OUTPUT_WIDTH, 3)
    assert canonicalize_preprocess_profile("unknown-profile") == LEGACY_PREPROCESS_PROFILE
    assert infer_preprocess_profile([{"role": "primary_rgb", "type": "csi"}]) == CAM0_FISHEYE_PREPROCESS_PROFILE
    assert cam0.mean() > legacy.mean()
