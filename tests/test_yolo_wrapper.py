import builtins
import sys
from pathlib import Path

import numpy as np


CLIENT_API_DIR = Path(__file__).resolve().parents[1] / "fleet" / "fleet_management_app" / "client_api"
if str(CLIENT_API_DIR) not in sys.path:
    sys.path.insert(0, str(CLIENT_API_DIR))

from models import ObjectDetector


class _FakeResult:
    def __init__(self):
        class _Box:
            cls = np.array([5], dtype=np.float32)
            xyxy = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
            conf = np.array([0.75], dtype=np.float32)

        self.boxes = [_Box()]


class _FakeModel:
    def __call__(self, frame, verbose=False):
        assert frame.shape == (8, 8, 3)
        return [_FakeResult()]


def test_object_detector_normalizes_detection_output():
    detector = ObjectDetector({"detection_model": "definitely_missing.pt"})
    detector.model = _FakeModel()

    detections = detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))

    assert detections == [
        {"class": 5, "bbox": [10.0, 20.0, 30.0, 40.0], "conf": 0.75}
    ]


def test_object_detector_gracefully_handles_missing_ultralytics(monkeypatch, tmp_path):
    weight_path = tmp_path / "weights.pt"
    weight_path.write_bytes(b"placeholder")

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ultralytics":
            raise ImportError("simulated missing ultralytics")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    detector = ObjectDetector({"detection_model": str(weight_path)})

    assert detector.model is None
    assert detector.detect(np.zeros((4, 4, 3), dtype=np.uint8)) == []
