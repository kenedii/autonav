import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import os
import cv2
import re


SUPPORTED_ARCHITECTURES = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "resnet152": 2048,
}


def _normalize_architecture(architecture):
    architecture = (architecture or "resnet18").lower()
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError("Unsupported architecture: %s" % architecture)
    return architecture


def _torchvision_kwargs():
    kwargs = {}
    import torchvision
    try:
        from packaging.version import Version
    except ImportError:
        Version = None

    if Version is not None and Version(torchvision.__version__) >= Version("0.13.0"):
        kwargs["weights"] = None
    else:
        kwargs["pretrained"] = False
    return kwargs


def _build_backbone(architecture):
    architecture = _normalize_architecture(architecture)
    backbone = getattr(models, architecture)(**_torchvision_kwargs())
    return backbone, SUPPORTED_ARCHITECTURES[architecture]


def _strip_module_prefix(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        normalized[key[7:] if key.startswith("module.") else key] = value
    return normalized


def unwrap_checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format: %s" % type(checkpoint))
    return _strip_module_prefix(checkpoint)


def infer_control_model_layout(state_dict):
    if "1.8.weight" in state_dict:
        return "sequential"
    if "head.6.weight" in state_dict:
        return "named"
    raise ValueError("Unable to infer model layout from checkpoint keys")


def infer_control_output_dim(state_dict, layout=None):
    layout = layout or infer_control_model_layout(state_dict)
    output_key = "1.8.weight" if layout == "sequential" else "head.6.weight"
    return int(state_dict[output_key].shape[0])


def _block_counts_from_state_dict(state_dict, layout):
    prefix = "0" if layout == "sequential" else "features"
    pattern = re.compile(r"^%s\.(4|5|6|7)\.(\d+)\." % re.escape(prefix))
    counts = {4: 0, 5: 0, 6: 0, 7: 0}
    for key in state_dict.keys():
        match = pattern.match(key)
        if not match:
            continue
        layer_idx = int(match.group(1))
        block_idx = int(match.group(2)) + 1
        counts[layer_idx] = max(counts[layer_idx], block_idx)
    return [counts[idx] for idx in (4, 5, 6, 7)]


def infer_control_architecture(state_dict, layout=None):
    layout = layout or infer_control_model_layout(state_dict)
    layer_counts = _block_counts_from_state_dict(state_dict, layout)
    if not any(layer_counts):
        return None

    uses_bottleneck = any(".conv3.weight" in key for key in state_dict.keys())
    if uses_bottleneck:
        stage3_blocks = layer_counts[2]
        if stage3_blocks >= 36:
            return "resnet152"
        if stage3_blocks >= 23:
            return "resnet101"
        return "resnet50"

    if layer_counts == [2, 2, 2, 2]:
        return "resnet18"
    if layer_counts == [3, 4, 6, 3]:
        return "resnet34"
    return "resnet34" if layer_counts[2] > 2 else "resnet18"


def is_autonav_checkpoint_path(model_path):
    return "autonav-v" in (model_path or "").lower()


def default_invert_steering(model_path):
    return is_autonav_checkpoint_path(model_path)


def default_throttle_output_scale(model_path, output_dim):
    model_path = (model_path or "").lower()
    if "autonav-v2" in model_path and (output_dim is None or output_dim >= 2):
        return 3.33
    return 1.0


def build_control_model(architecture="resnet18", num_outputs=1, layout="named"):
    backbone, feature_dim = _build_backbone(architecture)
    if layout == "sequential":
        features = nn.Sequential(*list(backbone.children())[:-2])
        head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_outputs),
            nn.Tanh(),
        )
        return nn.Sequential(features, head)
    return ControlModel(architecture=architecture, num_outputs=num_outputs)

# Wrapper for RKNN
class RKNNWrapper:
    def __init__(self, model_path, target='rk3588'):
        try:
            from rknnlite.api import RKNNLite
        except ImportError:
            from rknn.api import RKNN as RKNNLite

        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError('Load RKNN model failed!')
            
        ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError('Init runtime environment failed!')

    def __call__(self, x):
        # x is [1, 3, H, W]
        outputs = self.rknn.inference(inputs=[x])
        return torch.from_numpy(outputs[0])

# PyTorch ResNet definition matching training
class ControlModel(nn.Module):
    def __init__(self, architecture='resnet18', num_outputs=1):
        super().__init__()
        architecture = _normalize_architecture(architecture)
        backbone, feature_dim = _build_backbone(architecture)

        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, num_outputs),   nn.Tanh()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.head(x)

class AutonomousDriver:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        self.backend = config.get('control_model_type', 'pytorch')
        self.model_path = config.get('control_model')
        self.architecture = config.get('architecture', 'resnet18')
        self.model_layout = "named"
        self.output_dim = None
        self.invert_steering = config.get('invert_steering')
        self.throttle_output_scale = config.get('throttle_output_scale')

        self.model = self._load_model()
        if self.invert_steering is None:
            self.invert_steering = default_invert_steering(self.model_path)
        if self.throttle_output_scale is None:
            self.throttle_output_scale = default_throttle_output_scale(self.model_path, self.output_dim)

    def _prediction_to_array(self, output):
        if hasattr(output, "detach"):
            output = output.detach().cpu().numpy()
        array = np.asarray(output, dtype=np.float32).reshape(-1)
        return array

    def _format_prediction(self, output):
        values = self._prediction_to_array(output)
        if values.size == 0:
            raise RuntimeError("Model returned an empty prediction")

        steering = float(values[0])
        if self.invert_steering:
            steering *= -1.0

        prediction = {
            "steering": steering,
            "throttle": None,
            "raw_outputs": [float(v) for v in values.tolist()],
        }
        if values.size >= 2:
            throttle = float(values[1]) * float(self.throttle_output_scale)
            prediction["throttle"] = float(np.clip(throttle, -1.0, 1.0))
        return prediction

    def _load_model(self):
        print(f"[Model] Loading {self.backend} model from: {self.model_path}")
        if not os.path.exists(self.model_path):
            print(f"[Model] Error: File does not exist at {self.model_path}")
            return None

        if self.backend == 'rockchip':
            return RKNNWrapper(self.model_path)
        
        elif self.backend == 'tensorrt':
            try:
                from torch2trt import TRTModule
                model = TRTModule()
                # TRT models on Jetson often load directly with torch.load if they are TRTModules
                model.load_state_dict(torch.load(self.model_path))
                return model
            except Exception as e:
                print(f"TensorRT load failed: {e}. Falling back to PyTorch...")
                self.backend = 'pytorch'
                return self._load_model()
            
        else: # pytorch
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                state_dict = unwrap_checkpoint_state_dict(checkpoint)
                self.model_layout = infer_control_model_layout(state_dict)
                self.output_dim = infer_control_output_dim(state_dict, self.model_layout)
                inferred_architecture = infer_control_architecture(state_dict, self.model_layout)
                if inferred_architecture and inferred_architecture != self.architecture:
                    print(
                        "[Model] Inferred checkpoint architecture %s (configured: %s)"
                        % (inferred_architecture, self.architecture)
                    )
                    self.architecture = inferred_architecture
                if self.invert_steering is None:
                    self.invert_steering = default_invert_steering(self.model_path)
                if self.throttle_output_scale is None:
                    self.throttle_output_scale = default_throttle_output_scale(self.model_path, self.output_dim)

                model = build_control_model(
                    architecture=self.architecture,
                    num_outputs=self.output_dim,
                    layout=self.model_layout,
                )
                model.to(self.device)
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"[Model] Error loading state dict: {e}")
                return None

            model.eval()
            return model

    def predict(self, frame):
        # Preprocess
        # Assuming frame is RGB HxW
        # Speed up: only resize if dimensions differ from typical input
        h, w = frame.shape[:2]
        if w != 160 or h != 120:
             img = cv2.resize(frame, (160, 120))
        else:
             img = frame

        img = img.transpose(2, 0, 1) # HWC -> CHW
        
        if self.backend == 'rockchip':
            img = img.astype(np.float32) / 255.0
            input_tensor = img.reshape(1, 3, 120, 160)
            output = self.model(input_tensor)
            return self._format_prediction(output)
        
        else:
            # pin_memory or other torch optimizations if needed, but keeping it simple for now
            tensor = torch.from_numpy(img).float().div(255.0)
            input_tensor = tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(input_tensor)
            return self._format_prediction(output)

class ObjectDetector:
    def __init__(self, config):
        self.model_path = config.get('detection_model')
        self.conf_threshold = float(config.get('yolo_confidence_threshold', 0.25))
        self.iou_threshold = float(config.get('yolo_iou_threshold', 0.45))
        self.max_detections = int(config.get('yolo_max_detections', 100))
        # Placeholder for YOLOv8
        # In a real scenario, we might use 'ultralytics'
        self.model = None
        self.class_names = {}
        try:
            from ultralytics import YOLO
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                self.class_names = dict(getattr(self.model.model, "names", {}))
        except ImportError:
            print("Ultralytics YOLO not installed, detection will be dummy.")

    def detect(self, frame):
        if self.model:
            results = self.model(
                frame,
                verbose=False,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
            )
            # Process results to return standard format
            # e.g. list of {"class": "stop_sign", "bbox": [x1, y1, x2, y2], "conf": 0.9}
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0].item())
                    detections.append({
                        "class": class_id,
                        "label": self.class_names.get(class_id, str(class_id)),
                        "bbox": box.xyxy[0].tolist(),
                        "conf": box.conf[0].item()
                    })
            return detections
        return []
