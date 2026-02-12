import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import os
import cv2

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
    def __init__(self, architecture='resnet18'):
        super().__init__()
        
        # Compatibility for older torchvision (Jetson Nano / Python 3.6)
        kwargs = {}
        import torchvision
        from pkg_resources import parse_version
        if parse_version(torchvision.__version__) >= parse_version("0.13.0"):
            kwargs['weights'] = None
        else:
            kwargs['pretrained'] = False

        if architecture == 'resnet18':
            backbone = models.resnet18(**kwargs)
            feature_dim = 512
        elif architecture == 'resnet101':
            backbone = models.resnet101(**kwargs)
            feature_dim = 2048
        # Add others as needed
        else:
             # Fallback
             backbone = models.resnet18(**kwargs)
             feature_dim = 512
             
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, 1),   nn.Tanh()
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
        
        self.model = self._load_model()
        
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
            model = ControlModel(self.architecture)
            model.to(self.device)
            # Load weights
            if os.path.exists(self.model_path):
                try:
                    ckpt = torch.load(self.model_path, map_location=self.device)
                    model.load_state_dict(ckpt if not isinstance(ckpt, dict) else ckpt.get('model_state_dict', ckpt))
                except Exception as e:
                    print(f"[Model] Error loading state dict: {e}")
                    return None
            else:
                print(f"[Model] Model file not found: {self.model_path}")
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
            return output.item()
        
        else:
            # pin_memory or other torch optimizations if needed, but keeping it simple for now
            tensor = torch.from_numpy(img).float().div(255.0)
            input_tensor = tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(input_tensor)
            
            # Prediction -1 to 1
            res = output.item()
            return res

class ObjectDetector:
    def __init__(self, config):
        self.model_path = config.get('detection_model')
        # Placeholder for YOLOv8
        # In a real scenario, we might use 'ultralytics'
        self.model = None
        try:
            from ultralytics import YOLO
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
        except ImportError:
            print("Ultralytics YOLO not installed, detection will be dummy.")

    def detect(self, frame):
        if self.model:
            results = self.model(frame, verbose=False)
            # Process results to return standard format
            # e.g. list of {"class": "stop_sign", "bbox": [x1, y1, x2, y2], "conf": 0.9}
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    detections.append({
                        "class": int(box.cls[0].item()),
                        "bbox": box.xyxy[0].tolist(),
                        "conf": box.conf[0].item()
                    })
            return detections
        return []
