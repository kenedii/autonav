# trt_optimize.py
# Run this ONCE on your Jetson Nano to convert your trained model to TensorRT for faster inference. 

import os
import torch
import torch.nn as nn
import argparse
from torchvision import models

parser = argparse.ArgumentParser(description='Optimize PyTorch model to TensorRT')
parser.add_argument('--exp', type=int, default=1, choices=[1,2,3,4,5,6,7,8], help='Experiment ID (1-8)')
parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], help='Model architecture')
args = parser.parse_args()

# ------------------------------------------------------------
# USER CONFIG
# ------------------------------------------------------------
EXPERIMENT_ID = args.exp
MODEL_ARCHITECTURE = args.arch

EXPERIMENTS = {
    1: ['rgb_path', 'cam1_path', 'ir_path', 'depth_path'],
    2: ['rgb_path', 'ir_path', 'depth_path'],
    3: ['rgb_path'],
    4: ['rgb_path', 'cam1_path'],
    5: ['rgb_path', 'cam1_path', 'ir_path', 'depth_path'],
    6: ['rgb_path', 'cam1_path'],
    7: ['rgb_path', 'cam1_path', 'ir_path'],
    8: ['rgb_path', 'ir_path']
}

features = EXPERIMENTS[EXPERIMENT_ID]
IN_CHANNELS = sum(1 if ('depth' in f or 'ir' in f) else 3 for f in features)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_NAME = f"exp{EXPERIMENT_ID}_{MODEL_ARCHITECTURE}"
PYTORCH_MODEL_PATH = os.path.join(BASE_DIR, "best_model.pth")
TRT_MODEL_PATH     = os.path.join(BASE_DIR, "best_model_trt.pth")

# ------------------------------------------------------------
# 1. Install torch2trt automatically if not present
# ------------------------------------------------------------
try:
    from torch2trt import torch2trt, TRTModule
    print("[OK] torch2trt already installed")
except ImportError:
    print("[INFO] torch2trt not found → installing from source...")
    os.system("cd ~ && git clone https://github.com/NVIDIA-AI-IOT/torch2trt")
    os.system("cd ~/torch2trt && sudo python3 setup.py install")
    from torch2trt import torch2trt, TRTModule
    print("[OK] torch2trt installed successfully!")

# ------------------------------------------------------------
# 2. Exact same model class as in your training script — now dynamic
# ------------------------------------------------------------
def get_resnet_model(arch_name, in_channels):
    if arch_name == 'resnet18':
        model = models.resnet18(pretrained=False)
        out_features = 512
    elif arch_name == 'resnet34':
        model = models.resnet34(pretrained=False)
        out_features = 512
    elif arch_name == 'resnet50':
        model = models.resnet50(pretrained=False)
        out_features = 2048
    elif arch_name == 'resnet101':
        model = models.resnet101(pretrained=False)
        out_features = 2048
    elif arch_name == 'resnet152':
        model = models.resnet152(pretrained=False)
        out_features = 2048
    else:
        raise ValueError(f"Unsupported architecture: {arch_name}")
        
    # Modify first layer if in_channels != 3
    if in_channels != 3:
        original_conv = model.conv1
        model.conv1 = nn.Conv2d(in_channels, original_conv.out_channels, 
                                kernel_size=original_conv.kernel_size, 
                                stride=original_conv.stride, 
                                padding=original_conv.padding, 
                                bias=False)
        nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
        
    features = nn.Sequential(*list(model.children())[:-2])
    head = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(out_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, 2),  # 2 values for Steering and Throttle
        nn.Tanh()
    )
    
    return nn.Sequential(features, head)

# ------------------------------------------------------------
# 4. Load model + weights
# ------------------------------------------------------------
device = torch.device("cuda")

print(f"[INFO] Loading PyTorch model ({MODEL_ARCHITECTURE}) from: {PYTORCH_MODEL_PATH}")
model = get_resnet_model(MODEL_ARCHITECTURE, IN_CHANNELS).to(device)

# Handle both plain state_dict and dict with 'model_state_dict'
ckpt = torch.load(PYTORCH_MODEL_PATH, map_location=device)
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt['model_state_dict'])
else:
    model.load_state_dict(ckpt)

model.eval()
print("[OK] Model loaded and set to eval mode")

# ------------------------------------------------------------
# 5. Create dummy input (must match inference size)
# ------------------------------------------------------------
dummy_input = torch.ones((1, IN_CHANNELS, 120, 160)).to(device)

# ------------------------------------------------------------
# 6. Convert to TensorRT (FP16 = huge speed boost on Nano)
# ------------------------------------------------------------
print("[INFO] Converting to TensorRT with FP16... (30–90 seconds)")
model_trt = torch2trt(
    model,
    [dummy_input],
    fp16_mode=True,           # ← critical for speed on Jetson Nano
    max_workspace_size=1 << 25,  # 512MB workspace
    use_onnx=False
)

# ------------------------------------------------------------
# 7. Save the optimized engine
# ------------------------------------------------------------
os.makedirs(os.path.dirname(TRT_MODEL_PATH), exist_ok=True)
torch.save(model_trt.state_dict(), TRT_MODEL_PATH)
print(f"[SUCCESS] TensorRT engine saved → {TRT_MODEL_PATH}")
print("")
print("You can now run your autonomous script — it will automatically use the fast TRT version!")
