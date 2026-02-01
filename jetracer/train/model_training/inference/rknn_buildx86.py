import os
import argparse
import torch
import torch.nn as nn
from torchvision import models

# ControlModel matching run_autonomous_resnet.py
class ControlModel(nn.Module):
    def __init__(self, architecture='resnet101'):
        super().__init__()
        if architecture == 'resnet18':
            backbone = models.resnet18(pretrained=False)
            feature_dim = 512
        elif architecture == 'resnet34':
            backbone = models.resnet34(pretrained=False)
            feature_dim = 512
        elif architecture == 'resnet50':
            backbone = models.resnet50(pretrained=False)
            feature_dim = 2048
        elif architecture == 'resnet101':
            backbone = models.resnet101(pretrained=False)
            feature_dim = 2048
        else:
            raise ValueError("Unsupported architecture")
        # take all layers except avgpool and fc to match run_autonomous_resnet
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, 1), nn.Tanh()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.head(x)

def export_onnx(model, onnx_path, input_shape=(1,3,120,160), device='cpu'):
    model.eval()
    dummy = torch.ones(input_shape, device=device)
    os.makedirs(os.path.dirname(onnx_path) or '.', exist_ok=True)
    torch.onnx.export(model, dummy, onnx_path,
                      input_names=['input'], output_names=['output'],
                      opset_version=11)
    print(f"[OK] ONNX exported: {onnx_path}")

def build_rknn(onnx_path, rknn_path, target_platform='rk3588', use_fp16=True):
    try:
        from rknn.api import RKNN
    except Exception as e:
        print("ERROR: rknn toolkit not found in this environment:", e)
        print("Install RKNN Toolkit (x86) before running this script.")
        raise

    rknn = RKNN()
    print("[INFO] Loading ONNX...")
    ret = rknn.load_onnx(onnx_path)
    if ret != 0:
        raise RuntimeError(f"rknn.load_onnx failed with code {ret}")

    print("[INFO] Configuring RKNN (target_platform=%s, dtype=%s) ..." %
          (target_platform, 'fp16' if use_fp16 else 'fp32'))
    rknn.config(mean_values=[[0,0,0]], std_values=[[1,1,1]],
                target_platform=target_platform,
                quantized_dtype='fp16' if use_fp16 else 'fp32')

    print("[INFO] Building RKNN model (this may take a while)...")
    # For FP16 build, do_quantization=False is fine. For INT8 quant, pass dataset and do_quantization=True.
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        raise RuntimeError(f"rknn.build failed with code {ret}")

    rknn.export_rknn(rknn_path)
    print(f"[OK] RKNN exported: {rknn_path}")
    rknn.release()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pytorch-path", required=True, help="Path to best_model.pth")
    p.add_argument("--onnx-path", default="checkpoints/model_rknn/model.onnx")
    p.add_argument("--rknn-path", default="checkpoints/model_rknn/model.rknn")
    p.add_argument("--arch", default="resnet101")
    p.add_argument("--fp16", action="store_true", help="Build FP16 engine (recommended)")
    args = p.parse_args()

    device = torch.device("cpu")
    print("[INFO] Loading PyTorch checkpoint:", args.pytorch_path)
    model = ControlModel(architecture=args.arch).to(device)
    ckpt = torch.load(args.pytorch_path, map_location=device)
    state = ckpt if not isinstance(ckpt, dict) else ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model.eval()

    export_onnx(model, args.onnx_path, device=device)
    build_rknn(args.onnx_path, args.rknn_path, use_fp16=args.fp16)

if __name__ == "__main__":
    main()
