import os
import sys
import time
import json
import argparse
import tempfile
import multiprocessing as mp

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
            raise ValueError("Unsupported architecture: %s" % architecture)

        # use backbone up to the last conv block (exclude avgpool & fc)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
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


def export_onnx(model, onnx_path, input_shape=(1, 3, 120, 160), device='cpu'):
    model.eval()
    dummy = torch.ones(input_shape, device=device)
    os.makedirs(os.path.dirname(onnx_path) or '.', exist_ok=True)
    torch.onnx.export(model, dummy, onnx_path,
                      input_names=['input'], output_names=['output'],
                      opset_version=11, verbose=False)
    print(f"[OK] ONNX exported: {onnx_path}")


def _rknn_build_worker(onnx_path, rknn_path, target_platform, use_fp16, status_path):
    """
    Child process worker: import RKNN, load ONNX, config, build, export.
    Writes status JSON to status_path.
    """
    status = {'ok': False, 'step': None, 'time': time.time(), 'error': None}
    try:
        # import here so child process loads toolkit separately
        from rknn.api import RKNN
    except Exception as e:
        status.update(step='import', error=str(e))
        with open(status_path, 'w') as f:
            json.dump(status, f)
        return

    try:
        status['step'] = 'init'
        rknn = RKNN()

        # IMPORTANT: some RKNN toolkit versions require config BEFORE load_onnx.
        status['step'] = 'config'
        # Some RKNN toolkit versions expect specific quantized_dtype values
        # The user requested FP16; many toolkits will select the best internal format
        # if quantized_dtype is omitted. To maximize compatibility we avoid passing
        # an unsupported quantized_dtype string and instead only set the target platform.
        if use_fp16:
            rknn.config(mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]],
                        target_platform=target_platform)
        else:
            # leave quantized_dtype unset for default behavior
            rknn.config(mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]],
                        target_platform=target_platform)

        status['step'] = 'load_onnx'
        ret = rknn.load_onnx(onnx_path)
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx returned {ret}")

        status['step'] = 'build'
        ret = rknn.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError(f"rknn.build returned {ret}")

        status['step'] = 'export'
        rknn.export_rknn(rknn_path)
        rknn.release()

        status.update(ok=True, step='done', time=time.time())
    except Exception as e:
        status.update(error=str(e), time=time.time())
    finally:
        try:
            with open(status_path, 'w') as f:
                json.dump(status, f)
        except Exception:
            pass


def build_rknn_with_timeout(onnx_path, rknn_path, target_platform='rk3588', use_fp16=True, timeout=None):
    """
    Launch RKNN build in a child process and wait for completion or timeout (seconds).
    Returns status dict read from child process.
    """
    os.makedirs(os.path.dirname(rknn_path) or '.', exist_ok=True)
    fd, status_path = tempfile.mkstemp(prefix='rknn_status_', suffix='.json')
    os.close(fd)

    p = mp.Process(target=_rknn_build_worker, args=(onnx_path, rknn_path, target_platform, use_fp16, status_path))
    p.start()
    print(f"[INFO] RKNN build started (pid={p.pid}), timeout={timeout}s")
    p.join(timeout)
    if p.is_alive():
        print("[WARN] RKNN build exceeded timeout, terminating child process...")
        p.terminate()
        p.join(5)

    # read status file if present
    status = {'ok': False, 'error': 'no status file'}
    try:
        if os.path.exists(status_path):
            with open(status_path, 'r') as f:
                status = json.load(f)
    except Exception as e:
        status = {'ok': False, 'error': f'failed to read status file: {e}'}
    finally:
        try:
            os.remove(status_path)
        except Exception:
            pass

    return status


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pytorch-path", required=True, help="Path to best_model.pth")
    p.add_argument("--onnx-path", default="checkpoints/model_rknn/model.onnx")
    p.add_argument("--rknn-path", default="checkpoints/model_rknn/model.rknn")
    p.add_argument("--arch", default="resnet101")
    p.add_argument("--fp16", action="store_true", help="Build FP16 engine (recommended)")
    p.add_argument("--timeout", type=int, default=3600, help="Max seconds to wait for RKNN build (default 3600)")
    args = p.parse_args()

    device = torch.device("cpu")
    print("[INFO] Loading PyTorch checkpoint:", args.pytorch_path)
    model = ControlModel(architecture=args.arch).to(device)

    ckpt = torch.load(args.pytorch_path, map_location=device)
    # handle multiple checkpoint styles
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state = ckpt['state_dict']
        else:
            state = ckpt
    else:
        state = ckpt

    # If checkpoint keys are prefixed (module.), try to remove prefix automatically
    try:
        model.load_state_dict(state)
    except Exception as e1:
        # try stripping common 'module.' prefix
        new_state = {}
        for k, v in state.items():
            new_state[k.replace('module.', '')] = v
        try:
            model.load_state_dict(new_state)
        except Exception as e2:
            print("[ERROR] Failed to load state_dict:", e1, "and retry:", e2)
            sys.exit(1)

    model.eval()
    export_onnx(model, args.onnx_path, device=device)

    # Optionally test ONNX with onnxruntime (quick smoke test)
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.onnx_path, providers=['CPUExecutionProvider'])
        print("[OK] ONNX runtime test: session created")
        # run one inference
        inp = torch.ones((1, 3, 120, 160)).numpy().astype('float32')
        out = sess.run(None, {'input': inp})
        print("[OK] ONNX smoke-run OK, output shape:", [o.shape for o in out])
    except Exception:
        print("[WARN] onnxruntime not available or ONNX smoke-run failed; continuing to RKNN build")

    # Run RKNN build in child process with timeout
    status = build_rknn_with_timeout(args.onnx_path, args.rknn_path, target_platform='rk3588', use_fp16=args.fp16, timeout=args.timeout)
    print("[RESULT]", json.dumps(status, indent=2))
    if not status.get('ok'):
        print("[ERROR] RKNN build failed or timed out. See status JSON above.")
        sys.exit(1)
    print("[OK] RKNN build completed successfully, file:", args.rknn_path)


if __name__ == "__main__":
    main()
