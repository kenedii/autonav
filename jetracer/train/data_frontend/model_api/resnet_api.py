# api_predict_upload.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import uvicorn
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import io
import os
import zipfile
import shutil
import numpy as np
from typing import Optional

app = FastAPI(
    title="JetRacer Steering Prediction API (with Model Upload)",
    description="Upload both an image and your trained .pth model → get instant steering prediction",
    version="2.0"
)

@app.post("/upload_dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a zip file containing dataset folders.
    Unzips to ./data/
    """
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="File must be a .zip")
    
    try:
        # Create data directory if it doesn't exist
        extract_root = os.path.join(os.getcwd(), "data")
        os.makedirs(extract_root, exist_ok=True)
        
        # Read zip file content
        content = await file.read()
        with zipfile.ZipFile(io.BytesIO(content)) as zip_ref:
            zip_ref.extractall(extract_root)
            
        return {
            "message": "Dataset uploaded and extracted successfully",
            "extract_path": extract_root,
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process zip file: {str(e)}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mapping: architecture → feature dimension after backbone
ARCH_TO_DIM = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
}

class ControlModel(nn.Module):
    def __init__(self, architecture: str):
        super().__init__()
        arch = architecture.lower()
        if arch not in ARCH_TO_DIM:
            raise ValueError(f"Unsupported architecture: {arch}")

        if arch == "resnet18":
            backbone = models.resnet18(pretrained=False)
        elif arch == "resnet34":
            backbone = models.resnet34(pretrained=False)
        elif arch == "resnet50":
            backbone = models.resnet50(pretrained=False)
        elif arch == "resnet101":
            backbone = models.resnet101(pretrained=False)

        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(ARCH_TO_DIM[arch], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


def preprocess_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((160, 120))
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return tensor


@app.get("/")
def root():
    return {"message": "JetRacer Prediction API — Upload model + image!"}


@app.post("/predict")
async def predict(
    image: UploadFile = File(..., description="Input image (JPG/PNG)"),
    model_file: UploadFile = File(..., description="Your trained .pth model (best_model.pth)"),
    architecture: Optional[str] = Form("resnet18", description="Model architecture if not auto-detectable")
):
    """
    Upload both a model file and an image → get steering prediction instantly.
    """
    # --- Validate image ---
    if image.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        raise HTTPException(status_code=400, detail="Image must be JPG or PNG")

    # --- Validate model file ---
    if not model_file.filename.endswith(".pth"):
        raise HTTPException(status_code=400, detail="Model file must be a .pth file")

    try:
        # Read files
        image_bytes = await image.read()
        model_bytes = await model_file.read()

        # Preprocess image
        input_tensor = preprocess_image(image_bytes)

        # Load model from uploaded bytes
        model_buffer = io.BytesIO(model_bytes)
        ckpt = torch.load(model_buffer, map_location=DEVICE)

        # Extract state_dict
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt

        # Build model with specified or fallback architecture
        model = ControlModel(architecture).to(DEVICE)
        model.load_state_dict(state_dict)
        model.eval()

        # Run inference
        with torch.no_grad():
            pred = model(input_tensor)
            steering = pred.item()

        direction = "left" if steering > -0.05 else "right" if steering < 0.05 else "straight"

        return JSONResponse({
            "filename": image.filename,
            "model_used": model_file.filename,
            "architecture": architecture,
            "steering_normalized": round(steering, 5),
            "direction": direction,
            "confidence": round(abs(steering), 3)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")


@app.get("/health")
def health():
    return {"status": "healthy", "device": str(DEVICE), "supported_architectures": list(ARCH_TO_DIM.keys())}


if __name__ == "__main__":
    uvicorn.run("fastapi_resnet:app", host="0.0.0.0", port=8000, reload=True)
