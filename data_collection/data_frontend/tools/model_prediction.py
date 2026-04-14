# tools/model_prediction.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import os

# ------------------- Model Definition (same as training/inference) -------------------
ARCH_TO_DIM = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
}

class ControlModel(nn.Module):
    def __init__(self, arch: str):
        super().__init__()
        arch = arch.lower()
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

# ------------------- Preprocessing -------------------
def preprocess_image(pil_img):
    img = pil_img.convert("RGB").resize((160, 120))
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor

# ------------------- Page -------------------
def show():
    st.title("Model Prediction")
    st.markdown("### Upload an image and your trained model → get instant steering prediction")

    col1, col2 = st.columns(2)

    with col1:
        architecture = st.selectbox(
            "Model Architecture",
            options=["resnet18", "resnet34", "resnet50", "resnet101"],
            index=0
        )

        weights_file = st.file_uploader("Upload your trained `.pth` file", type=["pth"])

        image_file = st.file_uploader("Upload image to predict", type=["jpg", "jpeg", "png"])

    with col2:
        prediction_mode = st.radio("Run prediction via:", ["Local (PyTorch)", "FastAPI Server"])

        if prediction_mode == "FastAPI Server":
            api_url = st.text_input(
                "FastAPI Endpoint",
                value="http://127.0.0.1:8000/predict",
                help="Make sure your FastAPI server is running"
            )
        else:
            api_url = None

    if st.button("Run Prediction", type="primary", use_container_width=True):
        if not image_file:
            st.error("Please upload an image.")
            return
        if not weights_file:
            st.error("Please upload a model weights file (.pth).")
            return

        pil_img = Image.open(image_file)

        with st.spinner("Loading model and running inference..."):
            if prediction_mode == "Local (PyTorch)":
                # --- Local inference ---
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = ControlModel(architecture).to(device)
                model.eval()

                # Load weights
                weights_bytes = weights_file.getvalue()
                buffer = BytesIO(weights_bytes)
                ckpt = torch.load(buffer, map_location=device)
                state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                model.load_state_dict(state_dict)

                # Preprocess + predict
                input_tensor = preprocess_image(pil_img).to(device)
                with torch.no_grad():
                    pred = model(input_tensor).item()

            else:
                # --- FastAPI inference ---
                files = {
                    "image": (image_file.name, image_file.getvalue(), image_file.type),
                    "model_file": (weights_file.name, weights_file.getvalue(), "application/octet-stream"),
                }
                data = {"architecture": architecture}

                try:
                    response = requests.post(api_url, files=files, data=data, timeout=30)
                    if response.status_code != 200:
                        st.error(f"API Error {response.status_code}: {response.text}")
                        return
                    result = response.json()
                    pred = result["steering_normalized"]
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to reach FastAPI server: {e}")
                    return

            # --- Display results ---
            direction = "LEFT" if pred > 0.05 else "RIGHT" if pred < -0.05 else "STRAIGHT"
            confidence = abs(pred)

            st.success("Prediction Complete!")
            colA, colB = st.columns([1, 1])
            with colA:
                st.image(pil_img, caption="Input Image", use_column_width=True)
            with colB:
                st.metric("Steering Output (normalized)", f"{pred:+.4f}")
                st.write(f"**Direction:** {direction}")
                st.progress(confidence)
                st.write(f"**Confidence:** {confidence:.3f}")

                if prediction_mode == "FastAPI Server":
                    st.caption(f"via {api_url}")

    st.markdown("---")
    st.info("""
    **Tip:** Use the FastAPI version (`api_predict_upload.py`)for remote/server deployment.  
    Local mode works great on Jetson Nano or laptop with GPU.
    """)
