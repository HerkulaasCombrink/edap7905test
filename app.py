import streamlit as st
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import tempfile
import os
import pandas as pd

st.set_page_config(page_title="🧠 Hand Signal Detector", layout="wide")
st.title("🤖 Full-Frame Hand Signal Detection in Video")

# --- Upload model and video ---
uploaded_model = st.file_uploader("📥 Upload your trained model (.pth)", type=["pth"])
uploaded_video = st.file_uploader("🎥 Upload a video to analyze", type=["mp4", "mov", "avi"])

# --- CNN model definition (must match training) ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# --- Set your class names here ---
classes = ["signal_1"]  # Add more classes if needed
NUM_CLASSES = len(classes)

# --- Inference settings ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# --- Run detection ---
if uploaded_model and uploaded_video and st.button("🚀 Run Full-Frame Detection"):
    with st.spinner("Processing video..."):

        # Save uploaded files to temp paths
        model_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pth").name
        with open(model_path, "wb") as f:
            f.write(uploaded_model.read())

        video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup output video writer
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Load model
        model = SimpleCNN(NUM_CLASSES)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()

        # Store predictions
        frame_log = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Convert full frame to PIL, then preprocess
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

            # Predict
            with torch.no_grad():
                output = model(input_tensor)
                _, pred = torch.max(output, 1)
                label = classes[pred.item()]

            # Log frame and prediction
            frame_log.append({"frame": frame_idx, "prediction": label})

            # Annotate frame
            cv2.putText(frame, f"Frame {frame_idx}: {label}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            out.write(frame)

        cap.release()
        out.release()

    st.success("✅ Detection complete!")

    # --- Download annotated video ---
    with open(output_path, "rb") as f:
        st.download_button("📥 Download Annotated Video", f, file_name="annotated_video.mp4")

    # --- Show frame log as table ---
    st.write("🧾 Detection Log")
    df_log = pd.DataFrame(frame_log)
    st.dataframe(df_log)

    # --- Download detection log as CSV ---
    csv = df_log.to_csv(index=False).encode("utf-8")
    st.download_button("📄 Download Detection Log (.csv)", data=csv, file_name="frame_predictions.csv", mime="text/csv")
