import streamlit as st
import cv2
import tempfile
import os
import zipfile
from PIL import Image
import numpy as np
import pandas as pd

st.set_page_config(page_title="Hand Signal Dataset Generator", layout="wide")
st.title("🖼️ Hand Signal Image Generator (Slider + Preview)")

# Step 1: Upload video
uploaded_video = st.file_uploader("📤 Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    # Read video using OpenCV
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    if not ret:
        st.error("❌ Could not read video.")
    else:
        # Convert frame to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.subheader("Step 1: Select Bounding Box")

        # Show the first frame
        st.image(frame_rgb, caption="📷 First Frame of Video", use_column_width=True)

        # Sliders to define bounding box
        img_h, img_w = frame.shape[:2]

        st.write("### Bounding Box Coordinates (sliders)")
        x = st.slider("🟥 X (left)", 0, img_w - 1, int(img_w * 0.25))
        y = st.slider("🟥 Y (top)", 0, img_h - 1, int(img_h * 0.25))
        w = st.slider("🟥 Width", 1, img_w - x, int(img_w * 0.5))
        h = st.slider("🟥 Height", 1, img_h - y, int(img_h * 0.5))

        st.session_state.coords = (x, y, w, h)
        st.info(f"📐 Box selected: x={x}, y={y}, w={w}, h={h}")

        # Show cropped preview
        cropped_preview = frame[y:y+h, x:x+w]
        if cropped_preview.size > 0:
            preview_resized = cv2.resize(cropped_preview, (224, 224))
            preview_rgb = cv2.cvtColor(preview_resized, cv2.COLOR_BGR2RGB)
            st.image(preview_rgb, caption="🖼️ Cropped Preview (224x224)", channels="RGB")
        else:
            st.warning("⚠️ Cropped area is empty or out of bounds. Please adjust your sliders.")

        # Generate dataset
        if st.button("📦 Generate Dataset from Video"):
            output_dir = tempfile.mkdtemp()
            image_dir = os.path.join(output_dir, "images")
            os.makedirs(image_dir, exist_ok=True)

            labels = []

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            pbar = st.progress(0)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Crop and resize
                crop = frame[y:y+h, x:x+w]
                if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
                    continue  # skip invalid crops

                try:
                    crop_resized = cv2.resize(crop, (224, 224))
                except:
                    continue  # skip if resize fails

                img_pil = Image.fromarray(cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB))

                filename = f"signal_1_frame_{frame_idx:04d}.jpg"
                img_pil.save(os.path.join(image_dir, filename))
                labels.append({"filename": filename, "label": "signal_1"})

                frame_idx += 1
                pbar.progress(min(frame_idx / total_frames, 1.0))

            # Save labels CSV
            labels_df = pd.DataFrame(labels)
            labels_df.to_csv(os.path.join(output_dir, "labels.csv"), index=False)

            # Zip everything
            zip_path = os.path.join(output_dir, "dataset.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for root, _, files in os.walk(output_dir):
                    for file in files:
                        if file != "dataset.zip":
                            zipf.write(os.path.join(root, file),
                                       os.path.relpath(os.path.join(root, file), output_dir))

            st.success("✅ Dataset created!")

            with open(zip_path, "rb") as f:
                st.download_button("📥 Download ZIP", f, file_name="hand_signal_dataset.zip")

            cap.release()
