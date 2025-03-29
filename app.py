import streamlit as st
import cv2
import tempfile
import os
import shutil
import zipfile
from PIL import Image
import numpy as np
import pandas as pd

st.title("🖼️ Synthetic Hand Signal Image Generator")

# Step 1: Upload video
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

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
        # Show first frame to draw bounding box
        st.subheader("Step 1: Draw a bounding box on the first frame")

        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bbox = st.selectbox("Bounding box drawing mode", options=["Manual (click and drag)"], index=0)
        box = st.image(frame_rgb, caption="Draw a box on this frame and click below")

        if 'coords' not in st.session_state:
            st.session_state.coords = None

        if st.button("Click here after drawing bounding box"):
            st.warning("Use OpenCV window in future versions to draw interactively.")

            # TEMP: Ask for manual coordinates (since Streamlit doesn't support interactive box yet)
            x = st.number_input("Top-left X", 0, frame.shape[1])
            y = st.number_input("Top-left Y", 0, frame.shape[0])
            w = st.number_input("Width", 1, frame.shape[1])
            h = st.number_input("Height", 1, frame.shape[0])
            st.session_state.coords = (int(x), int(y), int(w), int(h))
        
        if st.session_state.coords:
            x, y, w, h = st.session_state.coords

            st.subheader("Step 2: Process all frames with the selected crop")

            # Output dirs
            output_dir = tempfile.mkdtemp()
            image_dir = os.path.join(output_dir, "images")
            os.makedirs(image_dir, exist_ok=True)

            labels = []

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            pbar = st.progress(0)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Crop and resize
                crop = frame[y:y+h, x:x+w]
                crop_resized = cv2.resize(crop, (224, 224))
                img_pil = Image.fromarray(cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB))

                # Save image
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
                st.download_button("📦 Download ZIP", f, file_name="hand_signal_dataset.zip")

            # Clean up
            cap.release()
