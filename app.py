import streamlit as st
import cv2
import tempfile
import os
import shutil
import zipfile
from PIL import Image
import numpy as np
import pandas as pd
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Hand Signal Dataset Generator", layout="wide")
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

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",  # Transparent red
            stroke_width=3,
            background_image=frame_pil,
            update_streamlit=True,
            height=frame.shape[0],
            width=frame.shape[1],
            drawing_mode="rect",
            key="canvas",
        )

        if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
            obj = canvas_result.json_data["objects"][0]
            left = int(obj["left"])
            top = int(obj["top"])
            width = int(obj["width"])
            height = int(obj["height"])
            st.session_state.coords = (left, top, width, height)
            st.success(f"Box drawn: x={left}, y={top}, w={width}, h={height}")

        if 'coords' in st.session_state:
            x, y, w, h = st.session_state.coords

            if st.button("📦 Generate Dataset from Video"):
                # Output dirs
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
                    st.download_button("📥 Download ZIP", f, file_name="hand_signal_dataset.zip")

                cap.release()


            with open(zip_path, "rb") as f:
                st.download_button("📦 Download ZIP", f, file_name="hand_signal_dataset.zip")

            # Clean up
            cap.release()
