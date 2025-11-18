# app.py
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
import time
import io
import base64

# Use torch hub to load YOLOv5s
import torch

st.set_page_config(
    page_title="YOLOv5s Object Detection",
    page_icon="🚀",
    layout="wide",
)

st.title("🚀 YOLOv5s Object Detection (Image · Video · Webcam snapshot)")
st.markdown(
    """
Upload an image or video, or take a webcam snapshot. The app runs **YOLOv5s** (pretrained)
to detect objects and returns annotated results with bounding boxes and labels.

Notes:
- For **real-time continuous webcam streaming**, see README — it requires `streamlit-webrtc` (optional).
- Video processing displays frames progressively (can be slow on long videos).
"""
)


@st.cache_resource(show_spinner=False)
def load_model():
    # This loads from torch hub (ultralytics). Requires internet on first run.
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    model.eval()
    return model

model = load_model()

# Utility: convert annotated result (numpy) -> streamlit image bytes
def numpy_to_bytes(img_arr, fmt="JPEG"):
    pil = Image.fromarray(img_arr)
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    byte_im = buf.getvalue()
    return byte_im

# Run detection on a PIL image (returns annotated numpy array and raw results)
def detect_on_image(pil_img):
    # Convert PIL to RGB np array
    img = np.array(pil_img.convert("RGB"))
    # YOLOv5 accepts numpy arrays or lists
    results = model(img)
    # results.render() draws boxes and labels on results.imgs (in-place)
    rendered = results.render()
    annotated = rendered[0]  # numpy array (RGB)
    return annotated, results

# Sidebar: choose mode
mode = st.sidebar.radio("Select input type", ("Image Upload", "Video Upload", "Webcam snapshot"))

# Common options
show_conf = st.sidebar.checkbox("Show confidence in labels", value=True)
draw_thickness = st.sidebar.slider("Box thickness", 1, 6, 2)

# Make model draw settings reflect checkboxes (we'll use results.xyxy for manual draw if needed)
# BUT simpler: use results.render() which already includes labels and conf. For label formatting we can rewrite
# Using ultralytics render, but allow toggling confidence by re-labeling manually if show_conf is False.


if mode == "Image Upload":
    st.header("Image Upload")
    uploaded = st.file_uploader("Upload an image (jpg, png)", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        st.subheader("Original Image")
        st.image(img, use_column_width=True)

        with st.spinner("Running YOLOv5s..."):
            annotated, results = detect_on_image(img)

        # optionally remove confidences from labels
        if not show_conf:
            # rebuild labels without confidences
            # results.names -> dict index->label
            labels = []
            for det in results.xyxy[0]:
                cls = int(det[5].item())
                labels.append(results.names[cls])
            # draw boxes manually without confidences
            annotated = np.array(img.convert("RGB")).copy()
            for det, label in zip(results.xyxy[0], labels):
                x1, y1, x2, y2 = map(int, det[:4].tolist())
                color = (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, draw_thickness)
                cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            # Optionally adjust thickness of rendered boxes
            # results.render already drew; to change thickness would require manual draw — skip for speed.
            pass

        st.subheader("Annotated Image")
        st.image(annotated, use_column_width=True)

        # Download annotated image
        annotated_bytes = numpy_to_bytes(annotated)
        st.download_button(
            label="Download annotated image",
            data=annotated_bytes,
            file_name="annotated_image.jpg",
            mime="image/jpeg",
        )

        # Show table of detections
        st.subheader("Detections")
        if len(results.xyxy[0]) == 0:
            st.info("No objects detected.")
        else:
            dets = []
            for det in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = det.tolist()
                dets.append(
                    {
                        "label": results.names[int(cls)],
                        "confidence": float(conf),
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                    }
                )
            st.table(pd.DataFrame(dets))



elif mode == "Video Upload":
    st.header("Video Upload")
    uploaded_vid = st.file_uploader("Upload a video (mp4, mov). Small files recommended.", type=["mp4", "mov", "avi"])
    max_frames = st.sidebar.slider("Max frames to process (for preview)", 50, 1000, 150)
    if uploaded_vid:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_vid.read())
        tfile.flush()
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        st.write(f"Video FPS: {fps:.2f} — Total frames: {total_frames}")

        preview_placeholder = st.empty()
        progress = st.progress(0)

        processed_frames = []
        frame_count = 0
        pbar_total = min(total_frames, max_frames)

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
         
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         
            results = model(img_rgb)
            results.render()
            annotated = results.imgs[0]  # numpy RGB
          
            preview_placeholder.image(annotated, use_column_width=True)
            processed_frames.append(annotated)
            frame_count += 1
            progress.progress(int(frame_count / pbar_total * 100))
        cap.release()
        progress.empty()
        st.success(f"Processed {frame_count} frames (preview).")

      
        save_gif = st.checkbox("Create GIF of annotated frames (may take time)", value=False)
        if save_gif:
            try:
                import imageio
                gif_path = "annotated_preview.gif"
                imageio.mimsave(gif_path, [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in processed_frames], fps=int(fps))
                with open(gif_path, "rb") as f:
                    st.download_button("Download GIF", f.read(), file_name="annotated_preview.gif", mime="image/gif")
            except Exception as e:
                st.error(f"Failed to create GIF: {e}")

        # Clean temp file
        try:
            os.remove(tfile.name)
        except Exception:
            pass


else:
    st.header("Webcam snapshot (single frame)")
    st.markdown("Use your webcam to take a single snapshot. For continuous webcam streaming, see README for `streamlit-webrtc` instructions.")
    img_file = st.camera_input("Take a snapshot")
    if img_file:
        pil_img = Image.open(img_file)
        st.subheader("Captured Image")
        st.image(pil_img, use_column_width=True)

        with st.spinner("Running YOLOv5s..."):
            annotated, results = detect_on_image(pil_img)

        st.subheader("Annotated Snapshot")
        st.image(annotated, use_column_width=True)

        # Download annotated snapshot
        annotated_bytes = numpy_to_bytes(annotated)
        st.download_button(
            label="Download annotated snapshot",
            data=annotated_bytes,
            file_name="annotated_snapshot.jpg",
            mime="image/jpeg",
        )


        st.subheader("Detections")
        if len(results.xyxy[0]) == 0:
            st.info("No objects detected.")
        else:
            dets = []
            for det in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = det.tolist()
                dets.append(
                    {
                        "label": results.names[int(cls)],
                        "confidence": float(conf),
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                    }
                )
            st.table(pd.DataFrame(dets))


st.markdown("---")
st.markdown(
    "If you want **real-time webcam streaming** (continuous live detection) install `streamlit-webrtc` and use the optional script in the README."
)
