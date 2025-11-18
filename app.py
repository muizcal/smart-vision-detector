
import streamlit as st
from PIL import Image
import numpy as np
import io
import pandas as pd

from ultralytics import YOLO

st.set_page_config(page_title="Smart Vision Detector (YOLOv8)", page_icon="🤖", layout="wide")
st.title("🤖 Smart Vision Detector — YOLOv8s (Cloud-friendly)")
st.markdown(
    """
Upload an image or take a webcam snapshot. The app runs **YOLOv8s** (ultralytics) to detect objects,
renders annotated results, shows detections in a table, and allows you to download the annotated image.
"""
)




@st.cache_resource(show_spinner=False)
def load_model():


    model = YOLO("yolov8s.pt")
    return model

model = load_model()




def pil_to_bytes(pil_img, fmt="JPEG"):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()

def run_yolo_on_pil(pil_img, conf=0.25):
    """
    Run YOLO on a PIL image. Return annotated PIL image and list of detections.
    """
    img_np = np.array(pil_img.convert("RGB"))
 
    results = model.predict(source=img_np, conf=conf, imgsz=640, verbose=False)
    r = results[0]  # first (and only) result
  
    annotated_np = r.plot()  
    annotated_pil = Image.fromarray(annotated_np)

    dets = []
    if r.boxes is not None and len(r.boxes) > 0:
        for box in r.boxes:
            # box.xyxy, box.conf, box.cls
            xyxy = box.xyxy.cpu().numpy().tolist()[0]
            conf_score = float(box.conf.cpu().numpy())
            cls_id = int(box.cls.cpu().numpy())
            label = model.model.names[cls_id] if hasattr(model, "model") else str(cls_id)
            dets.append({
                "label": label,
                "confidence": round(conf_score, 4),
                "x1": int(xyxy[0]),
                "y1": int(xyxy[1]),
                "x2": int(xyxy[2]),
                "y2": int(xyxy[3]),
            })
    return annotated_pil, dets



st.sidebar.header("Detection Options")
confidence = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.01)
show_table = st.sidebar.checkbox("Show detections table", value=True)
download_format = st.sidebar.selectbox("Download format", ["JPEG", "PNG"])




mode = st.radio("Input mode", ["Upload Image", "Webcam snapshot"], horizontal=True)

if mode == "Upload Image":
    uploaded = st.file_uploader("Upload an image (jpg, png)", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        pil_img = Image.open(uploaded).convert("RGB")
        st.subheader("Original Image")
        st.image(pil_img, use_column_width=True)

        with st.spinner("Running YOLOv8..."):
            annotated_pil, dets = run_yolo_on_pil(pil_img, conf=confidence)

        st.subheader("Annotated Image")
        st.image(annotated_pil, use_column_width=True)

        if show_table:
            if len(dets) == 0:
                st.info("No objects detected.")
            else:
                st.subheader("Detections")
                st.table(pd.DataFrame(dets))


        img_bytes = pil_to_bytes(annotated_pil, fmt=download_format)
        st.download_button(
            label="Download annotated image",
            data=img_bytes,
            file_name=f"annotated.{download_format.lower()}",
            mime=f"image/{download_format.lower()}",
        )

else:  
    st.info("Take a webcam snapshot (single frame). For continuous video streaming use `streamlit-webrtc` — see README.")
    img_file = st.camera_input("Take a snapshot")
    if img_file is not None:
        pil_img = Image.open(img_file).convert("RGB")
        st.subheader("Captured Snapshot")
        st.image(pil_img, use_column_width=True)

        with st.spinner("Running YOLOv8..."):
            annotated_pil, dets = run_yolo_on_pil(pil_img, conf=confidence)

        st.subheader("Annotated Snapshot")
        st.image(annotated_pil, use_column_width=True)

        if show_table:
            if len(dets) == 0:
                st.info("No objects detected.")
            else:
                st.subheader("Detections")
                st.table(pd.DataFrame(dets))

        img_bytes = pil_to_bytes(annotated_pil, fmt=download_format)
        st.download_button(
            label="Download annotated snapshot",
            data=img_bytes,
            file_name=f"annotated_snapshot.{download_format.lower()}",
            mime=f"image/{download_format.lower()}",
        )

# Footer
st.markdown("---")
st.markdown(
    "Notes: model weights download happens automatically on first run. If deployment environment blocks downloads, run locally or upload the weight file to the repo and change the model loader."
)
