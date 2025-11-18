# app.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Smart Vision Detector",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Smart Vision Detector (Cloud-Compatible)")
st.markdown("""
Upload an image and detect objects using YOLOv8.  
No OpenCV GUI needed — fully works on Streamlit Cloud.
""")

# --- Upload Image ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # --- Load YOLOv8 model ---
    st.info("Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")  # tiny model, fast and lightweight

    # --- Run detection ---
    results = model.predict(np.array(img), verbose=False)
    
    # --- Display results ---
    st.subheader("Detection Results")
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        st.write(f"Detected {len(boxes)} objects")
        st.write("Classes:", classes)
        st.write("Confidence scores:", np.round(scores, 2))

        annotated_img = result.plot()  # returns PIL image
        st.image(annotated_img, caption="Detected Objects", use_column_width=True)
