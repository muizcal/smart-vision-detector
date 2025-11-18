import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile

st.set_page_config(
    page_title="Smart Vision Detector",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Smart Vision Detector")
st.markdown("""
Upload an image and let the AI detect objects using YOLO.
""")

# Load YOLO model
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")  # You can use yolov8n.pt (small, fast)
    return model

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        image.save(tmp_file.name)
        tmp_path = tmp_file.name

    # Run detection
    results = model(tmp_path)

    # Annotate image
    annotated_image = results[0].plot()
    st.image(annotated_image, caption="Detected Objects", use_column_width=True)

    # Download button
    st.download_button(
        label="Download Annotated Image",
        data=Image.fromarray(annotated_image).tobytes(),
        file_name="annotated_image.png",
        mime="image/png"
    )

st.info("Note: This version works on Streamlit Cloud; live webcam detection is not supported.")
