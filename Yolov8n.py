import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

@st.cache_resource
def load_model():
    model = YOLO("yolov8n_trai_cay.pt")  # Thay bằng đường dẫn đến file .pt của bạn
    return model

model = load_model()

def detect_objects(image, conf_thresh=0.25):
    results = model.predict(image, conf=conf_thresh, verbose=False)
    names = model.names
    img_array = np.array(image)
    annotated_img = img_array.copy()
    if results:
        boxes = results[0].boxes
        if boxes is not None:
            for box, cls, conf in zip(boxes.xyxy.cpu(), boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist()):
                x1, y1, x2, y2 = map(int, box)
                label = f"{names[int(cls)]} {conf:.2f}"
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return annotated_img

# ✅ Hàm show() để gọi từ app.py
def show():
    # st.title("🍎 Fruit Detection with YOLOv8n")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    # conf = st.slider("Confidence threshold", 0.05, 1.0, 0.25, 0.01)
    conf = 0.09
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Detect Fruits"):
            with st.spinner("Running YOLOv8n detection..."):
                result_img = detect_objects(image, conf_thresh=conf)
                st.image(result_img, caption="Detected Objects", use_column_width=True)
