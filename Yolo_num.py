import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

@st.cache_resource
def load_model():
    return YOLO("yolov8n_num.pt")  # Đường dẫn đến model của bạn

model = load_model()

# Xử lý ảnh tĩnh
def detect_image_mode(image_pil, conf_thresh=0.25):
    image = np.array(image_pil)
    results = model.predict(image, conf=conf_thresh, verbose=False)
    names = model.names
    annotated_img = image.copy()
    boxes = results[0].boxes
    if boxes is not None:
        for box, cls, conf in zip(boxes.xyxy.cpu(), boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist()):
            x1, y1, x2, y2 = map(int, box)
            label = f"{names[int(cls)]} {conf:.2f}"
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return annotated_img

# Xử lý camera realtime trong Streamlit
def detect_camera_mode():
    cap = cv2.VideoCapture(0)
    st_frame = st.empty()
    stop_btn = st.button("🛑 Dừng camera")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.25, verbose=False)
        names = model.names
        annotated_img = frame.copy()
        boxes = results[0].boxes
        if boxes is not None:
            for box, cls, conf in zip(boxes.xyxy.cpu(), boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist()):
                x1, y1, x2, y2 = map(int, box)
                label = f"{names[int(cls)]} {conf:.2f}"
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        st_frame.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), channels="RGB")

        if stop_btn:
            break
    cap.release()
