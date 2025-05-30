# Reognition.py

import cv2
import streamlit as st
from PIL import Image
import numpy as np

# Load model như cũ
face_detector = cv2.FaceDetectorYN.create(
    model="face_detection_yunet_2023mar.onnx",
    config="",
    input_size=(320, 320),
    score_threshold=0.9,
    nms_threshold=0.3,
    top_k=5000
)

face_recognizer = cv2.FaceRecognizerSF.create(
    model="arcfaceresnet100-8.onnx",
    config=""
)

# Load known faces
known_faces = []
known_names = []
def face_recognition_image_mode(image):
    st.subheader("📷 Nhận diện khuôn mặt từ ảnh")
    load_known_faces()

    h, w = image.shape[:2]
    face_detector.setInputSize((w, h))
    _, faces = face_detector.detect(image)

    if faces is not None:
        for face in faces:
            x, y, w, h = face[:4].astype(int)
            aligned_face = face_recognizer.alignCrop(image, face)
            feature = face_recognizer.feature(aligned_face)

            name = "Unknown"
            max_sim = 0
            for known_feat, known_name in zip(known_faces, known_names):
                sim = face_recognizer.match(feature, known_feat, cv2.FaceRecognizerSF_FR_COSINE)
                if sim > 0.363 and sim > max_sim:
                    name = known_name
                    max_sim = sim

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return image

def register_face(img_path, name):
    img = cv2.imread(img_path)
    if img is None:
        st.warning(f"❌ Không tìm thấy ảnh tại: {img_path}")
        return
    h, w = img.shape[:2]
    face_detector.setInputSize((w, h))
    _, faces = face_detector.detect(img)
    if faces is not None:
        aligned_face = face_recognizer.alignCrop(img, faces[0])
        feature = face_recognizer.feature(aligned_face)
        known_faces.append(feature)
        known_names.append(name)
    else:
        st.warning(f"⚠️ Không phát hiện khuôn mặt trong ảnh: {name}")


# Đăng ký khuôn mặt
def load_known_faces():
    register_face("Image_train/cuong.jpg", "Cuong")
    register_face("Image_train/trump.jpg", "Trump")
    register_face("Image_train/duy.jpg", "Duy")
    register_face("Image_train/putin.jpg", "Putin")
    register_face("Image_train/jack.jpg", "Jack")
    register_face("Image_train/sontung.jpg", "Sơn Tùng")
    register_face("Image_train/quanglinh.jpg", "Quang Linh")
    register_face("Image_train/Chau.jpg", "Chau")
    register_face("Image_train/CongDanh.jpg", "CongDanh")
    register_face("Image_train/TheDong.jpg", "TheDong")

def face_recognition_streamlit():
    st.subheader("🎥 Nhận diện khuôn mặt từ camera")
    load_known_faces()

    FRAME_WINDOW = st.empty()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Không thể mở camera.")
        return

    st.info("Camera đang chạy... Nhấn Stop để dừng.")

    stop = st.button("⛔ Dừng camera")
    while not stop:
        ret, frame = cap.read()
        if not ret:
            st.warning("Không lấy được khung hình")
            break

        h, w = frame.shape[:2]
        face_detector.setInputSize((w, h))
        _, faces = face_detector.detect(frame)

        if faces is not None:
            for face in faces:
                x, y, w, h = face[:4].astype(int)
                aligned_face = face_recognizer.alignCrop(frame, face)
                feature = face_recognizer.feature(aligned_face)

                name = "Unknown"
                max_sim = 0
                for known_feat, known_name in zip(known_faces, known_names):
                    sim = face_recognizer.match(feature, known_feat, cv2.FaceRecognizerSF_FR_COSINE)
                    if sim > 0.363 and sim > max_sim:
                        name = known_name
                        max_sim = sim

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(rgb_frame)

    cap.release()
    cv2.destroyAllWindows()

