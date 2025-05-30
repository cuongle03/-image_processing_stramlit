import streamlit as st
from PIL import Image
st.set_page_config(page_title="Final Project", layout="wide")

import Yolov8n
import Chapter3
import Chapter4
import Chapter5
import Chapter9
import Reognition
import Nhan_dien_hinh
import Yolo_num

# --- Hiển thị logo và tiêu đề dự án ---
title_col,logo_col, _ = st.columns([1, 4, 1])
with title_col:
    st.markdown("<h1 style='text-align: left; font-size: 30px;'>Final Image Processing Project</h1>", unsafe_allow_html=True)


# Đặt logo ở giữa
with logo_col:
    st.image("logo.jpg", width=600)  # Đặt logo.png vào cùng thư mục với file app.py

# --- Khởi tạo session_state để lưu trạng thái ---
if "current_page" not in st.session_state:
    st.session_state.current_page = ""

# --- Bố cục 2 cột ---
col1, col2 = st.columns([1, 4])

# --- MENU bên trái ---
with col1:
    st.header("🧠 Menu")

    if st.button("🚀 Object Recognition By Yolov8n"):
        st.session_state.current_page = "YOLO"

    if st.button("Fingerprint Recognition with Yolov8n"):
        st.session_state.current_page = "Yolo_num"

    if st.button("📸 Face Recognition"):
        st.session_state.current_page = "FaceRecog"
    if st.button("Shape Classification"):
        st.session_state.current_page = "Shape"

    chapter_3_option = st.selectbox("📚 Chapter 3", ["None","Negative","NegativeColor","Logarit","Gamma","PicewiseLine", "Histogram", "Hist Equal", "Local Histogram", "Histogram Static", "Smooth Box", "Smooth Gaussian", "median filter", "Sharp", "Gradient"])
    if chapter_3_option != "None":
        st.session_state.current_page = f"Chapter3_{chapter_3_option}"

    chapter_4_option = st.selectbox("📚 Chapter 4", ["None", "Spectrum", "Draw Not Filter","Remove Moire Simple", " Draw Period Noise Filter", " Remove Period Noise"])
    if chapter_4_option != "None":
        st.session_state.current_page = f"Chapter4_{chapter_4_option}"

    chapter_5_option = st.selectbox("📚 Chapter 5", ["None", "Create Motion", "DeMotion","DeMotionNoise"])
    if chapter_5_option != "None":
        st.session_state.current_page = f"Chapter5_{chapter_5_option}"

    chapter_9_option = st.selectbox("📚 Chapter 9", ["None", "Erosion", "Dilation","Duality", "Coutour","ConvexHull","Defect_Detect","Hole_Fill","Connected_Components","Remove_Small_Rice"])
    if chapter_9_option != "None":
        st.session_state.current_page = f"Chapter9_{chapter_9_option}"

# --- NỘI DUNG bên phải ---
with col2:
    page = st.session_state.current_page

    if page == "YOLO":
        st.header("🍎 Fruit Detection with YOLOv8n")  # Đổi tiêu đề tại đây
        Yolov8n.show()

    elif page == "Yolo_num":
        st.header("Num Detection with YOLOv8n")
        mode = st.radio("Chọn chế độ", ["📷 Ảnh", "🎥 Camera real-time"])
        if mode == "📷 Ảnh":
            uploaded_file = st.file_uploader("Tải ảnh chứa bàn tay", type=["jpg", "png"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Ảnh gốc", use_column_width=True)

                if st.button("🔍 Detect Num"):
                    result = Yolo_num.detect_image_mode(image)
                    st.image(result, caption="Kết quả nhận diện", use_column_width=True)

        elif mode == "🎥 Camera real-time":
            Yolo_num.detect_camera_mode()


    elif page == "Shape":
        st.header("Shape Classification")  # Đổi tiêu đề tại đây
        Nhan_dien_hinh.show()
        
    elif page.startswith("Chapter3_"):
        st.header(f"📸 Chapter 3 - {page.split('_')[1]}")
        Chapter3.show()

    elif page.startswith("Chapter4_"):
        st.header(f"🧪 Chapter 4 - {page.split('_')[1]}")
        Chapter4.show()

    elif page.startswith("Chapter5_"):
        st.header(f"🔬 Chapter 5 - {page.split('_')[1]}")
        Chapter5.show()

    elif page.startswith("Chapter9_"):
        st.header(f"📈 Chapter 9 - {page.split('_')[1]}")
        Chapter9.show()

    elif page == "FaceRecog":
        st.header("📸 Nhận diện khuôn mặt")
        mode = st.radio("Chọn chế độ", ["📷 Ảnh", "🎥 Camera real-time"])

        if mode == "📷 Ảnh":
            uploaded_file = st.file_uploader("Tải ảnh chứa khuôn mặt", type=["jpg", "png"])
            if uploaded_file is not None:
                import numpy as np
                import cv2

                # Đọc ảnh đã tải lên
                image_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
                image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

                # Lưu ảnh gốc để không thay đổi
                original_image = image.copy()

                # Xử lý ảnh và đánh nhãn
                result = Reognition.face_recognition_image_mode(image)

                # Tạo 2 cột để hiển thị ảnh gốc và ảnh đã xử lý
                col1, col2 = st.columns(2)

                # Cột 1 - Ảnh gốc (Không thay đổi)
                with col1:
                    st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), caption="Ảnh gốc", use_column_width=True)

                # Cột 2 - Ảnh đã xử lý (Đã đánh nhãn)
                with col2:
                    st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Kết quả nhận diện", use_column_width=True)

        elif mode == "🎥 Camera real-time":
            if st.button("🎬 Bắt đầu camera"):
                Reognition.face_recognition_streamlit()


    else:
        st.info("Vui lòng chọn một chức năng từ menu bên trái.") 